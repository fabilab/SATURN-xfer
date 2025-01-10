'''
Created on Nov 7, 2022

@author: Yanay Rosen
@author: Fabio Zanini (2024-2025)
'''


import os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"


import os
import scanpy as sc
import pandas as pd
from anndata import AnnData
import warnings
from builtins import int
warnings.filterwarnings('ignore')
import losses, miners, distances, reducers, testers
import torch
import torch.optim as optim
import numpy as np
import utils.logging_presets as logging_presets
import record_keeper
from data.gene_embeddings import load_gene_embeddings_adata, load_gene_embeddings
from data.multi_species_data import ExperimentDatasetMulti, multi_species_collate_fn, ExperimentDatasetMultiEqualCT
from data.multi_species_data import ExperimentDatasetMultiEqual

from model.saturn_model import SATURNPretrainModel, SATURNMetricModel, TransferModel, make_centroids, score_genes_against_centroids
import torch.nn.functional as F
from tqdm import trange, tqdm
from pretrain_utils import *

import argparse
from datetime import datetime
from pathlib import Path
import sys
sys.path.append('../')

# SATURN
from sklearn.cluster import KMeans
from scipy.stats import rankdata
import pickle
from copy import deepcopy
from utils import stop_conditions
import random


def train(
    model, loss_func, mining_func, device,
    train_loader,
    optimizer,
    epoch,
    mnn, 
    sorted_species_names,
    guide_species,
    use_ref_labels=False,
    indices_counts={},
    equalize_triplets_species=False,
):
    '''
    Train one epoch for a model with Transfer Learning
    
    Keyword arguments:
    model -- the pretrain model, class is saturnMetricModel
    loss_func -- the loss function, cosine similarity distance
    mining_func -- mining function, triplet margin miner
    device -- the current torch device
    train_loader -- train loader, returns the macrogene values and label categorical codes
    optimizer -- torch optimizer for model
    epoch -- current epoch
    mnn -- use mutual nearest neighbors for metric learning mining
    sorted_species_names -- names of the species that are being aligned
    use_ref_labels -- if metric learning should increase supervision by using shared coarse labels, stored in the ref labels values
    equalize_triplets_species -- if metric learning should mine triples in a balanced manner from each species
    
    '''
    
    # Set the model in training mode, i.e. generating the autograds (only for the free parameters)
    model.train()
    torch.autograd.set_detect_anomaly(True)
    for batch_idx, batch_dict in enumerate(train_loader):
        # flatten grad for each batch, i.e. optimise them independently as if distinct epochs
        optimizer.zero_grad()
        embs = []
        labs = []
        spec = []
        ref_labs = []

        # NOTE: during xfer learning, we have two species, a new to be embedded and a guide which is already embedded
        for species, (data, labels, ref_labels, _) in batch_dict.items():
            if data is None:
                continue
            data, labels, ref_labels = data.to(device), labels.to(device), ref_labels.to(device)
            
            # Guide species is already embedded
            if species == guide_species:
                embeddings = data
            else:
                # We don't need the macrogene space info during training
                # NOTE: this would change if we "unfreeze" the macrogenes, of course
                _, embeddings = model(data, species)
            embeddings = F.normalize(embeddings)
            embs.append(embeddings)
            labs.append(labels)
            ref_labs.append(ref_labels)
            spec.append(np.argmax(np.array(sorted_species_names) == species) * torch.ones_like(labels))

        # Concat the embedding and metadta from both guide and new species
        embeddings = torch.cat(embs)
        labels = torch.cat(labs)
        ref_labels = torch.cat(ref_labs)
        species = torch.cat(spec)

        # Make indices_tuple to instruct the triplet loss
        if use_ref_labels:
            indices_tuple = mining_func(embeddings, labels, species, mnn=mnn, ref_labels=ref_labels)
        else:
            indices_tuple = mining_func(embeddings, labels, species, mnn=mnn)
            
        indices_mapped = [labels[i] for i in indices_tuple] # map to labels for only the purpose of writing to triplets file
        for j in range(len(indices_mapped[0])):
            key = f"{indices_mapped[0][j]},{indices_mapped[1][j]},{indices_mapped[2][j]}"
            indices_counts[key] = indices_counts.get(key, 0) + 1

        # Compute loss function based on the batch's embeddings (output) and metadata (labels)
        loss = loss_func(embeddings, labels, indices_tuple, embs_list=embs)
        
        # This is a special option in case one worries about inbalances. Honestly the model has
        # worse issues with that in the autobatcher ATM.
        if equalize_triplets_species:
            species_mapped = [species[i] for i in indices_tuple] # a,p,n species vectors
            a_spec = species_mapped[0]
            p_spec = species_mapped[1]
            a_uq, a_inv, a_ct = torch.unique(a_spec, return_counts=True, return_inverse=True)
            p_uq, p_inv, p_ct = torch.unique(p_spec, return_counts=True, return_inverse=True)
            
            a_prop = a_ct / torch.sum(a_ct) # Proportions of total ie 1/4
            p_prop = p_ct / torch.sum(p_ct) # Proportions of total ie 3/4
            
            a_balance = torch.reciprocal(a_prop) / len(a_prop) # balancing ie * 4 / 1, then divide by num species
            p_balance = torch.reciprocal(p_prop) / len(p_prop) # balancing ie 4 / 3, then divide by num species        
            
            a_bal_inv = a_balance[a_inv]
            p_bal_inv = p_balance[p_inv]
            
            # WEIGHT THE LOSS BY THE NUMBER OF TRIPLETS MINED PER SPECIES
            loss = torch.mul(torch.mul(loss, a_bal_inv), p_bal_inv).mean()                                         
                                                       
        # ...and finally (auto)compute the backward gradient to update the model parameters
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print("Epoch {} Iteration {}: Loss = {}, Number of mined triplets "
                  "= {}".format(epoch, batch_idx, loss,
                                mining_func.num_triplets))


def get_all_embeddings(dataset, model, device, use_batch_labels=False, obs_names=None):
    '''
    Get the embeddings and other metadata for a pretraining model.

    Keyword arguments:
    model -- the pretrain model, class is saturnPretrainModel
    dataset -- count values
    use_batch_labels -- if we add batch labels as a categorical covariate
    obs_names: sanity check argument to verify the order of cells is correct  # FIXME: added by @iosonofabio
    '''
    test_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=multi_species_collate_fn,
        batch_size=1024,
        shuffle=False,
    )

    # NOTE: this is inference time, so we use a combo of model.eval and torch.no_grad
    model.eval()
    embs = []
    macrogenes = []
    labs = []
    spec = []
    refs = []
    if use_batch_labels:
        batch_labs = []
    
    # This is inference, not training, so we don't need gradients (in fact, we want the weights frozen)
    with torch.no_grad():
        for batch_idx, batch_dict in tqdm(enumerate(test_loader), total=len(test_loader)):
            for species, (data, labels, ref_labels, batch_labels) in batch_dict.items():
                if data is None:
                    continue
                if use_batch_labels:
                    data, labels, ref_labels, batch_labels = data.to(device), labels.to(device), \
                                                             ref_labels.to(device), batch_labels.to(device)
                    encoder_inputs, encodeds, mus, log_var, px_rate, px_r, px_drop = model(data, species, batch_labels)
                else:
                    data, labels, ref_labels = data.to(device), labels.to(device), ref_labels.to(device)

                    # NOTE: This is where inference is actually carried out (!)
                    # Calling the model looks like it's calling the forward function, wrapped in some no_grad magic
                    # NOTE: this is always ONE species at a time.
                    encoder_inputs, encodeds = model(data, species)

                # These are the outputs of the model and how we store them:
                # 1. the encoded output (the embedding)
                # 2. the projection onto macrogenes (encoder input)
                # 3. metadata (species, labels, barch if requested)
                for encoded in encodeds:
                    embs.append(encoded.detach().cpu())
                for encoder_input in encoder_inputs:
                    macrogenes.append(encoder_input.detach().cpu())

                spec += [species] * data.shape[0]
                labs = labs + list(labels.cpu().numpy())
                refs = refs + list(ref_labels.cpu().numpy())
                
                if use_batch_labels:
                    batch_labs = batch_labs + list(batch_labels.cpu().numpy())

    # ... and at the end of the day, we stack vertically everything and return
    if use_batch_labels:
        return torch.stack(embs).cpu().numpy(), np.array(labs), np.array(spec), torch.stack(macrogenes),\
                                                                np.array(refs), np.array(batch_labs)
    else:
        return torch.stack(embs).cpu().numpy(), np.array(labs), np.array(spec), torch.stack(macrogenes), np.array(refs)


def create_output_anndata(train_emb, train_lab, train_species, train_macrogenes, train_ref, celltype_id_map, reftype_id_map, use_batch_labels=False, batchtype_id_map=None, train_batch=None, obs_names=None):
    '''
    Create an AnnData from SATURN results
    '''
    adata = AnnData(train_emb)
    labels = train_lab.squeeze()
    id2cell_type = {v:k for k,v in celltype_id_map.items()}
    adata.obs['labels'] = pd.Categorical([id2cell_type[int(l)]
                                           for l in labels])
    adata.obs['labels2'] = pd.Categorical([l.split('_')[-1]
                                           for l in adata.obs['labels']])
    
    
    ref_labels = train_ref.squeeze()
    id2ref_type = {v:k for k,v in reftype_id_map.items()}
    adata.obs['ref_labels'] = pd.Categorical([id2ref_type[int(l)]
                                           for l in ref_labels])
    
    adata.obs['species'] = pd.Categorical(train_species)
    
    adata.obsm["macrogenes"] = train_macrogenes
    if use_batch_labels:
        batch_labels = train_batch.squeeze()
        id2batch_type = {v:k for k,v in batchtype_id_map.items()}
        adata.obs['batch_labels'] = pd.Categorical([id2batch_type[int(l)]
                                               for l in batch_labels])
    if obs_names is not None:
        adata.obs_names = obs_names
    return adata

def inferrer(args):
    '''
    Runs the inference pipeline
    '''

    # Set some infra things
    hooks = logging_presets.get_hook_container(record_keeper)
    device = torch.device(args.device)
    dt = str(datetime.now())[5:19].replace(' ', '_').replace(':', '-')

    print("Load data")
    species_to_path = {args.species: args.in_adata_path}
    species_to_adata = {species:sc.read(path) for species,path in species_to_path.items()}
    species_to_embedding_paths = {args.species: args.in_embeddings_path}
        
    print("Preprocess data")
    if args.tissue_subset is not None:
        for species in species_to_adata.keys():
            ad = species_to_adata[species]
            species_to_adata[species] = ad[ad.obs[args.tissue_column] == args.tissue_subset]
            if species_to_adata[species].X.shape[0] == 0:
                raise ValueError(f"No cells in {args.tissue_subset} found in the input dataset")
    
    if args.in_label_col is None:
        # Assume in_label_col is set as the in_label_col column in the run DF 
        species_to_label = {args.species: args.in_label_col}
        species_to_label_col = {species:col for species,col in species_to_label.items()}
        
    # Add species to celltype name
    # NOTE: it is important to keep alphabetical order here because this list will later be joined with
    # get_all_embeddings which follows pytorch's default auto-batcher, which iterates using alphabetical order
    # added by @iosonofabio
    sorted_species_names = sorted(species_to_adata.keys())
    all_obs_names = []
    for species in sorted_species_names:
        adata = species_to_adata[species]
        adata_label = args.in_label_col
        if args.in_label_col is None:
            adata_label = species_to_label_col[species]
        species_str = pd.Series([species] * adata.obs.shape[0])
        species_str.index = adata.obs[adata_label].index
        adata.obs["species"] = species_str
        species_specific_celltype = species_str.str.cat(adata.obs[adata_label], sep="_")
        adata.obs["species_type_label"] = species_specific_celltype
        all_obs_names += list(adata.obs_names)
    
    print("Create truth, ref, and batch labels (as requested)")
    # Create mapping from cell type to ID
    unique_cell_types = set()
    for adata in species_to_adata.values():
        unique_cell_types = (unique_cell_types | set(adata.obs["species_type_label"]))
    unique_cell_types = sorted(unique_cell_types)
    celltype_id_map = {cell_type: index for index, cell_type in enumerate(unique_cell_types)}
    for adata in species_to_adata.values():
        adata.obs["truth_labels"] = pd.Categorical(
            values=[celltype_id_map[cell_type] for cell_type in adata.obs["species_type_label"]]
        )

    # If we are using batch labels, add them as a column in our output anndatas and pass them as a categorical covariate to pretraining
    num_batch_labels = 0
    use_batch_labels = args.non_species_batch_col is not None
    if use_batch_labels:
        unique_batch_types = set()
        for adata in species_to_adata.values():
            unique_batch_types = (unique_batch_types | set(adata.obs[args.non_species_batch_col]))

        unique_batch_types = sorted(unique_batch_types)
        batchtype_id_map = {batch_type: index for index, batch_type in enumerate(unique_batch_types)}
        for adata in species_to_adata.values():
            adata.obs["batch_labels"] = pd.Categorical(
                values=[batchtype_id_map[batch_type] for batch_type in adata.obs[args.non_species_batch_col]]
            )
        num_batch_labels = len(unique_batch_types)
        print(f"Using Batch Labels, {num_batch_labels}")
                
    # make the ref labels column categorical and mapped
    unique_ref_types = set()
    for adata in species_to_adata.values():
        unique_ref_types = (unique_ref_types | set(adata.obs[args.ref_label_col]))
    unique_ref_types = sorted(unique_ref_types)
    reftype_id_map = {ref_type: index for index, ref_type in enumerate(unique_ref_types)}
    for adata in species_to_adata.values():
        adata.obs["ref_labels"] = pd.Categorical(
            values=[reftype_id_map[ref_type] for ref_type in adata.obs[args.ref_label_col]]
        )
  
    # Load gene embeddings (which also requires filtering data genes to those with embeddings)
    species_to_gene_embeddings = {}
    for species, adata in species_to_adata.items():
        adata, species_gene_embeddings = load_gene_embeddings_adata(
            adata=adata,
            species=[species],
            embedding_model=args.embedding_model,
            embedding_path=species_to_embedding_paths[species]
        )

        species_to_gene_embeddings.update(species_gene_embeddings)
        species_to_adata[species] = adata
        print("After loading the anndata", species, adata)
    
    sorted_species_names = sorted(species_to_gene_embeddings.keys())
    
    print("Filter high-variable genes (hard feature selection)")
    # NOTE: perhaps there would be a better way to do this, using marker genes or Fatemeh's algo
    species_to_gene_idx_hv = {}
    ct = 0
    for species in sorted_species_names:
        adata = species_to_adata[species]
        if use_batch_labels:
            sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=args.hv_genes, \
                                        batch_key=args.non_species_batch_col, span=args.hv_span)  # Expects Count Data
        else:
            sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=args.hv_genes)  # Expects Count Data
        hv_index = adata.var["highly_variable"]
        species_to_adata[species] = adata[:, hv_index]
        species_to_gene_embeddings[species] = species_to_gene_embeddings[species][hv_index]
        species_to_gene_idx_hv[species] = (ct, ct+species_to_gene_embeddings[species].shape[0])
        ct+=species_to_gene_embeddings[species].shape[0]
        
    # List of species_ + gene_name for macrogenes
    all_gene_names = []
    for species in sorted_species_names:
        adata = species_to_adata[species]
        species_str = pd.Series([species] * adata.var_names.shape[0])
        gene_names = pd.Series(adata.var_names)
        all_gene_names += list(species_str.str.cat(gene_names, sep="_"))
    
    # stacked embeddings
    X = torch.cat([species_to_gene_embeddings[species] for species in sorted_species_names])
    
    print("Load the centroid weights")
    with open(args.centroids_init_path, "rb") as f:
        tmp = pickle.load(f)
        species_genes_scores_trained = tmp['scores']
        centroids_coords = tmp['centroids']
        centroid_score_func = tmp.get('score_func', args.centroid_score_func)
        sorted_species_names_trained = tmp['sorted_species_names']
        species_to_gene_idx_hv_trained = tmp['species_to_gene_idx_hv']
        # These are the genes from all species on which the model was trained,
        # which inculde only HVGs anyway
        all_gene_names_trained = tmp['all_gene_names']
        del tmp
    print("Loaded centroids and existing scores")

    print("Score genes of new species against cluster centers")
    species_genes_scores = score_genes_against_centroids(
        X, centroids_coords, all_gene_names, score_function=centroid_score_func,
    )
    
    # Initialize macrogenes, i.e. the weights from the initial gene embeddings to the centroid space
    # The initial (i.e. before pretraining) weights are the scores/distances of the genes from each centroid
    centroid_weights_trained = torch.stack([torch.tensor(species_genes_scores_trained[gn]) for gn in all_gene_names_trained])
    centroid_weights = torch.stack([torch.tensor(species_genes_scores[gn]) for gn in all_gene_names])

    if args.guide_species is not None:
        species_closest = args.guide_species
    else:
        print("Find the closest species within the training set")
        metric = {}
        for species in sorted_species_names_trained:
            centroid_weights_trained_species = torch.stack(
                [torch.tensor(species_genes_scores_trained[gn]) for gn in all_gene_names_trained if gn.startswith(species)],
            )
            cdist_species = torch.cdist(centroid_weights, centroid_weights_trained_species)
            dis_closest = cdist_species.min(axis=1).values
            nconserved = min(len(dis_closest), 50)
            dis_mean = dis_closest[:nconserved].mean()
            metric[species] = float(dis_mean)
        metric = pd.Series(metric)
        species_closest = metric.idxmin()
        del metric, centroid_weights_trained_species, cdist_species, dis_closest, nconserved,  dis_mean
        print(f"Closest species within the training set: {species_closest}.")

    print("Connect genes of new species with genes of closest species for inference")
    hvgs_guide_species = [gn for gn in all_gene_names_trained if gn.startswith(species_closest)]
    # Strip the "species-" part in front of each gene name
    hvgs_guide_plain = [x[len(species_closest) + 1:] for x in hvgs_guide_species]

    # Match genes against HVGs in the closest training species, using the original embeddings
    # NOTE: we need the anndata to match the indices because the embedding loading function
    # returns a tensor, not a dictionary
    # TODO: we could load only the genes we want, that'd be easier
    embedding_path_closest = pd.read_csv(args.training_csv_path, index_col="species").at[species_closest, "embedding_path"]
    gene_embedding_dict_guide = load_gene_embeddings(
        species=species_closest,
        genes=hvgs_guide_plain,
        embedding_model=args.embedding_model,
        embedding_path=embedding_path_closest,
    )
    gene_embeddings_closest_hvg = torch.tensor([gene_embedding_dict_guide[gn] for gn in hvgs_guide_plain])

    # Get the matix matching genes to the guide species genes. "one-hot" means winner takes all (two 1st prizes for equal distance)
    # This is the only one of the three currently used metrics that becomes exact in the limit of inferring a species that is
    # already in the database. Otherwise one could write new scoring systems that diverge for d -> 0+, so that upon Normalisation
    # (next line of code) only the closest match wins.
    # NOTE: For exact species matches, this is a sparse matrix (1-1 matching of genes). For nonexact matches, it will be dense.
    matrix_match_gene_guides = score_genes_against_centroids(
        species_to_gene_embeddings[args.species],
        gene_embeddings_closest_hvg,
        all_gene_names,
        return_dict=False,
        score_function="one_hot",
    )
    # Normalise the scores to one, to maintain overall gene expression onto the new species
    matrix_match_gene_guides_norm = (matrix_match_gene_guides.T / matrix_match_gene_guides.sum(axis=1)).T
    
    if use_batch_labels:
        sorted_batch_labels_names=list(unique_batch_types)
    else:
        sorted_batch_labels_names = None

    print("***STARTING TRANSFER LEARNING***")
    # Set a few infra things
    df_names = []
    for species in sorted_species_names:
        p = species_to_path[species]
        df_names += [p.split('/')[-1].split('.h5ad')[0]]
    args.dir_ += '_'.join(df_names) + \
                ('_org_'+ str(args.org) if args.org is not None else '') +\
                ('_'+dt if args.time_stamp else '') +\
                ('_'+args.tissue_subset if args.tissue_subset else '') +\
                ('_seed_'+str(args.seed))
    metric_dir = Path(args.work_dir)
    metric_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.dir_.split(args.log_dir)[-1]

    if args.unfreeze_macrogenes:
        print("***MACROGENE WEIGHTS UNFROZEN***")
        raise NotImplementedError("Unfreezing macrogenes is not implemented yet")

    # Load the xfer_model parameters from:
    # 1. the manual gene -> guide gene approach above (guide_weights, guide_layer_norm)
    # 2. the trained pretrain model (p_weights, cl_layer_norm)
    # 3. the trained metric model (encoder)
    xfer_state_dict = {}
     # Copy the gene -> guide gene weights from above. This needs not be diagonal even for identical
    # species due to ordering of genes (the projection is onto HVG space of the target species, because
    # those are the only genes with a macrogene weight)..
    xfer_state_dict['guide_weights'] = torch.tensor(matrix_match_gene_guides_norm.T)
    # leave an open multiplier for the guide gene space, decide later whether to freeze it or not
    xfer_state_dict['guide_layer_norm.weight'] = torch.ones(matrix_match_gene_guides_norm.shape[1])
    xfer_state_dict['guide_layer_norm.bias'] = torch.zeros(matrix_match_gene_guides_norm.shape[1])   # The latter could be further trained later on with an appropriate triplet loss.
    # Copy the guide gene -> macrogene weights and normalistion bias for all species
    pretrain_state_dict = torch.load(args.pretrain_model_path)
    xfer_state_dict['p_weights'] = deepcopy(pretrain_state_dict['p_weights'])
    xfer_state_dict['cl_layer_norm.weight'] = deepcopy(pretrain_state_dict['cl_layer_norm.weight'])
    xfer_state_dict['cl_layer_norm.bias'] = deepcopy(pretrain_state_dict['cl_layer_norm.bias'])
    # Copy the encoder parameters from either the pretrain or the metric model
    if args.encoder == 'pretrain':
        encoder_state_dict = pretrain_state_dict
    else:
        encoder_state_dict = torch.load(args.metric_model_path)
    for key, val in encoder_state_dict.items():
        if key.startswith('encoder'):
            xfer_state_dict[key] = deepcopy(val)

    xfer_model = TransferModel(
        input_dim=species_to_adata[args.species].n_vars,
        num_macrogenes=train_macrogenes.shape[1],
        dropout=0.1,
        hidden_dim=hidden_dim,
        embed_dim=model_dim,
        guide_species=species_closest,
        species_to_gene_idx=species_to_gene_idx_hv_trained,
        vae=args.vae,
    )
    xfer_model.load_state_dict(xfer_state_dict)

    print("***ZERO-SHOT INFERENCE***")
    if use_batch_labels: # we have a batch column to use for the pretrainer
        xfer_dataset = ExperimentDatasetMultiEqual(
            all_data = species_to_adata,
            all_ys = {species:adata.obs["truth_labels"] for (species, adata) in species_to_adata.items()},
            all_refs = {species:adata.obs["ref_labels"] for (species, adata) in species_to_adata.items()},
            all_batch_labs = {species:adata.obs["batch_labels"] for (species, adata) in species_to_adata.items()}
        )
        zeroshot_emb, zeroshot_lab, zeroshot_species, zeroshot_macrogenes, zeroshot_ref, zeroshot_batch = get_all_embeddings(
            xfer_dataset, xfer_model, device, use_batch_labels,
        )
        adata_zeroshot = create_output_anndata(
            zeroshot_emb, zeroshot_lab, zeroshot_species,
            zeroshot_macrogenes.cpu().numpy(), zeroshot_ref,
            celltype_id_map, reftype_id_map, use_batch_labels, batchtype_id_map, zeroshot_batch, obs_names=all_obs_names,
        )
    else:
        xfer_dataset = ExperimentDatasetMultiEqual(
            all_data = species_to_adata,
            all_ys = {species:adata.obs["truth_labels"] for (species, adata) in species_to_adata.items()},
            all_refs = {species:adata.obs["ref_labels"] for (species, adata) in species_to_adata.items()},
            all_batch_labs = {}
        )
        zeroshot_emb, zeroshot_lab, zeroshot_species, zeroshot_macrogenes, zeroshot_ref = get_all_embeddings(
            xfer_dataset, xfer_model, device, use_batch_labels,
        )
        adata_zeroshot = create_output_anndata(
            zeroshot_emb, zeroshot_lab, zeroshot_species,
            zeroshot_macrogenes.cpu().numpy(), zeroshot_ref,
            celltype_id_map, reftype_id_map, obs_names=all_obs_names,
        )
    # FIXME: do we still need this one?
    adata_zeroshot.obs['species'] = args.species

    if len(run_name) > 50:
        zeroshot_adata_fn = 'inference_zeroshot.h5ad'
    else:
        zeroshot_adata_fn = f'{run_name}_inference_zeroshot.h5ad'
    zeroshot_adata_path = metric_dir / zeroshot_adata_fn
    adata_zeroshot.write(zeroshot_adata_path)

    print(f"Zero-shot AnnData Path: {zeroshot_adata_path}")

     # Write outputs to file
    if (not args.train) and (args.xfer_model_path is not None):
        # Save the transfer model if asked to
        print(f"Saving Transfer Model to {args.xfer_model_path}")
        torch.save(xfer_model.state_dict(), args.xfer_model_path)

    xfer_model.to(device)

    # If requested, train the xfer model
    # NOTE: Only the first encoder, which projects onto the closest species, is trained here.
    if args.train:

        # NOTE: The key thing is that to make the triplet loss meaningful, at least unidirectionally,
        # we must run the metric (xfer) training with at least two species (the guide species is the
        # obvious candidate AND also shuffle to ensure both species are found in EACH batch. For now
        # we just use autobatching as the rest of SATURN but we'll have to fix both anyway
        if args.trained_adata_path is None:
            raise ValueError("Must provide a trained adata path for training the transfer model.")

        adata_trained = sc.read(args.trained_adata_path)
        # NOTE: this is a sanity check to ensure that the guide species is in the training set
        if species_closest not in adata_trained.obs['species'].unique():
            raise ValueError(f"Guide species {species_closest} not found in training set.")

        # Restrict the adata to the guide species
        adata_guide = adata_trained[adata_trained.obs['species'] == species_closest]
        adata_guide.obs['species_type_label'] = adata_guide.obs['labels']
        adata_guide.obs[args.ref_label_col] = adata_guide.obs['ref_labels']

        # Build dataset for data loader.
        # NOTE: the new data is in HVG space, the guide one in metric embedding space, but that's taken care of
        # in the train function: only the new data is passed through the model.
        species_to_adata_xfer = {
            args.species: species_to_adata[args.species],
            species_closest: adata_guide,
        }
        sorted_species_names_xfer = sorted(species_to_adata_xfer.keys())

        # Create the "truth_labels" column which is an integer representation of all cell types
        unique_cell_types_xfer = set()
        for adata in species_to_adata_xfer.values():
            unique_cell_types_xfer = (unique_cell_types_xfer | set(adata.obs["species_type_label"]))
        unique_cell_types_xfer = sorted(unique_cell_types_xfer)
        celltype_id_map_xfer = {cell_type: index for index, cell_type in enumerate(unique_cell_types_xfer)}
        for adata in species_to_adata_xfer.values():
            adata.obs["truth_labels"] = pd.Categorical(
                values=[celltype_id_map_xfer[cell_type] for cell_type in adata.obs["species_type_label"]]
            )

        # Create the "ref_labels" column which is also an integer representation of all cell types
        unique_ref_types_xfer = set()
        for adata in species_to_adata_xfer.values():
            unique_ref_types_xfer = (unique_ref_types_xfer | set(adata.obs[args.ref_label_col]))
        unique_ref_types_xfer = sorted(unique_ref_types_xfer)
        reftype_id_map_xfer = {ref_type: index for index, ref_type in enumerate(unique_ref_types_xfer)}
        for adata in species_to_adata_xfer.values():
            adata.obs["ref_labels"] = pd.Categorical(
                values=[reftype_id_map_xfer[ref_type] for ref_type in adata.obs[args.ref_label_col]]
            )

        # Make the guide species data loader
        if use_batch_labels: # we have a batch column to use for the pretrainer
            train_dataset = ExperimentDatasetMultiEqual(
                all_data = species_to_adata_xfer,
                all_ys = {species:adata.obs["truth_labels"] for (species, adata) in species_to_adata_xfer.items()},
                all_refs = {species:adata.obs["ref_labels"] for (species, adata) in species_to_adata_xfer.items()},
                all_batch_labs = {species:adata.obs["batch_labels"] for (species, adata) in species_to_adata_xfer.items()}
            )
        else:
            train_dataset = ExperimentDatasetMultiEqual(
                all_data = species_to_adata_xfer,
                all_ys = {species:adata.obs["truth_labels"] for (species, adata) in species_to_adata_xfer.items()},
                all_refs = {species:adata.obs["ref_labels"] for (species, adata) in species_to_adata_xfer.items()},
                all_batch_labs = {}
            )
        # Load data with shuffling
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            collate_fn=multi_species_collate_fn,
            batch_size=1024,
            shuffle=True,
        )

        # freeze everything except the first encoder (gene -> guide gene)
        for parameter in xfer_model.parameters():
            parameter.requires_grad = False
        xfer_model.guide_weights.requires_grad = True
        # TODO: decide whether to free this layer
        xfer_model.guide_layer_norm.requires_grad = True
        # Create the optimizer
        optimizer = optim.Adam(xfer_model.parameters(), lr=args.metric_lr)

        distance = distances.CosineSimilarity()
        reducer = reducers.ThresholdReducer(low = 0)

        # TripletMarginMMDLoss
        loss_func = losses.TripletMarginLoss(
            margin=0.2,
            distance=distance,
            reducer=reducer,
        )

        mining_func = miners.TripletMarginMiner(
            margin=0.2,
            distance=distance,
            type_of_triplets="semihard",
            miner_type="cross_species",
        )

        print("***STARTING FINE-TUNING***")
        all_indices_counts = pd.DataFrame(columns=["Epoch", "Triplet", "Count"])
        for epoch in range(1, args.epochs+1):
            epoch_indices_counts = {}
            train(
                xfer_model,
                loss_func,
                mining_func,
                device,
                train_loader,
                optimizer,
                epoch,
                args.mnn,
                sorted_species_names_xfer,
                species_closest,
                use_ref_labels=args.use_ref_labels,
                indices_counts=epoch_indices_counts,
                equalize_triplets_species=args.equalize_triplets_species,
            )

            # Collect output metrics for this one epoch
            epoch_df = pd.DataFrame.from_records(list(epoch_indices_counts.items()), columns=["Triplet", "Count"])
            epoch_df["Epoch"] = epoch
            all_indices_counts = pd.concat((all_indices_counts, epoch_df))

        if args.xfer_model_path is not None:
            # Save the transfer model if asked to
            print(f"Saving trained Transfer Model to {args.xfer_model_path}")
            torch.save(xfer_model.state_dict(), args.xfer_model_path)

    print("***FINE-TUNED INFERENCE***")
    if use_batch_labels:
        train_emb, train_lab, train_species, train_macrogenes, train_ref, train_batch = get_all_embeddings(
            xfer_dataset, xfer_model, device, use_batch_labels,
        )
        adata = create_output_anndata(
            train_emb, train_lab, train_species,
            train_macrogenes.cpu().numpy(), train_ref,
            celltype_id_map, reftype_id_map, use_batch_labels, batchtype_id_map, train_batch, obs_names=all_obs_names,
        )
    else:
        train_emb, train_lab, train_species, train_macrogenes, train_ref = get_all_embeddings(
            xfer_dataset, xfer_model, device, use_batch_labels,
        )
        adata = create_output_anndata(
            train_emb, train_lab, train_species,
            train_macrogenes.cpu().numpy(), train_ref,
            celltype_id_map, reftype_id_map, obs_names=all_obs_names,
        )

    if len(run_name) > 50:
        final_adata_fn = "finetuned_adata.h5ad"
    else:
        final_adata_fn = f'{run_name}_finetuned.h5ad'
    final_path = metric_dir / final_adata_fn
    adata.write(final_path)
    print(f"Fine-tuned AnnData Path: {final_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Set model hyperparametrs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Run Setup
    parser.add_argument('--in_adata_path', type=str, required=True,
                        help='Path to h5ad of the data to infer')
    parser.add_argument('--in_embeddings_path', type=str, required=True,
                        help='Path to embeddings summary pt file for the peptides of the species to infer')
    parser.add_argument('--training_csv_path', type=str, required=True,
                        help='Path to csv containing training adata and embedding paths and species names')
    parser.add_argument('--species', type=str, help="Data to infer belongs to this species (optional)")
    parser.add_argument('--guide-species', type=str, help="Choose a guide species to project gene expression onto. Default is to take the one with the closest embeddings for conserved proteins.")
    parser.add_argument('--device', type=str,
                    help='Set GPU/CPU')
    parser.add_argument('--device_num', type=int,
                        help='Set GPU Number')
    parser.add_argument('--time_stamp', type=bool,
                        help='Add time stamp in file name')
    parser.add_argument('--org', type=str,
                        help='Add organization to filename')
    parser.add_argument('--log_dir', type=str,
                        help='Log directory')
    parser.add_argument('--work_dir', type=str,
                        help='Working directory')
    parser.add_argument('--seed', type=int,
                        help='Init Seed')
    parser.add_argument('--in_label_col', type=str,
                        help='Label column for input data')
    parser.add_argument('--ref_label_col', type=str,
                        help='Reference label column for input data')
    parser.add_argument('--tissue_subset', type=str,
                        help='Subset the input anndatas by the column args.tissue_column to just be this tissue')
    parser.add_argument('--tissue_column', type=str,
                        help='When subsetting the input anndatas by the column, use this column name.')
    
    # SATURN Setup
    parser.add_argument('--hv_genes', type=int,
                        help='Number of highly variable genes')
    parser.add_argument('--hv_span', type=float,
                        help='Fraction of cells to use when calculating highly variable genes, scanpy defeault is 0.3.')
    parser.add_argument('--centroids_init_path', type=str, required=True,
                        help='Path to existing centroids pretraining weights, or location to save to.')
    parser.add_argument('--embedding_model', type=str, choices=['ESM1b', 'MSA1b', 'protXL', 'ESM1b_protref', 'ESM2'],
                    help='Gene embedding model whose embeddings should be loaded if using gene_embedding_method')
    parser.add_argument('--centroid_score_func', type=str, choices=['default', 'one_hot', 'smoothed'],
                    help='Gene embedding model whose embeddings should be loaded if using gene_embedding_method')
    
    # Model Setup
    parser.add_argument('--hidden_dim', type=int,
                        help='Model first layer hidden dimension')
    parser.add_argument('--model_dim', type=int,
                        help='Model latent space dimension')
    
    # Metric Learning Arguments
    parser.add_argument('--use_ref_labels', type=bool, nargs='?', const=True, 
                    help='Use reference labels when aligning')
    parser.add_argument('--batch_size', type=int,
                        help='batch size')
    parser.add_argument('--equalize_triplets_species', type=bool, nargs='?', const=True,
                        help='Balance species\' weighting in the metric learning model')
    parser.add_argument('--encoder', type=str, choices=['pretrain', 'metric'],
                        help='Which encoder to use for the transfer model. Default is to use the final metric model.')
    parser.add_argument('--train', action='store_true',
                        help="Train the transfer model instead of zero-shot inference.")
    parser.add_argument('--trained_adata_path', type=str,
                        help="If training is requested, this path must contained the trained adata for at least the guide species.")

    # Model paths
    parser.add_argument('--pretrain_model_path', type=str, required=True,
                        help='Path to load a pretraining model from')
    parser.add_argument('--metric_model_path', type=str, required=True,
                        help='Path to load a metric (macrogene -> embedding) model from')
    parser.add_argument('--xfer_model_path', type=str,
                        help='Path to store the transfer learning model to')

    
    # Defaults
    parser.set_defaults(
        org='saturn',
        species="new",
        guide_species=None,
        in_label_col=None,
        ref_label_col="CL_class_coarse",
        non_species_batch_col=None,
        device= torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        device_num=0,
        batch_size=4096,
        model_dim=256,
        hidden_dim=256,
        hv_genes=8000,
        epochs=50,
        metric_lr=0.001,
        pretrain_epochs=200,
        log_dir='tboard_log/',
        work_dir='./out/',
        time_stamp=False,
        mnn=True,
        pretrain=True,
        vae=False,
        use_ref_labels=False,
        embedding_model='ESM1b',
        gene_embedding_method=None,
        centroids_init_path=None,
        seed=0,
        unfreeze_macrogenes=False,
        score_ref_labels=False,
        tissue_subset=None,
        tissue_column="tissue_type",
        equalize_triplets_species=False,
        hv_span=0.3,
        centroid_score_func="default",
        train=False,
        encoder='metric',
    )

    args = parser.parse_args()
    torch.cuda.set_device(args.device_num)
    print(f"Using Device {args.device_num}")

    # Numpy seed
    np.random.seed(args.seed)
    # Torch Seed
    torch.manual_seed(args.seed)
    # Default random seed
    random.seed(args.seed)
    
    print(f"Set seed to {args.seed}")
    # Don't Balance the species numbers
    ExperimentDatasetMultiEqual = ExperimentDatasetMulti

    inferrer(args)

