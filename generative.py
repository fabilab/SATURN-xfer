"""
Created on Feb 28, 2025

@author: Fabio Zanini
"""

import os

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"

import os
import scanpy as sc
import pandas as pd
from anndata import AnnData
import warnings
from builtins import int

warnings.filterwarnings("ignore")
import losses, miners, distances, reducers, testers
import torch
import torch.optim as optim
import numpy as np
import utils.logging_presets as logging_presets
import record_keeper
from data.gene_embeddings import (
    load_gene_embeddings_one_species,
)
from data.multi_species_data import (
    ExperimentDatasetMulti,
    multi_species_collate_fn,
    ExperimentDatasetMultiEqualCT,
)
from data.multi_species_data import ExperimentDatasetMultiEqual

from model.saturn_model import make_centroids, score_genes_against_centroids
from model.gen_model import GenerativeModel
import torch.nn.functional as F
from tqdm import trange, tqdm
from pretrain_utils import *

import argparse
from datetime import datetime
from pathlib import Path
import sys

sys.path.append("../")

# SATURN
from sklearn.cluster import KMeans
from scipy.stats import rankdata
import pickle
from copy import deepcopy
from score_adata import get_all_scores
from utils import stop_conditions
import random


def train(
    model,
    train_loader,
    optimizer,
    device,
    nepochs,
    embeddings_tensor=None,
):
    """Train a SATURN-gen model.

    embeddings_tensor -- dictionary containing species:protein embeddings
    """

    print("Train new macrogene weights...")
    model.train()

    pbar = tqdm(np.arange(1, nepochs + 1))
    all_ave_loss = []
    for epoch in pbar:
        model.train()
        epoch_ave_loss = []

        for batch_idx, batch_tuple in enumerate(train_loader):
            # Obviously, each minibatch is used independently to speed up convergence
            optimizer.zero_grad()

            (data, labels, refs, batch_labels) = batch_tuple

            if data is None:
                continue

            data, labels, refs = data.to(device), labels.to(device), refs.to(device)

            # FIXME: this needs to change to something useful (backward)
            encoder_input, encoded, px_rates, px_rs, px_drops = model(data, species)

            if px_rates.dim() != 2:
                px_rates = px_rates.unsqueeze(0)
            if px_rs.dim() != 2:
                px_rs = px_rs.unsqueeze(0)
            if px_drops.dim() != 2:
                px_drops = px_drops.unsqueeze(0)

            gene_weights = model.p_weights.exp()

            # NOTE: the loss has three components:
            # - the reconstruction loss (ZINB adhaerence)
            # - a LASSO on the gene -> macrogene weights
            # - a rank loss on the gene -> macrogene weights
            # The last two are ok. The first one, let's take a look whether any change is needed given this is generated,
            # therefore there is no "library size" per se.
            # Ok, the reconstruction loss is the log likelihood of the ZINB applied with organism-specific input counts
            # and parameters estimated by the training. In normal SATURN, that means only the parameters are trained, whereas
            # here also the backward weights can be trained.
            l = model.loss_vae(
                data, None, None, 0, px_rates, px_rs, px_drops
            )  # This loss also works for non vae loss
            rec_loss = l["loss"] / data.shape[0]
            l1_loss = model.l1_penalty * model.lasso_loss(model.p_weights.exp())
            rank_loss = model.pe_sim_penalty * model.gene_weight_ranking_loss(
                model.p_weights.exp(), embeddings_tensor
            )

            batch_loss = rec_loss + l1_loss + rank_loss

            epoch_ave_loss.append(float(rec_loss.detach().cpu()))

            # TODO: why is there zero_grad here as well?
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        # This looks like a legit reporting bug
        loss_string = [
            f"Epoch {epoch}: L1 Loss {l1_loss.detach()} Rank Loss {rank_loss.detach()}"
        ]
        loss_string += [f"Avg Loss: {round(np.average(epoch_ave_loss))}"]
        loss_string = ", ".join(loss_string)
        pbar.set_description(loss_string)
        all_ave_loss.append(np.mean(epoch_ave_loss))

        return model


def create_output_anndata(
    train_counts,
    train_lab,
    train_species,
    species_guide,
    train_macrogenes,
    train_embed,
    train_ref,
    use_batch_labels=False,
    train_batch=None,
    obs_names=None,
):
    """
    Create an AnnData from the generative results
    """
    adata = AnnData(train_counts)
    labels = train_lab
    adata.obs["labels"] = pd.Categorical(labels)
    adata.obs["labels2"] = pd.Categorical(
        [l.split("_")[-1] for l in adata.obs["labels"]]
    )

    ref_labels = train_ref
    adata.obs["ref_labels"] = pd.Categorical(ref_labels)

    adata.obs["species"] = train_species
    adata.uns["guide_species"] = species_guide

    adata.obsm["macrogenes"] = train_macrogenes
    adata.obsm["embed"] = train_embed
    if use_batch_labels:
        batch_labels = train_batch
        adata.obs["batch_labels"] = pd.Categorical(batch_labels)
    if obs_names is not None:
        adata.obs_names = obs_names
    return adata


def trainer(args):
    """
    Runs the generative pipeline
    """
    device = torch.device(args.device)
    species = args.species
    embedding_path = args.in_embeddings_path

    print("Load the macrogene expression data, embedded by the autoencoder")
    adata_embed = sc.read(args.in_adata_path)
    print("Loaded macrogene expression data")

    if args.guide_species is not None:
        species_guide = args.guide_species
        adata_embed = adata_embed[adata_embed.obs["species"] == species_guide]
    else:
        species_guide = "all"

    print("Load the protein embeddings for the new species")
    embedding_dict = load_gene_embeddings_one_species(
        species=species,
        embedding_path=embedding_path,
        embedding_model=args.embedding_model,
    )
    # NOTE: the order of genes in the new species is arbitrary
    all_gene_names = list(embedding_dict.keys())
    # stacked embeddings
    X = torch.stack([embedding_dict[gene_symbol] for gene_symbol in all_gene_names])
    print("Loaded protein embeddings")

    # NOTE: there is no such thing as HVG strictu sensu. We can do HVG after we get the first pass at generating a synthetic atlas

    print("Load the centroids")
    with open(args.centroids_init_path, "rb") as f:
        tmp = pickle.load(f)
        centroids_coords = tmp["centroids"]
        centroid_score_func = tmp.get("score_func", args.centroid_score_func)
    print("Loaded centroids and existing scores")

    print("Score genes of new species against cluster centers")
    centroids_coords = torch.tensor(centroids_coords).type(X.dtype)
    gene_scores = score_genes_against_centroids(
        X,
        centroids_coords,
        all_gene_names,
        score_function=centroid_score_func,
    )
    # stacked gene scores
    centroid_weights = torch.stack(
        [torch.tensor(gene_scores[gn]) for gn in all_gene_names]
    )

    print("***STARTING GENERATIVE LEARNING***")
    pretrain_state_dict = torch.load(args.pretrain_model_path)
    # Copy the encoder parameters from either the pretrain or the metric model
    if args.encoder == "pretrain":
        encoder_state_dict = pretrain_state_dict
    else:
        encoder_state_dict = torch.load(args.metric_model_path)

    gen_model = GenerativeModel(
        gene_scores=centroid_weights,
        dropout=0.1,
        hidden_dim=encoder_state_dict["encoder.0.0.bias"].shape[0],
        embed_dim=encoder_state_dict["encoder.1.1.bias"].shape[0],
    )
    # Set frozen layers from the pretrained models
    gen_state_dict = gen_model.state_dict()
    gen_state_dict["cl_layer_norm.weight"] = deepcopy(
        pretrain_state_dict["cl_layer_norm.weight"]
    )
    gen_state_dict["cl_layer_norm.bias"] = deepcopy(
        pretrain_state_dict["cl_layer_norm.bias"]
    )
    for key, val in encoder_state_dict.items():
        if key.startswith("encoder"):
            gen_state_dict[key] = deepcopy(val)

    # Reload state dict with pretrained initialisations
    gen_model.load_state_dict(gen_state_dict)

    # Ship to GPU
    gen_model.to(device)

    # Set a few infra things
    args.dir_ = (
        args.work_dir
        + args.log_dir
        + "test"
        + str(args.model_dim)
        + "_data_"
        + args.species
        + ("_org_" + str(args.org) if args.org is not None else "")
        + ("_" + dt if args.time_stamp else "")
        + ("_" + args.tissue_subset if args.tissue_subset else "")
        + ("_guide_" + args.guide_species if args.guide_species else "")
        + ("_seed_" + str(args.seed))
    )
    metric_dir = Path(args.work_dir)
    metric_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.dir_.split(args.log_dir)[-1]

    print("***ZERO-SHOT GENERATION***")
    zeroshot_macrogenes = torch.tensor(adata_embed.obsm["macrogenes"])
    zeroshot_embed = torch.tensor(adata_embed.X)

    # TODO: invert from macrogene to gene space
    zeroshot_counts = None

    zeroshot_species = species
    zeroshot_lab = adata_embed.obs["labels2"].values
    zeroshot_ref = adata_embed.obs["ref_labels"].values
    all_obs_names = adata_embed.obs_names

    adata_zeroshot = create_output_anndata(
        zeroshot_counts,
        zeroshot_lab,
        zeroshot_species,
        species_guide,
        zeroshot_macrogenes.cpu().numpy(),
        zeroshot_embed.cpu().numpy(),
        zeroshot_ref,
        obs_names=all_obs_names,
    )
    adata_zeroshot.obs["species"] = species
    if species_guide is not None:
        adata_zeroshot.uns["guide_species"] = species_guide

    if len(run_name) > 70:
        zeroshot_adata_fn = "zeroshot.h5ad"
    else:
        zeroshot_adata_fn = f"{run_name}_zeroshot.h5ad"
    zeroshot_adata_path = metric_dir / zeroshot_adata_fn
    adata_zeroshot.write(zeroshot_adata_path)

    # If no training is requested, we can stop here
    if not args.train:
        return

    # Make the guide species data loader
    if use_batch_labels:  # we have a batch column to use for the pretrainer
        train_dataset = ExperimentDatasetMultiEqual(
            all_data=species_to_adata_xfer,
            all_ys={
                species: adata.obs["truth_labels"]
                for (species, adata) in species_to_adata_xfer.items()
            },
            all_refs={
                species: adata.obs["ref_labels"]
                for (species, adata) in species_to_adata_xfer.items()
            },
            all_batch_labs={
                species: adata.obs["batch_labels"]
                for (species, adata) in species_to_adata_xfer.items()
            },
        )
    else:
        train_dataset = ExperimentDatasetMultiEqual(
            all_data=species_to_adata_xfer,
            all_ys={
                species: adata.obs["truth_labels"]
                for (species, adata) in species_to_adata_xfer.items()
            },
            all_refs={
                species: adata.obs["ref_labels"]
                for (species, adata) in species_to_adata_xfer.items()
            },
            all_batch_labs={},
        )
    # Load data with shuffling
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=multi_species_collate_fn,
        batch_size=1024,
        shuffle=True,
    )

    # TODO: freeze only the parameters that need no retraining (common encoders/decoders)
    # The species-specific parts should be obviously unfrozen
    for parameter in gen_model.parameters():
        parameter.requires_grad = False
    gen_model.p_weights.requires_grad = True

    # Create the optimizer
    # lr is the learning rate for the metric model
    optimizer = optim.Adam(gen_model.parameters(), lr=args.lr)

    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)

    print("***STARTING FINE-TUNING***")
    # TODO: do something about estimating HVGs in the new species
    train(
        gen_model,
        train_loader,
        optimizer,
        device,
        args.epochs,
        embeddings_tensor=X,
    )

    if args.gen_model_path is not None:
        # Save the transfer model if asked to
        print(f"Saving trained Generative Model to {args.gen_model_path}")
        torch.save(gen_model.state_dict(), args.gen_model_path)

    print("***FINE-TUNED INFERENCE***")
    if use_batch_labels:
        (
            train_emb,
            train_lab,
            train_species,
            train_macrogenes,
            train_ref,
            train_batch,
        ) = get_all_embeddings(
            gen_dataset,
            gen_model,
            device,
            use_batch_labels,
        )
        adata_finetuned = create_output_anndata(
            train_emb,
            train_lab,
            train_species,
            train_macrogenes.cpu().numpy(),
            train_ref,
            use_batch_labels,
            train_batch,
            obs_names=all_obs_names,
        )
    else:
        train_emb, train_lab, train_species, train_macrogenes, train_ref = (
            get_all_embeddings(
                gen_dataset,
                gen_model,
                device,
                use_batch_labels,
            )
        )
        adata_finetuned = create_output_anndata(
            train_emb,
            train_lab,
            train_species,
            train_macrogenes.cpu().numpy(),
            train_ref,
            obs_names=all_obs_names,
        )
    adata_finetuned.obs["species"] = args.species
    if species_guide is not None:
        adata_finetuned.uns["guide_species"] = species_guide
    adata_finetuned.uns["gen_training_epoch"] = args.epochs

    if len(run_name) > 70:
        final_adata_fn = f"epoch_{args.epochs}_finetuned_adata.h5ad"
    else:
        final_adata_fn = f"{run_name}_epoch_{args.epochs}_finetuned.h5ad"
    final_path = metric_dir / final_adata_fn
    adata_finetuned.write(final_path)
    print(f"Fine-tuned AnnData Path: {final_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Set model hyperparametrs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Run Setup
    parser.add_argument(
        "--in_adata_path",
        type=str,
        required=True,
        help="Path to h5ad of the data in macrogene and embedding space, used to generate synthetic cell profiles.",
    )
    parser.add_argument(
        "--in_embeddings_path",
        type=str,
        required=True,
        help="Path to embeddings summary pt file for the peptides of the species to infer",
    )
    parser.add_argument(
        "--species", type=str, help="Data to infer belongs to this species (optional)"
    )
    parser.add_argument(
        "--guide_species",
        type=str,
        help="Choose a guide species to generate synthetic cell profiles from. Default is to generate profiles from all species.",
    )
    parser.add_argument("--device", type=str, help="Set GPU/CPU")
    parser.add_argument("--device_num", type=int, help="Set GPU Number")
    parser.add_argument("--time_stamp", type=bool, help="Add time stamp in file name")
    parser.add_argument("--org", type=str, help="Add organization to filename")
    parser.add_argument("--log_dir", type=str, help="Log directory")
    parser.add_argument("--work_dir", type=str, help="Working directory")
    parser.add_argument("--seed", type=int, help="Init Seed")
    parser.add_argument("--in_label_col", type=str, help="Label column for input data")
    parser.add_argument(
        "--ref_label_col", type=str, help="Reference label column for input data"
    )
    parser.add_argument(
        "--tissue_subset",
        type=str,
        help="Subset the input anndatas by the column args.tissue_column to just be this tissue",
    )
    parser.add_argument(
        "--tissue_column",
        type=str,
        help="When subsetting the input anndatas by the column, use this column name.",
    )

    # SATURN Setup
    parser.add_argument("--hv_genes", type=int, help="Number of highly variable genes")
    parser.add_argument(
        "--hv_span",
        type=float,
        help="Fraction of cells to use when calculating highly variable genes, scanpy defeault is 0.3.",
    )
    parser.add_argument(
        "--centroids_init_path",
        type=str,
        required=True,
        help="Path to existing centroids pretraining weights, or location to save to.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        choices=["ESM1b", "MSA1b", "protXL", "ESM1b_protref", "ESM2", "ESMc"],
        help="Gene embedding model whose embeddings should be loaded if using gene_embedding_method",
    )
    parser.add_argument(
        "--centroid_score_func",
        type=str,
        choices=["default", "one_hot", "smoothed"],
        help="Gene embedding model whose embeddings should be loaded if using gene_embedding_method",
    )

    # Model Setup
    parser.add_argument(
        "--hidden_dim", type=int, help="Model first layer hidden dimension"
    )
    parser.add_argument("--model_dim", type=int, help="Model latent space dimension")
    parser.add_argument(
        "--random_weights",
        action="store_true",
        help="Initialise the top layer with random weights.",
    )

    # Metric Learning Arguments
    parser.add_argument(
        "--use_ref_labels",
        type=bool,
        nargs="?",
        const=True,
        help="Use reference labels when aligning",
    )
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["pretrain", "metric"],
        help="Which encoder to use for the transfer model. Default is to use the final metric model.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the transfer model instead of zero-shot inference.",
    )
    parser.add_argument("--epochs", type=int, help="How many epochs to train for")

    # Model paths
    parser.add_argument(
        "--pretrain_model_path",
        type=str,
        required=True,
        help="Path to load a pretraining model from",
    )
    parser.add_argument(
        "--metric_model_path",
        type=str,
        required=True,
        help="Path to load a metric (macrogene -> embedding) model from",
    )
    parser.add_argument(
        "--gen_model_path",
        type=str,
        help="Path to store the transfer learning model to",
    )

    # Defaults
    parser.set_defaults(
        org=None,
        species="new",
        guide_species=None,
        in_label_col=None,
        ref_label_col="CL_class_coarse",
        non_species_batch_col=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        device_num=0,
        batch_size=4096,
        model_dim=256,
        hidden_dim=256,
        hv_genes=8000,
        epochs=50,
        metric_lr=0.001,
        pretrain_epochs=200,
        log_dir="tboard_log/",
        work_dir="./out/",
        time_stamp=False,
        mnn=True,
        pretrain=True,
        vae=False,
        use_ref_labels=False,
        embedding_model="ESM1b",
        gene_embedding_method=None,
        centroids_init_path=None,
        seed=0,
        score_ref_labels=False,
        tissue_subset=None,
        tissue_column="tissue_type",
        hv_span=0.3,
        centroid_score_func="default",
        train=False,
        encoder="metric",
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

    trainer(args)
