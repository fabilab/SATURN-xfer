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
            # - the vae loss (ZINB adhaerence)
            # - a LASSO on the gene -> macrogene weights
            # - a rank loss on the gene -> macrogene weights
            # The last two are ok. The first one, let's take a look whether any change is needed given this is generated,
            # therefore there is no "library size" per se.
            l = model.loss_vae(
                data, None, None, 0, px_rates, px_rs, px_drops
            )  # This loss also works for non vae loss
            spec_loss = l["loss"] / data.shape[0]
            l1_loss = model.l1_penalty * model.lasso_loss(model.p_weights.exp())
            rank_loss = model.pe_sim_penalty * model.gene_weight_ranking_loss(
                model.p_weights.exp(), embeddings_tensor
            )

            batch_loss = spec_loss + l1_loss + rank_loss

            epoch_ave_loss.append(float(spec_loss.detach().cpu()))

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


def trainer(args):
    """
    Runs the generative pipeline
    """
    device = torch.device(args.device)
    species = args.species
    embedding_path = args.protein_embedding_path

    print("Load the macrogene expression data, embedded by the autoencoder")
    adata_embed = sc.read(args.adata_embed)
    print("Loaded macrogene expression data")

    if args.species_guide is not None:
        species_guide = args.species_guide
        adata_embed = adata_embed[adata_embed.obs["species"] == species_guide]
    else:
        species_guide = "unknown"

    print("Load the protein embeddings for the new species")
    embedding_dict = load_gene_embeddings_one_species(
        species=species,
        embedding_path=embedding_path,
        embedding_model=args.embedding_model,
    )
    # NOTE: the order of genes in the new species is arbitrary
    all_gene_names = list(embedding_dict.keys())
    # stacked embeddings
    X = torch.stack(
        [embedding_dict[gene_symbol.lower()] for gene_symbol in all_gene_names]
    )
    print("Loaded protein embeddings")

    # NOTE: there is no such thing as HVG strictu sensu. We can do HVG after we get the first pass at generating a synthetic atlas

    print("Load the centroids")
    with open(args.centroids_init_path, "rb") as f:
        tmp = pickle.load(f)
        centroids_coords = tmp["centroids"]
        centroid_score_func = tmp.get("score_func", args.centroid_score_func)
    print("Loaded centroids and existing scores")

    print("Score genes of new species against cluster centers")
    centroids_coords = torch.tensor(centroids_coords).type("f4")
    gene_scores = score_genes_against_centroids(
        X,
        centroids_coords,
        all_gene_names,
        score_function=centroid_score_func,
    )
    centroid_weights = torch.stack(
        [torch.tensor(species_genes_scores[gn]) for gn in all_gene_names]
    )

    print("***STARTING GENERATIVE LEARNING***")
    pretrain_state_dict = torch.load(args.pretrain_model_path)
    gen_state_dict = {}
    gen_state_dict["cl_layer_norm.weight"] = deepcopy(
        pretrain_state_dict["cl_layer_norm.weight"]
    )
    gen_state_dict["cl_layer_norm.bias"] = deepcopy(
        pretrain_state_dict["cl_layer_norm.bias"]
    )
    # Copy the encoder parameters from either the pretrain or the metric model
    if args.encoder == "pretrain":
        encoder_state_dict = pretrain_state_dict
    else:
        encoder_state_dict = torch.load(args.metric_model_path)
    for key, val in encoder_state_dict.items():
        if key.startswith("encoder"):
            gen_state_dict[key] = deepcopy(val)

    gen_model = GenerativeModel(
        species=species,
        gene_scores,
        hidden_dim=xfer_state_dict["encoder.0.0.bias"].shape[0],
        embed_dim=xfer_state_dict["encoder.1.1.bias"].shape[0],
        dropout=0.1,
    )
    gen_model.load_state_dict(gen_state_dict)
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
