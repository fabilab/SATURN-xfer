'''
Created on Feb 28, 2025

@author: Fabio Zanini
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
from data.gene_embeddings import load_gene_embeddings_adata
from data.multi_species_data import ExperimentDatasetMulti, multi_species_collate_fn, ExperimentDatasetMultiEqualCT
from data.multi_species_data import ExperimentDatasetMultiEqual

from model.saturn_model import SATURNPretrainModel, SATURNMetricModel, make_centroids
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
from score_adata import get_all_scores
from utils import stop_conditions
import random


def train(
    model,
    train_loader,
    optimizer,
    device,
    nepochs,
    ):
    '''Train a SATURN-gen model.'''

    print("Train new macrogene weights...")
    model.train()

    pbar = tqdm(np.arange(1, nepochs+1))
    for epoch in pbar:
        model.train()

        for batch_idx, batch_tuple in enumerate(train_loader):
            # Obviously, each minibatch is used independently to speed up convergence
            optimizer.zero_grad()
            batch_loss = 0

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
