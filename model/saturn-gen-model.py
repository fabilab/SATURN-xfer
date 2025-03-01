'''
Created on Feb 28, 2025

@author: Fabio Zanini
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from scvi.distributions import ZeroInflatedNegativeBinomial
try:
    from kmeans_pytorch import kmeans
except ImportError:
    kmeans = None
from sklearn.cluster import KMeans
from scipy.stats import rankdata
import scanpy as sc
import numpy as np


from .saturn_model import (
    full_block
)


class SATURNGenModel(torch.nn.Module):
    def __init__(
        self,
        gene_scores,
        dropout=0,
        hidden_dim=128,
        embed_dim=10,
        random_weights=False,
        l1_penalty=0.1,
        pe_sim_penalty=1.0,
    ):
        '''Generate synthetic cell gene expression vectors for new species.'''
        super.__init__()

        self.num_gene_scores = len(gene_scores)
        self.p_weights = nn.Parameter(gene_scores.float().t().log())
        if random_weights: # for the genes to centroids weights
            nn.init.xavier_uniform_(self.p_weights, gain=nn.init.calculate_gain('relu'))

        # num_cl is the number of macrogenes
        self.num_cl = gene_scores.shape[1]
        self.expr_filler = nn.Parameter(torch.zeros(self.num_genes), requires_grad=False) # pad exprs with zeros        


        # Z Encoder
        self.encoder = nn.Sequential(
                full_block(self.num_cl, self.hidden_dim, self.dropout),
                full_block(self.hidden_dim, self.embed_dim, self.dropout),
        )

        self.px_decoder = nn.Sequential(
            full_block(self.embed_dim, self.hidden_dim, self.dropout),
        )
        
        self.cl_scale_decoder = full_block(self.hidden_dim, self.num_cl)
        
        self.px_dropout_decoder = nn.Sequential(
                nn.Linear(self.hidden_dim, gene_idxs[1] - gene_idxs[0]) 
        )
        
        self.px_r = torch.nn.Parameter(torch.randn(gene_idxs[1] - gene_idxs[0]))

        # Gene to Macrogene modifiers
        self.l1_penalty = l1_penalty
        self.pe_sim_penalty = pe_sim_penalty
        
        self.p_weights_embeddings = nn.Sequential(
            full_block(self.num_cl, 256, self.dropout) # This embedding layer will be used in metric learning to encode
                                                       # similarity in the protein embedding space
        )


    def forward(self, inp):
        """The actual layering of the model.

        Args:
            inp: the macrogene expression values for the guide species (NOT the embeddings!)

        Returns:
            A tuple of a few encoded/decoded results used for different things.


        The architechture is the following:
            PRE-ENCODER (gene expression + protein embedding -> weighted macrogenes)

        NOTE: The attempt here is to start from known expression values in macrogene space
        i.e. coming from real data (typically cells from a guide species). We want to
        construct macrogene -> gene weights onto NEW genes from an unseen species, and
        we have protein embeddings from those NEW genes.
        """
        batch_size = inp.shape[0]

        expr = inp
        expr = torch.log1p(expr)
        expr = expr.unsqueeze(1)
        clusters = []
        expr_and_genef = expr 
        x = nn.functional.linear(expr_and_genef.squeeze(), self.p_weights.exp())
        x = self.cl_layer_norm(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout)

        encoder_input = x.squeeze()

        encoded = self.encoder(encoder_input)
        if encoded.dim() != 2:
            encoded = encoded.unsqueeze(0)


        decoded = self.px_decoder(encoded)
        library = torch.log(inp.sum(1)).unsqueeze(1)
        cl_scale = self.cl_scale_decoder(decoded)

        # Decode macrogenes to genes of this one NEW species
        cl_to_px = nn.functional.linear(cl_scale.unsqueeze(0), self.p_weights.exp().t())[:, :, idx[0]:idx[1]]


        # Distribute the means by cluster, based on the current weights of gene_to_macrogene
        px_scale_decode = nn.Softmax(-1)(cl_to_px.squeeze())
        px_rate =  torch.exp(library) * px_scale_decode


        # Dropout decoder for this one species, used to get the expected dropout rate in ZINB
        px_drop = self.px_dropout_decoder(decoded)

        # Species-specific theta used in ZINB
        px_r = torch.exp(self.px_r)

        return encoder_input, encoded, px_rate, px_r, px_drop

    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout):
        '''https://github.com/scverse/scvi-tools/blob/main/src/scvi/module/_vae.py


        NOTE: whether negative binomimals are a good idea for this is questionable.
        '''
        return -ZeroInflatedNegativeBinomial(
            mu=px_rate, theta=px_r, zi_logits=px_dropout
        ).log_prob(x).sum(dim=-1)
