"""
Created on Feb 28, 2025

@author: Fabio Zanini
"""

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


from .saturn_model import full_block


class GenerativeModel(torch.nn.Module):
    def __init__(
        self,
        gene_scores,
        dropout=0,
        hidden_dim=128,
        embed_dim=10,
        random_weights=False,
        l1_penalty=0.1,
        pe_sim_penalty=1.0,
        vae=False,
    ):
        """Generate synthetic cell gene expression vectors for new species."""
        super().__init__()

        # TODO: unused for now
        self.vae = vae

        self.dropout = dropout
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_gene_scores = len(gene_scores)
        # num_cl is the number of macrogenes
        self.num_cl = gene_scores.shape[1]

        # Gene -> macrogene weights
        self.p_weights_rev = nn.Parameter(gene_scores.float().log())
        if random_weights:
            nn.init.xavier_uniform_(
                self.p_weights_rev, gain=nn.init.calculate_gain("relu")
            )
        self.cl_layer_norm = nn.LayerNorm(self.num_cl)

        # Decoders
        self.px_decoder = nn.Sequential(
            full_block(self.embed_dim, self.hidden_dim, self.dropout),
        )
        self.cl_scale_decoder = full_block(self.hidden_dim, self.num_cl)
        self.px_dropout_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.num_gene_scores),
        )
        # Theta rate for ZINB
        self.px_r = torch.nn.Parameter(torch.randn(self.num_gene_scores))

        # Weight for the Lasso on gene -> weights
        self.l1_penalty = l1_penalty
        # Weight for the similarity on gene -> weights, which biases connections towards similar macrogenes/genes
        self.pe_sim_penalty = pe_sim_penalty

        # This is used in the similarity loss
        # The shape is macrogenes x 256
        self.p_weights_embeddings = nn.Sequential(
            full_block(
                self.num_cl, 256, self.dropout
            )  # This embedding layer will be used in metric learning to encode
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

        # Input is in embedded space
        encoded = inp

        # expr = inp
        # expr = torch.log1p(expr)
        # expr = expr.unsqueeze(1)
        # clusters = []
        # expr_and_genef = expr
        # x = nn.functional.linear(expr_and_genef.squeeze(), self.p_weights.exp())
        # x = self.cl_layer_norm(x)
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout)

        # encoder_input = x.squeeze()
        # encoded = self.encoder(encoder_input)

        if encoded.dim() != 2:
            encoded = encoded.unsqueeze(0)

        decoded = self.px_decoder(encoded)

        cl_scale = self.cl_scale_decoder(decoded)

        # Decode macrogenes to genes of this one NEW species
        cl_to_px = nn.functional.linear(cl_scale.unsqueeze(0), self.p_weights_rev.exp())

        # Distribute the means by cluster, based on the current weights of gene_to_macrogene
        px_scale_decode = nn.Softmax(-1)(cl_to_px.squeeze())

        # Dropout decoder for this one species, used to get the expected dropout rate in ZINB
        px_drop = self.px_dropout_decoder(decoded)

        # Species-specific theta used in ZINB
        px_r = torch.exp(self.px_r)

        return px_scale_decode, px_r, px_drop

    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout):
        """https://github.com/scverse/scvi-tools/blob/main/src/scvi/module/_vae.py


        NOTE: whether negative binomimals are a good idea for this is questionable.
        """
        return (
            -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
            .log_prob(x)
            .sum(dim=-1)
        )

    def loss_vae(
        self, inp, mu, log_var, kld_weight, px_rate, px_r, px_drop, weights=None
    ):
        if weights is None:
            recons_loss = torch.sum(
                self.get_reconstruction_loss(inp, px_rate, px_r, px_drop)
            )
        else:
            recons_loss = torch.sum(
                self.get_reconstruction_loss(inp, px_rate, px_r, px_drop) * weights
            )  # weight by CT abundancy

        loss = recons_loss
        if self.vae:
            kld_loss = torch.sum(
                -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
            )

            loss = loss + (kld_weight * kld_loss)
            return {
                "loss": loss,
                "Reconstruction_Loss": recons_loss.detach(),
                "KLD": -kld_loss.detach(),
            }
        else:
            return {"loss": loss}

    def lasso_loss(self, weights):
        """Lasso Loss used to regularize (sparsify) the gene to macrogene weights"""
        loss = torch.nn.L1Loss(reduction="sum")
        return loss(weights, torch.zeros_like(weights))

    def gene_weight_ranking_loss(self, weights, embeddings):
        """Ranking loss used to regularize the gene to macrogene weights"""
        # weights is M x G, the gene -> macrogene weights
        # p_weights_embeddings is M x 256, macrogenes to a standardised latent space
        # x1 is G x 256, genes to a standardised latent space
        # This way we can compute similarity between gene/protein embeddings in a latent space
        x1 = self.p_weights_embeddings(weights.t())

        # Mean squared error loss
        loss = nn.MSELoss(reduction="sum")
        similarity = torch.nn.CosineSimilarity()

        # Resample randon G x 256 weights (i.e. randomise genes)
        idx1 = torch.randint(low=0, high=x1.shape[0], size=(x1.shape[0],))
        x2 = x1[idx1, :]

        # Compute self-similarity of these weights after randomisation
        target = similarity(embeddings, embeddings[idx1, :])

        return loss(similarity(x1, x2), target)
