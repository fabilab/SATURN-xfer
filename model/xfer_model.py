"""
@author: Fabio Zanini
@date: 01/01/2025

Transfer model based on a combo of pretrain and metric models from SATURN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def full_block(in_features, out_features, p_drop=0.1, bias=True):
    """This is a block of 4 layers that is reused in the code a few times."""
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.LayerNorm(out_features),
        nn.ReLU(),
        nn.Dropout(p=p_drop),
    )


class TransferModel(torch.nn.Module):
    def __init__(
        self,
        input_dim=2000,
        num_macrogenes=2000,
        dropout=0,
        hidden_dim=128,
        embed_dim=10,
        guide_species=None,
        species_to_gene_idx={},
        vae=False,
    ):
        """This is essentially a transfer learning model that starts from one species' genes and encodes into embedding space.

        Args:
            input_dim: The number of genes in the species to learn about.
            dropout: dropout rate (0-1).
            hidden_dim: the hidden dimension of the encoder.
            embed_dim: the embedding dimension. The model output at inference time is #cells x embed_dim
            vae: Whether to use a VAE or not.
        """
        super().__init__()

        self.dropout = dropout
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_macrogenes = num_macrogenes
        self.guide_species = guide_species
        self.vae = vae

        # If no guide species is given, project straight into macrogene space
        # Ottherwise, project onto guide species and from there onto macrogenes
        if self.guide_species is None:
            self.p_weights = nn.Parameter(
                torch.zeros(self.num_macrogenes, self.input_dim)
            )
        else:
            self.species_to_gene_idx = species_to_gene_idx
            # Number of genes in the guide species
            tmp = self.species_to_gene_idx[self.guide_species]
            self.num_guide_genes = tmp[1] - tmp[0]
            del tmp

            # NOTE: see note above for the meaning and rationale for this
            self.num_genes = 0
            for k, v in self.species_to_gene_idx.items():
                self.num_genes = max(self.num_genes, v[1])

            self.guide_weights = nn.Parameter(
                torch.zeros(self.input_dim, self.num_guide_genes)
            )
            self.guide_layer_norm = nn.LayerNorm(self.num_guide_genes)
            self.p_weights = nn.Parameter(
                torch.zeros(self.num_macrogenes, self.num_genes)
            )
        self.cl_layer_norm = nn.LayerNorm(self.num_macrogenes)

        if self.vae:
            # Z Encoder
            self.encoder = nn.Sequential(
                full_block(self.input_dim, hidden_dim, self.dropout),
            )
            self.fc_var = nn.Linear(self.hidden_dim, self.embed_dim)
            self.fc_mu = nn.Linear(self.hidden_dim, self.embed_dim)

        else:
            self.encoder = nn.Sequential(
                full_block(self.num_macrogenes, self.hidden_dim, self.dropout),
                full_block(self.hidden_dim, self.embed_dim, self.dropout),
            )

    def forward(self, inp, species=None):
        """Perform the forward pass, a straightout encoder, no more, no less.

        Args:
            inp: input, in macrogene space now (after the pretraining).
            species: the species to predict for. This argument is not actually used.
        Returns:
            The transformed input in embedding (latent) space, no decoding needed.
        """

        # First, project the new species genes onto the guide species genes
        # NOTE: This is basically a custom encoder, just like the beginning of the
        # pretraining model, however it's unclear whether dropout is needed here.
        # The default one-hot weights are 1 for the closest gene match and 0
        # elsewhere, therefore they are already normalised and relu'd.
        # NOTE: the exp() here and below is a cheap way to ensure all weights "as used"
        # are positive, and we are just splashing gene expression - which is
        # nonnegative - onto another basis that is also nonnegative.
        expr = torch.log1p(inp)

        # If there is no guide species, project directly onto macrogene space
        # using positive weights. If there is a guide, project onto the
        # guide and then use (mostly pre-trained) weights to project onto
        # macrogene space
        if self.guide_species is not None:
            x = nn.functional.linear(expr, self.guide_weights.exp())
            x = self.guide_layer_norm(x)
            x = F.relu(x)  # all pos
            x = F.dropout(x, self.dropout)

            # NOTE: This chunk comes straight from the pretrain model
            batch_size = inp.shape[0]
            expr = torch.zeros(batch_size, self.num_genes).to(inp.device)
            filler_idx = self.species_to_gene_idx[self.guide_species]
            expr[:, filler_idx[0] : filler_idx[1]] = x
            expr = torch.log1p(expr)

        # Finally, the custom encoder from guide species to macrogene space3
        x = nn.functional.linear(expr, self.p_weights.exp())
        x = self.cl_layer_norm(x)
        x = F.relu(x)  # all pos
        x = F.dropout(x, self.dropout)
        encoder_input = x.squeeze()

        # Encode the macrogene input (inp -> hidden_dim, hidden_dim -> embed_dim)
        # using weights from the trained metric model
        encoded = self.encoder(encoder_input)
        if self.vae:
            return encoder_input, self.fc_mu(encoded)
        else:
            return encoder_input, encoded
