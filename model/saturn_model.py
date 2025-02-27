'''
Created on Nov 7, 2022

@author: Yanay Rosen
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


def full_block(in_features, out_features, p_drop=0.1, bias=True):
    """This is a block of 4 layers that is reused in the code a few times."""
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.LayerNorm(out_features),
        nn.ReLU(),
        nn.Dropout(p=p_drop),
    )


class SATURNPretrainModel(torch.nn.Module):
    def __init__(
        self,
        gene_scores,
        dropout=0,
        hidden_dim=128,
        embed_dim=10,
        species_to_gene_idx={},
        vae=False,
        random_weights=False,
        sorted_batch_labels_names=None,
        l1_penalty=0.1,
        pe_sim_penalty=1.0,
    ):
        super().__init__()
        
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        self.sorted_batch_labels_names = sorted_batch_labels_names
        if self.sorted_batch_labels_names is not None:
            self.num_batch_labels = len(sorted_batch_labels_names)
        else:
            self.num_batch_labels = 0
        
        self.num_gene_scores = len(gene_scores)
        self.num_species = len(species_to_gene_idx)
        self.species_to_gene_idx = species_to_gene_idx
        self.sorted_species_names = sorted(self.species_to_gene_idx.keys())
        
        self.vae = vae

        # Num genes is the TOTAL number of genes in the dataset, sum of all species together.
        # NOTE: species_to_gene_idx is like a CSR matrix data structure that keeps track of the boundaries,
        # so the highest right edge is what we want. In practice, because the indices are in a dict rather
        # than an actual CSR structure, one has to (i) store most edges twice, and (ii) iterate over all
        # just to figure out the rightmost edge. Not great, but meh.
        self.num_genes = 0
        for k, v in self.species_to_gene_idx.items():
            self.num_genes = max(self.num_genes, v[1])

        self.p_weights = nn.Parameter(gene_scores.float().t().log())
        if random_weights: # for the genes to centroids weights
            nn.init.xavier_uniform_(self.p_weights, gain=nn.init.calculate_gain('relu'))

        # num_cl is the number of macrogenes
        self.num_cl = gene_scores.shape[1]
            
        self.cl_layer_norm = nn.LayerNorm(self.num_cl)
        self.expr_filler = nn.Parameter(torch.zeros(self.num_genes), requires_grad=False) # pad exprs with zeros        
        
        if self.vae:
            # Z Encoder
            self.encoder = nn.Sequential(
                    full_block(self.num_cl, self.hidden_dim, self.dropout),
            )
            self.fc_var = nn.Linear(self.hidden_dim, self.embed_dim)
            self.fc_mu = nn.Linear(self.hidden_dim, self.embed_dim)
            
        else:
            self.encoder = nn.Sequential(
                    full_block(self.num_cl, self.hidden_dim, self.dropout),
                    full_block(self.hidden_dim, self.embed_dim, self.dropout),
            )
            
        # Decoder
        self.px_decoder = nn.Sequential(
            full_block(self.embed_dim + self.num_species + self.num_batch_labels, self.hidden_dim, self.dropout),
        )
        
        self.cl_scale_decoder = full_block(self.hidden_dim, self.num_cl)
        
        self.px_dropout_decoders = nn.ModuleDict({
            species: nn.Sequential(
                nn.Linear(self.hidden_dim, gene_idxs[1] - gene_idxs[0]) 
            ) for species, gene_idxs in species_to_gene_idx.items()}
        )
        
        self.px_rs = nn.ParameterDict({
            species: torch.nn.Parameter(torch.randn(gene_idxs[1] - gene_idxs[0]))
            for species, gene_idxs in species_to_gene_idx.items()}
        )
        
        # This is kind of a hack, see the forward function as of why this exists.
        self.metric_learning_mode = False

        # Gene to Macrogene modifiers
        self.l1_penalty = l1_penalty
        self.pe_sim_penalty = pe_sim_penalty
        
        self.p_weights_embeddings = nn.Sequential(
            full_block(self.num_cl, 256, self.dropout) # This embedding layer will be used in metric learning to encode
                                                       # similarity in the protein embedding space
        )

    def forward(self, inp, species, batch_labels=None):
        """The actual layering of the model.

        Args:
            inp: the gene expression values for the species (NOT the embeddings!)
            species: the species name
            batch_labels: the batch labels for the species (if present)

        Returns:
            A tuple of a few encoded/decoded results used for different things.

        The architechture is the following:

            PRE-ENCODER (tokenisation gene expression + embedding -> weighted macrogenes)

            ENCODER (VAE works slightly differently):
              FULL BLOCK: (weighted macrogenes -> hidden dim)
              FULL BLOCK: (hidden dim -> embed_dim)

            DECODER:
              one-hot encode the species among all species
              FULL BLOCK: (embed_dim + num_species (because OH encoding) -> hidden dim)
                1. (CL-DECODER) FULL BLOCK: (hidden dim -> weighted macrogenes)
                   LINEAR: (weighted macrogenes -> genes of one species)
                   SOFTMAX: (transform the output into positive, normalised weights)
                   (a manual renormalisation step by library size (# of UMIs), to satisfy
                     the formal statistical constraints for negative binomial, even though
                     neg binomial might be flawed in the first place.
                2. (DROPOUT DECODERS): (hidden_dim -> genes of one species) this one skips
                     the final weights and shortcuts the macrogenes back to square one.

        Note that the input is *only* expression values, not embeddings. However, they are expression
        values for one species at a time, so mixing species in inference might be messy. Moreover,
        because the embeddings are used at __init__ time instead of here, we cannot add unseen genes
        during inference in this model.
        """
        batch_size = inp.shape[0]
        
        # Pad the appened expr with 0s to fill all gene nodes
        # num_genes is the sum of all genes (HVGs) from all species. We set to nonzero only the expression for the
        # dimensions related to this one species.
        expr = torch.zeros(batch_size, self.num_genes).to(inp.device)
        # species_to_gene_idx is like a CSR sparse matrix, it contains the edge indices of the tensor for each species
        filler_idx = self.species_to_gene_idx[species]
        # These are the actual gene expression values for that species
        expr[:, filler_idx[0]:filler_idx[1]] = inp
        # we logp1 the values
        expr = torch.log1p(expr)
        
        # concatenate the gene embeds with the expression as the last item in the embed
        # unsqueeze means: [1,2,3] -> [[1], [2], [3]]
        expr = expr.unsqueeze(1)
        
        # GNN and cluster weights
        clusters = []
        expr_and_genef = expr 

        # We multiply the gene expression times the scores from each centroid/macrogene
        # (initially), later on these weights get rebalanced by the pretraining regimen.
        # NOTE: scores are NOT the distances from the macrogenes. They are computed from
        # the distances through a monotonically decreasing function (see "default" and
        # other scoring functions at the end of this file).
        # This is a simple matrix multiplication with NO affine component: we are saying
        # that the expression of gene A is felt by each macrogene proportionally to the
        # "score" A bears upon each macrogene. In a way, it's a funny concept that a
        # single gene can influence multiple macrogenes as that's the opposite of what
        # the average reader would expect - obviously this gives the model quite some freedom.
        x = nn.functional.linear(expr_and_genef.squeeze(), self.p_weights.exp())

        # Normalisation layer
        x = self.cl_layer_norm(x)

        # Activation
        x = F.relu(x) # all pos

        # Dropout
        x = F.dropout(x, self.dropout)

        # NOTE: the last four components are what is called a "full_block" at the top of this file.
        # Basically it's an encoder into macrogene/centroid space that "tokenises" the gene expression
        # and is upstream of the actual encoder/decoder layers (below).
            
        # Here we start the actual encoder layers
        encoder_input = x.squeeze()
        encoded = self.encoder(encoder_input)
        
        if self.vae:
            # VAE 
            mu = self.fc_mu(encoded)
            log_var = self.fc_var(encoded)
        
            encoded = self.reparameterize(mu, log_var)
        else:
            mu = None
            log_var = None
        
        # One hot encode each species and paste it at the end of the embed_dim
        spec_1h = torch.zeros(batch_size, self.num_species).to(inp.device)
        #spec_idx = np.argmax(np.array(self.sorted_species_names) == species) # Fix for one hot
        spec_idx = 0
        spec_1h[:, spec_idx] = 1.
        
        if self.num_batch_labels > 0:
            # construct the one hot encoding of the batch labels
            # also a categorical covariate
            batch_1h = torch.zeros(batch_size, self.num_batch_labels).to(inp.device)
            batch_idx = np.argmax(np.array(self.sorted_batch_labels_names) == batch_labels)
            batch_1h[:, batch_idx] = 1.
            spec_1h = torch.hstack((spec_1h, batch_1h)) # should already be one hotted
        
        if encoded.dim() != 2:
            encoded = encoded.unsqueeze(0)
        
        # This happens when we unfreeze the macrogene weights during metric learning. Instead of
        # making two versions of the metric model, one with varying macrogenes and one with
        # frozen ones, we recycle the output of the first custom encoder of the pretrain model
        # as input for the standard model, and we recycle this class as an "unfrozen metric model"
        # for the case that the weights in the early custom encoder need to be further trained
        # (this time, using a different loss, namely the triplet loss).
        if self.metric_learning_mode:
            return encoded
        
        # This is where the pasting of the speceis one-hot encoding happens (hstack)
        # followed by the general decoder into hidden_dim
        decoded = self.px_decoder(torch.hstack((encoded, spec_1h)))
        
        # This is (log of) library size: either number of reads/UMIs or 10000 is prenormalised
        # (we probably should NOT prenormalise given this and the negative binamial assumption)
        library = torch.log(inp.sum(1)).unsqueeze(1)
        
        # Decode from hidden_dim to the macrogenes
        cl_scale = self.cl_scale_decoder(decoded) # num_cl output

        # index genes for mu
        idx = self.species_to_gene_idx[species]

        # Decode macrogenes to genes of this one species
        # NOTE: the transpose in the p_weights gives it away: you have to multiply
        # the decoded macrogenes by the transcpose of the weight matrix (exponentiated, just like
        # above, presumably to gain precision even though floating point numbers are intrinsically "logged")
        # to get back the original space (all genes from all species), followed by indexing using the CSR-like
        # data structure (and discarding the rest).
        cl_to_px = nn.functional.linear(cl_scale.unsqueeze(0), self.p_weights.exp().t())[:, :, idx[0]:idx[1]]

        # distribute the means by cluster, based on the current weights of gene_to_macrogene
        # in other words, each macrogene has a total weight of ONE and it gets smeared across various original
        # genes. Multiple macrogenes can contribute to the same original gene, in agreement with the idea above
        # that a single gene can affect multiple macrogenes.
        px_scale_decode = nn.Softmax(-1)(cl_to_px.squeeze())
        px_rate =  torch.exp(library) * px_scale_decode
        
        # dropout decoder for this one species, used to get the expected dropout rate in ZINB
        px_drop = self.px_dropout_decoders[species](decoded)

        # species-specific thneta used in ZINB
        px_r = torch.exp(self.px_rs[species])        
        
        # In the end, we return a few encoded result and the ZINB parameters:
        # - px_rate: the rate of the negative binomial
        # - px_r: the theta of the negative binomial
        # - px_drop: the dropout rates
        # NOTE: we also return the encoder_input, which is the gene expression in macrogene space. This is actually
        # the input of the metric model below, which is a vanilla encoder from macrogne space.
        # All there are species-specific. Check out the next method to figure out how they are used to compute the loss
        
        return encoder_input, encoded, mu, log_var, px_rate, px_r, px_drop
    
    
    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout):
        '''https://github.com/scverse/scvi-tools/blob/main/src/scvi/module/_vae.py


        NOTE: whether negative binomimals are a good idea for this is questionable.
        '''
        return -ZeroInflatedNegativeBinomial(
            mu=px_rate, theta=px_r, zi_logits=px_dropout
        ).log_prob(x).sum(dim=-1)
    
    def loss_vae(self, inp, mu, log_var, kld_weight, px_rate, px_r, px_drop, weights=None):
        if weights is None:
            recons_loss = torch.sum(self.get_reconstruction_loss(inp, px_rate, px_r, px_drop))
        else:
            recons_loss = torch.sum(self.get_reconstruction_loss(inp, px_rate, px_r, px_drop) * weights) # weight by CT abundancy
        
        loss = recons_loss
        if self.vae:
            kld_loss = torch.sum(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

            loss = loss + (kld_weight * kld_loss)
            return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
        else:
            return {'loss': loss}
    
    def lasso_loss(self, weights):
        # Lasso Loss used to regularize (sparsify) the gene to macrogene weights
        loss = torch.nn.L1Loss(reduction="sum")
        return loss(weights, torch.zeros_like(weights))
    
    def gene_weight_ranking_loss(self, weights, embeddings):
        # weights is M x G
        x1 = self.p_weights_embeddings(weights.t())
        # genes x 256
        loss = nn.MSELoss(reduction="sum")
        similarity = torch.nn.CosineSimilarity()
        
        idx1 = torch.randint(low=0, high=x1.shape[0], size=(x1.shape[0],))      
        x2 = x1[idx1, :]
        target = similarity(embeddings, embeddings[idx1, :])
        
        return loss(similarity(x1, x2), target)
        
    def loss_ae(self, pred, y):
        loss = F.mse_loss(pred, y) # MSE loss for numerical values
        return loss
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    
class SATURNMetricModel(torch.nn.Module):
    def __init__(
        self,
        input_dim=2000,
        dropout=0,
        hidden_dim=128,
        embed_dim=10,
        species_to_gene_idx={},
        vae=False,
    ):
        """This is essentially an encoder-only version of the above without preembedding.

        As such, the input here is already macrogenes and the output is embed_dim (usually 256), which
        is the actual output of the model and used for PCA, UMAP, etc.
        """
        super().__init__()
        
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        self.num_species = len(species_to_gene_idx)
        self.species_to_gene_idx = species_to_gene_idx
        self.sorted_species_names = sorted(self.species_to_gene_idx.keys())
        
        self.vae = vae

        # NOTE: see note above for the meaning and rationale for this
        self.num_genes = 0
        for k,v in self.species_to_gene_idx.items():
            self.num_genes = max(self.num_genes, v[1])
            
        # FIXME: I don't think this is used at all.
        self.cl_layer_norm = nn.LayerNorm(self.input_dim)
        
        if self.vae:
            # Z Encoder
            self.encoder = nn.Sequential(
                    full_block(self.input_dim, hidden_dim, self.dropout),
            )
            self.fc_var = nn.Linear(self.hidden_dim, self.embed_dim)
            self.fc_mu = nn.Linear(self.hidden_dim, self.embed_dim)
            
        else:
            self.encoder = nn.Sequential(
                    full_block(self.input_dim, self.hidden_dim, self.dropout),
                    full_block(self.hidden_dim, self.embed_dim, self.dropout),
            )
        

    def forward(self, inp, species=None):
        """Perform the forward pass, a straightout encoder, no more, no less.

        Args:
            inp: input, in macrogene space now (after the pretraining).
            species: the species to predict for.
        Returns:
            The transformed input in embedding (latent) space, no decoding needed.
        """

        # Encode the macrogene input (inp -> hidden_dim, hidden_dim -> embed_dim)
        encoded = self.encoder(inp)
        if self.vae:
            return self.fc_mu(encoded)
        else:
            return encoded
    
    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout):
        '''https://github.com/scverse/scvi-tools/blob/master/scvi/module/_vae.py'''
        return -ZeroInflatedNegativeBinomial(
            mu=px_rate, theta=px_r, zi_logits=px_dropout
        ).log_prob(x).sum(dim=-1)
    
    def loss_vae(self, inp, mu, log_var, kld_weight, px_rate, px_r, px_drop):
        recons_loss = torch.sum(self.get_reconstruction_loss(inp, px_rate, px_r, px_drop))
        if self.vae:
            kld_loss = torch.sum(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

            loss = recons_loss + kld_weight * kld_loss
            return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
        else:
            return {'loss': recons_loss}

    def loss_ae(self, pred, y):
        loss = F.mse_loss(pred, y) # MSE loss for numerical values
        return loss
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


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
            self.p_weights = nn.Parameter(torch.zeros(self.num_macrogenes, self.input_dim))
        else:
            self.species_to_gene_idx = species_to_gene_idx
            # Number of genes in the guide species
            tmp = self.species_to_gene_idx[self.guide_species]
            self.num_guide_genes = tmp[1] - tmp[0]
            del tmp

            # NOTE: see note above for the meaning and rationale for this
            self.num_genes = 0
            for k,v in self.species_to_gene_idx.items():
                self.num_genes = max(self.num_genes, v[1])

            self.guide_weights = nn.Parameter(torch.zeros(self.input_dim, self.num_guide_genes))
            self.guide_layer_norm = nn.LayerNorm(self.num_guide_genes)
            self.p_weights = nn.Parameter(torch.zeros(self.num_macrogenes, self.num_genes))
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
            x = F.relu(x) # all pos
            x = F.dropout(x, self.dropout)

            # NOTE: This chunk comes straight from the pretrain model
            batch_size = inp.shape[0]
            expr = torch.zeros(batch_size, self.num_genes).to(inp.device)
            filler_idx = self.species_to_gene_idx[self.guide_species]
            expr[:, filler_idx[0]:filler_idx[1]] = x
            expr = torch.log1p(expr)

        # Finally, the custom encoder from guide species to macrogene space3
        x = nn.functional.linear(expr, self.p_weights.exp())
        x = self.cl_layer_norm(x)
        x = F.relu(x) # all pos
        x = F.dropout(x, self.dropout)
        encoder_input = x.squeeze()

        # Encode the macrogene input (inp -> hidden_dim, hidden_dim -> embed_dim)
        # using weights from the trained metric model
        encoded = self.encoder(encoder_input)
        if self.vae:
            return encoder_input, self.fc_mu(encoded)
        else:
            return encoder_input, encoded


#####################################
# PREEMBEDDING AND UTILITY FUNCTIONS
#####################################
def make_centroids(embeds, species_gene_names, num_centroids=2000, normalize=False, seed=0, score_function="default", device="cuda:0"):
    print("Making Centroids using KMeans (sklearn, not on GPU) and scoring genes against them")
    if normalize:
        row_sums = embeds.sum(axis=1)
        embeds = embeds / row_sums[:, np.newaxis]

    # NOTE: (@iosonofabio) I verified that torch.cdist is equivalent to kmeans_obj.transform
    # Note that the distances are NOT the scores; scores are computed from and monotonically
    # negatively related to the distances, such that higher distance leads to lower score.
    # We never actually use the distance space outside of this one function.
    if kmeans is not None:
        _, cluster_centers = kmeans(
            X=embeds,
            num_clusters=num_centroids,
            distance='euclidean',
            device=torch.device(device),
        )
        dd = torch.cdist(embeds, cluster_centers, p=2)
    elif KMeans is not None:
        embeds_cpu = embeds.to("cpu")
        kmeans_obj = KMeans(n_clusters=num_centroids, random_state=seed).fit(embeds_cpu)
        # dd is distance frome each gene to centroid
        cluster_centers = kmeans_obj.cluster_centers_
        dd = kmeans_obj.transform(embeds_cpu)
    else:
        raise ImportError("kmeans_pytorch or sklearn.cluster.KMeans is required for this function")
    
    if score_function == "default":
        to_scores = default_centroids_scores(dd)
    elif score_function == "one_hot":
        to_scores = one_hot_centroids_scores(dd)
    elif score_function == "smoothed":
        to_scores = smoothed_centroids_score(dd)
    else:
        raise ValueError("score_function must be one of 'default', 'one_hot', or 'smoothed'")
    
    species_genes_scores = {}
    for i, gene_species_name in enumerate(species_gene_names):
        species_genes_scores[gene_species_name] = to_scores[i, :]
    return species_genes_scores, cluster_centers


def score_genes_against_centroids(embeds, cluster_centers, species_gene_names, score_function="default", return_dict=True):
    """As make_centroids, but for inference."""
    # NOTE: This is ok as per the comment in the previous function. Just like up there, distance
    # space is not actually broadcast outside of this function, only the scores are.
    dd = torch.cdist(embeds, cluster_centers, p=2)

    if score_function == "default":
        to_scores = default_centroids_scores(dd)
    elif score_function == "one_hot":
        to_scores = one_hot_centroids_scores(dd)
    elif score_function == "smoothed":
        to_scores = smoothed_centroids_score(dd)
    else:
        raise ValueError("score_function must be one of 'default', 'one_hot', or 'smoothed'")

    if not return_dict:
        return to_scores

    species_genes_scores = {}
    for i, gene_species_name in enumerate(species_gene_names):
        species_genes_scores[gene_species_name] = to_scores[i, :]
    return species_genes_scores


def default_centroids_scores(dd):
    """
    Convert KMeans distances to centroids to scores.
    :param dd: distances from gene to centroid.
    """
    ranked = rankdata(dd, axis=1) # rank 1 is close rank 2000 is far

    to_scores = np.log1p(1 / ranked) # log 1 is close log 1/2000 is far

    to_scores = ((to_scores) ** 2)  * 2
    return to_scores


def one_hot_centroids_scores(dd):
    """
    Convert KMeans distances to centroids to scores. All or nothing, so closest centroid has score 1, others have score 0.
    :param dd: distances from gene to centroid.
    """
    ranked = rankdata(dd, axis=1) # rank 1 is close rank 2000 is far
    
    to_scores = (ranked == 1).astype(float) # true, which is rank 1, is highest, everything else is 0
    return to_scores


def smoothed_centroids_score(dd):
    """
    Convert KMeans distances to centroids to scores. Smoothed version of original function, so later ranks have larger values.
    :param dd: distances from gene to centroid.
    """
    ranked = rankdata(dd, axis=1) # rank 1 is close rank 2000 is far
    to_scores = 1 / ranked # 1/1 is highest, 1/2 is higher than before, etc.
    return to_scores


#####################################################
# THE FOLLOWING IS NOT PART OF THE STANDARD PIPELINE
#####################################################
### ABLATION (INPUT IS ORTHOLOG GENES) ###
class OrthologPretrainModel(torch.nn.Module):
    def __init__(self, input_dim, dropout=0, hidden_dim=256, embed_dim=256, species_names=[], vae=False):
        super().__init__()
        
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # self.num_gene_scores = len(gene_scores)
        self.num_species = len(species_names)
        # self.species_to_gene_idx = species_to_gene_idx
        self.sorted_species_names = sorted(species_names)
        
        self.num_genes = 0
        self.vae = vae
        self.num_genes = input_dim
        
        
        if self.vae:
            # Z Encoder
            self.encoder = nn.Sequential(
                    full_block(self.num_genes, hidden_dim, self.dropout),
            )
            self.fc_var = nn.Linear(self.hidden_dim, self.embed_dim)
            self.fc_mu = nn.Linear(self.hidden_dim, self.embed_dim)
            
        else:
            self.encoder = nn.Sequential(
                    full_block(self.num_genes, self.hidden_dim, self.dropout),
                    full_block(self.hidden_dim, self.embed_dim, self.dropout),
            )
            
        # Decoder
        self.px_decoder = nn.Sequential(
            full_block(self.embed_dim + self.num_species, self.hidden_dim, self.dropout),
        )
        
        self.px_scale_decoder = full_block(self.hidden_dim, self.num_genes)
        
        self.px_dropout_decoders =  nn.Sequential(
                nn.Linear(self.hidden_dim, self.num_genes)
        )
        
        self.px_rs = torch.nn.Parameter(torch.randn(self.num_genes))
        

    def forward(self, inp, species):
        batch_size = inp.shape[0]
        
        # Pad the appened expr with 0s to fill all gene nodes
        #expr = torch.zeros(batch_size, self.num_genes).to(inp.device)
        expr = torch.log(inp + 1)
        
        # concatenate the gene embeds with the expression as the last item in the embed
        expr = expr.unsqueeze(1)
        
        # GNN and cluster weights
        clusters = []
        expr_and_genef = expr
            
        encoder_input = expr.squeeze()
        encoded = self.encoder(encoder_input)
        
        if self.vae:
            # VAE 
            mu = self.fc_mu(encoded)
            log_var = self.fc_var(encoded)
        
            encoded = self.reparameterize(mu, log_var)
        else:
            mu = None
            log_var = None
        
        spec_1h = torch.zeros(batch_size, self.num_species).to(inp.device)
        spec_idx = np.argmax(self.sorted_species_names == species)
        spec_1h[:, spec_idx] = 1.
        
        if encoded.dim() != 2:
            encoded = encoded.unsqueeze(0)
        
        decoded = self.px_decoder(torch.hstack((encoded, spec_1h)))
        
        library = torch.log(inp.sum())
        
        # modfiy                
        px_scale = self.px_scale_decoder(decoded)
        px_scale_decode = nn.Softmax(-1)(px_scale.squeeze())
        
        px_drop = self.px_dropout_decoders(decoded)
        px_rate =  torch.exp(library) * px_scale_decode
        px_r = torch.exp(self.px_rs)        
        
        return encoder_input, encoded, mu, log_var, px_rate, px_r, px_drop
    
    
    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout):
        '''https://github.com/scverse/scvi-tools/blob/master/scvi/module/_vae.py'''
        return -ZeroInflatedNegativeBinomial(
                                                mu=px_rate, theta=px_r, zi_logits=px_dropout
                                            ).log_prob(x).sum(dim=-1)
    
    def loss_vae(self, inp, mu, log_var, kld_weight, px_rate, px_r, px_drop):
        recons_loss = torch.sum(self.get_reconstruction_loss(inp, px_rate, px_r, px_drop))
        if self.vae:
            kld_loss = torch.sum(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

            loss = recons_loss + kld_weight * kld_loss
            return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
        else:
            return {'loss': recons_loss}
            

    def loss_ae(self, pred, y):
        loss = F.mse_loss(pred, y) # MSE loss for numerical values
        return loss
    
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    
    
class OrthologMetricModel(torch.nn.Module):
    def __init__(self, input_dim=2000, dropout=0, hidden_dim=256, embed_dim=256, species_names=[], vae=False):
        super().__init__()
        
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        self.num_species = len(species_names)
        self.sorted_species_names = sorted(species_names)
        
        self.num_genes = 0
        self.vae = vae
        self.num_genes = input_dim
        
        if self.vae:
            # Z Encoder
            self.encoder = nn.Sequential(
                    full_block(self.input_dim, hidden_dim, self.dropout),
            )
            self.fc_var = nn.Linear(self.hidden_dim, self.embed_dim)
            self.fc_mu = nn.Linear(self.hidden_dim, self.embed_dim)
            
        else:
            self.encoder = nn.Sequential(
                    full_block(self.input_dim, self.hidden_dim, self.dropout),
                    full_block(self.hidden_dim, self.embed_dim, self.dropout),
            )
        

    def forward(self, inp, species):
        batch_size = inp.shape[0]
        
        # input is now the anchor values themselves
        encoded = self.encoder(inp)
        
        if self.vae:
            # VAE 
            mu = self.fc_mu(encoded)
            return mu
        else:
            return encoded
    
    
    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout):
        '''https://github.com/scverse/scvi-tools/blob/master/scvi/module/_vae.py'''
        return -ZeroInflatedNegativeBinomial(
                                                mu=px_rate, theta=px_r, zi_logits=px_dropout
                                            ).log_prob(x).sum(dim=-1)
    
    def loss_vae(self, inp, mu, log_var, kld_weight, px_rate, px_r, px_drop):
        recons_loss = torch.sum(self.get_reconstruction_loss(inp, px_rate, px_r, px_drop))
        if self.vae:
            kld_loss = torch.sum(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

            loss = recons_loss + kld_weight * kld_loss
            return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
        else:
            return {'loss': recons_loss}
            

    def loss_ae(self, pred, y):
        loss = F.mse_loss(pred, y) # MSE loss for numerical values
        return loss
    
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
