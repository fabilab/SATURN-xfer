# SATURN-xfer / SATURN-gen: Transfer and generative learning for cross-species single cell data
This repository contains the implementations of SATURN-xfer and SATURN-gen as described in the preprint by [Xu et al. 2025](https://doi.org/10.1101/2025.02.19.639005) (currently in review):

https://doi.org/10.1101/2025.02.19.639005

This repository *also* contains a fork of the original SATURN code with a number of bug fixes. We have attempted to upstream those fixes but our attempts to collaborate with the original author have been, for now, unsuccessful. One of the most important fixes was that the original codebase fails to run for more than a handful of species or, occasionally, it will produce scrambled results that are not interpretable.

## Who is this repository for?
- If you are trying to run SATURN-xfer to perform transfer learning with a pretrained SATURN model, this repo is for you.
- If you are trying to run SATURN-gen to generate virtual cell atlases/synthetic cell atlases/single cell digital twins with a pretrained SATURN model, this repo is for you.
- If you are trying to run SATURN, we recommend you use this repo since the original codebase has critical bugs that compromise its results. 

## Conceptual overview
This repo hosts three model architectures: SATURN (with bugfixes), SATURN-xfer, and SATURN-gen.

### SATURN
This is the original model published in [Nature Methods](https://www.nature.com/articles/s41592-024-02191-z) by Rosen et al. It takes two types of input:
- Single cell transcriptomic data from multiple species (one h5ad file per species). Cell metadata must include a column describing cell type labels, which are used for training. They need to be strings. It is allowed to use uninterpretable labels such as "cluster_0", "cluster_1", etc. although doing so is likely to make downstream analysis more challenging.
- Embeddings of protein sequences from multiple species through a protein language model (PLM). This is traditionally ESM1b, ESM2, ESMC (we tested all three), or similar models. The code assumes the embeddings are floating point numbers.

The three conceptual steps are as follows:
1. Preprocess counts, labels, and protein embeddings: This prepares data structures for model training. Among other things, this includes a hard feature selection for each species (typically the top 1,000 or 2,000 overdispersed genes): not all genes are considered for each species.
2. Pretraining of the autoencoder: This adjusts all layers of the model so that counts from each species are reconstructed after passing through the latent space bottleneck. Data from multiple species are only very weakly harmonised after this step.
3. "Metric" training: This does the heavy lifting of the harmonisation across species. It does so by pulling together cells, in latent space, that respect cell type annotation after certain triangulations between pairs of species. Data from multiple species are fully harmonised after this step, with "sibling" cell types ideally fully overlapping.

SATURN stores model and inference on the training data after pretraining (step 2) as well as after metric training (step 3), so one can look at the results of both and compare if desired. In practice, the data before metric training is hardly usable, and the model before metric training is only useful for testing purposes, namely trying out many variations of the metric training schedule and figuring out the best one for a given purpose.

**NOTE:** Because of the triplet loss used by SATURN, cellular granularity within annotated cell types might actually decrease *within* each species compared to the original data. The upside is that within each cell type, cells are harmonised *across* species. This is an intrinsic design choice of SATURN and not an accident of specific training runs.

### SATURN-xfer
This is the transfer learning model introduced by [Xu et al. 2025](https://doi.org/10.1101/2025.02.19.639005). It takes the following inputs:
- A fully trained (pretraining + metric training) SATURN model on multiple species.
- Output data from inference on the original training data through the trained model.
- A single cell dataset from a new species. Cell type annotations are neither needed nor used if present.
- Embeddings of protein sequences from the new species through the *same* PLM as the original, trained model.

The conceptual steps of SATURN-xfer are as follows:
- Load the trained model and training inference data.
- Load and preprocess the new species data and PLM embeddings (same note as above on feature selection).
- Fine-tuning of the model: This adjust the model to accomodate for the new species using a similar loss via triangulation exclusively with the new species. The original data from the training species is untouched.

**NOTE:** You do need single-cell transcriptomic data from the new species to run SATURN-xfer. If you want to predict cell embeddings without experimental data, use SATURN-gen below.

### SATURN-gen
This is the generative model introduced by [Xu et al. 2025](https://doi.org/10.1101/2025.02.19.639005). It takes the following inputs:
- A fully trained (pretraining + metric training) SATURN model on multiple species.
- Output data from inference on the original training data through the trained model, for *one* species which you would like to use as a guide.
- Embeddings of protein sequences from the new species through the *same* PLM as the original, trained model.

The conceptual steps of SATURN-gen are as follows:
- Load the trained model and training inference data (for the guide species only).
- Load and prerocess the new species' PLM embeddings.
- Infer virtual cell atlas for the new species guided by its own PLM embeddings on top of the guide species' counts.

**NOTE:** SATURN-gen is a zeroshot prediction with no training. Ideally you would choose a species that is evolutionarily close to the new target species as a guide: the model cannot predict new cell types that might have emerged since their most recent common ancestor.

## Installation
Please refer to the original SATURN repository for instructions, cloning this repo instead of the original one. PyTorch is notoriously fragile across GPU generations, software versions, and CUDA versions. If you cannot get it to install at all, please open an issue on this repository and we will attempt to help you.

## Usage
### SATURN
To use SATURN, see the original repo for instructions. In a nutshell, you need to run the following Python script from an environment that has the dependencies and drivers installed:

- `train-saturn.py`

This can be used for pretraining, metric training, or both in series. The latter would be the most common usage case apart from testing.

Please refer to the paper or the [original repo](https://github.com/snap-stanford/saturn) for more details.

### SATURN-xfer
To use SATURN-xfer, run the following Python script from an environment that has the dependencies and drivers installed:

- `saturn-xfer.py`

The most notable command-line options for a transfer learning run are:

- `--species`: Name of the new species, for convenience only. This does not affect the computations.
- `--in_adata_path`: Path to the h5ad file for the new species.
- `--in_embeddings_path`: Path to the pt embedding summary file (from ESM etc.) for the new species. These must be in the same file as for a normal SATURN run.
- `--training_csv_path`: Path to the csv file previously used to train the SATURN base model. This file contains itself paths to h5ad and pt files for each training species.
- `--guide_species`: Name of the species used as a guide for the transfer learning. This species must be present in the csv file above.
- `--pretrain_model_path`: Path to the SATURN model after pretraining. This contains the decoder layers used for transfer learning.
- `--metric_model_path`: Path to the SATURN model after metric training. This contains the encoder layers used for transfer learning.
- `--trained_adata_path`: Path to an h5ad file containing the **latent space** counts for the guide species after metric training. It is allowed for this file to contains more than the guide species, as data from all other species will be silently ignored. Typically, people would just run the initial SATURN training and then use the output h5ad in encoder space, containing all species, for all kinds of transfer learning downstream.
- `--train`: If used, fine-tune the transfer learning model (see below). If unused, run zeroshot inference.

The output path for the data is controlled by `--work_dir`. The output path for the trained (fine-tuned) transfer model is controlled by `--xfer_model_path`.

#### Fine-tuning or not?
SATURN-xfer can run in either one of two modes: zeroshot and fine-tuning:

- **zeroshot** is faster because it does not require any additional training: it loads the new species data, guesses the relationships of its PLM embeddings onto the autoencoder, and runs inference with all weights frozen.
- **fine-tuning** (triggered by the `--train` command-line flag) leads to more accurate results because it loads the new species data, guesses the relationships of its PLM embeddings onto the autoencoder, runs a training schedule aimed at improving the harmonisation of the new species with the guide species, and finally runs inference.

**Both modes are much, much faster than retraining SATURN from scratch**. In general, we recommend fine tuning since the few additional minutes needed for this task are rarely a bottleneck in a research project. However, if you are doing exotic things like running hundreds of SATURN-xfer instances for large scale predictions, benchmarking, etc. you might want to check out the zeroshot mode. Zeroshot SATURN-xfer is also conceptually quite close to SATURN-gen, so it has some educational value if you are trying to learn how these models operate.

### SATURN-gen
To use SATURN-gen, run the following Python script from an environment that has the dependencies and drivers installed:

- `saturn-gen.py`

The most notable command-line options for a generative learning run are:

- `--species`: Name of the new species, for convenience only. This does not affect the computations.
- `--in_embeddings_path`: Path to the pt embedding summary file (from ESM etc.) for the new species. These must be in the same file as for a normal SATURN run.
- `--training_csv_path`: Path to the csv file previously used to train the SATURN base model. This file contains itself paths to h5ad and pt files for each training species.
- `--guide_species`: Name of the species used as a guide for the transfer learning. This species must be present in the csv file above.
- `--pretrain_model_path`: Path to the SATURN model after pretraining. This contains the decoder layers used for transfer learning.
- `--metric_model_path`: Path to the SATURN model after metric training. This contains the encoder layers used for transfer learning.
- `--trained_adata_path`: Path to an h5ad file containing the **latent space** counts for the guide species after metric training. It is allowed for this file to contains more than the guide species, as data from all other species will be silently ignored. Typically, people would just run the initial SATURN training and then use the output h5ad in encoder space, containing all species, for all kinds of transfer learning downstream.

The output path for the data is controlled by `--work_dir`. The output path for the trained (fine-tuned) transfer model is controlled by `--gen_model_path`.

**NOTE**: Notice that unlike SATURN-xfer, SATURN-gen does not have the option `--in_adata_path` because it is a *generative* model: it generates virtual/synthetic data rather than requiring it as input.

## Citing

If you find this code useful, please cite the [preprint](https://doi.org/10.1101/2025.02.19.639005), below in BibTex format:

```
@article {Xu2025.02.19.639005,
	author = {Xu, Ying and Gatt, Catherine and Kaymak, Esra and Kikuchi, Kensei and Bourguignon, Thomas and Zanini, Fabio},
	title = {Exploration and generation of cell transcriptomes over deep evolutionary time},
	elocation-id = {2025.02.19.639005},
	year = {2025},
	doi = {10.1101/2025.02.19.639005},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Whole-organism cell atlases have painted the cellular landscapes of individual species; however, comparing cells across the tree of life remains challenging. Most cross-species analyses are restricted to orthologous genes, while the complexity of atlas data constitutes an access barrier for many researchers. We developed a computational strategy to accelerate the exploration of cellular identities at scale. We integrated 30 atlases detailing the expression of 861,013 genes by 2,645,508 animal and plant cells and trained a universal model of cellular transcriptomes to track cell type diversification over evolutionary times. Transfer learning achieved cell type annotation of a de novo atlas of the insect Cryptocercus punctulatus within minutes. We devised a generative artificial intelligence approach to construct virtual cell atlases from genome sequences and used it to synthesise an atlas of the Tasmanian tiger, extinct since 1936. We then reduced the footprint of extant atlases by 100 times while retaining crucial information and developed interfaces to answer dozens of query types within seconds, boosting atlas exploration by \~{}50,000 times.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2025/04/07/2025.02.19.639005},
	eprint = {https://www.biorxiv.org/content/early/2025/04/07/2025.02.19.639005.full.pdf},
	journal = {bioRxiv}
}
```

## Data Availability
TODO

|Link|Description|
|----|-----------|
|http://snap.stanford.edu/saturn/data/ah_atlas_export.tar.gz|5 Species Alignment of Aqueous Humor Outflow (AH) Atlas|
|http://snap.stanford.edu/saturn/data/frog_zebrafish_export.tar.gz|Frog and Zebrafish Embryogenesis Alignment|
|http://snap.stanford.edu/saturn/data/tabula_mammal_export.tar.gz|Tabula Sapiens, Muris and Microcebus Coarse Whole Atlas Alignments and Individual Tissue alignemnts|
|http://snap.stanford.edu/saturn/data/protein_embeddings.tar.gz|Protein Embeddings for analyzed species|







