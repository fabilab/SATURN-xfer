# SATURN-xfer / SATURN-gen: Transfer and generative learning for cross-species single cell data
This repository contains the implementations of SATURN-xfer and SATURN-gen as described in the preprint by [Xu et al. 2025](https://doi.org/10.1101/2025.02.19.639005) (currently in review):

https://doi.org/10.1101/2025.02.19.639005

This repository *also* contains a fork of the original SATURN code with a number of bug fixes. We have attempted to upstream those fixes but our attempts to collaborate with the original author have been, for now, unsuccessful. One of the most important fixes was that the original codebase fails to run for more than a handful of species or, occasionally, it will produce scrambled results that are not interpretable.

## Who is this repository for?
- If you are trying to run SATURN-xfer to perform transfer learning with a pretrained SATURN model, this repo is for you.
- If you are trying to run SATURN-gen to generate virtual cell atlases/synthetic cell atlases/single cell digital twins with a pretrained SATURN model, this repo is for you.
- If you are trying to run SATURN, we recommend you use this repo since the original codebase has critical bugs that compromise its results. 

## Conceptual overview
This repo hosts three model architectures: SATURN (with fixes), SATURN-xfer, and SATURN-gen.

### SATURN
This is the original model published in [Nature Methods](https://www.nature.com/articles/s41592-024-02191-z) by Rosen et al. SATURN takes two types of input:
- Single cell transcriptomic data from multiple species (one h5ad file per species). Cell metadata must include a column describing cell type labels, which are used for training. They need to be strings. It is allowed to use uninterpretable labels such as "cluster_0", "cluster_1", etc. although doing so is likely to make downstream analysis more challenging.
- Embeddings of protein sequences from multiple species through a protein language model (PLM). This is traditionally ESM1b, ESM2, ESMC (we tested all three), or similar models. The code assumes the embeddings are floating point numbers.

The conceptual steps of SATURN are as follows:
- Preprocess counts, labels, and protein embeddings: This prepares data structures for model training. Among other things, this includes a hard feature selection for each species (typically the top 1,000 or 2,000 overdispersed genes): not all genes are considered for each species.
- Pretraining of the autoencoder: This adjusts all layers of the model so that counts from each species are reconstructed after passing through the latent space bottleneck. Data from multiple species are only very weakly harmonised after this step.
- "Metric" training: This does the heavy lifting of the harmonisation across species. It does so by pulling together cells, in latent space, that respect cell type annotation after certain triangulations between pairs of species. Data from multiple species are fully harmonised after this step, with "sibling" cell types ideally fully overlapping.

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

### SATURN-xfer
To use SATURN-xfer, run the following Python script from an environment that has the dependencies and drivers installed:

- `saturn-xfer.py`

To run SATURN-xfer, you usually want to XXX (TODO TODO)

SATURN-xfer can run in either one of two modes: zeroshot and fine-tuning:

- **zeroshot** is faster because it does not require any additional training: it loads the new species data, guesses the relationships of its PLM embeddings onto the autoencoder, and runs inference with all weights frozen.
- **fine-tuning** is more accurate because it loads the new species data, guesses the relationships of its PLM embeddings onto the autoencoder, then runs a training schedule aimed at improving the harmonisation of the new species with the rest of the model, and finally runs inference.

Both modes are much, much faster than retraining SATURN from scratch. In general, we recommend fine tuning since the few additional minutes needed for this task are rarely a bottleneck in a research project. However, if you are doing exotic things like running hundreds of SATURN-xfer instances for large scale predictions, benchmarking, etc. you might want to check out the zeroshot mode. Zeroshot SATURN-xfer is also conceptually quite close to SATURN-gen, so it has some educational value if you are trying to learn how these models operate.


Please refer to the paper or the [original repo](https://github.com/snap-stanford/saturn) for more details. The original repo appears unmaintained since 2024.


PyTorch implementation of [SATURN](https://www.biorxiv.org/content/10.1101/2023.02.03.526939v1), a deep learning approach that couples gene expression with protein representations learnt using large protein language models for cross-species integration. The key idea in SATURN is to map cells from all datasets to a shared space of functionally related genes that we name macrogenes. Using macrogenes, SATURN is uniquely able to detect functionally related genes co-expressed across species.

<p align="center">
<img src="https://github.com/snap-stanford/saturn/blob/main/saturn_fig.png" width="1100" align="center">
</p>

SATURN takes as an input:
- multiple scRNA-seq *count* datasets from different species (AnnDatas), with cell type annotations.
- protein embeddings generated by large language models (TorchDicts)

SATURN is composed of three modules:
- Macrogene initialization with Kmeans (scipy)
- Pretraining conditional autoencoder (scVI ZINB loss)
- Fine tuning cell clusters with weakly supervised metric learning

`Vignettes/frog_zebrafish_embryogenesis/Train SATURN.ipynb` has an example of running SATURN, scoring the results, and running differential expression on Macrogenes.

`protein_embeddings/Generate Protein Embeddings.ipynb` has an example of creating and formatting protein embeddings.


## Setting up SATURN

### Protein Embeddings
To run SATURN you will need protein embeddings. To generate your own protein embeddings, please see instructions in the `protein_embeddings` directory. 

Alternatively, you can use one of the publicly available protein embedding datasets we have provided.

### Requirements

SATURN requires installation of a number of python modules. Please install them via:

```
pip install -r requirements.txt
pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

This install should talke under 10 minutes.

## Running SATURN

To run SATURN, use the `train-saturn.py` file.

`Vignettes/frog_zebrafish_embryogenesis/Train SATURN.ipynb` has an example of running SATURN, scoring the results, and running differential expression on Macrogenes.

`train-saturn.py` accepts a number of required arguments:

- `in_data`: a csv file with the following columns

`path`: path to an Scanpy h5 file with scRNA-seq count values in the `X` field, and at least one column of cell type labels.
`species`: name of the Adata species
`embedding_path`: optionally, you can specify a path to gene embedding torch files per adata. Otherwise, SATURN will map protein embeddings based on the default paths in `data/gene_embeddings.py`

- `in_label_col`: the name of a column present in all input AnnDatas that contains cell type labels that should be used for metric learning.
- `ref_label_col`: the name of a column present in all input AnnDatas that contains an additional cell type label that can be used to aid metric learning if the additional argumetn `--use_ref_labels` is also passed, otherwise this column will just be included in SATURN's output AnnData. If you don't have an additional column, just set this value to the same column as `in_label_col` and do not pass the `--use_ref_labels` argument.

### Optional Arguments

`train-saturn.py` also contains a number of optional arguments, a number of which are detailed below:

- `hv_genes`: Number of highly variable genes each AnnData should be subset to. Defaulted to 8000
- `num_macrogenes`: The number of macrogenes.
- `score_adatas`: should the pretraining AnnData and metric learning AnnDatas be scored? See the section on scoring AnnDatas for more info.
- `centroids_init_path`: This is a path to a `.pkl` file that contains a copy of gene to macrogene scores after HV subset and centroids initialization, but before SATURN pretraining. When you pass this argument, SATURN will look for a file at this location and use that file to initialize macrogene weights. If the file does not exist, SATURN will instead create centroids and save to this location. This can be used to skip the centroids creation step if running SATURN multiple times on the same datasets with the same genes (after HV gene selection).
- `device_num` which GPU to use

## SATURN Output

`train-saturn.py` will output a number of files during and following training.

All of these files will be in the same directory, and have the same prefix, `run_name`, which SATURN will display at the conclusion of training.

**AnnDatas:**
- Final integrated AnnData: `{run_name}.h5ad`

This AnnData will have SATURN embeddings in the `.X` slot.
In `.obs`, there will be columns for the original unmodified labels from `in_label_col` in the slot `labels2`, and the slot `labels` will contain those labels but with their corresponding `species` value preprended. The `ref_labels` column will contain values from the `ref_label_col`, and the `species` column will contain species values.

In the AnnData's `.obsm`, there will be a slot called `macrogenes` that will contain the macrogene values for each cell.

- Pretraining AnnData: `{run_name}_pretrain.h5ad`

The pretraining AnnData has the same format the final AnnData.

**Final Macrogene Weights:**
- Gene to macrogene final weights file: `{run_name}_genes_to_macrogenes.pkl`

**Log Files:**

There are a number of additional log files outputted:
- `{run_name}_triplets.csv` A csv with information about which triplets were mined during metric learning
- `{run_name}_epoch_scores.csv` A csv with information about scoring during metric learning
- `{run_name}_celltype_id.pkl` A pkl of a dictionary containing cell type to categorical codings used for interpreting the other log files

## Scoring SATURN outputs

To score SATURN outputs, either after training or during training, you will need a csv file that maps cell types between each species.

This csv should have the columns:

`{species 1 name}_cell_type`, `{species 2 name}_cell_type`... `{species n name}_cell_type`

and each row should contain the cell types from each species that should be mapped to eachother.

If a cell type is unique, you can just leave the other species' values blank.

To score while training SATURN, add the argument `--score_adatas` and pass the name of this csv file to `--ct_map_path`. 

To score SATURN outputs after training is finished, use the `score_adata.py` file.

`score_adata.py` takes the following arguments:

- `adata`: path to the SATURN formatted AnnData to score
- `species1`: The species whose embeddings will be used to train a simple cell type classifier
- `species2`: The species whose embeddings will be used to test a simple cell type classifier
- `ct_map_path`: path to the csv file mapping cell types between species
- `label`: the label column for cell types. If scoring a SATURN output, this should be `labels2`
- `scores`: the number of scores that should be calculated.

## Citing

If you find our paper and code useful, please consider citing the [preprint](https://www.biorxiv.org/content/10.1101/2023.02.03.526939v1):

```
@article{saturn2023,
  title={Towards Universal Cell Embeddings: Integrating Single-cell RNA-seq Datasets across Species with SATURN},
  author={Rosen, Yanay and Brbi{\'c}, Maria and Roohani, Yusuf and Swanson, Kyle and Ziang, Li and Leskovec, Jure},
  journal={bioRxiv},
  doi = {10.1101/2023.02.03.526939},
  year={2023},
}
```


## Data Availability

|Link|Description|
|----|-----------|
|http://snap.stanford.edu/saturn/data/ah_atlas_export.tar.gz|5 Species Alignment of Aqueous Humor Outflow (AH) Atlas|
|http://snap.stanford.edu/saturn/data/frog_zebrafish_export.tar.gz|Frog and Zebrafish Embryogenesis Alignment|
|http://snap.stanford.edu/saturn/data/tabula_mammal_export.tar.gz|Tabula Sapiens, Muris and Microcebus Coarse Whole Atlas Alignments and Individual Tissue alignemnts|
|http://snap.stanford.edu/saturn/data/protein_embeddings.tar.gz|Protein Embeddings for analyzed species|







