"""
Created on March 5, 2025

@author: Fabio Zanini
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch.utils.data as data
import torch
import numpy as np
import anndata
import scanpy as sc


def data_to_torch_X(X):
    if isinstance(X, sc.AnnData):
        X = X.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    return torch.from_numpy(X).float()


class ExperimentDatasetSingle(data.Dataset):
    def __init__(
        self,
        data: Union[anndata.AnnData, np.ndarray],
        library: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray],
        ref: Union[torch.Tensor, np.ndarray],
        species: Union[torch.Tensor, np.ndarray],
        batch_lab: Union[None, torch.Tensor, np.ndarray],
    ) -> None:
        super().__init__()

        X = data_to_torch_X(data)
        num_cells, num_genes = X.shape
        self.x = X
        self.num_cells = num_cells
        self.num_genes = num_genes

        self.library = torch.LongTensor(library)
        self.y = torch.LongTensor(y)
        self.ref_labels = torch.LongTensor(ref)
        self.species = torch.LongTensor(species)
        # if we have an additional batch column like for tissue
        if batch_lab is not None:
            self.batch_labels = torch.LongTensor(batch_lab)
        else:
            self.batch_labels = None

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx < self.num_cells:
                if self.batch_labels is not None:
                    batch_ret = self.batch_labels[idx]
                else:
                    batch_ret = None

                return (
                    self.x[idx],
                    self.library[idx],
                    self.y[idx],
                    self.ref_labels[idx],
                    self.species[idx],
                    batch_ret,
                )
            else:
                idx -= self.num_cells
            raise IndexError
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return self.num_cells

    def get_dim(self) -> int:
        return self.num_genes


# NOTE: this is kind of awkward, ok for now
def single_species_collate_fn(
    batch: List[Tuple[torch.FloatTensor, torch.LongTensor, str]]
) -> tuple:
    has_batch_labels = False

    # NOTE: this is like a fancy zipping with optional last column
    res = [[], [], [], [], [], None]
    for data, library, labels, refs, species, batch_labels in batch:
        res[0].append(data)
        res[1].append(library)
        res[2].append(labels)
        res[3].append(refs)
        res[4].append(species)

        if batch_labels is not None:
            has_batch_labels = True
            if res[5] is None:
                res[5] = []
            res[5].append(batch_labels) 

    res = tuple(torch.stack(x) if x is not None else None for x in res)

    return res
