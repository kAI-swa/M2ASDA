import os
import anndata2ri
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn import metrics
from rpy2.robjects import r, globalenv
from rpy2.robjects.packages import importr


def monocle_v3(train: ad.AnnData, random_state: int):
    importr("monocle3")
    importr("SingleCellExperiment")
    globalenv['seed'] = random_state
    globalenv['train'] = anndata2ri.py2rpy(train)
    r("""
    set.seed(seed)
    expression_matrix <- assay(train, 'X')
    cell_metadata = colData(train)
    gene_metadata = rowData(train)
    
    cds <- new_cell_data_set(expression_matrix,
                         cell_metadata = cell_metadata,
                         gene_metadata = gene_metadata)
    cds <- preprocess_cds(cds, num_dim = 50)
    cds <- reduce_dimension(cds)
    plot_cells(cds,
               color_cells_by = "cell.type",
               label_groups_by_cluster=FALSE,
               label_leaves=FALSE,
               label_branch_points=FALSE)
    """)
    