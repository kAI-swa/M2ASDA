import os
import anndata2ri
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import io
from sklearn import metrics
from rpy2.robjects import r, globalenv
from rpy2.robjects.packages import importr
from typing import Optional


def monocle_prepare(train: ad.AnnData, random_state: int, save_dir: Optional[str] = None, save_seurat: bool = False):
    if not save_dir:
        if os.path.exists("./temp"):
            print("Save into current temp file")
        else:
            os.mkdir("./temp")
        save_dir = "./temp"
    else:
        if not os.path.exists(save_dir):
            print("Not exists yet")
            os.mkdir(save_dir)

    # Convert Anndata to Seurat Object
    try:
        train.obsm["X_umap"] == None
    except KeyError as e:
        print("Not computing Umap yet")
        sc.tl.pca(train)
        sc.pp.neighbors(train)
        sc.tl.umap(train)
    else:
        print("Use computed Umap")
    finally:
        train.layers["raw"] = train.X.copy()
        io.mmwrite(save_dir + "/counts.mtx", train.layers['raw'])
        cell_metadata = train.obs.copy()
        cell_metadata['Barcode'] = cell_metadata.index
        barcode = cell_metadata.index.to_series()
        barcode.name = 'barcode'
        barcode.to_csv(save_dir + "/barcode.csv", index=None)
        cell_metadata['UMAP1'] = train.obsm['X_umap'][:, 0]
        cell_metadata['UMAP2'] = train.obsm['X_umap'][:, 1]
        cell_metadata.to_csv(save_dir + "/cell_metadata.csv", index=None)

        gene_metadata = train.var.copy()
        gene_metadata['gene'] = gene_metadata.index
        gene_metadata.to_csv(save_dir + "/gene_metadata.csv", index=None)
    
    globalenv['seed'] = random_state
    globalenv['train'] = anndata2ri.py2rpy(train)
    r.saveRDS(anndata2ri.py2rpy(train), file= save_dir + "/sce_data.rds")
    