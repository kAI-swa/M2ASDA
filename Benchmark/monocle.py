import os
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import io
from sklearn import metrics
from rpy2.robjects import r, globalenv
from rpy2.robjects.packages import importr
from typing import Optional


def monocle_v3(train: ad.AnnData, random_state: int, save_dir: Optional[str] = None, save_seurat: bool = False):
    if not save_dir:
        os.mkdir("./temp")
        save_dir = "./temp"
    else:
        if not os.path.exists(save_dir):
            raise FileNotFoundError("Can not find Filepath")

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
    
    importr("monocle3")
    importr("Matrix")
    importr("Seurat")
    globalenv['seed'] = random_state
    globalenv['save_seurat'] = save_seurat
    globalenv['save_dir'] = save_dir
    globalenv['counts_dir'] = os.path.join(save_dir + "/counts.mtx")
    globalenv['barcode_dir'] = os.path.join(save_dir + "/barcode.csv")
    globalenv['cellmeta_dir'] = os.path.join(save_dir + "/cell_metadata.csv")
    globalenv['genemeta_dir'] = os.path.join(save_dir + "/gene_metadata.csv")

    r("""
    set.seed(seed)
    ## load the data from disk into R environment
    counts <- ReadMtx(mtx=counts_dir, cells=barcode_dir, features=genemeta_dir, feature.column=1)
    cell_metadata = read.csv(cellmeta_dir)
    gene_metadata = read.csv(genemeta_dir)
    
    rownames(counts) <- cell_metadata$Barcode
    colnames(counts) <- gene_metadata$gene
    
    seurat <- CreateSeuratObject(counts = t(counts))
    ## Set the meta data
    seurat@meta.data <- cbind(cell_metadata, seurat@meta.data)
    rownames(seurat@meta.data) <- colnames(seurat)
    seurat <- AddMetaData(object = seurat, metadata = gene_metadata, key = "gene")
    
    ## Reset the UMAP embeddings with previous runned umap embeddings
    seurat <- RunPCA(seurat, features = VariableFeatures(object = seurat))
    seurat <- RunUMAP(seurat, dims = 1:30)
    runned_umap <- seurat@meta.data[, c('UMAP1', 'UMAP2')]
    colnames(runned_umap) <- c('UMAP_1', 'UMAP_2')
    seurat@reductions$umap@cell.embeddings <- as.matrix(runned_umap)
    
    ## if want to save the seurat object into disks
    if (save_seurat == True) {
        saveRDS(seurat, file.path(save_dir,'seo_annotated.rds'))
    }
    
    ## Build the cell_data_set object for monocle3

    ### gene annotations
    gene_annotation <- as.data.frame(rownames(seurat@reductions$pca@feature.loadings), \
    row.names = rownames(seurat@reductions$pca@feature.loadings))
    colnames(gene_annotation) <- "gene_short_name"
    
    ### gene expression matrix
    expression_matrix <- GetAssayData(object = seurat)
    
    cds <- new_cell_data_set(expression_matrix, \
    cell_metadata = cell_metadata, gene_metadata = gene_annotation)
    cds@reducedDims@listData[["UMAP"]] <-seurat@reductions[["umap"]]@cell.embeddings

    cds <- learn_graph(cds)
    plot_cells(cds,
               color_cells_by = "cell.type",
               label_groups_by_cluster=FALSE,
               label_leaves=FALSE,
               label_branch_points=FALSE)
    """)
