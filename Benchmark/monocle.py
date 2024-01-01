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


def monocle_v3(train: ad.AnnData, random_state: int, save_dir: Optional[str] = None, save_seurat: bool = False):
    if not save_dir:
        if os.path.exists("./temp"):
            print("Save into current temp file")
        else:
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
    globalenv['train'] = anndata2ri.py2rpy(train)
    globalenv['save_seurat'] = save_seurat
    globalenv['save_dir'] = save_dir
    globalenv['counts_dir'] = os.path.join(save_dir + "/counts.mtx")
    globalenv['barcode_dir'] = os.path.join(save_dir + "/barcode.csv")
    globalenv['cellmeta_dir'] = os.path.join(save_dir + "/cell_metadata.csv")
    globalenv['genemeta_dir'] = os.path.join(save_dir + "/gene_metadata.csv")

    r("""
    set.seed(seed)
    options(warn=-1)
    options(browser = "firefox")
    ## load the data from disk into R environment
    counts <- assay(train,'X')
    cell_metadata = read.csv(cellmeta_dir)
    gene_metadata = read.csv(genemeta_dir)
    
    colnames(counts) <- cell_metadata$Barcode
    rownames(counts) <- gene_metadata$gene
    
    seurat <- CreateSeuratObject(counts=counts)
    ## Set the meta data
    seurat@meta.data <- cbind(cell_metadata, seurat@meta.data)
    rownames(seurat@meta.data) <- colnames(seurat)
    
    ## Reset the UMAP embeddings with previous runned umap embeddings
    seurat <- FindVariableFeatures(seurat, selection.method = "vst", nfeatures = 2000)
    seurat <- ScaleData(seurat)
    seurat <- RunPCA(seurat, features = VariableFeatures(object = seurat), npcs=30, verbose=FALSE)
    seurat <- RunUMAP(seurat, dims = 1:10)
    runned_umap <- seurat@meta.data[, c('UMAP1', 'UMAP2')]
    colnames(runned_umap) <- c('UMAP_1', 'UMAP_2')
    seurat@reductions$umap@cell.embeddings <- as.matrix(runned_umap)
    
    ## if want to save the seurat object into disks
    if (save_seurat == TRUE) {
        saveRDS(seurat, file.path(save_dir,'seo_annotated.rds'))
    }
    
    ## Build the cell_data_set object for monocle3

    ### gene annotations
    gene_annotation <- as.data.frame(rownames(seurat@reductions$pca@feature.loadings), \
    row.names = rownames(seurat@reductions$pca@feature.loadings))
    colnames(gene_annotation) <- "gene_short_name"
    
    ### cell_metadata
    cell_metadata <- as.data.frame(seurat@assays[["RNA"]]@counts@Dimnames[[2]], row.names = seurat@assays[["RNA"]]@counts@Dimnames[[2]])
    colnames(cell_metadata) <- "barcode"
    
    ### expression matrix
    New_matrix <- seurat@assays[["RNA"]]@counts
    New_matrix <- New_matrix[rownames(seurat@reductions[["pca"]]@feature.loadings), ]
    expression_matrix <- New_matrix
    
    cds <- new_cell_data_set(expression_matrix, \
    cell_metadata = cell_metadata, gene_metadata = gene_annotation)
    
    cds <- preprocess_cds(cds, num_dim = 50)
    cds <- reduce_dimension(cds, reduction_method="UMAP")
    
    recreate.partition <- c(rep(1, length(cds@colData@rownames)))
    names(recreate.partition) <- cds@colData@rownames
    recreate.partition <- as.factor(recreate.partition)

    cds@clusters@listData[["UMAP"]][["partitions"]] <- recreate.partition
    cds@int_colData@listData[["reducedDims"]][["UMAP"]] <-seurat@reductions[["umap"]]@cell.embeddings
    
    cds <- cluster_cells(cds, cluster_method="louvain")
    cds <- learn_graph(cds)
    plot_cells(cds, color_cells_by = "partition")
    cds <- order_cells(cds, reduction_method = "UMAP")
    
    plot_cells(cds,
               color_cells_by = "pseudotime",
               label_cell_groups=FALSE,
               label_leaves=FALSE,
               label_branch_points=FALSE,
               graph_label_size=1.5)
    """)
    
