import os
import anndata2ri
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn import metrics
from rpy2.robjects import r, globalenv
from rpy2.robjects.packages import importr


def CAMLU(train: ad.AnnData, test: ad.AnnData, random_state: int):
    importr("CAMLU")
    globalenv['seed'] = random_state
    globalenv['train'] = anndata2ri.py2rpy(train)
    globalenv['test'] = anndata2ri.py2rpy(test)
    r("""
    set.seed(seed)
    label <- CAMLU(x_train = assay(train,'X'),
                   x_test = assay(test,'X'),
                   full_annotation = FALSE,
                   ngene=3000, lognormalize=TRUE)
    """)
    label = list(r('label'))
    return label


def SCEVAN(data: ad.AnnData, random_state: int):
    importr("SCEVAN")
    globalenv['seed'] = random_state
    globalenv['data'] = anndata2ri.py2rpy(data)
    r("""
    set.seed(seed)
    data <- as.matrix(assay(data))
    result <- pipelineCNA(data, par_cores=1, SUBCLONES=FALSE)
    """)
    label = list(r('result$class'))
    return label


def copyKAT(data: ad.AnnData, random_state: int):
    importr("copykat")
    globalenv['seed'] = random_state
    globalenv['data'] = anndata2ri.py2rpy(data)
    r("""
    set.seed(seed)
    data <- as.matrix(assay(data))
    copykat.test <- copykat(rawmat=data, 
                        id.type="S", 
                        cell.line="no", 
                        ngene.chr=5, 
                        win.size=25, 
                        KS.cut=0.1, 
                        sam.name="test", 
                        distance="euclidean", 
                        n.cores=1)

    label <- data.frame(copykat.test$prediction)
    """)
    label = list(r('label$copykat.pred'))

    datanames = os.listdir(os.getcwd())
    for dataname in datanames:
        if dataname.startswith('test_copykat'):
            os.remove(os.getcwd() + f'/{dataname}')

    return [i.split(':')[1] if ':' in i else i for i in label]


def scPred(train: ad.AnnData, test: ad.AnnData, random_state: int):
    importr("scPred")
    importr("Seurat")
    importr("magrittr")
    globalenv['seed'] = random_state
    globalenv['train'] = anndata2ri.py2rpy(train)
    globalenv['test'] = anndata2ri.py2rpy(test)
    r("""
    set.seed(seed)
    cell_type <- colData(train)
    train <- CreateSeuratObject(assay(train, 'X'),
                                cell_metadata=colData(train),
                                feat_metadata=rowData(train))
    train <- train %>%
        NormalizeData() %>%
        FindVariableFeatures() %>%
        ScaleData() %>%
        RunPCA()
    train@meta.data <- data.frame(train@meta.data, cell_type)

    train <- getFeatureSpace(train, 'cell.type')
    train <- trainModel(train)

    test <- CreateSeuratObject(assay(test, 'X'),
                               cell_metadata=colData(test),
                               feat_metadata=rowData(test))
    test <- NormalizeData(test)
    test <- scPredict(test, train, seed=seed)
    """)
    pre_label = list(r('test@meta.data$scpred_prediction'))
    pre_label = [1 if i == 'unassigned' else 0 for i in pre_label]
    result = {'cell_type': test.obs['cell.type'].values,
              'label': test.obs['label'].values,
              'diff': pre_label}
    result = pd.DataFrame(result, index=test.obs.index)
    return result

def CHETAH(train: ad.AnnData, test: ad.AnnData, random_state: int):
    importr("CHETAH")
    globalenv['seed'] = random_state
    globalenv['train'] = anndata2ri.py2rpy(train)
    globalenv['test'] = anndata2ri.py2rpy(test)
    r("""
    set.seed(seed)

    colnames(colData(train)) <- 'celltypes'
    test <- CHETAHclassifier(input = test, ref_cells = train)
    """)
    pre_label = list(r('colData(test)$celltype_CHETAH'))
    pre_label = [1 if i == 'Unassigned' else 0 for i in pre_label]
    result = {'cell_type': test.obs['cell.type'].values,
              'label': test.obs['label'].values,
              'diff': pre_label}
    result = pd.DataFrame(result, index=test.obs.index)
    return result

def scmap(train: ad.AnnData, test: ad.AnnData, random_state: int):
    importr("scmap")
    globalenv['seed'] = random_state
    globalenv['train'] = anndata2ri.py2rpy(train)
    globalenv['test'] = anndata2ri.py2rpy(test)
    r("""
    set.seed(seed)

    logcounts(train) <- assay(train, 'X')
    rowData(train)$feature_symbol <- rownames(train)
    colData(train)$cell_type1 = colData(train)$cell.type
    train <- selectFeatures(train, suppress_plot = TRUE)
    train <- indexCluster(train)

    logcounts(test) <- assay(test, 'X')
    rowData(test)$feature_symbol <- rownames(test)
    scmapCluster_results <- scmapCluster(
      projection = test,
      index_list = list(
        metadata(train)$scmap_cluster_index
      )
    )
    """)

    pre_label = list(r('scmapCluster_results$scmap_cluster_labs'))
    pre_label = [1 if i == 'unassigned' else 0 for i in pre_label]
    result = {'cell_type': test.obs['cell.type'].values,
              'label': test.obs['label'].values,
              'diff': pre_label}
    result = pd.DataFrame(result, index=test.obs.index)
    return result

'''
After getting the result, save it into adata.obs
    anndata2ri.activate()
    result = detect_scPred(ref, adata, 100)
    adata.obs['scPred'] = result['diff']
    result = detect_CHETAH(ref, adata, 100)
    adata.obs['CHETAH'] = result['diff']
    result = detect_scmap_cluster(ref, adata, 100)
    adata.obs['scmap'] = result['diff']
'''