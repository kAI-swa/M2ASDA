import numpy as np
import anndata as ad
import scanpy as sc
import argparse
from detect import Detect_SC
from sklearn.metrics import precision_recall_curve, precision_score, recall_score

def parse_args():
    parser = argparse.ArgumentParser(
        description="Anomaly Detection for single cell"
    )
    parser.add_argument('--train', type=str, help="path of reference data for training")
    parser.add_argument('--test', type=str, help="Path of Target data for detecting anomalies")
    parser.add_argument('--num_epochs', default=600, type=int, help="Training Epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial Training Learning rate")
    parser.add_argument("--sample_rate", type=float, default=1)
    parser.add_argument("--memory_dim", type=int, default=1024, help="Number of intermediate embeddings in Queue(Memory Bank)")
    parser.add_argument("--update_size", type=int, default=64, help="Enqueue number when updating the queue, should able to be divided by batch size")
    parser.add_argument("--shrink_threshold", type=float, defaut=9e-3, help="shrikage relu threshold")
    parser.add_argument("--n_critic", type=int, default=1, help="Numeber of updating Discriminator per epoch")
    parser.add_argument("--pretrain", action="store_true", help="Determine whether pretrain before anomaly detection or not")
    parser.add_argument("--gpu", action="store_true", help="Training on GPU by default")
    parser.add_argument("--verbose", action="store_true", help="log")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--weight", default=None)

    args = parser.parse_args()

    return args


def outlier_detect(
    train: ad.AnnData,
    test: ad.AnnData,
):
    
    model = Detect_SC(**parameters)
    model.fit(train)
    result = model.predict(test)
    return result


if __name__ == "__main__":
    args = parse_args()
    parameters = {
        'n_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'sample_rate': args.sample_rate,
        'mem_dim': args.memory_dim,
        'update_size': args.update_size,
        'shrink_thres': args.shrink_threshold,
        'temperature': args.temperature,
        'n_critic': args.n_critic,
        'pretrain': args.pretrain,
        'GPU': args.gpu,
        'verbose': args.verbose,
        'log_interval': args.log_interval,
        'random_state': args.random_state,
        'weight': args.weight
    }

    model = Detect_SC(**parameters)
    model.fit(args.train)
    result = model.predict(args.test)

    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    adata = sc.read_h5ad(args.test)
    adata.obs['score'] = result['score'].values
    precision_scores = []
    recall_scores = []
    with tqdm(total=len(threshold_list), leave=False) as t:
        for threshold in threshold_list:
            t.set_description(f"Threhold:{threshold}")
            adata.obs[f'Pred_{threshold}_threshold'] = (adata.obs['score'] > threshold).astype('category')

            precision = precision_score(adata.obs['Pred_{threshold}_threshold'], adata.obs['label'])
            recall = recall_score(adata.obs['Pred_{threshold}_threshold'], adata.obs['label'])
            t.set_postfix(Precision = precision, Recall = recall)

            precision_scores.append(precision)
            recall_scores.append(recall)
    

