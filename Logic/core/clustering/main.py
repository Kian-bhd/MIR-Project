import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from ..word_embedding.fasttext_data_loader import FastTextDataLoader
from ..word_embedding.fasttext_model import FastText
from .dimension_reduction import DimensionReduction
from .clustering_metrics import ClusteringMetrics
from .clustering_utils import ClusteringUtils

# Main Function: Clustering Tasks

# 0. Embedding Extraction
ft_model = FastText()
path = 'IMDB_crawled_give.json'
ft_data_loader = FastTextDataLoader(path)
X, y = ft_data_loader.create_train_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
ft_model.prepare(None, 'load')
embeddings = [ft_model.get_query_embedding(sentence) for sentence in X_train]
# 1. Dimension Reduction
print('PCA')
dim = DimensionReduction()
pca_reduced_features = dim.pca_reduce_dimension(embeddings, 5)

print('TSNE')
tsne_reduced_features = dim.convert_to_2d_tsne(embeddings)

# 2. Clustering
## K-Means Clustering
# Implement the K-means clustering algorithm from scratch.
metrics = ClusteringMetrics()
utils = ClusteringUtils
centers, clusters = utils.cluster_kmeans(pca_reduced_features.tolist(), 20)
utils.visualize_kmeans_clustering_wandb(pca_reduced_features, 20, 'kmeans_prj', 'kmeans_run')

for k in range(1, 50):
    centers, clusters = utils.cluster_kmeans(pca_reduced_features.tolist(), k)
    utils.visualize_kmeans_clustering_wandb(pca_reduced_features, k, 'kmeans_prj', 'kmeans_run')

## Hierarchical Clustering
utils.wandb_plot_hierarchical_clustering_dendrogram(embeddings, 'dendrogram_prj', 'average', 'dendrogram_run')

# 3. Evaluation
utils.plot_kmeans_cluster_scores(pca_reduced_features, y_train, [i for i in range(50)], 'kmeans_prj', 'kmeans_run')
