# Week 3: Iris Flower Clustering Project

This project completes the Week 3 unsupervised learning assignment. It groups Iris flowers with **K-Means (k=3)**, reduces four measurements to two principal components with **PCA**, plots the clusters, and compares them with the true species labels. It also demonstrates feature scaling, the elbow method, and saving and reloading trained models.

## Results

| Measure | Result |
| --- | ---: |
| Flowers / numeric features | 150 / 4 |
| K-Means clusters | 3 |
| Variance retained by first two PCs | **95.81%** (PC1: 72.96%; PC2: 22.85%) |
| Silhouette score on standardized features | **0.460** |
| Adjusted Rand index against true species | **0.620** |
| Best label-matched agreement with species | **83.33%** (125 of 150) |

### Predicted clusters compared with true labels

| True species | Cluster 0 | Cluster 1 | Cluster 2 |
| --- | ---: | ---: | ---: |
| Setosa | 0 | 50 | 0 |
| Versicolor | 39 | 0 | 11 |
| Virginica | 14 | 0 | 36 |

Cluster numbers are arbitrary, so cluster 1 corresponds to setosa, cluster 0 mostly to versicolor, and cluster 2 mostly to virginica. Setosa is completely separated; versicolor and virginica overlap. The 83.33% figure is calculated **after** matching cluster numbers to species names for evaluation. The true species labels were never used to train K-Means or PCA. The adjusted Rand index is independent of how clusters are numbered.

## Visualizations

- [Elbow curve](results/elbow_curve.png): inertia for k=1 through k=10. The curve starts flattening around k=3, which is also the assignment's required value.
- [PCA cluster and species comparison](results/pca_clusters_vs_species.png): the same two-dimensional projection colored by predicted cluster and true species.

K-Means was fit to **all four standardized measurements**, while PCA was used to show the data in two dimensions. Clustering only on the two plotted components would be a different experiment.

## Run

Requires Python 3.10 or newer. From this folder:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python run_assignment.py
```

The script recreates the files in `results/` and `models/`. It prints the metrics and verifies that reloaded models reproduce the original cluster assignments and PCA coordinates.

## Files

- `data/iris.csv`: the 150 flower records and true species labels.
- `run_assignment.py`: complete preprocessing, elbow method, K-Means, PCA, plots, comparison, and model persistence.
- `results/metrics.json`: scores, PCA variance, cluster-to-species matching, and elbow values.
- `results/cluster_vs_species.csv`: full comparison table.
- `results/iris_cluster_assignments.csv`: every flower's measurements, species, cluster, and PCA coordinates.
- `results/pca_feature_loadings.csv`: contribution of each standardized measurement to PC1 and PC2.
- `models/iris_kmeans_pipeline.joblib`: fitted scaler and K-Means model.
- `models/iris_pca.joblib`: fitted two-component PCA model.

## Data source and reproducibility

The CSV was exported from [`sklearn.datasets.load_iris`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html), a bundled version of the classic [UCI Iris dataset](https://archive.ics.uci.edu/dataset/53/iris). This avoids a network download when rerunning the assignment. The random seed is 42 and K-Means uses 20 initializations. Model files should only be loaded from trusted sources.
# week3-ml-assignment
