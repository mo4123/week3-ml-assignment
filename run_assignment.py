"""Complete Week 3 Iris clustering, PCA, evaluation, and model persistence."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "iris.csv"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
FEATURES = [
    "sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"
]
SPECIES = ["setosa", "versicolor", "virginica"]
RANDOM_STATE = 42


def make_kmeans(k: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)),
    ])


def plot_elbow(x: pd.DataFrame) -> list[dict]:
    points = []
    for k in range(1, 11):
        model = make_kmeans(k).fit(x)
        points.append({"k": k, "inertia": float(model.named_steps["kmeans"].inertia_)})

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([p["k"] for p in points], [p["inertia"] for p in points],
            marker="o", linewidth=2)
    ax.axvline(3, color="#d62728", linestyle="--", label="Assignment choice: k=3")
    ax.set(xlabel="Number of clusters (k)", ylabel="Within-cluster sum of squares",
           title="Elbow method for Iris clustering", xticks=range(1, 11))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "elbow_curve.png", dpi=180)
    plt.close(fig)
    return points


def plot_pca_comparison(coordinates: np.ndarray, clusters: np.ndarray,
                        species: pd.Series, variance: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for cluster in range(3):
        mask = clusters == cluster
        axes[0].scatter(coordinates[mask, 0], coordinates[mask, 1],
                        color=colors[cluster], label=f"Cluster {cluster}",
                        s=35, alpha=0.8)
    axes[0].set_title("K-Means predicted clusters")

    for index, name in enumerate(SPECIES):
        mask = species.to_numpy() == name
        axes[1].scatter(coordinates[mask, 0], coordinates[mask, 1],
                        color=colors[index], label=name.title(), s=35, alpha=0.8)
    axes[1].set_title("True species (evaluation only)")

    for ax in axes:
        ax.set_xlabel(f"PC1 ({variance[0]:.1%} of variance)")
        ax.set_ylabel(f"PC2 ({variance[1]:.1%} of variance)")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Iris flowers projected from four features to two PCs")
    fig.tight_layout()
    fig.savefig(RESULTS / "pca_clusters_vs_species.png", dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    iris = pd.read_csv(DATA)
    if len(iris) != 150 or iris[FEATURES + ["species"]].isna().any().any():
        raise ValueError("Expected 150 complete Iris records")
    x = iris[FEATURES]
    y = iris["species"]

    elbow = plot_elbow(x)
    model = make_kmeans(3).fit(x)
    clusters = model.predict(x)
    scaled = model.named_steps["scaler"].transform(x)

    pca = PCA(n_components=2).fit(scaled)
    coordinates = pca.transform(scaled)
    plot_pca_comparison(coordinates, clusters, y, pca.explained_variance_ratio_)

    comparison = pd.crosstab(
        pd.Categorical(y, categories=SPECIES), clusters,
        rownames=["True species"], colnames=["Predicted cluster"]
    ).reindex(index=SPECIES, columns=range(3), fill_value=0)
    comparison.to_csv(RESULTS / "cluster_vs_species.csv")

    # Cluster numbers have no inherent species meaning. This matching is for
    # evaluation only; true labels were never provided to K-Means or PCA.
    row_indices, col_indices = linear_sum_assignment(-comparison.to_numpy())
    matched_count = int(comparison.to_numpy()[row_indices, col_indices].sum())
    cluster_to_species = {str(col): SPECIES[row] for row, col in zip(row_indices, col_indices)}

    assignments = iris.copy()
    assignments["predicted_cluster"] = clusters
    assignments["matched_species_for_evaluation"] = [cluster_to_species[str(c)] for c in clusters]
    assignments["PC1"] = coordinates[:, 0]
    assignments["PC2"] = coordinates[:, 1]
    assignments.to_csv(RESULTS / "iris_cluster_assignments.csv", index=False)

    pd.DataFrame(pca.components_.T, index=FEATURES, columns=["PC1", "PC2"]).to_csv(
        RESULTS / "pca_feature_loadings.csv", index_label="feature"
    )
    joblib.dump(model, MODELS / "iris_kmeans_pipeline.joblib")
    joblib.dump(pca, MODELS / "iris_pca.joblib")
    restored_model = joblib.load(MODELS / "iris_kmeans_pipeline.joblib")
    restored_pca = joblib.load(MODELS / "iris_pca.joblib")
    if not np.array_equal(restored_model.predict(x), clusters):
        raise RuntimeError("Reloaded K-Means model changed cluster assignments")
    if not np.allclose(restored_pca.transform(scaled), coordinates):
        raise RuntimeError("Reloaded PCA model changed the projection")

    metrics = {
        "dataset_rows": len(iris),
        "feature_count": len(FEATURES),
        "kmeans_k": 3,
        "standardized_before_clustering_and_pca": True,
        "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "pca_two_component_total_variance": float(pca.explained_variance_ratio_.sum()),
        "silhouette_score_on_scaled_features": float(silhouette_score(scaled, clusters)),
        "adjusted_rand_index_vs_true_species": float(adjusted_rand_score(y, clusters)),
        "matched_accuracy_for_evaluation_only": matched_count / len(iris),
        "cluster_to_species_for_evaluation_only": cluster_to_species,
        "elbow_inertia": elbow,
        "saved_models_reloaded_successfully": True,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
