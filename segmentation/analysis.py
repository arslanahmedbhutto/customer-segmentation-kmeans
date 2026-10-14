"""Choosing K, checking stability, and comparing clustering algorithms."""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from segmentation import SEED


def scale(X) -> np.ndarray:
    return StandardScaler().fit_transform(X)


def k_selection(X, k_range=range(2, 11), seed: int = SEED) -> pd.DataFrame:
    """Elbow (inertia) plus three internal validity indices for each K, on scaled data."""
    Z = scale(X)
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(Z, km.labels_),          # higher is better
            "davies_bouldin": davies_bouldin_score(Z, km.labels_),  # lower is better
            "calinski_harabasz": calinski_harabasz_score(Z, km.labels_),  # higher is better
        })
    return pd.DataFrame(rows)


def best_k(scores: pd.DataFrame) -> dict:
    return {
        "silhouette": int(scores.loc[scores["silhouette"].idxmax(), "k"]),
        "davies_bouldin": int(scores.loc[scores["davies_bouldin"].idxmin(), "k"]),
        "calinski_harabasz": int(scores.loc[scores["calinski_harabasz"].idxmax(), "k"]),
    }


def fit_kmeans(X, k: int, seed: int = SEED) -> Pipeline:
    return make_pipeline(StandardScaler(), KMeans(n_clusters=k, n_init=20, random_state=seed)).fit(X)


def bootstrap_stability(X, k: int, n_boot: int = 100, seed: int = SEED) -> np.ndarray:
    """Adjusted Rand Index between the full-data clustering and clusterings fitted on
    bootstrap resamples (compared on all customers). Near 1 = the segments are stable."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    reference = fit_kmeans(X, k, seed).predict(X)
    aris = []
    for b in range(n_boot):
        idx = rng.integers(0, len(X), len(X))
        model = fit_kmeans(X[idx], k, seed + b + 1)
        aris.append(adjusted_rand_score(reference, model.predict(X)))
    return np.array(aris)


def compare_algorithms(X, k: int, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """Fit K-Means, Ward hierarchical clustering and a Gaussian mixture with the same K.
    Agreement with K-Means is measured with the Adjusted Rand Index."""
    Z = scale(X)
    labels = {
        "K-Means": KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(Z),
        "Hierarchical (Ward)": AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z),
        "Gaussian mixture": GaussianMixture(n_components=k, n_init=5, random_state=seed).fit_predict(Z),
    }
    rows = [{
        "algorithm": name,
        "silhouette": silhouette_score(Z, lab),
        "davies_bouldin": davies_bouldin_score(Z, lab),
        "ARI vs K-Means": adjusted_rand_score(labels["K-Means"], lab),
    } for name, lab in labels.items()]
    return pd.DataFrame(rows), labels
