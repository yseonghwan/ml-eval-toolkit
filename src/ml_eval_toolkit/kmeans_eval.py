"""From-scratch NumPy K-Means implementation and quantitative evaluation pipeline."""

from __future__ import annotations

import numpy as np

from ml_eval_toolkit.spec import ClusterEvalResult


def min_max_normalize(data: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale each feature column independently to [0.0, 1.0]."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array (n_samples, n_features), got {arr.ndim}D.")
    col_min = arr.min(axis=0, keepdims=True)
    col_max = arr.max(axis=0, keepdims=True)
    denom = np.where((col_max - col_min) < eps, 1.0, col_max - col_min)
    return (arr - col_min) / denom


def z_score_normalize(data: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Standardize each feature column to mean 0.0 and standard deviation 1.0."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array (n_samples, n_features), got {arr.ndim}D.")
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    safe_std = np.where(std < eps, 1.0, std)
    return (arr - mean) / safe_std


def compute_inertia(data: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Compute within-cluster sum of squared Euclidean distances (WCSS / Inertia)."""
    assigned_centroids = centroids[labels]
    diffs = data - assigned_centroids
    return float(np.sum(diffs * diffs))


def compute_silhouette_score(data: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean Silhouette Coefficient in [-1.0, 1.0] using pure NumPy."""
    n_samples = data.shape[0]
    unique_labels = np.unique(labels)
    if unique_labels.size < 2 or unique_labels.size >= n_samples:
        raise ValueError("Number of clusters must be in [2, n_samples - 1].")

    # Pairwise Euclidean distance matrix
    diff = data[:, np.newaxis, :] - data[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff * diff, axis=-1))

    sil_scores = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        own_cluster = labels[i]
        own_mask = labels == own_cluster
        own_count = int(np.sum(own_mask))

        if own_count <= 1:
            sil_scores[i] = 0.0
            continue

        # Mean intra-cluster distance a(i) (excluding self distance 0)
        a_i = float(np.sum(dist_matrix[i, own_mask]) / (own_count - 1))

        # Minimum mean inter-cluster distance b(i)
        b_i = float("inf")
        for other_cluster in unique_labels:
            if other_cluster == own_cluster:
                continue
            other_mask = labels == other_cluster
            mean_other_dist = float(np.mean(dist_matrix[i, other_mask]))
            if mean_other_dist < b_i:
                b_i = mean_other_dist

        denom = max(a_i, b_i)
        sil_scores[i] = (b_i - a_i) / denom if denom > 0.0 else 0.0

    return float(np.mean(sil_scores))


class KMeansScratch:
    """Pure NumPy K-Means clustering with k-means++ initialization."""

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        if n_clusters < 2:
            raise ValueError("n_clusters must be >= 2.")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.n_iter_: int = 0

    def _init_centroids(self, data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n_samples, n_features = data.shape
        centroids = np.empty((self.n_clusters, n_features), dtype=np.float64)
        first_idx = int(rng.integers(0, n_samples))
        centroids[0] = data[first_idx]

        for c_id in range(1, self.n_clusters):
            diffs = data[:, np.newaxis, :] - centroids[np.newaxis, :c_id, :]
            sq_dists = np.sum(diffs * diffs, axis=-1)
            min_sq_dists = np.min(sq_dists, axis=1)
            total = float(np.sum(min_sq_dists))
            if total <= 0.0:
                idx = int(rng.integers(0, n_samples))
            else:
                probs = min_sq_dists / total
                idx = int(rng.choice(n_samples, p=probs))
            centroids[c_id] = data[idx]

        return centroids

    def fit(self, data: np.ndarray) -> KMeansScratch:
        arr = np.asarray(data, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        centroids = self._init_centroids(arr, rng)
        labels = np.zeros(arr.shape[0], dtype=np.int64)

        for iteration in range(1, self.max_iter + 1):
            diffs = arr[:, np.newaxis, :] - centroids[np.newaxis, :, :]
            sq_dists = np.sum(diffs * diffs, axis=-1)
            labels = np.argmin(sq_dists, axis=1)

            new_centroids = np.copy(centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if np.any(mask):
                    new_centroids[k] = arr[mask].mean(axis=0)
                else:
                    new_centroids[k] = arr[int(rng.integers(0, arr.shape[0]))]

            shift = float(np.sqrt(np.sum((new_centroids - centroids) ** 2)))
            centroids = new_centroids
            self.n_iter_ = iteration
            if shift <= self.tol:
                break

        self.centroids_ = centroids
        self.labels_ = labels
        return self


def evaluate_k_range(
    data: np.ndarray,
    k_min: int = 2,
    k_max: int = 6,
    scale_method: str = "zscore",
    random_state: int = 42,
) -> list[ClusterEvalResult]:
    """Run automated K-Means evaluation across [k_min, k_max] with enforced normalization."""
    if scale_method == "zscore":
        processed = z_score_normalize(data)
    elif scale_method == "minmax":
        processed = min_max_normalize(data)
    elif scale_method == "none":
        processed = np.asarray(data, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported scale_method: {scale_method}")

    results: list[ClusterEvalResult] = []
    for k in range(k_min, k_max + 1):
        model = KMeansScratch(n_clusters=k, random_state=random_state).fit(processed)
        assert model.labels_ is not None and model.centroids_ is not None
        inertia = compute_inertia(processed, model.labels_, model.centroids_)
        sil = compute_silhouette_score(processed, model.labels_)
        results.append(
            ClusterEvalResult(
                k=k,
                scale_method=scale_method,
                inertia=round(inertia, 4),
                silhouette_score=round(sil, 4),
                iterations=model.n_iter_,
            )
        )
    return results
