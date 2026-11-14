"""Unit tests for K-Means normalization and quantitative evaluation metrics."""

import unittest
import numpy as np

from ml_eval_toolkit.kmeans_eval import (
    KMeansScratch,
    compute_silhouette_score,
    evaluate_k_range,
    min_max_normalize,
    z_score_normalize,
)


class TestKMeansEval(unittest.TestCase):
    def test_min_max_normalize_bounds(self) -> None:
        raw = np.array([[10.0, 1000.0], [20.0, 2000.0], [30.0, 5000.0]])
        normed = min_max_normalize(raw)
        self.assertTrue(np.allclose(normed.min(axis=0), [0.0, 0.0]))
        self.assertTrue(np.allclose(normed.max(axis=0), [1.0, 1.0]))

    def test_z_score_normalize_stats(self) -> None:
        rng = np.random.default_rng(42)
        raw = rng.normal(loc=[50.0, -200.0], scale=[5.0, 80.0], size=(100, 2))
        standardized = z_score_normalize(raw)
        self.assertTrue(np.allclose(standardized.mean(axis=0), [0.0, 0.0], atol=1e-7))
        self.assertTrue(np.allclose(standardized.std(axis=0), [1.0, 1.0], atol=1e-7))

    def test_kmeans_recovers_true_clusters_with_zscore(self) -> None:
        rng = np.random.default_rng(7)
        c1 = rng.normal(loc=[0.0, 0.0], scale=[0.4, 0.4], size=(30, 2))
        c2 = rng.normal(loc=[5.0, 5.0], scale=[0.4, 0.4], size=(30, 2))
        c3 = rng.normal(loc=[-5.0, 5.0], scale=[0.4, 0.4], size=(30, 2))
        data = np.vstack([c1, c2, c3])

        results = evaluate_k_range(data, k_min=2, k_max=5, scale_method="zscore", random_state=7)
        best = max(results, key=lambda r: r.silhouette_score)
        self.assertEqual(best.k, 3)
        self.assertTrue(-1.0 <= best.silhouette_score <= 1.0)
        self.assertGreater(best.silhouette_score, 0.7)

    def test_invalid_cluster_count_raises(self) -> None:
        data = np.zeros((10, 2))
        with self.assertRaises(ValueError):
            KMeansScratch(n_clusters=1)
        with self.assertRaises(ValueError):
            compute_silhouette_score(data, np.zeros(10, dtype=int))


if __name__ == "__main__":
    unittest.main()
