"""Command-Line Interface (CLI) for automated K-Means evaluation and CNN diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

import numpy as np

from ml_eval_toolkit.cnn_debugger import run_cnn_diagnostic_report
from ml_eval_toolkit.kmeans_eval import evaluate_k_range


def _generate_synthetic_clusters(n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c1 = rng.normal(loc=[0.0, 0.0], scale=[1.0, 100.0], size=(n_samples // 3, 2))
    c2 = rng.normal(loc=[8.0, 500.0], scale=[1.0, 100.0], size=(n_samples // 3, 2))
    c3 = rng.normal(loc=[-8.0, -500.0], scale=[1.0, 100.0], size=(n_samples - 2 * (n_samples // 3), 2))
    return np.vstack([c1, c2, c3])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-eval",
        description="CLI toolkit for quantitative K-Means evaluation and CNN feature-map debugging.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand 1: kmeans
    kmeans_parser = subparsers.add_parser(
        "kmeans",
        help="Run K-Means clustering evaluation pipeline across a range of K values.",
    )
    kmeans_parser.add_argument("--samples", type=int, default=180, help="Number of synthetic samples.")
    kmeans_parser.add_argument("--k-min", type=int, default=2, help="Minimum K to evaluate.")
    kmeans_parser.add_argument("--k-max", type=int, default=5, help="Maximum K to evaluate.")
    kmeans_parser.add_argument(
        "--scale",
        choices=["zscore", "minmax", "none"],
        default="zscore",
        help="Feature normalization strategy before Euclidean distance calculation.",
    )
    kmeans_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    kmeans_parser.add_argument("--json", action="store_true", help="Output structured JSON Spec.")

    # Subcommand 2: cnn-check
    cnn_parser = subparsers.add_parser(
        "cnn-check",
        help="Inspect CNN input normalization and intermediate Conv2D/MaxPooling2D feature maps.",
    )
    cnn_parser.add_argument(
        "--unnormalized-demo",
        action="store_true",
        help="Simulate raw [0, 255] input without /255.0 scaling to trigger diagnostic warnings.",
    )
    cnn_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    cnn_parser.add_argument("--json", action="store_true", help="Output structured JSON Spec.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "kmeans":
        data = _generate_synthetic_clusters(args.samples, args.seed)
        results = evaluate_k_range(
            data,
            k_min=args.k_min,
            k_max=args.k_max,
            scale_method=args.scale,
            random_state=args.seed,
        )
        best = max(results, key=lambda r: r.silhouette_score)
        payload = {
            "task": "kmeans_evaluation",
            "scale_method": args.scale,
            "recommended_k": best.k,
            "best_silhouette_score": best.silhouette_score,
            "evaluations": [r.to_dict() for r in results],
        }
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(f"[K-Means Evaluation Pipeline] scale={args.scale} | optimal_k={best.k}")
            for r in results:
                print(
                    f"  K={r.k}: inertia={r.inertia:.2f}, "
                    f"silhouette={r.silhouette_score:.4f}, iter={r.iterations}"
                )
        return 0

    if args.command == "cnn-check":
        rng = np.random.default_rng(args.seed)
        raw_input = rng.integers(0, 256, size=(8, 28, 28, 1)).astype(np.float64)
        input_batch = raw_input if args.unnormalized_demo else raw_input / 255.0

        # Simulate Conv2D (28x28x32) and Conv2D_1 (14x14x64) feature maps
        conv1_map = np.maximum(0.0, rng.normal(0.15, 0.3, size=(8, 28, 28, 32)))
        conv2_map = np.maximum(0.0, rng.normal(0.10, 0.25, size=(8, 14, 14, 64)))
        if args.unnormalized_demo:
            # Simulate dead channels when input scale is distorted
            conv2_map[:, :, :, :40] = 0.0

        report = run_cnn_diagnostic_report(
            input_batch=input_batch,
            layer_outputs={"conv2d": conv1_map, "conv2d_1": conv2_map},
        )
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(f"[CNN Diagnostic Report] normalized={report.is_input_normalized}")
            for stat in report.layer_stats:
                print(
                    f"  Layer {stat.layer_name} {stat.shape}: "
                    f"mean={stat.mean_activation:.4f}, dead_channels={stat.dead_channel_ratio*100:.1f}%"
                )
            for w in report.warning_messages:
                print(f"  [WARNING] {w}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
