"""Unit tests for CNN input normalization verification, feature map diagnostics, and CLI."""

import io
import json
import unittest
from contextlib import redirect_stdout
import numpy as np

from ml_eval_toolkit.cli import main
from ml_eval_toolkit.cnn_debugger import (
    analyze_feature_maps,
    inspect_input_scale,
    run_cnn_diagnostic_report,
)


class TestCNNDebugger(unittest.TestCase):
    def test_inspect_input_scale_detects_unnormalized_pixels(self) -> None:
        raw_images = np.random.default_rng(42).integers(0, 256, size=(4, 28, 28, 1))
        scaled_images = raw_images / 255.0

        raw_check = inspect_input_scale(raw_images)
        scaled_check = inspect_input_scale(scaled_images)

        self.assertFalse(raw_check["is_normalized"])
        self.assertTrue(scaled_check["is_normalized"])

    def test_analyze_feature_maps_dead_channel_ratio(self) -> None:
        fmap = np.ones((2, 14, 14, 4), dtype=np.float64)
        fmap[:, :, :, 0] = 0.0
        fmap[:, :, :, 2] = 0.0

        stat = analyze_feature_maps(fmap, layer_name="conv2d_1")
        self.assertEqual(stat.shape, (2, 14, 14, 4))
        self.assertEqual(stat.dead_channel_ratio, 0.5)

    def test_run_cnn_diagnostic_report_warnings(self) -> None:
        raw_images = np.full((2, 28, 28, 1), 255.0)
        fmap = np.zeros((2, 7, 7, 64), dtype=np.float64)
        report = run_cnn_diagnostic_report(raw_images, {"conv2d_1": fmap})

        self.assertFalse(report.is_input_normalized)
        self.assertEqual(len(report.warning_messages), 2)

    def test_cli_json_outputs(self) -> None:
        buf1 = io.StringIO()
        with redirect_stdout(buf1):
            exit_code_1 = main(["kmeans", "--samples", "60", "--k-min", "2", "--k-max", "4", "--json"])
        parsed_1 = json.loads(buf1.getvalue())
        self.assertEqual(exit_code_1, 0)
        self.assertEqual(parsed_1["task"], "kmeans_evaluation")
        self.assertEqual(len(parsed_1["evaluations"]), 3)

        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            exit_code_2 = main(["cnn-check", "--unnormalized-demo", "--json"])
        parsed_2 = json.loads(buf2.getvalue())
        self.assertEqual(exit_code_2, 0)
        self.assertFalse(parsed_2["is_input_normalized"])
        self.assertGreaterEqual(len(parsed_2["warning_messages"]), 1)


if __name__ == "__main__":
    unittest.main()
