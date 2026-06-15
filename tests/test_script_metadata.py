from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ScriptMetadataTest(unittest.TestCase):
    def test_external_eval_marks_keyword_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_external.py",
                    "--output_dir",
                    tmp,
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            metrics = json.loads((Path(tmp) / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["evaluator"], "keyword_weak_baseline")
            self.assertFalse(metrics["model_evaluation"])

    def test_counterfactual_marks_non_model_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/build_counterfactual_eval.py",
                    "--num_cases",
                    "2",
                    "--output_dir",
                    tmp,
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            metrics = json.loads((Path(tmp) / "counterfactual_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["evaluation_type"], "case_generation_and_keyword_baseline")
            self.assertFalse(metrics["model_evaluation"])


if __name__ == "__main__":
    unittest.main()
