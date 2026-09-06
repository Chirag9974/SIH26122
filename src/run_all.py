"""Rebuild the whole data + evaluation pipeline from scratch, in order.

    python run_all.py

data/regression/ holds the original 100-report benchmark (controlled
regression set) and is never regenerated -- only evaluated.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS = [
    ("1  schedule master", ["-m", "generation.gen_schedule"]),
    ("2  raw reports + 3  gold labels", ["-m", "generation.gen_reports"]),
    ("   terminology export", ["-m", "generation.export_terminology"]),
    ("   adversarial edge cases", ["-m", "generation.gen_edge_cases"]),
    ("   train/dev/test/test_hard splits", ["-m", "generation.make_splits"]),
    ("   dataset validation", ["-m", "quality.validate_dataset"]),
    ("4  extractor invariants", ["-m", "tests.test_extractor"]),
    ("   eval: synthetic (all)", ["-m", "evaluation.eval", "--errors", "0"]),
    ("   eval: frozen test split",
     ["-m", "evaluation.eval", "--split", "test", "--errors", "0",
      "--out", "../data/evaluation/metrics_test.json"]),
    ("   eval: hard generalization split",
     ["-m", "evaluation.eval", "--split", "test_hard", "--errors", "0",
      "--out", "../data/evaluation/metrics_test_hard.json"]),
    ("   eval: adversarial edge cases",
     ["-m", "evaluation.eval", "--paired", "../data/evaluation/test_set.jsonl", "--errors", "0",
      "--out", "../data/evaluation/metrics_edge.json"]),
    ("   eval: regression benchmark (original 100 reports)",
     ["-m", "evaluation.eval", "--data-dir", "../data/regression", "--errors", "0",
      "--out", "../data/evaluation/metrics_regression.json"]),
]


def main() -> None:
    for label, cmd in STEPS:
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}", flush=True)
        r = subprocess.run([sys.executable, *cmd], cwd=HERE)
        if r.returncode:
            sys.exit(f"step failed: {label}")
    print("\npipeline complete.", flush=True)


if __name__ == "__main__":
    main()
