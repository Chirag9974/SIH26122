"""Rebuild the whole data + evaluation pipeline from scratch, in order.

    python run_all.py

Steps 1-4 of the spec's section 15 plan. Matching (step 5) is not built yet.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS = [
    ("1  schedule master", ["gen_schedule.py"]),
    ("2  raw reports + 3  gold labels", ["gen_reports.py"]),
    ("   terminology export", ["export_terminology.py"]),
    ("   adversarial edge cases", ["gen_edge_cases.py"]),
    ("   train/val/test splits", ["make_splits.py"]),
    ("4  extractor invariants", ["test_extractor.py"]),
    ("   eval: synthetic (all)", ["eval.py", "--errors", "0"]),
    ("   eval: frozen test split",
     ["eval.py", "--split", "test", "--errors", "0",
      "--out", "../data/evaluation/metrics_test.json"]),
    ("   eval: adversarial edge cases",
     ["eval.py", "--paired", "../data/evaluation/test_set.jsonl", "--errors", "0",
      "--out", "../data/evaluation/metrics_edge.json"]),
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
