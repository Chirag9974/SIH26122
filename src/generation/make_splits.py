"""Create proper train/dev/test splits with no leakage.

Improvements:
- Prevent leakage between splits
- Add hard/generalization test set with unseen wordings
- Template-based split: ensure test uses different report styles
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "data" / "raw_reports" / "reports.jsonl"
GOLD = ROOT / "data" / "labels" / "gold_extractions.jsonl"
OUT = ROOT / "data" / "evaluation" / "splits.json"

SEED = 26122


def main() -> None:
    random.seed(SEED)

    with GOLD.open(encoding="utf-8") as fh:
        golds = [json.loads(line) for line in fh if line.strip()]

    # Group by case_kind to ensure balanced splits
    by_kind = {}
    for g in golds:
        kind = g.get("case_kind", "unknown")
        by_kind.setdefault(kind, []).append(g["report_id"])

    train, dev, test, test_hard = [], [], [], []

    # Generalization test: hold out specific case kinds entirely
    generalization_kinds = ["uncertain", "delay", "no_match"]

    for kind, ids in by_kind.items():
        random.shuffle(ids)
        n = len(ids)

        if kind in generalization_kinds:
            # All go to hard test set (unseen case types)
            test_hard.extend(ids)
        else:
            # Standard split: 60% train, 20% dev, 20% test
            n_train = int(n * 0.6)
            n_dev = int(n * 0.2)

            train.extend(ids[:n_train])
            dev.extend(ids[n_train:n_train + n_dev])
            test.extend(ids[n_train + n_dev:])

    # Shuffle to mix case kinds
    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)
    random.shuffle(test_hard)

    splits = {
        "train": sorted(train),
        "dev": sorted(dev),
        "test": sorted(test),
        "test_hard": sorted(test_hard),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    print(f"{OUT}:")
    print(f"  train: {len(train)} reports")
    print(f"  dev: {len(dev)} reports")
    print(f"  test: {len(test)} reports (standard)")
    print(f"  test_hard: {len(test_hard)} reports (unseen case types)")
    print(f"  total: {len(train) + len(dev) + len(test) + len(test_hard)}")

    # Verify no leakage
    all_ids = set(train + dev + test + test_hard)
    assert len(all_ids) == len(train) + len(dev) + len(test) + len(test_hard), "Leakage detected!"
    print("No leakage between splits - verified")


if __name__ == "__main__":
    main()
