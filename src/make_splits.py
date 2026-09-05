"""Spec section 12: 70/15/15 dev/val/test split, leak-free.

The leak the spec warns about: near-identical wording variants of the SAME
schedule activity landing in both dev and test, which inflates scores by
memorisation. We prevent it by grouping on the gold-matched activity_id (and,
for ambiguous/irrelevant reports with no match, on the report's event signature)
and assigning whole groups to one split.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "labels" / "gold_extractions.jsonl"
MATCHES = ROOT / "data" / "labels" / "gold_matches.jsonl"
OUT = ROOT / "data" / "evaluation" / "splits.json"

SEED = 26122
RATIOS = {"dev": 0.70, "val": 0.15, "test": 0.15}


def _jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def group_key(report_id: str, gold: dict, acts: list[str | None]) -> str:
    """Reports sharing any matched activity_id must share a split."""
    real = sorted(a for a in acts if a)
    if real:
        return "|".join(real)
    ev = gold["events"]
    if not ev:
        return f"irrelevant::{report_id}"
    return "unmatched::" + "|".join(
        sorted(f"{e['activity']['action']}@{e['context']['location']}" for e in ev))


def main() -> None:
    gold = {g["report_id"]: g for g in _jsonl(GOLD)}
    acts: dict[str, list[str | None]] = defaultdict(list)
    for m in _jsonl(MATCHES):
        acts[m["report_id"]].append(m["schedule_activity_id"])

    # union-find over activity ids so transitively-linked reports stay together
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for rid, g in gold.items():
        key = f"report::{rid}"
        find(key)
        real = [a for a in acts.get(rid, []) if a]
        if real:
            for a in real:
                union(key, f"act::{a}")
        else:
            union(key, f"grp::{group_key(rid, g, acts.get(rid, []))}")

    clusters: dict[str, list[str]] = defaultdict(list)
    for rid in gold:
        clusters[find(f"report::{rid}")].append(rid)

    rng = random.Random(SEED)
    order = sorted(clusters.values(), key=lambda c: (-len(c), c[0]))
    rng.shuffle(order)

    splits: dict[str, list[str]] = {k: [] for k in RATIOS}

    # Stratify by case_kind: split each kind 70/15/15 on its own. Unstratified
    # greedy filling left the test split with no ambiguous and no irrelevant
    # reports at all -- i.e. unable to measure abstention, the property the spec
    # cares about most. Clusters stay intact, so this does not reintroduce leaks.
    by_kind: dict[str, list[list[str]]] = defaultdict(list)
    for cluster in order:
        by_kind[min(gold[r].get("case_kind", "?") for r in cluster)].append(cluster)

    for kind in sorted(by_kind):
        n_k = sum(len(c) for c in by_kind[kind])
        deficit = {k: v * n_k for k, v in RATIOS.items()}
        for cluster in by_kind[kind]:
            pick = max(RATIOS, key=lambda k: (deficit[k], k))
            deficit[pick] -= len(cluster)
            splits[pick].extend(cluster)

    for k in splits:
        splits[k].sort()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    # leak audit
    owner: dict[str, str] = {}
    leaks = 0
    for name, ids in splits.items():
        for rid in ids:
            for a in (x for x in acts.get(rid, []) if x):
                if a in owner and owner[a] != name:
                    leaks += 1
                owner[a] = name

    print(f"{OUT}: " + "  ".join(f"{k}={len(v)}" for k, v in splits.items()))
    for name, ids in splits.items():
        kinds = Counter(gold[r].get("case_kind") for r in ids)
        print(f"  {name:<4} " + "  ".join(f"{k}:{n}" for k, n in sorted(kinds.items())))
    print(f"activity-id leaks across splits: {leaks}")
    assert leaks == 0, "split leak: same activity in two splits"


if __name__ == "__main__":
    main()
