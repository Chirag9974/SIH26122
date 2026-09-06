"""Evaluate the schedule matcher against data/labels/gold_matches.jsonl.

    python -m matching.evaluate_matcher            # table + JSON out
    python -m matching.evaluate_matcher --failures 30

Metrics (task spec): Recall@1/@3/@5, final match accuracy, no-match
accuracy, human-review accuracy, plus confusion counts and failure examples.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matching.matcher import Matcher  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data" / "labels" / "gold_matches.jsonl"
OUT = ROOT / "data" / "evaluation" / "metrics_matcher.json"


def load_event_bodies(path: Path) -> dict[str, dict]:
    bodies: dict[str, dict] = {}
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        for ev in rec.get("events", []):
            if ev.get("event_id"):
                bodies[ev["event_id"]] = ev
    return bodies


def rank_of(gold_id: str | None, res) -> int:
    """Position of the gold activity in the candidate ranking (99 = absent)."""
    if gold_id is None:
        return -1
    for i, c in enumerate(res.candidates, 1):
        if c.activity_id == gold_id:
            return i
    return 99


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(GOLD))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--failures", type=int, default=15,
                    help="failure examples to print/save")
    ap.add_argument("--limit", type=int, default=0, help="debug subset size")
    ap.add_argument("--auto-th", type=float, default=None)
    ap.add_argument("--auto-margin", type=float, default=None)
    ap.add_argument("--floor", type=float, default=None)
    args = ap.parse_args()
    import matching.matcher as M
    if args.auto_th is not None:
        M.AUTO_THRESHOLD = args.auto_th
    if args.auto_margin is not None:
        M.AUTO_MARGIN = args.auto_margin
    if args.floor is not None:
        M.NO_MATCH_FLOOR = args.floor

    gold = [json.loads(l) for l in open(Path(args.gold), encoding="utf-8")]
    if args.limit:
        gold = gold[: args.limit]
    bodies = load_event_bodies(ROOT / "data" / "labels" / "gold_extractions.jsonl")

    matcher = Matcher()
    results = []
    for g in gold:
        ev = bodies.get(g["event_id"], {"event_id": g["event_id"]})
        res = matcher.match(ev, candidate_pool=g.get("candidate_pool"))
        results.append((g, res))

    # ---- rank metrics (gold events only) ---------------------------------
    ranks = [rank_of(g["schedule_activity_id"], r) for g, r in results
             if g["schedule_activity_id"]]
    recall_at = {
        f"recall@{k}": round(sum(1 for r in ranks if r <= k) / max(1, len(ranks)), 4)
        for k in (1, 3, 5)
    }

    # ---- decision metrics -------------------------------------------------
    conf: Counter = Counter()
    no_match_total = no_match_correct = 0
    review_total = review_correct = 0
    auto_total = auto_correct = 0          # predicted auto with correct id
    auto_gold = auto_gold_correct = 0      # gold auto with correct id
    failures = []

    for g, res in results:
        gd, pd_ = g["decision"], res.decision
        conf[(gd, pd_)] += 1
        ok = pd_ == gd
        if gd == "no_match":
            no_match_total += 1
            no_match_correct += ok
        elif gd == "human_review":
            review_total += 1
            review_correct += ok
        elif gd == "auto_match":
            auto_gold += 1
            id_ok = res.schedule_activity_id == g["schedule_activity_id"]
            auto_gold_correct += id_ok
            if pd_ == "auto_match":
                ok = ok and id_ok
        if pd_ == "auto_match":
            auto_total += 1
            if res.schedule_activity_id == g["schedule_activity_id"]:
                auto_correct += 1
        if not ok and len(failures) < args.failures:
            failures.append({
                "event_id": g["event_id"],
                "gold_decision": gd,
                "pred_decision": pd_,
                "gold_id": g["schedule_activity_id"],
                "pred_id": res.schedule_activity_id,
                "reason": g.get("reason"),
                "top_candidates": [
                    {"id": c.activity_id, "score": round(c.score, 3),
                     "name": c.name}
                    for c in res.candidates[:3]
                ],
                "signals": res.candidates[0].signals if res.candidates else {},
            })

    n = len(results)
    decision_acc = sum(
        c for (g, p), c in conf.items() if g == p
        # decision-level accuracy; auto golds additionally require correct id
    )
    # recompute decision accuracy properly (auto needs id agreement)
    decision_correct = 0
    for g, res in results:
        if g["decision"] == "auto_match":
            decision_correct += (
                res.decision == "auto_match"
                and res.schedule_activity_id == g["schedule_activity_id"]
            )
        else:
            decision_correct += res.decision == g["decision"]

    summary = {
        "n": n,
        **recall_at,
        "final_match_accuracy": round(decision_correct / max(1, n), 4),
        "decision_accuracy": round(decision_acc / max(1, n), 4),
        "auto_precision": round(auto_correct / max(1, auto_total), 4),
        "auto_recall": round(auto_gold_correct / max(1, auto_gold), 4),
        "no_match_accuracy": round(no_match_correct / max(1, no_match_total), 4),
        "no_match_n": no_match_total,
        "human_review_recall": round(review_correct / max(1, review_total), 4),
        "human_review_n": review_total,
        "confusion": {f"{g}->{p}": c for (g, p), c in sorted(conf.items())},
    }
    print(json.dumps(summary, indent=2))
    if failures:
        print(f"\n--- {len(failures)} failure examples ---")
        for f in failures[: args.failures]:
            print(json.dumps(f, indent=1))

    out = {"summary": summary, "failures": failures}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
