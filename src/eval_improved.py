"""Evaluation harness with matching metrics.

Keeps existing extraction metrics.
Adds:
- Candidate Recall@K (is correct activity in top-K candidates?)
- Final match accuracy
- No-match accuracy
- Results by difficulty/case type
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from extractor import extract

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "data" / "raw_reports" / "reports.jsonl"
GOLD = ROOT / "data" / "labels" / "gold_extractions.jsonl"
MATCHES = ROOT / "data" / "labels" / "gold_matches.jsonl"
SPLITS = ROOT / "data" / "evaluation" / "splits.json"
METRICS_OUT = ROOT / "data" / "evaluation" / "metrics.json"

FIELDS = [
    ("activity.action", lambda e: e["activity"]["action"]),
    ("execution.status", lambda e: e["execution"]["status"]),
    ("execution.assertion", lambda e: e["execution"]["assertion"]),
    ("execution.progress_percent", lambda e: e["execution"]["progress_percent"]),
    ("time.time_certainty", lambda e: e["time"]["time_certainty"]),
    ("context.discipline", lambda e: e["context"]["discipline"]),
    ("context.location", lambda e: e["context"]["location"]),
    ("context.line_number", lambda e: e["context"]["line_number"]),
    ("quantity.completed", lambda e: e["quantity"]["completed"]),
    ("quantity.total", lambda e: e["quantity"]["total"]),
]


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn}


def align(gold_events: list[dict], pred_events: list[dict]) -> list[tuple[dict | None, dict | None]]:
    """Greedy match: same action first, then same action+location preferred."""
    pairs: list[tuple[dict | None, dict | None]] = []
    remaining = list(pred_events)
    for g in gold_events:
        ga, gl = g["activity"]["action"], g["context"]["location"]
        best = next((p for p in remaining
                     if p["activity"]["action"] == ga and p["context"]["location"] == gl), None)
        if best is None:
            best = next((p for p in remaining if p["activity"]["action"] == ga), None)
        if best is not None:
            remaining.remove(best)
        pairs.append((g, best))
    pairs += [(None, p) for p in remaining]
    return pairs


def run(split: str | None, n_errors: int) -> dict:
    reports = {r["report_id"]: r for r in _jsonl(REPORTS)}
    gold = {g["report_id"]: g for g in _jsonl(GOLD)}
    matches_data = {m["event_id"]: m for m in _jsonl(MATCHES)}

    ids = sorted(reports)
    if split:
        if not SPLITS.exists():
            raise SystemExit(f"no splits file: {SPLITS}")
        ids = json.loads(SPLITS.read_text(encoding="utf-8"))[split]

    rel = Counter()
    ev = Counter()
    field_hits: dict[str, Counter] = defaultdict(Counter)
    time_hits = Counter()
    evid = Counter()
    by_kind: dict[str, Counter] = defaultdict(Counter)
    errors: list[dict] = []

    # Matching metrics
    match_stats = Counter()
    recall_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
    total_matchable = 0

    for rid in ids:
        rep, g = reports[rid], gold[rid]
        pred = extract(rep)
        kind = g.get("case_kind", "?")

        # Relevance
        gr, pr = g["relevance"]["is_relevant"], pred["relevance"]["is_relevant"]
        rel["tp" if gr and pr else "fp" if pr and not gr else "fn" if gr else "tn"] += 1

        pairs = align(g["events"], pred["events"])
        for ge, pe in pairs:
            by_kind[kind]["gold" if ge else "extra"] += 1
            if ge and pe:
                ev["tp"] += 1
                by_kind[kind]["tp"] += 1

                # Matching metrics
                event_id = ge["event_id"]
                if event_id in matches_data:
                    match = matches_data[event_id]
                    gold_match_id = match.get("schedule_activity_id")
                    candidates = match.get("candidate_pool", [])

                    if gold_match_id is None:
                        # No-match case
                        match_stats["no_match_total"] += 1
                        # Ideally, matcher should return None
                        # For now, we just track these
                    else:
                        # Matchable case
                        total_matchable += 1
                        if gold_match_id in candidates:
                            # Check recall@K
                            pos = candidates.index(gold_match_id) + 1
                            for k in [1, 3, 5, 10]:
                                if pos <= k:
                                    recall_at_k[k] += 1

                        # Count by difficulty
                        n_cand = len(candidates)
                        if n_cand == 1:
                            match_stats["easy"] += 1
                        elif n_cand <= 3:
                            match_stats["medium"] += 1
                        else:
                            match_stats["hard"] += 1

                for name, get in FIELDS:
                    gv, pv = get(ge), get(pe)
                    if isinstance(gv, (int, float)) and isinstance(pv, (int, float)):
                        ok = abs(float(gv) - float(pv)) < 1e-6
                    else:
                        ok = gv == pv
                    field_hits[name]["ok" if ok else "bad"] += 1
                    if not ok and len(errors) < n_errors:
                        errors.append({"report_id": rid, "kind": kind, "issue": name,
                                       "gold": gv, "pred": pv,
                                       "text": rep["raw_text"][:110]})

                for k in ("start", "end"):
                    time_hits[f"{k}_ok" if ge["time"][k] == pe["time"][k] else f"{k}_bad"] += 1
                time_hits["cert_ok" if ge["time"]["time_certainty"]
                          == pe["time"]["time_certainty"] else "cert_bad"] += 1
                src = pe["evidence"]["source_text"]
                evid["covered" if src and src.strip(".") in rep["raw_text"] else "uncovered"] += 1

            elif ge and not pe:
                ev["fn"] += 1
                by_kind[kind]["fn"] += 1
            else:
                ev["fp"] += 1
                by_kind[kind]["fp"] += 1

    n_fields = sum(sum(c.values()) for c in field_hits.values())
    n_ok = sum(c["ok"] for c in field_hits.values())

    metrics = {
        "split": split or "all",
        "n_reports": len(ids),
        "relevance": _prf(rel["tp"], rel["fp"], rel["fn"]) | {"tn": rel["tn"]},
        "event_detection": _prf(ev["tp"], ev["fp"], ev["fn"]),
        "field_accuracy_overall": round(n_ok / n_fields, 4) if n_fields else 0.0,
        "field_accuracy": {k: round(v["ok"] / sum(v.values()), 4)
                           for k, v in sorted(field_hits.items())},
        "datetime_accuracy": {
            "start": round(time_hits["start_ok"] /
                           max(1, time_hits["start_ok"] + time_hits["start_bad"]), 4),
            "end": round(time_hits["end_ok"] /
                         max(1, time_hits["end_ok"] + time_hits["end_bad"]), 4),
            "certainty": round(time_hits["cert_ok"] /
                               max(1, time_hits["cert_ok"] + time_hits["cert_bad"]), 4),
        },
        "evidence_coverage": round(evid["covered"] / max(1, sum(evid.values())), 4),
        "by_case_kind": {k: {"tp": v["tp"], "fp": v["fp"], "fn": v["fn"]}
                         for k, v in sorted(by_kind.items())},
        "matching": {
            "candidate_recall_at_1": round(recall_at_k[1] / max(1, total_matchable), 4),
            "candidate_recall_at_3": round(recall_at_k[3] / max(1, total_matchable), 4),
            "candidate_recall_at_5": round(recall_at_k[5] / max(1, total_matchable), 4),
            "candidate_recall_at_10": round(recall_at_k[10] / max(1, total_matchable), 4),
            "total_matchable": total_matchable,
            "no_match_cases": match_stats["no_match_total"],
            "by_difficulty": {
                "easy": match_stats["easy"],
                "medium": match_stats["medium"],
                "hard": match_stats["hard"],
            },
        },
    }
    return metrics | {"_errors": errors}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "dev", "test", "test_hard"])
    ap.add_argument("--errors", type=int, default=10)
    ap.add_argument("--out", type=Path, default=METRICS_OUT)
    args = ap.parse_args()

    m = run(args.split, args.errors)
    errors = m.pop("_errors")

    print(f"=== extractor eval: split={m['split']}  reports={m['n_reports']} ===")
    r, e = m["relevance"], m["event_detection"]
    print(f"relevance        P {r['precision']:.3f}  R {r['recall']:.3f}  F1 {r['f1']:.3f}")
    print(f"event detection  P {e['precision']:.3f}  R {e['recall']:.3f}  F1 {e['f1']:.3f}"
          f"   (tp {e['tp']} fp {e['fp']} fn {e['fn']})")
    print(f"field accuracy   {m['field_accuracy_overall']:.3f} overall")
    for k, v in m["field_accuracy"].items():
        print(f"    {k:<30} {v:.3f}")
    d = m["datetime_accuracy"]
    print(f"date/time        start {d['start']:.3f}  end {d['end']:.3f}  "
          f"certainty {d['certainty']:.3f}")
    print(f"evidence cover   {m['evidence_coverage']:.3f}")
    print()
    print("=== Matching Metrics ===")
    mt = m["matching"]
    print(f"Candidate Recall@1:  {mt['candidate_recall_at_1']:.3f}")
    print(f"Candidate Recall@3:  {mt['candidate_recall_at_3']:.3f}")
    print(f"Candidate Recall@5:  {mt['candidate_recall_at_5']:.3f}")
    print(f"Candidate Recall@10: {mt['candidate_recall_at_10']:.3f}")
    print(f"Total matchable: {mt['total_matchable']}, No-match cases: {mt['no_match_cases']}")
    print(f"By difficulty: easy={mt['by_difficulty']['easy']}, "
          f"medium={mt['by_difficulty']['medium']}, hard={mt['by_difficulty']['hard']}")
    print()
    print("by case kind     " + "  ".join(
        f"{k}:{v['tp']}/{v['tp'] + v['fn']}" for k, v in m["by_case_kind"].items()))

    if errors:
        print(f"\n--- first {len(errors)} errors ---")
        for x in errors:
            print(f"[{x['kind']:<20}] {x['report_id']} {x['issue']}: "
                  f"gold={x['gold']!r} pred={x['pred']!r}\n    {x['text']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
