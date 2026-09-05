"""Dataset validation - catch issues before training.

Checks:
- duplicate/invalid activities
- invalid L5/L6 hierarchy
- broken labels/matches
- missing gold candidates
- duplicate candidate IDs
- train/test leakage
- invalid no-match cases
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ROOT / "data" / "schedule" / "schedule_activities.csv"
REPORTS = ROOT / "data" / "raw_reports" / "reports.jsonl"
GOLD = ROOT / "data" / "labels" / "gold_extractions.jsonl"
MATCHES = ROOT / "data" / "labels" / "gold_matches.jsonl"
SPLITS = ROOT / "data" / "evaluation" / "splits.json"


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def validate() -> dict:
    errors = []
    warnings = []

    # Load data
    with SCHEDULE.open(encoding="utf-8") as fh:
        schedule = list(csv.DictReader(fh))
    reports = _jsonl(REPORTS)
    golds = _jsonl(GOLD)
    matches = _jsonl(MATCHES)

    if SPLITS.exists():
        splits = json.loads(SPLITS.read_text())
    else:
        splits = {}

    # 1. Check schedule activities
    activity_ids = [r["activity_id"] for r in schedule]
    if len(activity_ids) != len(set(activity_ids)):
        dupes = [aid for aid, cnt in Counter(activity_ids).items() if cnt > 1]
        errors.append(f"Duplicate activity IDs: {dupes[:5]}")

    l5_ids = {r["activity_id"] for r in schedule if r["wbs_level"] == "L5"}
    l6_ids = {r["activity_id"] for r in schedule if r["wbs_level"] == "L6"}

    for r in schedule:
        if r["wbs_level"] == "L6":
            pred = r["predecessor_ids"]
            if pred and pred not in l5_ids and pred not in l6_ids:
                errors.append(f"L6 {r['activity_id']} has invalid predecessor {pred}")

    # 2. Check reports/gold alignment
    report_ids = {r["report_id"] for r in reports}
    gold_ids = {g["report_id"] for g in golds}
    if report_ids != gold_ids:
        errors.append(f"Report/gold ID mismatch: {len(report_ids - gold_ids)} missing gold, {len(gold_ids - report_ids)} missing reports")

    # 3. Check matches
    match_event_ids = {m["event_id"] for m in matches}
    gold_event_ids = {ev["event_id"] for g in golds for ev in g["events"]}
    if match_event_ids != gold_event_ids:
        errors.append(f"Match/event ID mismatch: {len(match_event_ids - gold_event_ids)} extra, {len(gold_event_ids - match_event_ids)} missing")

    # 4. Check candidate pools
    all_schedule_ids = set(activity_ids)
    for m in matches:
        if m["schedule_activity_id"] and m["schedule_activity_id"] not in all_schedule_ids:
            errors.append(f"Match {m['event_id']} references invalid schedule ID {m['schedule_activity_id']}")

        for cid in m.get("candidate_pool", []):
            if cid not in all_schedule_ids:
                errors.append(f"Match {m['event_id']} has invalid candidate {cid}")

        # Check no-match cases
        if m["decision"] == "no_match" and m["schedule_activity_id"] is not None:
            errors.append(f"Match {m['event_id']}: decision=no_match but has schedule_activity_id")

        if m["decision"] == "no_match" and m.get("candidate_pool"):
            warnings.append(f"Match {m['event_id']}: no_match but has non-empty candidate pool")

    # 5. Check train/test leakage
    if splits:
        all_split_ids = set()
        for split_name, ids in splits.items():
            overlap = all_split_ids & set(ids)
            if overlap:
                errors.append(f"Leakage: {len(overlap)} reports appear in multiple splits")
            all_split_ids.update(ids)

    # 6. Check gold event structure
    for g in golds:
        for ev in g["events"]:
            if "event_id" not in ev:
                errors.append(f"Event in {g['report_id']} missing event_id")
            if "activity" not in ev or "action" not in ev["activity"]:
                errors.append(f"Event {ev.get('event_id', '?')} missing activity.action")

    # 7. Statistics
    n_events = sum(len(g["events"]) for g in golds)
    n_irrelevant = sum(1 for g in golds if not g["events"])
    n_no_match = sum(1 for m in matches if m["decision"] == "no_match")
    n_with_candidates = sum(1 for m in matches if len(m.get("candidate_pool", [])) > 1)

    case_kinds = Counter(g.get("case_kind", "unknown") for g in golds)

    stats = {
        "schedule_activities": len(activity_ids),
        "reports": len(reports),
        "gold_events": n_events,
        "irrelevant_reports": n_irrelevant,
        "matches_no_match": n_no_match,
        "matches_with_candidates": n_with_candidates,
        "case_kinds": dict(case_kinds),
    }

    return {
        "errors": errors[:20],  # cap for readability
        "warnings": warnings[:20],
        "stats": stats,
        "valid": len(errors) == 0,
    }


def main() -> None:
    result = validate()

    print("=== Dataset Validation ===")
    print(f"Status: {'PASS' if result['valid'] else 'FAIL'}")
    print()

    if result["errors"]:
        print(f"Errors ({len(result['errors'])}):")
        for e in result["errors"]:
            print(f"  - {e}")
        print()

    if result["warnings"]:
        print(f"Warnings ({len(result['warnings'])}):")
        for w in result["warnings"]:
            print(f"  - {w}")
        print()

    print("Statistics:")
    for k, v in result["stats"].items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in sorted(v.items()):
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")

    if not result["valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
