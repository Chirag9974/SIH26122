"""End-to-end integration tests for the vertical slice.

Run:  python -m tests.test_pipeline

Covers the required flow matrix: strong match -> auto update, ambiguous
match -> review, no-match -> review/unmatched, accepted review -> schedule
update, rejected review -> no update, and an audit record for every update.
Uses a throwaway DB and the deterministic baseline extractor (no Ollama).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parents[2]))

from pipeline.db import PipelineDB  # noqa: E402
from pipeline.orchestrator import get_activity, process_report  # noqa: E402
from pipeline.review import ReviewService  # noqa: E402

RD = "2026-09-07"


def _db() -> PipelineDB:
    return PipelineDB(tempfile.mktemp(suffix=".db"))


def run() -> None:
    stats = {"ok": 0, "fail": 0}

    def check(name: str, cond: bool) -> None:
        if cond:
            stats["ok"] += 1
            print(f"  ok    {name}")
        else:
            stats["fail"] += 1
            print(f"  FAIL  {name}")

    # ------------------------------------------------ strong match -> auto
    db = _db()
    out = process_report({
        "report_id": "PIPE-T1", "report_date": RD, "source_type": "daily_report",
        "raw_text": "Piping crew completed erection of the 18in spool at Rack C "
                    "on Line 40-09. Work ran from 09:00 to 16:00.",
    }, db=db)
    ev1 = out["events"][0]
    check("strong match: auto decision", ev1["decision"] == "auto_match")
    check("strong match: auto-applied to schedule", ev1["auto_updated"])
    check("strong match: activity updated in DB",
          get_activity("PIP-L6-0019", db=db)["actual"] is not None)
    audit = get_activity("PIP-L6-0019", db=db)["audit"]
    check("strong match: audit record written", len(audit) >= 1)
    a0 = audit[0]
    check("audit row carries required fields",
          a0["report_id"] == "PIPE-T1" and a0["event_id"] == ev1["event_id"]
          and a0["schedule_activity_id"] == "PIP-L6-0019"
          and a0["evidence"] and a0["action_source"].startswith("pipeline")
          and a0["confidence"] is not None and a0["timestamp"])

    # ----------------------------------------------- ambiguous -> review
    db2 = _db()
    out2 = process_report({
        "report_id": "PIPE-T2", "report_date": RD, "source_type": "daily_report",
        "raw_text": "Pump installation was reported complete by the contractor. "
                    "Supervisor confirmation is still pending.",
    }, db=db2)
    ev2 = out2["events"][0]
    check("ambiguous match: not auto-applied", not ev2["auto_updated"])
    check("ambiguous match: parked for review", ev2["needs_review"])
    check("ambiguous match: listed in pending reviews",
          any(p["event_id"] == ev2["event_id"] for p in db2.pending_reviews()))

    # ------------------------------------------------ no-match -> review
    out3 = process_report({
        "report_id": "PIPE-T3", "report_date": RD, "source_type": "daily_report",
        "raw_text": "Concreting work was completed at Flare Area today.",
    }, db=db2)
    ev3 = out3["events"][0]
    check("no-match: decision is no_match", ev3["decision"] == "no_match")
    check("no-match: parked for review, nothing updated",
          ev3["needs_review"] and not ev3["auto_updated"])

    # ------------------------------------- accepted review -> update+audit
    svc = ReviewService(db2)
    before = get_activity("PIP-L6-0019", db=db2)["actual"]
    occ = svc.decide(ev2["event_id"], "accept", reviewer="supervisor1",
                     note="confirmed on site")
    check("accepted review: update applied", occ.updated
          and occ.schedule_activity_id is not None)
    after = get_activity(occ.schedule_activity_id, db=db2)
    check("accepted review: schedule changed", after["actual"] is not None)
    check("accepted review: audit written with reviewer source",
          any(a["action_source"] == "review:supervisor1" for a in after["audit"]))
    check("accepted review: review row recorded",
          any(r["action"] == "accept" for r in
              db2.conn.execute("SELECT action FROM reviews").fetchall()))

    # ------------------------------------- corrected review -> right activity
    occ2 = svc.decide(ev3["event_id"], "correct",
                      corrected_activity_id="CIV-L6-0003", reviewer="eng2",
                      note="Flare Area work actually belongs to CIV-L6-0003")
    check("corrected review: applied to corrected activity",
          occ2.updated and occ2.schedule_activity_id == "CIV-L6-0003")
    check("corrected review: audit trail on corrected activity",
          get_activity("CIV-L6-0003", db=db2)["audit"])

    # ------------------------------------- rejected review -> no update
    out4 = process_report({
        "report_id": "PIPE-T4", "report_date": RD, "source_type": "daily_report",
        "raw_text": "Painting of 8in line completed at Rack A today.",
    }, db=db2)
    ev4 = out4["events"][0]
    state_before = get_activity(ev4["schedule_activity_id"] or "PIP-L6-0019",
                                db=db2)["actual"]
    if ev4["schedule_activity_id"]:
        occ3 = svc.decide(ev4["event_id"], "reject", reviewer="eng2",
                          note="wrong activity")
        check("rejected review: nothing updated", not occ3.updated)
        state_after = get_activity(ev4["schedule_activity_id"], db=db2)["actual"]
        check("rejected review: schedule state unchanged",
              state_before == state_after or state_after is None)
    else:
        occ3 = svc.decide(ev4["event_id"], "reject", reviewer="eng2")
        check("rejected review: nothing updated", not occ3.updated)

    # ------------------------------------- mark_unmatched
    occ4 = svc.decide(ev4["event_id"], "mark_unmatched", reviewer="eng2",
                      note="new work, not in schedule")
    check("mark_unmatched: no schedule touched", not occ4.updated
          and occ4.schedule_activity_id is None)

    # ------------------------------------- validation/edge guards
    try:
        svc.decide("NOPE-EVT-01", "accept")
        check("unknown event id rejected", False)
    except KeyError:
        check("unknown event id rejected", True)
    try:
        svc.decide(ev1["event_id"], "correct")
        check("correct without activity id rejected", False)
    except (ValueError, KeyError):
        check("correct without activity id rejected", True)

    total = stats["ok"] + stats["fail"]
    print(f"\n{stats['ok']}/{total} pipeline tests passed")
    if stats["fail"]:
        print(f"{stats['fail']} FAILED")
        sys.exit(1)


if __name__ == "__main__":
    run()
