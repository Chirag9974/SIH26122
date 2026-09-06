"""Invariant checks for the extractor contract. Run: python test_extractor.py

These guard the rules the spec is strict about, independently of any dataset:
never invent a field, never turn a negation into a completion, never emit a
schedule_activity_id, always keep evidence traceable to the raw text.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path as _P
from datetime import date

sys.path.insert(0, str(_P(__file__).resolve().parents[2]))

from extraction.extractor import extract, parse_quantity, parse_times

D = "2026-09-01"
RD = date(2026, 9, 1)


def run(text: str, **kw) -> dict:
    return extract({"report_id": "T", "source_type": "daily_report",
                    "report_date": D, "raw_text": text, **kw})


def one(text: str, **kw) -> dict:
    out = run(text, **kw)
    assert len(out["events"]) == 1, f"expected 1 event, got {len(out['events'])}: {text}"
    return out["events"][0]


# --- time parsing ---------------------------------------------------------
def test_times():
    assert parse_times("from 10:00 to 16:00", RD) == (
        f"{D}T10:00:00", f"{D}T16:00:00", "explicit")
    assert parse_times("from 10 am to 4 pm", RD) == (
        f"{D}T10:00:00", f"{D}T16:00:00", "explicit")
    assert parse_times("1000-1600", RD) == (
        f"{D}T10:00:00", f"{D}T16:00:00", "explicit")
    assert parse_times("10 to 4", RD) == (
        f"{D}T10:00:00", f"{D}T16:00:00", "explicit")
    # overnight rolls to next day
    assert parse_times("from 22:00 to 02:00", RD)[1] == "2026-09-02T02:00:00"
    # no time -> null, never invented
    assert parse_times("work finished today", RD) == (None, None, "missing")
    # a quantity is not a clock time
    assert parse_times("around 5 of 12 spools", RD) == (None, None, "missing")


def test_quantity():
    assert parse_quantity("5 of 12 spools completed") == (5.0, 12.0, "spool")
    assert parse_quantity("300 of 400 m done") == (300.0, 400.0, "m")
    assert parse_quantity("no quantities here") == (None, None, None)
    # nonsense (done > total) is rejected rather than reported
    assert parse_quantity("20 of 12 spools") == (None, None, None)


# --- the two hard rules ---------------------------------------------------
def test_missing_fields_are_null():
    e = one("Welding finished.")
    assert e["context"]["location"] is None
    assert e["context"]["line_number"] is None
    assert e["context"]["equipment"] is None
    assert e["time"]["start"] is None and e["time"]["end"] is None
    assert e["time"]["time_certainty"] == "missing"
    assert e["quantity"] == {"completed": None, "total": None, "unit": None}
    assert "missing_location" in e["warnings"] and "missing_time" in e["warnings"]


def test_negation_is_not_completion():
    for text in ("No welding was done at Rack A today.",
                 "Hydrotest of Line 24 could not be carried out.",
                 "Painting of 12in Line 30 not yet started.",
                 "Grouting at Unit 100 has been cancelled."):
        e = one(text)
        assert e["execution"]["status"] != "completed", text
        assert e["execution"]["assertion"] == "negated", text
        assert e["execution"]["progress_percent"] is None, text
        assert "negated_statement" in e["warnings"], text


def test_suspension_is_affirmed():
    """A stoppage really happened -- it is not a negated statement."""
    e = one("Excavation at Area 20 stopped at 11:00 after hitting a buried cable.")
    assert e["execution"]["status"] == "suspended"
    assert e["execution"]["assertion"] == "affirmed"


def test_uncertainty_flagged():
    e = one("Cable pulling in trench T-4 probably completed, to be confirmed.")
    assert e["execution"]["assertion"] == "uncertain"
    assert "uncertain_statement" in e["warnings"]
    assert e["execution"]["progress_percent"] is None, "hedged claim is not 100%"


def test_irrelevant_yields_no_events():
    out = run("Lunch was delayed today because the canteen was crowded.")
    assert out["relevance"]["is_relevant"] is False
    assert out["events"] == []


def test_partial_progress_from_quantities_only():
    e = one("Started erecting 24in spool. Around 5 of 12 spools completed today.")
    assert e["execution"]["status"] == "in_progress"
    assert e["quantity"]["completed"] == 5.0 and e["quantity"]["total"] == 12.0
    assert e["execution"]["progress_percent"] == 42
    assert "quantity_partial" in e["warnings"]


def test_cross_sentence_start_then_finish():
    e = one("Crew started Line 22 spool erection at 10 AM. Work was completed by 4 PM.")
    assert e["execution"]["status"] == "completed"
    assert e["time"]["start"] == f"{D}T10:00:00"
    assert e["time"]["end"] == f"{D}T16:00:00"


def test_multi_event_report():
    out = run("Civil: shuttering completed at Area 20 from 07:00 to 12:00. "
              "Electrical: cable pulling started at MCC Room at 09:00.")
    assert len(out["events"]) == 2
    assert [e["activity"]["action"] for e in out["events"]] == ["shuttering", "cable pulling"]
    assert out["events"][0]["context"]["discipline"] == "Civil"
    assert out["events"][1]["context"]["discipline"] == "Electrical"


def test_conflict_downgrades_completion():
    e = one("Erection of 24in spool Line 24 at Rack B completed. "
            "Later found incomplete, two joints still open.")
    assert e["execution"]["status"] == "in_progress"
    assert e["execution"]["progress_percent"] is None
    assert "conflicting_report" in e["warnings"]


def test_duplicate_flagged_not_dropped():
    out = run("Backfilling at Area 10 completed from 08:00 to 12:00. "
              "Backfilling at Area 10 completed from 08:00 to 12:00.")
    assert len(out["events"]) == 2, "duplicates are flagged, never silently merged"
    assert "possible_duplicate" in out["events"][1]["warnings"]


# --- contract shape -------------------------------------------------------
def test_never_emits_schedule_id():
    """Matching is a separate stage (spec 0.1). The extractor must stay out of it."""
    blob = json.dumps(run("Piping crew completed erection of 24in spool Line 24 "
                          "at Rack B from 10 AM to 4 PM."))
    for banned in ("schedule_activity_id", "activity_id", "wbs_code", "match"):
        assert banned not in blob, f"extractor leaked {banned}"


def test_evidence_is_substring_of_raw():
    for text in ("Piping crew completed erection of 24in spool Line 24 at Rack B "
                 "from 10 AM to 4 PM.",
                 "Crew started Line 22 spool erection at 10 AM. Work completed by 4 PM."):
        out = run(text)
        for e in out["events"]:
            for part in e["evidence"]["source_text"].split(". "):
                assert part.strip(". ") in text, f"evidence not traceable: {part!r}"


def test_required_keys_present():
    e = one("Piping crew completed erection of 24in spool Line 24 at Rack B "
            "from 10 AM to 4 PM.")
    assert set(e) == {"event_id", "activity", "execution", "time", "context",
                      "quantity", "evidence", "warnings", "confidence"}
    assert set(e["confidence"]) == {"overall", "activity", "status", "time"}
    assert all(0.0 <= v <= 1.0 for v in e["confidence"].values())


def test_confidence_drops_without_identifier():
    strong = one("Piping crew completed erection of 24in spool Line 24 at Rack B "
                 "from 10 AM to 4 PM.")
    weak = one("Spool erection completed in Rack B.")
    assert weak["confidence"]["overall"] < strong["confidence"]["overall"]


def test_warnings_are_known_vocabulary():
    from common.vocab import WARNINGS
    for text in ("Welding finished.", "No welding was done at Rack A today.",
                 "Around 5 of 12 spools completed.", "Spool erection completed in Rack B."):
        for e in run(text)["events"]:
            unknown = set(e["warnings"]) - set(WARNINGS)
            assert not unknown, f"unknown warnings {unknown} for {text!r}"


def main() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
