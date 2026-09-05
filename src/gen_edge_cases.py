"""Hand-written adversarial edge cases (spec section 9) + their gold labels.

Why hand-written: gen_reports.py renders from templates that share vocab.py with
the extractor, so template-derived scores flatter the baseline. These texts were
written independently of the extractor's regexes. This is the honest test set.

Emits the spec-section-13 files (each line = {raw, gold} paired):
  data/edge_cases/{irrelevant,ambiguous,conflicting,negative,noise,relative}.jsonl
and merges everything into data/evaluation/test_set.jsonl.

Gold convention: only fields the TEXT supports. Absent -> null.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDGE_DIR = ROOT / "data" / "edge_cases"
TEST_OUT = ROOT / "data" / "evaluation" / "test_set.jsonl"

D = "2026-09-14"   # a Monday; all cases share it so relative dates are checkable


def ev(action, status, *, assertion="affirmed", pct=None, start=None, end=None,
       cert="missing", disc=None, loc=None, equip=None, line=None,
       qc=None, qt=None, unit=None, warn=()):
    return {
        "activity": {"action": action},
        "execution": {"status": status, "assertion": assertion, "progress_percent": pct},
        "time": {"start": start, "end": end, "time_certainty": cert},
        "context": {"discipline": disc, "location": loc, "equipment": equip,
                    "line_number": line},
        "quantity": {"completed": qc, "total": qt, "unit": unit},
        "warnings": sorted(warn),
    }


def T(hhmm, day=D):
    return f"{day}T{hhmm}:00"


# ---------------------------------------------------------------------------
# case packs. text = what a supervisor actually wrote.
# ---------------------------------------------------------------------------

IRRELEVANT = [
    ("Canteen menu changed from today, non-veg on Wednesdays.", []),
    ("Safety induction for 12 new joinees held in the training room.", []),
    ("Diesel bowser could not reach site, road blocked by village procession.", []),
    ("Client representative visited site office for document review.", []),
    ("Attendance for the day: 84 workmen, 6 staff.", []),
]

AMBIGUOUS = [
    # no identifier, no location: two schedule activities could fit
    ("Spool erection completed in Rack B.",
     [ev("erection", "completed", pct=100, loc="Rack B",
         warn=["ambiguous_activity", "missing_line_number", "missing_time"])]),
    ("Welding finished, rest tomorrow.",
     [ev("welding", "completed", pct=100,
         warn=["ambiguous_activity", "missing_line_number", "missing_location",
               "missing_time"])]),
    ("Concrete pour done at the usual place.",
     [ev("concreting", "completed", pct=100,
         warn=["ambiguous_activity", "missing_line_number", "missing_location",
               "missing_time"])]),
    # uncertain hedging: assertion must be 'uncertain', not affirmed
    # "probably completed" is not an explicit report -> progress_percent stays null
    ("Cable pulling in trench T-4 probably completed, to be confirmed with foreman.",
     [ev("cable pulling", "completed", assertion="uncertain",
         loc="Cable Trench T-4",
         warn=["ambiguous_activity", "missing_line_number", "missing_time",
               "uncertain_statement"])]),
    ("Around 5 of 12 spools erected, exact count to be confirmed.",
     [ev("erection", "in_progress", assertion="uncertain", pct=42,
         qc=5.0, qt=12.0, unit="spool",
         warn=["ambiguous_activity", "missing_line_number", "missing_location",
               "missing_time", "quantity_partial", "uncertain_statement"])]),
]

NEGATIVE = [
    ("Hydrotest of Line 24 could not be carried out, blind flange not available.",
     [ev("hydrotest", "not_started", assertion="negated", line="24",
         warn=["missing_location", "missing_time", "negated_statement"])]),
    ("No welding was done at Rack A today.",
     [ev("welding", "not_started", assertion="negated", loc="Rack A",
         warn=["ambiguous_activity", "missing_line_number", "missing_time",
               "negated_statement"])]),
    ("Grouting at Unit 100 has been cancelled, foundation rework required.",
     [ev("grouting", "cancelled", assertion="negated", loc="Unit 100",
         warn=["ambiguous_activity", "missing_line_number", "missing_time",
               "negated_statement"])]),
    # suspension is an AFFIRMED event about stoppage, not a negation
    ("Excavation at Area 20 stopped at 11:00 after hitting a buried cable.",
     [ev("excavation", "suspended", start=T("11:00"), cert="explicit", loc="Area 20",
         warn=["ambiguous_activity", "missing_line_number"])]),
    ("Painting of 12in Line 30 not yet started, surface prep pending.",
     [ev("painting", "not_started", assertion="negated", line="30",
         warn=["missing_location", "missing_time", "negated_statement"])]),
]

CONFLICTING = [
    # two statements about the same work, second contradicts the first
    ("Erection of 24in spool Line 24 at Rack B completed. Later found incomplete, "
     "two joints still open.",
     [ev("erection", "in_progress", line="24", loc="Rack B",
         warn=["conflicting_report", "missing_time"])]),
    ("Loop check of PT-210 done in the morning. Loop check of PT-210 could not be "
     "completed, DCS not ready.",
     [ev("loop check", "not_started", assertion="negated", equip="PT-210",
         warn=["conflicting_report", "missing_location", "missing_time"])]),
]

DUPLICATE = [
    # same work reported twice in one report -> flag, keep both
    ("Backfilling at Area 10 completed from 08:00 to 12:00. "
     "Backfilling at Area 10 completed from 08:00 to 12:00.",
     [ev("backfilling", "completed", pct=100, start=T("08:00"), end=T("12:00"),
         cert="explicit", loc="Area 10",
         warn=["ambiguous_activity", "missing_line_number"]),
      ev("backfilling", "completed", pct=100, start=T("08:00"), end=T("12:00"),
         cert="explicit", loc="Area 10",
         warn=["ambiguous_activity", "missing_line_number", "possible_duplicate"])]),
]

RELATIVE = [
    # relative / overnight dates -- resolved against report_date
    # overnight window: end rolls to the next calendar day. "continued" reports
    # ongoing work, so status is in_progress and progress stays null.
    ("Night shift continued 18in spool erection at Rack C from 22:00 to 02:00.",
     [ev("erection", "in_progress", start=T("22:00"), end=T("02:00", "2026-09-15"),
         cert="explicit", loc="Rack C",
         warn=["ambiguous_activity", "missing_line_number", "relative_date_resolved"])]),
    ("Yesterday's pending termination of C-431 was completed at 09:30 today.",
     [ev("termination", "completed", pct=100, start=T("09:30"), cert="explicit",
         equip="C-431", warn=["missing_location"])]),
]

NOISE = [
    # weather/delay/safety noise mixed with real progress
    ("Rain stopped work for 2 hours. After that, 3 of 8 spools were erected at Rack B.",
     [ev("erection", "in_progress", pct=38, qc=3.0, qt=8.0, unit="spool", loc="Rack B",
         warn=["ambiguous_activity", "missing_line_number", "missing_time",
               "quantity_partial"])]),
    # "PIP" is a discipline abbreviation the alias table must resolve
    ("PIP - 8in joints wldg @ rb-a, 0800-1730, compl. Crane idle 1 hr, no impact.",
     [ev("welding", "completed", pct=100, start=T("08:00"), end=T("17:30"),
         cert="explicit", disc="Piping", loc="Rack A",
         warn=["ambiguous_activity", "missing_line_number"])]),
    # cross-sentence: start in one sentence, finish in the next
    ("Crew started Line 22 spool erection at 10 AM. Work was completed by 4 PM.",
     [ev("erection", "completed", pct=100, start=T("10:00"), end=T("16:00"),
         cert="explicit", line="22", warn=["missing_location"])]),
    # start-only
    ("Line 32 hydrotest started at 10:15 AM, pressurising in progress.",
     [ev("hydrotest", "in_progress", start=T("10:15"), cert="explicit", line="32",
         warn=["missing_location"])]),
    # finish-only
    ("Remaining Line 40 spool was completed at 16:00.",
     [ev("erection", "completed", pct=100, start=T("16:00"), cert="explicit", line="40",
         warn=["missing_location"])]),
    # multi-discipline abbreviated one-liner
    ("CIV: shuttering Area 20 compl 0700-1200. ELE: cbl pulling mcc 300 of 400 m done.",
     [ev("shuttering", "completed", pct=100, start=T("07:00"), end=T("12:00"),
         cert="explicit", disc="Civil", loc="Area 20",
         warn=["ambiguous_activity", "missing_line_number"]),
      ev("cable pulling", "in_progress", pct=75, qc=300.0, qt=400.0, unit="m",
         disc="Electrical", loc="MCC Room",
         warn=["ambiguous_activity", "missing_line_number", "missing_time",
               "quantity_partial"])]),
]

PACKS = {
    "irrelevant": IRRELEVANT,
    "ambiguous": AMBIGUOUS,
    "negative": NEGATIVE,
    "conflicting": CONFLICTING + DUPLICATE,
    "relative": RELATIVE,
    "noise": NOISE,
}


def main() -> None:
    EDGE_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    n = 0

    for pack, cases in PACKS.items():
        rows = []
        for text, events in cases:
            n += 1
            rid = f"EDGE-{n:03d}"
            report = {"report_id": rid, "source_type": "daily_report",
                      "report_date": D, "discipline": None, "raw_text": text}
            gold = {
                "report_id": rid,
                "relevance": {"is_relevant": bool(events),
                              "reason": "contains project execution information" if events
                                        else "no project execution information"},
                "events": [e | {"event_id": f"{rid}-EVT-{i:02d}"}
                           for i, e in enumerate(events, 1)],
                "case_kind": f"edge_{pack}",
            }
            rows.append({"report": report, "gold": gold})
        (EDGE_DIR / f"{pack}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8")
        all_rows += rows
        print(f"{pack:<12} {len(rows):>2} cases")

    TEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    TEST_OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_rows),
                        encoding="utf-8")
    n_ev = sum(len(r["gold"]["events"]) for r in all_rows)
    print(f"\n{TEST_OUT}: {len(all_rows)} reports, {n_ev} gold events")


if __name__ == "__main__":
    main()
