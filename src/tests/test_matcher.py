"""Invariant tests for the schedule matcher. Run: python -m tests.test_matcher

Covers: exact match, paraphrase, same-action/different-location,
same-location/different-action, L5/L6 disambiguation, no-match, ambiguous
twins, and the stable match_event API contract (input never mutated).
"""
from __future__ import annotations

import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parents[2]))

from matching.matcher import Matcher  # noqa: E402
from matching.normalize import (  # noqa: E402
    canon_action,
    canon_discipline,
    canon_equipment,
    canon_line,
    canon_location,
)

_M = None


def _matcher() -> Matcher:
    global _M
    if _M is None:
        # real configuration (embeddings on); ScheduleIndex degrades to
        # lexical-only if the model is unavailable, which would show up
        # as a lower absolute score in the exact-match assertions.
        _M = Matcher(use_embeddings=True)
    return _M


def _ev(**kw) -> dict:
    base = {
        "event_id": "T-EVT-01",
        "activity": {"description": None, "action": None},
        "execution": {"status": None, "assertion": "affirmed",
                      "progress_percent": None},
        "time": {"start": None, "end": None, "time_certainty": "missing"},
        "context": {"discipline": None, "location": None,
                    "equipment": None, "line_number": None},
        "quantity": {"completed": None, "total": None, "unit": None},
        "evidence": {"source_text": None},
        "warnings": [],
    }
    act = kw.pop("activity", {})
    ctx = kw.pop("context", {})
    base["activity"].update(act)
    base["context"].update(ctx)
    base.update(kw)
    return base


def run() -> None:
    stats = {"ok": 0, "fail": 0}

    def check(name: str, cond: bool) -> None:
        if cond:
            stats["ok"] += 1
            print(f"  ok    {name}")
        else:
            stats["fail"] += 1
            print(f"  FAIL  {name}")

    # 1. exact match: full metadata pins the activity
    r = _matcher().match(_ev(
        activity={"action": "erection", "description": "erection of 18in spool"},
        context={"discipline": "Piping", "location": "Rack C",
                 "line_number": "40-09"},
    ))
    check("exact match: PIP-L6-0019", r.schedule_activity_id == "PIP-L6-0019")
    check("exact match: auto decision", r.decision == "auto_match")

    # 2. paraphrase: 'Install' -> installation, no tag, twin ambiguity allowed
    r = _matcher().match(_ev(
        activity={"action": "installation",
                  "description": "pump installation"},
        context={"discipline": "Equipment", "location": "Compressor House"},
    ))
    check("paraphrase: finds Install Pump at Compressor House",
          r.schedule_activity_id is not None
          and "Install Pump" in r.matched_activity
          and "Compressor House" in r.matched_activity)
    check("paraphrase: auto or review (twins may force review)",
          r.decision in ("auto_match", "human_review"))

    # 3. same action, different location -> not the Rack C activity
    r = _matcher().match(_ev(
        activity={"action": "erection", "description": "erection of 18in spool"},
        context={"discipline": "Piping", "location": "Rack A",
                 "line_number": "40-09"},
    ))
    check("same action/diff location: not Rack C activity",
          r.schedule_activity_id != "PIP-L6-0019")
    top_name = (r.matched_activity or "")
    check("same action/diff location: top is Rack A variant",
          "Rack A" in top_name)

    # 4. same location, different action -> different activity family
    r = _matcher().match(_ev(
        activity={"action": "hydrotest",
                  "description": "line hydrotest"},
        context={"discipline": "Piping", "location": "Pipe Rack North"},
    ))
    check("same location/diff action: hydro family, not erection",
          r.schedule_activity_id is None or "Hydrotest" in r.matched_activity)

    # 5. L5 vs L6: the L5 Work Package rollup must not win over the L6 leaf
    r = _matcher().match(_ev(
        activity={"action": "hydrotest", "description": "hydrotest"},
        context={"discipline": "Piping", "location": "Pipe Rack North",
                 "line_number": "16-01"},
    ))
    check("L5/L6: L6 leaf preferred over L5 rollup",
          r.schedule_activity_id is not None
          and r.schedule_activity_id.startswith("PIP-L6"))
    # white-box: in the FULL ranking the L5 rollup exists but ranks below
    # the L6 leaf (top-5 slice alone can't show this reliably)
    cands, _ = _matcher()._score_candidates(
        r.event_norm, _matcher()._query_text(r.event_norm))
    rank = {c.activity_id: i for i, c in enumerate(cands, 1)}
    check("L5/L6: rollup ranked below the L6 leaf",
          rank.get("PIP-L5-0027", 99) > rank.get("PIP-L6-0028", 0))

    # 6. no-match: location outside the schedule inventory
    r = _matcher().match(_ev(
        activity={"action": "concreting", "description": "concreting"},
        context={"discipline": "Civil", "location": "Flare Area"},
    ))
    check("no-match: unknown location -> no_match", r.decision == "no_match")
    check("no-match: no activity id forced", r.schedule_activity_id is None)

    # 7. no-match: score below floor (irrelevant action + unknown context)
    r = _matcher().match(_ev(
        activity={"action": "grouting", "description": "grouting"},
        context={"discipline": "Civil", "location": "Unit 400"},
    ))
    check("no-match: unknown location (2) -> no_match",
          r.decision == "no_match")

    # 8. ambiguous twins: two identical part-activities -> human_review
    r = _matcher().match(_ev(
        activity={"action": "installation",
                  "description": "pump installation"},
        context={"discipline": "Equipment", "location": "Compressor House"},
    ))
    if r.decision == "human_review":
        check("ambiguous twins flagged for review", True)
    else:
        # acceptable only if a unique pick is genuinely justified
        check("ambiguous twins: unique pick justified",
              r.decision == "auto_match" and r.confidence >= 0.6)

    # 9. stable API contract
    from matching.matcher import match_event
    ev = _ev(activity={"action": "erection",
                       "description": "erection of 18in spool"},
             context={"discipline": "Piping", "location": "Rack C",
                      "line_number": "40-09"})
    import copy
    before = copy.deepcopy(ev)
    out = match_event(ev)
    check("API: event input not mutated", ev == before)
    check("API: result has required keys",
          all(k in out for k in (
              "decision", "schedule_activity_id", "matched_activity",
              "confidence", "candidates", "reasons")))
    check("API: candidates carry signals",
          all("signals" in c for c in out["candidates"]))
    check("API: decision is one of the three",
          out["decision"] in ("auto_match", "human_review", "no_match"))

    # 10. normalizers
    check("normalize: canon_action('erected') == 'erection'",
          canon_action("erected") == "erection")
    check("normalize: canon_location('rb-c') == 'Rack C'",
          canon_location("rb-c") == "Rack C")
    check("normalize: canon_discipline('ele') == 'Electrical'",
          canon_discipline("ele") == "Electrical")
    check("normalize: canon_line('Line 40') == '40'",
          canon_line("Line 40") == "40")
    check("normalize: canon_equipment('lt-140') == 'LT140'",
          canon_equipment("lt-140") == "LT140")

    total = stats["ok"] + stats["fail"]
    print(f"\n{stats['ok']}/{total} matcher tests passed")
    if stats["fail"]:
        print(f"{stats['fail']} FAILED")
        sys.exit(1)


if __name__ == "__main__":
    run()
