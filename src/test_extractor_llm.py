"""Offline invariant tests for the LLM extractor (mocked client, no Ollama).

Run:  python test_extractor_llm.py

Guards the PDF section 3 architecture without touching the network: mock
outputs cover the happy path, dirty-but-repairable output, schema-failure
fallback, negation safety, uncertainty review, evidence traceability,
caching, and the no-schedule-id contract.
"""
from __future__ import annotations

import json

from extractor_llm import extract, extract_report
from vocab import WARNINGS

REPORT = {"report_id": "T-1", "source_type": "daily_report",
          "report_date": "2026-09-01", "discipline": "Piping",
          "raw_text": "Piping crew completed erection of 24in spool Line 24 "
                      "at Rack B from 10 AM to 4 PM."}


def good_generation() -> dict:
    return {
        "relevance": {"is_relevant": True, "confidence": 0.95, "reason": "work"},
        "events": [{
            "activity": {"description": "24in spool erection",
                         "action": "erection"},
            "execution": {"status": "completed", "assertion": "affirmed",
                          "progress_percent": 100},
            "time": {"start": "2026-09-01T10:00:00",
                     "end": "2026-09-01T16:00:00", "certainty": "explicit"},
            "context": {"discipline": "Piping", "location": "Rack B",
                        "line_number": "24", "equipment": None},
            "quantity": {"completed": None, "total": None, "unit": None},
            "issue": {"type": None, "reason": None},
            "evidence": {"source_text":
                         "completed erection of 24in spool Line 24 at Rack B"},
            "confidence": {"overall": 0.9, "activity": 0.9, "status": 0.9,
                           "time": 0.9},
            "needs_review": False,
            "warnings": [],
        }]}


def make_client(payload, stages=None):
    """Mock chat client: returns payload every stage, or the staged list."""
    calls = {"n": 0}

    def client(model, messages, *, schema_json, options, stage=0):
        calls["n"] += 1
        calls["stage"] = stage
        if isinstance(payload, str):
            return payload
        if stages is not None and calls["n"] <= len(stages):
            return json.dumps(stages[calls["n"] - 1])
        return json.dumps(payload)

    client.calls = calls
    return client


def test_happy_path():
    out = extract(REPORT, model="mock", use_cache=False,
                  client=make_client(good_generation()))
    assert out["_meta"]["fallback"] is False
    assert out["relevance"]["is_relevant"] is True
    assert len(out["events"]) == 1
    ev = out["events"][0]
    assert ev["activity"]["action"] == "erection"
    assert ev["execution"]["status"] == "completed"
    assert ev["time"]["end"] == "2026-09-01T16:00:00"
    assert ev["evidence"]["source_text"] in REPORT["raw_text"]


def test_dirty_output_repairs():
    g = good_generation()
    ev = g["events"][0]
    ev["execution"]["progress_percent"] = "100"
    ev["activity"]["action"] = "ERECTED"
    ev["quantity"]["completed"] = "5 of 12"
    del ev["context"]
    out = extract(REPORT, model="mock", use_cache=False,
                  client=make_client(g))
    ev = out["events"][0]
    assert ev["execution"]["progress_percent"] == 100
    assert ev["activity"]["action"] == "erection"
    assert ev["quantity"]["completed"] == 5.0
    assert ev["context"]["discipline"] is None


def test_fenced_json_accepted():
    text = "```json" + chr(10) + json.dumps(good_generation()) + chr(10) + "```"
    out = extract(REPORT, model="mock", use_cache=False,
                  client=make_client(text))
    assert out["events"] and out["_meta"]["fallback"] is False


def test_schema_failure_falls_back_not_crashes():
    out = extract(REPORT, model="mock", use_cache=False,
                  client=make_client("not json at all"))
    assert out["_meta"]["fallback"] is True
    assert out["events"] == []
    assert "schema failure" in out["_meta"]["fallback_reason"]


def test_fallback_flags_review_not_fabricates():
    out = extract(REPORT, model="mock", use_cache=False,
                  client=make_client("garbage"))
    assert out["relevance"]["is_relevant"] is True
    assert out["relevance"]["confidence"] <= 0.5


def test_negation_never_completion():
    g = good_generation()
    g["events"][0]["execution"]["assertion"] = "negated"
    out = extract(REPORT, model="mock", use_cache=False,
                  client=make_client(g))
    ev = out["events"][0]
    assert ev["execution"]["status"] is None
    assert ev["execution"]["progress_percent"] is None
    assert ev["needs_review"] is True
    assert "negated_statement" in ev["warnings"]


def test_uncertainty_needs_review():
    # Real hedge in the text: the review flag must survive (model hedging a
    # genuinely uncertain sentence is correct behavior).
    g = good_generation()
    g["events"][0]["execution"]["assertion"] = "uncertain"
    g["events"][0]["warnings"] = ["uncertain_statement"]
    g["events"][0]["evidence"]["source_text"] = (
        "erection of 24in spool Line 24 at Rack B from 10 AM to 4 PM")
    hedged = {"report_id": "T-1", "source_type": "daily_report",
              "report_date": "2026-09-01", "discipline": "Piping",
              "raw_text": "Piping crew says erection of 24in spool Line 24 "
                          "at Rack B from 10 AM to 4 PM appears complete."}
    out = extract(hedged, model="mock", use_cache=False,
                  client=make_client(g))
    assert out["events"][0]["needs_review"] is True
    assert out["events"][0]["execution"]["assertion"] == "uncertain"


def test_relevance_false_no_events():
    g = {"relevance": {"is_relevant": False, "confidence": 0.9,
                       "reason": "lunch"}, "events": []}
    out = extract(REPORT, model="mock", use_cache=False,
                  client=make_client(g))
    assert out["events"] == []


def test_unsupported_evidence_flagged():
    g = good_generation()
    g["events"][0]["evidence"]["source_text"] = "the moon is made of cheese"
    out = extract(REPORT, model="mock", use_cache=False,
                  client=make_client(g))
    ev = out["events"][0]
    assert "unsupported_evidence" in ev["warnings"]
    assert ev["needs_review"] is True


def test_overnight_rollover():
    g = good_generation()
    g["events"][0]["time"] = {"start": "2026-09-01T22:00:00",
                              "end": "2026-09-01T02:00:00",
                              "certainty": "explicit"}
    out = extract(REPORT, model="mock", use_cache=False,
                  client=make_client(g))
    assert out["events"][0]["time"]["end"] == "2026-09-02T02:00:00"


def test_warnings_closed_vocabulary():
    for client in (make_client(good_generation()),
                   make_client("garbage")):
        out = extract(REPORT, model="mock", use_cache=False, client=client)
        for ev in out["events"]:
            unknown = set(ev["warnings"]) - set(WARNINGS)
            assert not unknown, unknown


def test_never_emits_schedule_id():
    out = extract(REPORT, model="mock", use_cache=False,
                  client=make_client(good_generation()))
    blob = json.dumps(out)
    for banned in ("schedule_activity_id", "activity_id", "wbs_code"):
        assert banned not in blob


def test_cache_hit_avoids_second_call():
    client = make_client(good_generation())
    a = extract(REPORT, model="cachetest", use_cache=True, client=client)
    n_after_first = client.calls["n"]
    b = extract(REPORT, model="cachetest", use_cache=True, client=client)
    assert client.calls["n"] == n_after_first, "second call must hit cache"
    assert a["events"] == b["events"]


def test_retry_then_success():
    bad = dict(good_generation())
    bad["events"] = "not-a-list"
    client = make_client(None, stages=[bad, good_generation()])
    out = extract(REPORT, model="mock", use_cache=False, client=client)
    assert out["events"] and client.calls["n"] == 2


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
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)




# --- MVP additions: case coverage + stable API + fallback chain ----------

HINGLISH_REPORT = {
    "report_id": "T-HIN", "source_type": "daily_report",
    "report_date": "2026-09-01", "discipline": None,
    "raw_text": "Aaj welding kaam Rack A pe ho gaya 08:00 se 16:00 tak.",
}


def test_partial_progress_quantity():
    g = good_generation()
    ev = g["events"][0]
    ev["execution"]["status"] = "in_progress"
    ev["quantity"] = {"completed": 5.0, "total": 12.0, "unit": "spool"}
    ev["warnings"] = ["quantity_partial"]
    out = extract(REPORT, model="mock", use_cache=False,
                  client=make_client(g))
    ev = out["events"][0]
    assert ev["execution"]["status"] == "in_progress"
    assert ev["execution"]["progress_percent"] == 42
    assert "quantity_partial" in ev["warnings"]


def test_negative_stays_negated():
    g = good_generation()
    ev = g["events"][0]
    ev["execution"]["status"] = "not_started"
    ev["execution"]["assertion"] = "negated"
    ev["execution"]["progress_percent"] = None
    out = extract(REPORT, model="mock", use_cache=False,
                  client=make_client(g))
    ev = out["events"][0]
    assert ev["execution"]["assertion"] == "negated"
    assert ev["execution"]["status"] == "not_started"
    assert ev["execution"]["progress_percent"] is None
    assert "negated_statement" in ev["warnings"]
    assert ev["needs_review"] is False  # negation is explicit, not uncertain


def test_irrelevant_no_events():
    g = {"relevance": {"is_relevant": False, "confidence": 0.92,
                       "reason": "canteen logistics"}, "events": []}
    out = extract(REPORT, model="mock", use_cache=False,
                  client=make_client(g))
    assert out["relevance"]["is_relevant"] is False
    assert out["events"] == []


def test_hinglish_event_extracted():
    g = {
        "relevance": {"is_relevant": True, "confidence": 0.9, "reason": "work"},
        "events": [{
            "activity": {"description": "welding kaam", "action": "welding"},
            "execution": {"status": "completed", "assertion": "affirmed",
                          "progress_percent": 100},
            "time": {"start": "08:00", "end": "16:00", "certainty": "explicit"},
            "context": {"discipline": None, "location": "Rack A",
                        "line_number": None, "equipment": None},
            "quantity": {"completed": None, "total": None, "unit": None},
            "issue": {"type": None, "reason": None},
            "evidence": {"source_text":
                         "welding kaam Rack A pe ho gaya 08:00 se 16:00 tak"},
            "confidence": {"overall": 0.85, "activity": 0.9, "status": 0.9,
                           "time": 0.7},
            "needs_review": False,
            "warnings": [],
        }]}
    out = extract(HINGLISH_REPORT, model="mock", use_cache=False,
                  client=make_client(g))
    ev = out["events"][0]
    assert ev["activity"]["action"] == "welding"
    assert ev["execution"]["status"] == "completed"
    assert ev["time"]["start"] == "2026-09-01T08:00:00"
    assert ev["context"]["location"] == "Rack A"


def test_prompt_declares_multilingual_support():
    from prompt import SYSTEM_PROMPT
    low = SYSTEM_PROMPT.lower()
    for token in ("hindi", "hinglish", "shorthand"):
        assert token in low, token


def test_missing_fields_stay_null():
    g = good_generation()
    ev = g["events"][0]
    ev["context"] = {"discipline": None, "location": None,
                     "line_number": None, "equipment": None}
    ev["quantity"] = {"completed": None, "total": None, "unit": None}
    ev["execution"]["progress_percent"] = None
    ev["execution"]["status"] = None
    out = extract(REPORT, model="mock", use_cache=False,
                  client=make_client(g))
    ev = out["events"][0]
    assert ev["context"]["location"] is not None  # sole location recovered
    assert ev["context"]["line_number"] is not None  # sole "Line NN" recovered
    assert ev["quantity"] == {"completed": None, "total": None,
                              "unit": None}
    # Status IS resolvable from the text's completion cue -> recovered,
    # not invented (deterministic evidence-backed repair).
    assert ev["execution"]["status"] == "completed"
    assert ev["needs_review"] is False

    # A status the text cannot resolve still forces review.
    g2 = good_generation()
    g2["events"][0]["execution"]["status"] = None
    g2["events"][0]["evidence"]["source_text"] = "24in spool Line 24 at Rack B"
    plain = {"report_id": "T-2", "source_type": "daily_report",
             "report_date": "2026-09-01", "discipline": "Piping",
             "raw_text": "Piping crew worked on the 24in spool Line 24 "
                         "at Rack B from 10 AM to 4 PM."}
    out2 = extract(plain, model="mock", use_cache=False,
                   client=make_client(g2))
    assert out2["events"][0]["execution"]["status"] is None
    assert out2["events"][0]["needs_review"] is True


def test_retry_once_then_safe_result():
    client = make_client("still garbage")
    out = extract(REPORT, model="mock", use_cache=False, client=client)
    assert client.calls["n"] == 2, "exactly one retry after first failure"
    assert out["_meta"]["fallback"] is True and out["events"] == []


def test_extract_report_stable_api_llm():
    out = extract_report(REPORT["raw_text"],
                         metadata={"report_date": "2026-09-01"},
                         model="mock", use_cache=False,
                         client=make_client(good_generation()))
    assert out["_meta"]["engine"] == "llm"
    assert set(out) >= {"relevance", "events"}
    assert out["events"][0]["activity"]["action"] == "erection"


def test_extract_report_baseline_fallback():
    out = extract_report(REPORT["raw_text"],
                         metadata={"report_date": "2026-09-01"},
                         model="mock", use_cache=False,
                         client=make_client("garbage"))
    assert out["_meta"]["engine"] == "baseline"
    assert out["needs_review"] is True
    assert out["events"], "baseline must recover the obvious event"
    assert out["events"][0]["warnings"][-1] == "baseline_fallback"


def test_extract_report_nothing_usable_is_safe():
    out = extract_report("asdf qwer zxcv jkl", metadata=None,
                         model="mock", use_cache=False,
                         client=make_client("garbage"))
    assert out["needs_review"] is True
    assert out["events"] == []
    assert out["relevance"]["is_relevant"] is False


def test_extract_report_never_raises():
    # server-side explosion is contained by the API
    def exploding_client(*a, **k):
        raise ConnectionError("ollama down")
    out = extract_report(REPORT["raw_text"],
                         metadata={"report_date": "2026-09-01"},
                         model="mock", use_cache=False,
                         client=exploding_client)
    assert out["_meta"]["engine"] in ("baseline", "fallback")
    assert "needs_review" in out




def test_evaluate_captures_case_types_and_examples():
    from eval_extractor import evaluate
    gold = {"report_id": "G1", "case_kind": "partial",
            "relevance": {"is_relevant": True},
            "events": [{"event_id": "G1-EVT-01",
                        "activity": {"action": "welding"},
                        "execution": {"status": "in_progress",
                                      "progress_percent": 42},
                        "context": {"discipline": None, "location": "Rack A",
                                    "line_number": None, "equipment": None},
                        "time": {"start": None, "end": None},
                        "quantity": {"completed": 5.0, "total": 12.0}}]}
    report = {"raw_text": "welding at rack a, 5 of 12 done",
              "report_id": "G1"}
    pred = {"relevance": {"is_relevant": True},
            "events": [{"activity": {"action": "welding"},
                        "execution": {"status": "completed",
                                      "progress_percent": None},
                        "context": {"discipline": None, "location": "Rack A",
                                    "line_number": None, "equipment": None},
                        "time": {"start": None, "end": None},
                        "quantity": {"completed": 5.0, "total": 12.0},
                        "evidence": {"source_text": "welding at rack a"}}]}
    m, exs = evaluate(pred, gold, report)
    assert m["kind:partial"]["tp"] == 1
    statuses = [e for e in exs if e["field"] == "execution.status"]
    ok = statuses and statuses[0]["gold"] == "in_progress"
    ok = ok and statuses[0]["pred"] == "completed"
    assert ok
    assert m["field:activity.action"]["ok"] == 1


if __name__ == "__main__":
    main()
