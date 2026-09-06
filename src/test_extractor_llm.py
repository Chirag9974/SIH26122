"""Offline invariant tests for the LLM extractor (mocked client, no Ollama).

Run:  python test_extractor_llm.py

Guards the PDF section 3 architecture without touching the network: mock
outputs cover the happy path, dirty-but-repairable output, schema-failure
fallback, negation safety, uncertainty review, evidence traceability,
caching, and the no-schedule-id contract.
"""
from __future__ import annotations

import json

from extractor_llm import extract
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
    g = good_generation()
    g["events"][0]["execution"]["assertion"] = "uncertain"
    g["events"][0]["warnings"] = ["uncertain_statement"]
    out = extract(REPORT, model="mock", use_cache=False,
                  client=make_client(g))
    assert out["events"][0]["needs_review"] is True


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


if __name__ == "__main__":
    main()
