"""Report orchestration for the vertical slice.

process_report: extract (deterministic baseline or LLM) -> validate ->
match each event -> persist report + events -> auto-apply clean auto_matches.

ingest_pdf: pypdf text extraction -> process_report (PDF is supported input,
matching the attached SIH26122 test daily progress report).
"""
from __future__ import annotations

from pathlib import Path

from matching.matcher import Matcher
from pipeline.db import PipelineDB
from pipeline.review import ReviewService

_ROOT = Path(__file__).resolve().parents[2]
SCHEDULE_CSV = _ROOT / "data" / "schedule" / "schedule_activities.csv"


def ingest_pdf(path: str | Path, report_id: str | None = None,
               db: PipelineDB | None = None, engine: str = "baseline",
               use_embeddings: bool = True) -> dict:
    """PDF -> text -> process_report. Report id/date come from the text."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    text = "\n".join((p.extract_text() or "") for p in reader.pages)
    report = _report_from_text(text)
    if report_id:
        report["report_id"] = report_id
    return process_report(report, db=db, engine=engine,
                          use_embeddings=use_embeddings)


def _unwrap_pdf_text(text: str) -> str:
    """Rejoin lines the PDF wrapped mid-sentence so evidence spans match
    the raw text exactly ('Final activity is\\ncomplete.' -> one line)."""
    import re

    text = re.sub(r"(?<=[^\n.!?:])\n(?=[a-z0-9])", " ", text)
    return re.sub(r"[ \t]{2,}", " ", text)


def _report_from_text(text: str) -> dict:
    """Pull report_id / report_date out of the report header text."""
    import re

    text = _unwrap_pdf_text(text)
    report = {"source_type": "daily_report"}
    m = re.search(r"Report ID\s*:?\s*(\S+)", text)
    if m:
        report["report_id"] = m.group(1)
    m = re.search(r"Report Date\s*:?\s*(\d{4}-\d{2}-\d{2})", text)
    report["report_date"] = m.group(1) if m else None
    if not report.get("report_id"):
        report["report_id"] = "PDF-" + str(abs(hash(text[:200])) % 10_000)
    report["raw_text"] = text
    return report


_MATCHER_CACHE: dict[bool, Matcher] = {}


def _get_matcher(use_embeddings: bool) -> Matcher:
    if use_embeddings not in _MATCHER_CACHE:
        _MATCHER_CACHE[use_embeddings] = Matcher(use_embeddings=use_embeddings)
    return _MATCHER_CACHE[use_embeddings]


def process_report(
    report: dict,
    db: PipelineDB | None = None,
    engine: str = "baseline",
    use_embeddings: bool = True,
) -> dict:
    """REPORT -> EXTRACT -> MATCH -> STORE (+ auto-apply clean auto_matches)."""
    from extraction.extractor import extract as baseline_extract
    from extraction.validators import validate_extraction

    db = db or PipelineDB()

    if engine == "llm":
        from extraction.extractor_llm import extract_report
        ext = extract_report(
            report.get("raw_text", ""),
            metadata={"report_id": report.get("report_id"),
                      "report_date": report.get("report_date"),
                      "source_type": report.get("source_type")},
        )
        extraction = {
            "document": {
                "report_id": report.get("report_id"),
                "source_type": report.get("source_type"),
                "report_date": report.get("report_date"),
                "raw_text": report.get("raw_text"),
            },
            "relevance": ext.get("relevance"),
            "events": ext.get("events", []),
        }
    else:
        extraction = baseline_extract(report)

    # deterministic validation layer (contract unchanged; validation repairs
    # and flags; issues list is the audit of what it had to fix)
    extraction, _issues = validate_extraction(extraction, report)

    # the frozen schema has no event_id/report_id; re-stamp deterministically
    # so events can be reviewed and audited individually
    for i, ev in enumerate(extraction.get("events", []), start=1):
        ev.setdefault("event_id", f"{report.get('report_id')}-EVT-{i:02d}")
        ev["report_id"] = report.get("report_id")

    db.upsert_report(report, extraction, engine=engine)

    matcher = _get_matcher(use_embeddings)
    svc = ReviewService(db)
    events_out = []
    auto_applied = []
    for ev in extraction.get("events", []):
        match = matcher.match(ev).to_dict()
        db.upsert_event(ev, match)
        outcome = None
        if svc.auto_gate(ev, match):
            outcome = svc.auto_update(ev, match)
            auto_applied.append(outcome.__dict__)
        events_out.append({
            "event_id": ev.get("event_id"),
            "decision": match.get("decision"),
            "schedule_activity_id": match.get("schedule_activity_id"),
            "confidence": match.get("confidence"),
            "matched_activity": match.get("matched_activity"),
            "needs_review": bool(ev.get("needs_review"))
            or match.get("decision") != "auto_match",
            "auto_updated": outcome is not None,
        })

    return {
        "report_id": report.get("report_id"),
        "relevance": extraction.get("relevance"),
        "engine": engine,
        "events": events_out,
        "auto_applied": auto_applied,
        "pending_review": [e for e in events_out if e["needs_review"]],
    }


def get_activity(activity_id: str, db: PipelineDB | None = None) -> dict:
    """Schedule activity + latest actual state + audit history."""
    import csv

    db = db or PipelineDB()
    planned = None
    with open(SCHEDULE_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["activity_id"] == activity_id:
                planned = row
                break
    state = db.activity_state(activity_id)
    return {
        "activity_id": activity_id,
        "planned": planned,
        "actual": (state or {}).get("actual"),
        "audit": (state or {}).get("audit", []),
    }
