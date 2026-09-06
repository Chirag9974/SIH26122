"""System + user prompts for the LLM extractor (PDF section 6).

The system prompt is the PDF's specified rules. The user prompt carries only
metadata the extractor is actually allowed to see (report_date, source
discipline when the report header states it). Schedule facts and gold labels
must never appear here -- spec section 16.
"""
from __future__ import annotations

import json

SYSTEM_PROMPT = """You are a construction project progress extraction engine.
Read only the supplied field report. Return only the defined JSON schema.
Rules:
- Extract only explicitly stated or clearly implied facts.
- Never invent dates, quantities, locations, identifiers or activities.
- Missing information = null.
- One activity = one event; multiple activities = multiple events.
- Negated information must remain negated (assertion "negated", status like
  "not_started"/"cancelled"; never a completion).
- Uncertain information must remain uncertain (assertion "uncertain") and
  normally require review (needs_review=true).
- Irrelevant/non-project text -> is_relevant=false and events=[].
- Understand English, Hindi, Hinglish, shorthand and broken site language.
- Preserve supporting evidence: evidence.source_text must be copied word for
  word from the report text that supports the event.
- Do not output schedule IDs or matching decisions.
- time.certainty is "explicit" when a clock time or time window is stated,
  otherwise "missing".
- Leave execution.progress_percent null unless the text states or clearly
  implies a percentage or a completed/total quantity.
- activity.action must be a canonical work type such as erection, welding, concreting, cable pulling, calibration - never a status word like "done" or "completed".
- All timestamps must be full ISO "YYYY-MM-DDTHH:MM:SS". A stated clock time like 08:00 belongs on the report_date day.
- For conflicts (a later sentence contradicts an earlier one), follow the
  latest statement, set needs_review=true and warning "conflicting_report".
"""


def user_prompt(report: dict) -> str:
    """report: {report_id?, source_type?, report_date, discipline?, raw_text}

    Only allowed metadata is included: report_date and the discipline when
    the report header itself declares it. Never the schedule, never gold.
    """
    meta = {"report_date": report.get("report_date")}
    if report.get("discipline"):
        meta["discipline"] = report["discipline"]
    payload = {
        "metadata": meta,
        "report_text": report["raw_text"],
        "response_contract": {
            "top_level": ["relevance", "events"],
            "relevance": ["is_relevant", "confidence", "reason"],
            "event_fields": [
                "activity.description", "activity.action",
                "execution.status", "execution.assertion",
                "execution.progress_percent",
                "time.start", "time.end", "time.certainty",
                "context.discipline", "context.location",
                "context.line_number", "context.equipment",
                "quantity.completed", "quantity.total", "quantity.unit",
                "issue.type", "issue.reason",
                "evidence.source_text",
                "confidence.overall", "confidence.activity",
                "confidence.status", "confidence.time",
                "needs_review", "warnings",
            ],
            "null_means": "information absent from the text",
        },
    }
    return NL.join([
        "You are extracting execution events from this field report.",
        json.dumps(payload, ensure_ascii=False),
        "Respond with the JSON object only.",
    ])


def repair_prompt(report: dict, bad_json: str, error: str) -> str:
    """Second-chance prompt: same task with the validation error attached."""
    payload = {
        "metadata": {"report_date": report.get("report_date")},
        "report_text": report["raw_text"],
    }
    return NL.join([
        "Your previous response failed schema validation.",
        "Validation error:",
        error,
        "Return the corrected JSON object only, same schema, no commentary.",
        "Report to extract:",
        json.dumps(payload, ensure_ascii=False),
    ])

NL = chr(10)
