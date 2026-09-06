"""Frozen v1 extraction contract (PDF section 2) as Pydantic v2 models.

The extractor never emits schedule_activity_id (spec 0.1 / PDF section 2).
Missing information = null. Uncertainty -> needs_review=true.

`flat_schema()` exposes a *flat* version of the contract for Ollama structured
output: Ollama constrained generation wants every field present, so at wire
time `null` is the explicit value for "absent". `Extraction` is the canonical
nested contract; `normalize_nested()` converts the flat wire object into the
nested shape and clamps values to types.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vocab import ACTION_ALIASES

# canonical work types the extractor may emit as activity.action
ACTION_CANON = sorted(ACTION_ALIASES)

# ---------------------------------------------------------------------------
# canonical (nested) contract -- the PDF section 2 schema
# ---------------------------------------------------------------------------

Status = Literal["completed", "in_progress", "started", "suspended",
                 "cancelled", "not_started"]
Assertion = Literal["affirmed", "negated", "uncertain"]
Certainty = Literal["explicit", "inferred", "missing"]

ALLOWED_STATUS = ("completed", "in_progress", "started", "suspended",
                  "cancelled", "not_started")
ALLOWED_ASSERTION = ("affirmed", "negated", "uncertain")
ALLOWED_CERTAINTY = ("explicit", "inferred", "missing")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RelevanceOut(_Strict):
    is_relevant: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str | None = None


class ActivityOut(_Strict):
    description: str | None = None
    action: str | None = None


class ExecutionOut(_Strict):
    status: Status | None = None
    assertion: Assertion = "affirmed"
    progress_percent: int | float | None = Field(default=None, ge=0, le=100)


class TimeOut(_Strict):
    start: str | None = None
    end: str | None = None
    certainty: Certainty = "missing"


class ContextOut(_Strict):
    discipline: str | None = None
    location: str | None = None
    line_number: str | None = None
    equipment: str | None = None


class QuantityOut(_Strict):
    completed: float | None = None
    total: float | None = None
    unit: str | None = None


class IssueOut(_Strict):
    type: str | None = None
    reason: str | None = None


class EvidenceOut(_Strict):
    source_text: str | None = None


class ConfidenceOut(_Strict):
    overall: float = Field(default=0.5, ge=0.0, le=1.0)
    activity: float = Field(default=0.5, ge=0.0, le=1.0)
    status: float = Field(default=0.5, ge=0.0, le=1.0)
    time: float = Field(default=0.5, ge=0.0, le=1.0)


class EventOut(_Strict):
    event_id: str | None = None
    activity: ActivityOut = Field(default_factory=ActivityOut)
    execution: ExecutionOut = Field(default_factory=ExecutionOut)
    time: TimeOut = Field(default_factory=TimeOut)
    context: ContextOut = Field(default_factory=ContextOut)
    quantity: QuantityOut = Field(default_factory=QuantityOut)
    issue: IssueOut = Field(default_factory=IssueOut)
    evidence: EvidenceOut = Field(default_factory=EvidenceOut)
    confidence: ConfidenceOut = Field(default_factory=ConfidenceOut)
    needs_review: bool = False
    warnings: list[str] = Field(default_factory=list)


class Extraction(_Strict):
    relevance: RelevanceOut
    events: list[EventOut] = Field(default_factory=list)

    def to_dict(self) -> dict:
        return self.model_dump(exclude_none=False)


# ---------------------------------------------------------------------------
# flat generation schema for Ollama structured output (every field required,
# null = absent) -- the wire format the LLM must fill.
# ---------------------------------------------------------------------------

def flat_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "relevance": {
                "type": "object",
                "properties": {
                    "is_relevant": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reason": {"type": ["string", "null"]},
                },
                "required": ["is_relevant", "confidence", "reason"],
            },
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "activity": {
                            "type": "object",
                            "properties": {
                                "description": {"type": ["string", "null"]},
                                "action": {"type": ["string", "null"],
                                           "enum": [*ACTION_CANON, None]},
                            },
                            "required": ["description", "action"],
                        },
                        "execution": {
                            "type": "object",
                            "properties": {
                                "status": {"type": ["string", "null"],
                                           "enum": [*ALLOWED_STATUS, None]},
                                "assertion": {"type": "string",
                                              "enum": [*ALLOWED_ASSERTION]},
                                "progress_percent": {"type": ["number", "null"],
                                                     "minimum": 0, "maximum": 100},
                            },
                            "required": ["status", "assertion", "progress_percent"],
                        },
                        "time": {
                            "type": "object",
                            "properties": {
                                "start": {"type": ["string", "null"]},
                                "end": {"type": ["string", "null"]},
                                "certainty": {"type": "string",
                                              "enum": [*ALLOWED_CERTAINTY]},
                            },
                            "required": ["start", "end", "certainty"],
                        },
                        "context": {
                            "type": "object",
                            "properties": {
                                "discipline": {"type": ["string", "null"]},
                                "location": {"type": ["string", "null"]},
                                "line_number": {"type": ["string", "null"]},
                                "equipment": {"type": ["string", "null"]},
                            },
                            "required": ["discipline", "location", "line_number",
                                         "equipment"],
                        },
                        "quantity": {
                            "type": "object",
                            "properties": {
                                "completed": {"type": ["number", "null"]},
                                "total": {"type": ["number", "null"]},
                                "unit": {"type": ["string", "null"]},
                            },
                            "required": ["completed", "total", "unit"],
                        },
                        "issue": {
                            "type": "object",
                            "properties": {
                                "type": {"type": ["string", "null"]},
                                "reason": {"type": ["string", "null"]},
                            },
                            "required": ["type", "reason"],
                        },
                        "evidence": {
                            "type": "object",
                            "properties": {
                                "source_text": {"type": ["string", "null"]},
                            },
                            "required": ["source_text"],
                        },
                        "confidence": {
                            "type": "object",
                            "properties": {
                                "overall": {"type": "number", "minimum": 0.0,
                                            "maximum": 1.0},
                                "activity": {"type": "number", "minimum": 0.0,
                                             "maximum": 1.0},
                                "status": {"type": "number", "minimum": 0.0,
                                           "maximum": 1.0},
                                "time": {"type": "number", "minimum": 0.0,
                                         "maximum": 1.0},
                            },
                            "required": ["overall", "activity", "status", "time"],
                        },
                        "needs_review": {"type": "boolean"},
                        "warnings": {"type": "array",
                                     "items": {"type": "string"}},
                    },
                    "required": ["activity", "execution", "time", "context",
                                 "quantity", "issue", "evidence", "confidence",
                                 "needs_review", "warnings"],
                },
            },
        },
        "required": ["relevance", "events"],
    }


# ---------------------------------------------------------------------------
# flat -> nested conversion + type normalization
# ---------------------------------------------------------------------------

def _num(x) -> float | None:
    """Parse a number, tolerating string slips like "12" or "5 of 12"."""
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    if isinstance(x, str):
        m = re.match(r"\s*(\d+(?:\.\d+)?)", x)
        if m:
            return float(m.group(1))
    return None


def _pct(x) -> int | float | None:
    """Progress percent clamp: 0-100, tolerant of 0-1 fraction slips."""
    fx = _num(x)
    if fx is None:
        return None
    if fx in (0.0, 1.0):
        fx *= 100  # model said 1 meaning 100% (0 stays 0)
    if not (0 <= fx <= 100):
        return None
    return int(fx) if float(fx).is_integer() else fx


def _str(x) -> str | None:
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, str):
        x = x.strip()
        return x or None
    if isinstance(x, (int, float)):
        return str(x)
    return None


def _lstr(x) -> str | None:
    s = _str(x)
    return s.lower() if s else None


def _clamp01(x, default: float = 0.5) -> float:
    fx = _num(x)
    if fx is None:
        return default
    return round(max(0.0, min(1.0, fx)), 3)


def normalize_nested(d: dict) -> dict:
    """Coerce a flat/dirty generation dict into the canonical nested shape.

    Accepts the model's imperfections (numbers-as-strings, missing
    sub-objects, enum slips, relevance at top level) and clamps to the frozen
    contract. Never invents content: unparseable values become null.
    """
    rel = d.get("relevance")
    if not isinstance(rel, dict):
        rel = d if ("is_relevant" in d or "confidence" in d) else {}
    rel = rel if isinstance(rel, dict) else {}
    out_rel = {
        "is_relevant": bool(rel.get("is_relevant", False)),
        "confidence": _clamp01(rel.get("confidence"), default=0.9),
        "reason": _str(rel.get("reason")),
    }

    raw_events = d.get("events")
    if raw_events is None and isinstance(d.get("activity"), dict):
        raw_events = [d]  # tolerance: single event object instead of a list
    out_events: list[dict] = []
    for ev in raw_events or []:
        if not isinstance(ev, dict):
            continue
        _g = lambda key: ev.get(key) if isinstance(ev.get(key), dict) else {}
        act, exx = _g("activity"), _g("execution")
        tim, ctx = _g("time"), _g("context")
        qty, iss = _g("quantity"), _g("issue")
        evd, cnf = _g("evidence"), _g("confidence")

        status = _lstr(exx.get("status"))
        if status not in ALLOWED_STATUS:
            status = None
        assertion = _lstr(exx.get("assertion")) or "affirmed"
        if assertion not in ALLOWED_ASSERTION:
            assertion = "affirmed"
        certainty = _lstr(tim.get("certainty")) or "missing"
        if certainty not in ALLOWED_CERTAINTY:
            certainty = "missing"

        warnings = [w for w in (ev.get("warnings") or [])
                    if isinstance(w, str) and w]
        ev_text = " ".join(str(x) for x in (evd.get("source_text"),
                                            act.get("description")) if x)

        out_events.append({
            "activity": {"description": _str(act.get("description")),
                         "action": _lstr(act.get("action"))},
            "execution": {"status": status,
                          "assertion": assertion,
                          "progress_percent": _pct(exx.get("progress_percent"))},
            "time": {"start": _str(tim.get("start")),
                     "end": _str(tim.get("end")),
                     "certainty": certainty},
            "context": {"discipline": _str(ctx.get("discipline")),
                        "location": _str(ctx.get("location")),
                        "line_number": _str(ctx.get("line_number")),
                        "equipment": _str(ctx.get("equipment"))},
            "quantity": {"completed": _num(qty.get("completed")),
                         "total": _num(qty.get("total")),
                         "unit": _str(qty.get("unit"))},
            "issue": {"type": _str(iss.get("type")),
                      "reason": _str(iss.get("reason"))},
            "evidence": {"source_text": _str(evd.get("source_text"))},
            "confidence": {k: _clamp01(cnf.get(k)) for k in
                           ("overall", "activity", "status", "time")},
            "needs_review": bool(ev.get("needs_review", False)),
            "warnings": warnings,
            "_ev_text": ev_text,
        })

    return {"relevance": out_rel, "events": out_events}


def parse_extraction(raw: dict) -> tuple[Extraction | None, str | None]:
    """Validate a raw generation dict against the frozen contract.

    Returns (Extraction, None) or (None, error_string). Normalizes first so
    ordinary model slips repair silently; anything structurally invalid is
    reported for the retry loop.
    """
    if not isinstance(raw, dict):
        return None, "top level is not a JSON object"
    if raw.get("events") is not None and not isinstance(raw.get("events"), list):
        return None, "events must be an array"
    clean = normalize_nested(raw)
    for ev in clean["events"]:
        ev.pop("_ev_text", None)
    try:
        return Extraction(**clean), None
    except ValidationError as e:
        return None, str(e)
