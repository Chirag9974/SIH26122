"""Step 4: relevance detection + event extraction.

Contract (spec section 5): extract(report) -> document/relevance/events JSON.
This is the *baseline* extractor: deterministic, alias-driven, no LLM. It exists
so the eval harness has a measurable floor before any model is wired in -- an
LLM extractor should implement this same `extract()` signature and be scored by
the same eval.py.

Two rules the spec is strict about, enforced here:
  1. Missing field -> null. Never guessed.
  2. Negated statement -> never a completion.

The extractor never emits schedule_activity_id. That is matcher.py's job.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from vocab import (ACTION_ALIASES, DISCIPLINE_ALIASES, LOCATION_ALIASES,
                   NEGATION_CUES, STATUS_CUES, UNCERTAIN_CUES)

# ---------------------------------------------------------------------------
# alias indexes, longest-surface-first so "fit up" beats "fit"
# ---------------------------------------------------------------------------

def _index(alias_map: dict[str, list[str]]) -> list[tuple[str, str]]:
    pairs = [(s.lower(), canon) for canon, surfaces in alias_map.items() for s in surfaces]
    return sorted(pairs, key=lambda p: -len(p[0]))


ACTION_IDX = _index(ACTION_ALIASES)
DISC_IDX = _index(DISCIPLINE_ALIASES)
LOC_IDX = _index(LOCATION_ALIASES)

# irrelevant-report signals: site life, logistics, weather with no work reported
IRRELEVANT_HINTS = [
    "lunch", "canteen", "toolbox talk", "pickup vehicle", "printer", "toner",
    "visitor pass", "drinking water", "labour colony", "took shelter",
    "safety induction", "medical checkup", "attendance for the day",
    "document review", "road blocked", "menu changed",
]

# Physical work objects. A completion statement about one of these is execution
# progress even when the action verb is elided: "Remaining Line 40 spool was
# completed at 16:00." -> erection.
WORK_OBJECTS = {
    "spool": "erection", "joint": "welding", "foundation": "concreting",
    "vessel": "erection", "steel frame": "erection", "platform": "grating",
    "grating": "grating", "panel": "installation", "transmitter": "installation",
    "pump": "installation", "skid": "grouting", "loop": "loop check",
    "impulse line": "tubing",
}

_PROGRESS_VERB_RE = re.compile(
    r"\b(?:completed|complete|done|finished|started|commenced|in progress|erected|"
    r"welded|installed|poured|cast|pulled|tested|terminated|calibrated)\b")


def _find(text: str, index: list[tuple[str, str]]) -> str | None:
    """First alias hit on a word boundary."""
    for surface, canon in index:
        if re.search(rf"(?<![a-z0-9]){re.escape(surface)}(?![a-z0-9])", text):
            return canon
    return None


def _find_all_actions(text: str) -> list[tuple[int, str, str]]:
    """All action hits as (position, canonical, matched surface), de-overlapped."""
    hits: list[tuple[int, str, str]] = []
    taken: list[tuple[int, int]] = []
    for surface, canon in ACTION_IDX:
        for m in re.finditer(rf"(?<![a-z0-9]){re.escape(surface)}(?![a-z0-9])", text):
            span = (m.start(), m.end())
            if any(span[0] < b and span[1] > a for a, b in taken):
                continue
            taken.append(span)
            hits.append((m.start(), canon, surface))
    if hits:
        return sorted(hits)
    return _infer_action_from_object(text)


def _infer_action_from_object(text: str) -> list[tuple[int, str, str]]:
    """Elided action: "Remaining Line 40 spool was completed at 16:00." -> erection.

    Only fires when a work object AND a progress verb are both present, so
    ordinary site chatter does not become an event.
    """
    if not _PROGRESS_VERB_RE.search(text):
        return []
    for obj in sorted(WORK_OBJECTS, key=len, reverse=True):
        m = re.search(rf"(?<![a-z]){re.escape(obj)}s?(?![a-z])", text)
        if m:
            return [(m.start(), WORK_OBJECTS[obj], obj)]
    return []


# ---------------------------------------------------------------------------
# field parsers
# ---------------------------------------------------------------------------

_LINE_RE = re.compile(r"\bline\s*(?:no\.?\s*)?(\d{1,3})(?:-\d+)?\b", re.I)
_TAG_RE = re.compile(r"\b((?:P|V|PT|FT|LT|TK|C|E|F)-\d{2,3})\b")
_SIZE_RE = re.compile(r"\b(\d{1,2})\s*(?:in|inch|\")\b", re.I)

# time windows: "from 10:00 to 16:00", "10 AM to 4 PM", "1000-1600", "10 to 4"
_T_RANGE_24 = re.compile(r"\b(\d{1,2}):(\d{2})\s*(?:to|-|till|until|–)\s*(\d{1,2}):(\d{2})\b")
_T_RANGE_MIL = re.compile(r"\b(\d{2})(\d{2})\s*(?:to|-|till|until|–)\s*(\d{2})(\d{2})\b")
_T_RANGE_AMPM = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:to|-|till|until|–)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
    re.I)
_T_RANGE_BARE = re.compile(r"\b(\d{1,2})\s*(?:to|-|till|until|–)\s*(\d{1,2})\b")
# Single clock time. "around 5 of 12 spools" is a QUANTITY, not 05:00 -- so a bare
# number needs a clock signal (:MM, am/pm, hrs) unless introduced by at/by.
_T_SINGLE = re.compile(
    r"\b(?:at|by)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?(?!\s*(?:of|out\s+of))\b"
    r"|\b(?:at|by|around|from)\s+(\d{1,2})(?::(\d{2})|\s*(am|pm)|\s*hrs)\b", re.I)

_QTY_RE = re.compile(
    r"\b(?:around|approx\.?|about|nearly)?\s*(\d+(?:\.\d+)?)\s*(?:out\s+)?of\s+(\d+(?:\.\d+)?)"
    r"\s*([a-z0-9]+)?", re.I)


def _h24(h: int, ampm: str | None, *, pm_bias: bool = False) -> int:
    if ampm:
        h = h % 12 + (12 if ampm.lower() == "pm" else 0)
    elif pm_bias and h <= 7:      # "10 to 4" -> 4 means 16:00 on a work shift
        h += 12
    return min(h, 23)


def parse_times(text: str, rdate: date) -> tuple[str | None, str | None, str]:
    """Return (start_iso, end_iso, certainty). Never invents a time."""
    def iso(h: int, m: int = 0, day_off: int = 0) -> str:
        d = rdate + timedelta(days=day_off)
        return f"{d.isoformat()}T{h:02d}:{m:02d}:00"

    m = _T_RANGE_24.search(text)
    if m:
        sh, sm, eh, em = (int(g) for g in m.groups())
        return iso(sh, sm), iso(eh, em, 1 if eh < sh else 0), "explicit"

    m = _T_RANGE_MIL.search(text)
    if m:
        sh, sm, eh, em = (int(g) for g in m.groups())
        if sh < 24 and eh < 24:
            return iso(sh, sm), iso(eh, em, 1 if eh < sh else 0), "explicit"

    m = _T_RANGE_AMPM.search(text)
    if m:
        sh, sm, sap, eh, em, eap = m.groups()
        e = _h24(int(eh), eap)
        s = _h24(int(sh), sap or (None if int(sh) > 12 else ("am" if e > int(sh) else None)))
        if not sap and s > e:
            s = int(sh)
        return iso(s, int(sm or 0)), iso(e, int(em or 0), 1 if e < s else 0), "explicit"

    m = _T_RANGE_BARE.search(text)
    if m:
        sh, eh = int(m.group(1)), int(m.group(2))
        if sh <= 24 and eh <= 24:
            e = _h24(eh, None, pm_bias=eh < sh)
            return iso(sh), iso(e, 0, 1 if e < sh else 0), "explicit"

    m = _T_SINGLE.search(text)
    if m:
        hh, mm, ap = (m.group(1), m.group(2), m.group(3)) if m.group(1) \
            else (m.group(4), m.group(5), m.group(6))
        return iso(_h24(int(hh), ap), int(mm or 0)), None, "explicit"

    return None, None, "missing"


def parse_quantity(text: str) -> tuple[float | None, float | None, str | None]:
    m = _QTY_RE.search(text)
    if not m:
        return None, None, None
    done, total, unit = float(m.group(1)), float(m.group(2)), m.group(3)
    if total <= 0 or done > total:
        return None, None, None
    unit = unit.lower() if unit and unit.lower() not in {"the", "and", "was", "were"} else None
    if unit and unit.endswith("s") and len(unit) > 3:
        unit = unit[:-1]
    return done, total, unit


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!;])\s+", text.strip())
    return [p for p in (s.strip() for s in parts) if p]


# "no welding was done", "could not be carried out", "did not start", "nil progress"
_NEGATED_RE = re.compile(
    r"(?:\b(?:could|did|does|was|were|has|have|had|is|are)\s+n[o']?t\b"
    r"|\bcannot\b|\bcouldn't\b|\bdidn't\b|\bwasn't\b"
    r"|\bno\s+(?:\w+\s+){0,3}?(?:was|were|has|have)?\s*(?:done|carried out|taken up|"
    r"completed|executed|progress)\b"
    r"|\bnot\s+(?:yet\s+)?(?:done|completed|started|carried out|taken up|executed)\b"
    r"|\bnil\s+progress\b)")

_CANCEL_RE = re.compile(r"\b(?:cancelled|canceled|called off|dropped)\b")


def _status(seg: str, has_qty_partial: bool, *, negated: bool = False) -> str:
    """Cue-based status.

    Negation wins over completion cues: "No welding was done" must not become
    completed just because the word "done" appears (spec section 1).
    """
    if negated:
        return "cancelled" if _CANCEL_RE.search(seg) else "not_started"
    for status in ("not_started", "cancelled", "suspended", "completed",
                   "in_progress", "started"):
        for cue in STATUS_CUES[status]:
            if re.search(rf"(?<![a-z]){re.escape(cue)}(?![a-z])", seg):
                if status == "completed" and has_qty_partial:
                    return "in_progress"
                return status
    return "in_progress" if has_qty_partial else "started"


def _is_negated(seg: str) -> bool:
    return bool(_NEGATED_RE.search(seg) or _CANCEL_RE.search(seg))


def _assertion(seg: str, negated: bool) -> str:
    """Suspension ("stopped midway") is an affirmed stoppage, not a negation."""
    if any(c in seg for c in UNCERTAIN_CUES):
        return "uncertain"
    return "negated" if negated else "affirmed"


def _location(seg: str) -> str | None:
    return _find(seg, LOC_IDX)


def _line_number(seg: str) -> str | None:
    m = _LINE_RE.search(seg)
    return m.group(1) if m else None


def _equipment(seg: str) -> str | None:
    m = _TAG_RE.search(seg)
    return m.group(1) if m else None


def _description(seg: str, action: str) -> str:
    """Normalized description of the physical work, drawn from the segment."""
    size = _SIZE_RE.search(seg)
    bits = []
    if size:
        bits.append(f"{size.group(1)}in")
    obj = re.search(r"\b(spool|joint|foundation|cable|panel|transmitter|instrument|loop|"
                    r"pump|vessel|skid|steel frame|platform|grating|line|tubing|grid)s?\b", seg)
    if obj:
        bits.append(obj.group(1))
    bits.append(action)
    return " ".join(bits)


# ---------------------------------------------------------------------------
# confidence: transparent penalty model, not a learned score
# ---------------------------------------------------------------------------

def _confidence(ev: dict, seg: str) -> dict:
    act = 0.95
    if not ev["context"]["line_number"] and not ev["context"]["equipment"]:
        act -= 0.20
    if not ev["context"]["location"]:
        act -= 0.10
    if "ambiguous_activity" in ev["warnings"]:
        act -= 0.15

    st = 0.95 if ev["execution"]["assertion"] == "affirmed" else 0.80
    if ev["execution"]["assertion"] == "uncertain":
        st = 0.55

    tm = {"explicit": 0.95, "inferred": 0.70, "missing": 0.30}[ev["time"]["time_certainty"]]

    clip = lambda x: round(max(0.05, min(0.99, x)), 2)
    act, st, tm = clip(act), clip(st), clip(tm)
    overall = clip(0.5 * act + 0.3 * st + 0.2 * tm)
    return {"overall": overall, "activity": act, "status": st, "time": tm}


# ---------------------------------------------------------------------------
# relevance
# ---------------------------------------------------------------------------

def detect_relevance(text: str) -> dict:
    low = text.lower()
    actions = _find_all_actions(low)
    hint = next((h for h in IRRELEVANT_HINTS if h in low), None)

    if not actions:
        return {"is_relevant": False, "confidence": 0.93 if hint else 0.75,
                "reason": f"no execution action detected ({hint})" if hint
                          else "no execution action detected"}
    if hint and len(actions) == 1:
        # e.g. "printer ... replacement requested" -- action word without work context
        if not re.search(r"\b(?:completed|done|started|in progress|erected|welded|"
                         r"installed|poured|pulled|tested)\b", low):
            return {"is_relevant": False, "confidence": 0.70,
                    "reason": f"site-logistics content ({hint}), no work progress"}
    return {"is_relevant": True, "confidence": 0.95,
            "reason": "contains project execution information"}


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------

def extract(report: dict) -> dict:
    """report: {report_id, source_type, report_date, discipline?, raw_text}"""
    text = report["raw_text"]
    low = text.lower()
    rdate = datetime.fromisoformat(report["report_date"]).date()
    relevance = detect_relevance(text)

    doc = {
        "report_id": report["report_id"],
        "source_type": report.get("source_type", "daily_report"),
        "report_date": report["report_date"],
        "discipline": report.get("discipline") or _find(low, DISC_IDX),
        "raw_text": text,
    }
    if not relevance["is_relevant"]:
        return {"document": doc, "relevance": relevance, "events": []}

    doc_disc = report.get("discipline") or _find(low, DISC_IDX)
    events: list[dict] = []

    # one event per (sentence, distinct action). Cross-sentence carry: a sentence
    # with a status cue but no action inherits the previous sentence's action.
    prev_action: str | None = None
    prev_ctx: dict | None = None

    for seg in _sentences(text):
        seg_low = seg.lower()
        hits = _find_all_actions(seg_low)
        qty = parse_quantity(seg_low)
        has_partial = qty[0] is not None and qty[1] is not None and qty[0] < qty[1]

        # A continuation sentence updates the previous event instead of opening a
        # new one: "Work was completed by 4 PM.", "19 of 22 completed today.",
        # "Later found incomplete." Rule: no new action, or the same action with
        # no new location/identifier -- or the same identifier as the last event
        # (a correction/contradiction about the very same work).
        actions_here = {a for _, a, _ in hits}
        same_ident = bool(prev_ctx) and (
            (_line_number(seg_low) or _equipment(seg)) is not None
            and (_line_number(seg_low) == prev_ctx.get("line_number")
                 or _equipment(seg) == prev_ctx.get("equipment"))
        )
        is_continuation = bool(events) and prev_action is not None and (
            not hits
            or (actions_here == {prev_action}
                and (same_ident
                     or (not _location(seg_low) and not _line_number(seg_low)
                         and not _equipment(seg))))
        ) and (has_partial or _is_negated(seg_low) or re.search(
            r"\b(?:completed|complete|done|finished|stopped|suspended|incomplete|"
            r"still open|still pending)\b", seg_low))

        if is_continuation:
            tgt = events[-1]
            s, e, cert = parse_times(seg_low, rdate)
            negated = _is_negated(seg_low)
            tgt["execution"]["status"] = _status(seg_low, has_partial, negated=negated)
            tgt["execution"]["assertion"] = _assertion(seg_low, negated)

            # a later sentence contradicting an earlier one is a conflict, and
            # the conflicting claim must not stay recorded as done
            if tgt["execution"]["status"] != "completed":
                tgt["execution"]["progress_percent"] = None
            tgt["warnings"] = sorted(set(
                tgt["warnings"]
                + (["conflicting_report"] if tgt["execution"]["status"]
                   in {"in_progress", "not_started", "cancelled", "suspended"} else [])
                + (["negated_statement"] if negated else [])))

            if has_partial:
                tgt["quantity"] = {"completed": qty[0], "total": qty[1], "unit": qty[2]}
                tgt["execution"]["progress_percent"] = round(qty[0] * 100 / qty[1])
                tgt["warnings"] = sorted(set(tgt["warnings"] + ["quantity_partial"]))
                if "conflicting_report" in tgt["warnings"]:
                    tgt["warnings"].remove("conflicting_report")
            elif tgt["execution"]["status"] == "completed" \
                    and tgt["quantity"]["completed"] is None:
                tgt["execution"]["progress_percent"] = 100

            if s and not tgt["time"]["start"]:
                tgt["time"]["start"] = s
            if s and tgt["time"]["start"] and not tgt["time"]["end"]:
                tgt["time"]["end"] = s
            if e:
                tgt["time"]["end"] = e
            if cert == "explicit":
                tgt["time"]["time_certainty"] = "explicit"
                if "missing_time" in tgt["warnings"]:
                    tgt["warnings"].remove("missing_time")
            tgt["evidence"]["source_text"] += " " + seg
            tgt["confidence"] = _confidence(tgt, seg_low)
            continue

        if not hits:
            continue

        seen: set[str] = set()
        for _, action, _surface in hits:
            if action in seen:
                continue
            seen.add(action)

            start, end, cert = parse_times(seg_low, rdate)
            negated = _is_negated(seg_low)
            status = _status(seg_low, has_partial, negated=negated)
            assertion = _assertion(seg_low, negated)
            loc = _location(seg_low) or (prev_ctx or {}).get("location")
            line = _line_number(seg_low)
            equip = _equipment(seg)

            progress: int | None = None
            if qty[0] is not None:
                progress = round(qty[0] * 100 / qty[1])
            elif status == "completed" and assertion == "affirmed":
                progress = 100

            warnings: list[str] = []
            if cert == "missing":
                warnings.append("missing_time")
            if not loc:
                warnings.append("missing_location")
            if not line and not equip:
                warnings.append("missing_line_number")
                warnings.append("ambiguous_activity")
            if assertion == "negated":
                warnings.append("negated_statement")
            if assertion == "uncertain":
                warnings.append("uncertain_statement")
            if has_partial:
                warnings.append("quantity_partial")

            ev = {
                "event_id": f"{report['report_id']}-EVT-{len(events) + 1:02d}",
                "activity": {"description": _description(seg_low, action), "action": action},
                "execution": {"status": status, "assertion": assertion,
                              "progress_percent": progress},
                "time": {"start": start, "end": end, "time_certainty": cert},
                "context": {
                    "discipline": _find(seg_low, DISC_IDX) or doc_disc,
                    "location": loc,
                    "equipment": equip,
                    "line_number": line,
                },
                "quantity": {"completed": qty[0], "total": qty[1], "unit": qty[2]},
                "evidence": {"source_text": seg},
                "warnings": sorted(set(warnings)),
            }
            ev["confidence"] = _confidence(ev, seg_low)
            events.append(ev)
            prev_action, prev_ctx = action, ev["context"]

    # duplicate events inside one report (same action+location) -> flag, don't drop
    seen_keys: dict[tuple, str] = {}
    for ev in events:
        key = (ev["activity"]["action"], ev["context"]["location"])
        if key in seen_keys:
            ev["warnings"] = sorted(set(ev["warnings"] + ["possible_duplicate"]))
        else:
            seen_keys[key] = ev["event_id"]

    return {"document": doc, "relevance": relevance, "events": events}


if __name__ == "__main__":
    import json
    demo = {"report_id": "REP-DEMO", "source_type": "daily_report",
            "report_date": "2026-09-01", "discipline": "Piping",
            "raw_text": "Piping crew completed erection of 24 inch spool Line 24 "
                        "at Rack B from 10 AM to 4 PM."}
    print(json.dumps(extract(demo), indent=2))
