"""Deterministic consistency checks (PDF section 10).

Runs after the LLM structured output and before final Pydantic validation.
Repairs what can be repaired deterministically, forces needs_review when
something is unsafe, and never invents content: unresolvable fields become
null. Every emitted warning stays inside the closed vocab.WARNINGS set.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from schema import normalize_nested
from vocab import ACTION_ALIASES, LOCATION_ALIASES


def _parse_iso(s):
    """True when s is a parseable ISO timestamp."""
    if not s or not isinstance(s, str):
        return False
    try:
        datetime.fromisoformat(s)
        return True
    except ValueError:
        return False


# alias -> canonical maps for deterministic canonicalization
_ALIAS_TO_ACTION: dict[str, str] = {}
for _canon, _surfaces in ACTION_ALIASES.items():
    _ALIAS_TO_ACTION[_canon.lower()] = _canon
    for _s in _surfaces:
        _ALIAS_TO_ACTION[_s.lower()] = _canon

_LOC_TO_CANON: dict[str, str] = {}
for _canon, _surfaces in LOCATION_ALIASES.items():
    _LOC_TO_CANON[_canon.lower()] = _canon
    for _s in _surfaces:
        _LOC_TO_CANON[_s.lower()] = _canon

# words that describe execution status, never a work type
_STATUS_WORDS = {
    "done", "completed", "complete", "compl", "cmpltd", "completd", "finished",
    "started", "under way", "in progress", "ongoing", "over", "achieved",
    "closed out", "all done", "wrapped up", "carried out", "executed",
    "performed", "held", "conducted", "delayed", "halted", "suspended",
}


def _canon_action(value):
    """Map a model action string to the canonical work-type vocabulary."""
    if not value or not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in _STATUS_WORDS:
        return None
    if v in _ALIAS_TO_ACTION:
        return _ALIAS_TO_ACTION[v]
    best = None
    for alias, canon in _ALIAS_TO_ACTION.items():
        if len(alias) >= 4 and alias in v:
            if best is None or len(alias) > len(best[0]):
                best = (alias, canon)
    return best[1] if best else None


def _canon_location(value):
    """Map a location shorthand to the canonical location name."""
    if not value or not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in _LOC_TO_CANON:
        return _LOC_TO_CANON[v]
    best = None
    for alias, canon in _LOC_TO_CANON.items():
        if len(alias) >= 4 and alias in v:
            if best is None or len(alias) > len(best[0]):
                best = (alias, canon)
    return best[1] if best else None


def _repair_ts(s, rdate: str | None):
    """Resolve a stated clock time to a full ISO timestamp.

    Bare "08:00" or "08:00:00" -> report_date + T08:00:00 (the report describes
    that day; resolution, not invention). "YYYY-MM-DD HH:MM(:SS)" gets its
    separator normalized. Returns (iso_or_original, repaired_bool).
    """
    if not s or not isinstance(s, str):
        return s, False
    s2 = s.strip()
    if " " in s2 and len(s2) >= 11 and s2[:4].isdigit():
        s2 = s2.replace(" ", "T", 1)
    if "T" in s2:
        date_part, _, time_part = s2.partition("T")
        bits = time_part.split(":")
        if len(date_part) == 10 and len(bits) in (2, 3)                 and all(b.isdigit() for b in bits):
            hh = min(int(bits[0]), 23)
            mm = int(bits[1])
            ss = int(bits[2]) if len(bits) == 3 else 0
            return f"{date_part}T{hh:02d}:{mm:02d}:{ss:02d}", True
        return s, False
    bits = s2.split(":")
    if rdate and len(bits) in (2, 3) and all(b.isdigit() for b in bits):
        hh = min(int(bits[0]), 23)
        mm = int(bits[1])
        ss = int(bits[2]) if len(bits) == 3 else 0
        return f"{rdate}T{hh:02d}:{mm:02d}:{ss:02d}", True
    return s, False


def _recover_location(raw: str):
    """Sole canonical location present in the raw text, else None.

    Evidence-backed recovery when the model omitted location: if exactly one
    known location (or alias) occurs in the report, it is clearly implied.
    """
    low = raw.lower()
    found = set()
    for alias, canon in _LOC_TO_CANON.items():
        idx = low.find(alias)
        while idx >= 0:
            before = low[idx - 1] if idx else " "
            end = idx + len(alias)
            after = low[end] if end < len(low) else " "
            if not before.isalnum() and not after.isalnum():
                found.add(canon)
                break
            idx = low.find(alias, idx + 1)
    return found.pop() if len(found) == 1 else None


def _locate(raw: str, seg: str):
    """Start index of seg in raw (case-insensitive, word-bounded preferred)."""
    low, segl = raw.lower(), seg.lower().strip()
    if not segl or len(segl) > len(low):
        return None
    cands = [i for i in range(len(low) - len(segl) + 1)
             if low.startswith(segl, i)]
    if not cands:
        return None
    for i in cands:
        before = low[i - 1] if i else " "
        after = low[i + len(segl)] if i + len(segl) < len(low) else " "
        if not before.isalnum() and not after.isalnum():
            return i
    return cands[0]


def _evidence_span(raw: str, ev: dict):
    """Resolve the event's evidence to an exact span of the raw text.

    Tries evidence.source_text, then the activity description. Returns
    (span, None) or (None, "unsupported_evidence").
    """
    evd = ev.get("evidence") or {}
    act = ev.get("activity") or {}
    for seg in (evd.get("source_text"), act.get("description")):
        if not seg or not isinstance(seg, str):
            continue
        i = _locate(raw, seg)
        if i is not None:
            return raw[i:i + len(seg.strip())].strip(), None
    return None, "unsupported_evidence"


def validate_extraction(pred: dict, report: dict) -> tuple[dict, list[str]]:
    """Apply PDF section 10 checks. Returns (normalized prediction, issues).

    issues are failure tags for error_analysis.py; they describe what the
    validator had to repair or flag, not what it invented.
    """
    raw = report["raw_text"]
    issues: list[str] = []
    n = normalize_nested(pred)
    rel, events = n["relevance"], n["events"]

    # Rule: relevance=false means zero events. If the model emitted events
    # anyway, the events are the stronger signal: flip relevance to true.
    if not rel["is_relevant"] and events:
        rel["is_relevant"] = True
        rel["reason"] = rel.get("reason") or "events present in text"
        issues.append("relevance_false_with_events")

    for ev in events:
        flags = {w for w in ev.get("warnings", []) if isinstance(w, str)}
        nr = bool(ev.get("needs_review", False))
        exx, qty, tim = ev["execution"], ev["quantity"], ev["time"]

        # Evidence traceability: the claim must exist in the raw text.
        span, miss = _evidence_span(raw, ev)
        if span is None:
            flags.add("unsupported_evidence")
            nr = True
            issues.append("unsupported_evidence")
        else:
            ev["evidence"]["source_text"] = span

        # Progress in 0-100 only (normalizer clamps; guard the residual).
        pct = exx["progress_percent"]
        if pct is not None and not (0 <= pct <= 100):
            exx["progress_percent"] = None
            nr = True
            issues.append("progress_out_of_range")

        # Quantity: non-negative; completed <= total when comparable.
        c, t = qty.get("completed"), qty.get("total")
        if c is not None and t is not None and c > t:
            ev["quantity"] = {"completed": None, "total": None, "unit": None}
            exx["progress_percent"] = None
            nr = True
            flags.add("quantity_partial")
            issues.append("quantity_completed_gt_total")

        # Resolve stated clock times against the report date first.
        for tk in ("start", "end"):
            fixed, did = _repair_ts(tim[tk], report.get("report_date"))
            if did:
                tim[tk] = fixed
                flags.add("relative_date_resolved")
                issues.append("time_resolved_from_report_date")

        # Times must be parseable when present.
        bad_ts = any(x is not None and not _parse_iso(x)
                     for x in (tim["start"], tim["end"]))
        if bad_ts:
            if tim["start"] is not None and not _parse_iso(tim["start"]):
                tim["start"] = None
            if tim["end"] is not None and not _parse_iso(tim["end"]):
                tim["end"] = None
            tim["certainty"] = "missing"
            nr = True
            issues.append("invalid_timestamp")

        # Overnight rollover: end may roll to the next day when evidence
        # supports it (end < start). More than one day of gap is unsafe.
        if tim["start"] and tim["end"]:
            s = datetime.fromisoformat(tim["start"])
            e = datetime.fromisoformat(tim["end"])
            if e < s:
                e2 = e + timedelta(days=1)
                if (e2 - s) <= timedelta(days=1):
                    tim["end"] = e2.isoformat()
                    flags.add("relative_date_resolved")
                    issues.append("overnight_rollover")
                else:
                    tim["end"] = None
                    nr = True
                    issues.append("overnight_gap_too_large")

        # Negation can never stand next to a completion-like status.
        if exx["assertion"] == "negated" and exx["status"] in (
                "completed", "started", "in_progress"):
            exx["status"] = None
            exx["progress_percent"] = None
            nr = True
            flags.add("negated_statement")
            issues.append("negation_as_completion")

        # Uncertainty stays uncertain and always needs review.
        if "uncertain_statement" in flags:
            exx["assertion"] = "uncertain"
        if exx["assertion"] == "uncertain":
            nr = True
        if "conflicting_report" in flags:
            nr = True

        # A relevant event with no resolvable status is unsafe to auto-accept.
        if exx["status"] is None:
            nr = True
            issues.append("missing_status")

        # Missing-time warning consistency.
        if tim["start"] is None and tim["end"] is None:
            flags.add("missing_time")

        # Canonicalize activity.action: the model may put a status word or
        # paraphrase there. Alias-map it; recover from the description when it
        # collapses to a status word; null it when unrecognized.
        act_raw = ev["activity"].get("action")
        act_canon = _canon_action(act_raw)
        if act_canon is None and act_raw:
            act_canon = _canon_action(ev["activity"].get("description"))
            if act_canon is not None:
                issues.append("action_recovered_from_description")
        if act_canon is not None and act_canon != act_raw:
            ev["activity"]["action"] = act_canon
            issues.append("action_canonicalized")
        elif act_canon is None and act_raw:
            ev["activity"]["action"] = None
            issues.append("action_unrecognized")
            nr = True

        # Canonicalize location shorthand ("u-300" -> "Unit 300").
        loc_raw = ev["context"].get("location")
        loc_canon = _canon_location(loc_raw)
        if loc_canon is not None and loc_canon != loc_raw:
            ev["context"]["location"] = loc_canon
            issues.append("location_canonicalized")

        # Location recovery: sole known location in the text fills an empty
        # context.location (clearly implied, never invented).
        if ev["context"].get("location") is None:
            recovered = _recover_location(raw)
            if recovered is not None:
                ev["context"]["location"] = recovered
                issues.append("location_recovered_from_text")

        # Progress derivation per gold convention:
        #   completed + affirmed + no quantity -> 100
        #   quantity present -> round(completed * 100 / total)
        if (exx["status"] == "completed" and exx["assertion"] == "affirmed"
                and "conflicting_report" not in flags
                and exx["progress_percent"] is None
                and ev["quantity"]["completed"] is None):
            exx["progress_percent"] = 100
        c, t = ev["quantity"]["completed"], ev["quantity"]["total"]
        if c is not None and t is not None and t > 0:
            exx["progress_percent"] = round(c * 100 / t)

        ev["warnings"] = sorted(flags)
        ev["needs_review"] = nr
        ev.pop("_ev_text", None)

    return {"relevance": rel, "events": events}, issues
