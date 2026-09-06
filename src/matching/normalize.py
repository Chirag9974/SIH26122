"""Normalize extracted events and schedule activities into comparable forms.

Pure deterministic layer: canonical actions/locations/disciplines come from
the shared vocabulary in common.vocab (same alias tables the extractor uses),
so both sides of the match speak the same language before any scoring.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from common.vocab import ACTION_ALIASES, DISCIPLINE_ALIASES, LOCATION_ALIASES

_WS = re.compile(r"\s+")
_NONALNUM = re.compile(r"[^a-z0-9\s]")
_TRAILING_WP = re.compile(r"\s*-\s*work\s+package\s*$", re.IGNORECASE)

# surface -> canonical lookup tables (multi-word surfaces checked first)
def _build_lookup(aliases: dict[str, list[str]]) -> dict[str, str]:
    return {
        norm_surface(surface): canon
        for canon, surfaces in aliases.items()
        for surface in surfaces
    }


def norm_text(s: str | None) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not s:
        return ""
    return _WS.sub(" ", _NONALNUM.sub(" ", s.lower())).strip()


def norm_surface(s: str) -> str:
    return norm_text(s)


_ACTION_LOOKUP = _build_lookup(ACTION_ALIASES)
_LOCATION_LOOKUP = _build_lookup(LOCATION_ALIASES)
_DISCIPLINE_LOOKUP = _build_lookup(DISCIPLINE_ALIASES)


def canon_action(text: str | None) -> str | None:
    """Map a free-text action/description to a canonical action, if any.

    Multi-word surfaces are tried first so 'fitting up' wins over 'fit'.
    """
    if not text:
        return None
    t = norm_text(text)
    if not t:
        return None
    for surface in sorted(_ACTION_LOOKUP, key=len, reverse=True):
        if surface in t:
            return _ACTION_LOOKUP[surface]
    return None


def canon_location(text: str | None) -> str | None:
    """Canonicalize a location string ('rb-c' -> 'Rack C')."""
    if not text:
        return None
    t = norm_text(text)
    if not t:
        return None
    if t in _LOCATION_LOOKUP:
        return _LOCATION_LOOKUP[t]
    for surface in sorted(_LOCATION_LOOKUP, key=len, reverse=True):
        if surface in t:
            return _LOCATION_LOOKUP[surface]
    return None


def canon_discipline(text: str | None) -> str | None:
    if not text:
        return None
    t = norm_text(text)
    if not t:
        return None
    if t in _DISCIPLINE_LOOKUP:
        return _DISCIPLINE_LOOKUP[t]
    for surface in sorted(_DISCIPLINE_LOOKUP, key=len, reverse=True):
        if surface in t:
            return _DISCIPLINE_LOOKUP[surface]
    return None


def canon_line(s: str | None) -> str | None:
    """Line numbers are compared as bare digits ('Line 40' / '40-02' -> 40/40-02)."""
    if not s:
        return None
    m = re.search(r"\d+(?:[-/]\d+)*", str(s))
    return m.group(0) if m else None


def canon_equipment(s: str | None) -> str | None:
    """Equipment tags compared uppercase without separators (P-204 == P204)."""
    if not s:
        return None
    t = re.sub(r"[^A-Za-z0-9]", "", str(s)).upper()
    return t or None


@dataclass
class EventNorm:
    """Canonical view of an extracted event for matching."""

    event_id: str | None = None
    action: str | None = None
    description: str | None = None
    discipline: str | None = None
    location: str | None = None
    line_number: str | None = None
    equipment: str | None = None
    status: str | None = None
    needs_review: bool = False


def normalize_event(event: dict) -> EventNorm:
    """Accept any event-shaped dict (extractor output or gold); never raises."""
    event = event or {}
    act = event.get("activity") or {}
    ctx = event.get("context") or {}
    exe = event.get("execution") or {}
    desc = act.get("description") or None
    action = act.get("action") or None
    if not action:
        action = canon_action(desc)
    location = ctx.get("location") or None
    location = canon_location(location) or (location.strip() if location else None)
    return EventNorm(
        event_id=event.get("event_id"),
        action=action,
        description=desc,
        discipline=canon_discipline(ctx.get("discipline")) or ctx.get("discipline"),
        location=location,
        line_number=canon_line(ctx.get("line_number")) or None,
        equipment=canon_equipment(ctx.get("equipment")) or None,
        status=exe.get("status") or None,
        needs_review=bool(event.get("needs_review")),
    )


_NAME_LINE = re.compile(r"line\s*(\d+(?:[-/]\d+)*)", re.IGNORECASE)
_NAME_EQUIP = re.compile(r"\b([A-Z]{1,4}-\d{1,5}[A-Z]?)\b")


def line_from_text(text: str | None) -> str | None:
    """Extract the 'Line NN' token from free text (activity names embed the
    line number that the CSV column often omits)."""
    if not text:
        return None
    m = _NAME_LINE.search(text)
    return m.group(1) if m else None


def equipment_from_text(text: str | None) -> str | None:
    """Extract an instrument/equipment tag (LT-140, P-204) from free text."""
    if not text:
        return None
    m = _NAME_EQUIP.search(text)
    return canon_equipment(m.group(1)) if m else None


@dataclass
class ActivityNorm:
    """Canonical view of one schedule activity row."""

    activity_id: str
    name: str
    name_core: str  # name without the ' - Work Package' suffix
    wbs_level: str  # 'L5' | 'L6'
    discipline: str | None
    location: str | None
    line_number: str | None
    equipment: str | None
    name_line: str | None = None       # 'Line 40-09' inside the name
    name_equipment: str | None = None  # tag inside the name (LT-140)
    planned_start: str | None = None
    planned_finish: str | None = None
    tokens: set[str] = field(default_factory=set)

    @property
    def is_l6(self) -> bool:
        return self.wbs_level.upper() == "L6"

    def lines_match(self, other: str | None) -> bool:
        """Prefix-compatible line comparison: '40' ~ '40-09', '40-08' ~ '40'."""
        if not other:
            return False
        for own in (self.line_number, self.name_line):
            if not own:
                continue
            if own == other or own.startswith(other + "-") or other.startswith(own + "-"):
                return True
        return False

    def equipment_match(self, other: str | None) -> bool:
        if not other:
            return False
        for own in (self.equipment, self.name_equipment):
            if own and own == other:
                return True
        return False


def discipline_from_name(name: str) -> str | None:
    """Fall back to the leading word of a schedule name for discipline."""
    first = norm_text(name).split(" ")[0] if name else ""
    return canon_discipline(first)


def parse_activity(row: dict) -> ActivityNorm:
    name = (row.get("activity_name") or "").strip()
    name_core = _TRAILING_WP.sub("", name).strip()
    disc = canon_discipline(row.get("discipline")) or discipline_from_name(name)
    loc = canon_location(row.get("location")) or (
        (row.get("location") or "").strip() or None
    )
    return ActivityNorm(
        activity_id=row["activity_id"],
        name=name,
        name_core=name_core,
        wbs_level=(row.get("wbs_level") or "L6").strip().upper(),
        discipline=disc,
        location=loc,
        line_number=canon_line(row.get("line_number")) or None,
        equipment=canon_equipment(row.get("equipment_tag")) or None,
        name_line=line_from_text(name_core),
        name_equipment=equipment_from_text(name),
        planned_start=row.get("planned_start") or None,
        planned_finish=row.get("planned_finish") or None,
    )
