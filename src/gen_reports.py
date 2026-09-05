"""Steps 2+3: generate raw field reports AND their gold extraction labels together.

Why paired generation: the gold label is built from the facts we *chose to render*,
not re-parsed from the text afterwards. If a style omits the location, the label
carries location=null plus a missing_location warning -- so "null is correct" is
enforced by construction, per spec section 1.

Outputs:
  data/raw_reports/reports.jsonl        raw text only (extractor input)
  data/labels/gold_extractions.jsonl    ground-truth structured events
  data/labels/gold_matches.jsonl        event_id -> schedule activity_id (or null)

Gold labels carry no confidence scores: confidence is a model output, not truth.
"""
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from vocab import DISCIPLINES, LOCATION_ALIASES

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ROOT / "data" / "schedule" / "schedule_activities.csv"
OUT_REPORTS = ROOT / "data" / "raw_reports" / "reports.jsonl"
OUT_GOLD = ROOT / "data" / "labels" / "gold_extractions.jsonl"
OUT_MATCH = ROOT / "data" / "labels" / "gold_matches.jsonl"

SEED = 26122
N_REPORTS = 100
MIX = {  # spec section 8
    "normal": 40,
    "noisy": 20,
    "multi": 10,
    "partial": 10,
    "ambiguous": 10,
    "irrelevant_negative": 10,
}

# --------------------------------------------------------------------------
# work-item lookup: recover the vocab work entry behind a schedule activity
# --------------------------------------------------------------------------

def _work_for(disc: str, activity_name: str) -> dict:
    verb = activity_name.split()[0]
    for w in DISCIPLINES[disc]["works"]:
        if w["verb"].split()[0].lower() == verb.lower():
            return w
    return DISCIPLINES[disc]["works"][0]


def _field_noun(work: dict, act: dict) -> str:
    size = next((s for s in ("6in", "8in", "12in", "18in", "24in")
                 if s in act["activity_name"]), "")
    return work["field_noun"].format(size=size).strip()


def _work_phrase(work: dict, act: dict) -> str:
    """Field wording that always carries the action verb exactly once.

    Some field_nouns already embed the action ("cable pulling"); appending the
    action again produced "cable pulling of cable pulling".
    """
    noun = _field_noun(work, act)
    head = work["action"].split()[0]
    return noun if head in noun.lower() else f"{work['action']} of {noun}"


def _loc_alias(rng: random.Random, loc: str) -> str:
    return rng.choice(LOCATION_ALIASES.get(loc, [loc.lower()]))


# --------------------------------------------------------------------------
# event plan -> the single source of truth for both text and gold
# --------------------------------------------------------------------------

@dataclass
class EventPlan:
    activity: dict          # {"activity_id": .., "activity_name": .., ...} schedule row
    description: str
    action: str
    status: str
    assertion: str = "affirmed"
    progress_percent: int | None = None
    start: str | None = None
    end: str | None = None
    time_certainty: str = "missing"
    discipline: str | None = None
    location: str | None = None
    equipment: str | None = None
    line_number: str | None = None
    qty_completed: float | None = None
    qty_total: float | None = None
    qty_unit: str | None = None
    evidence: str = ""
    warnings: list[str] = field(default_factory=list)
    match_id: str | None = None       # gold schedule activity, None => needs_review
    match_reason: str = "exact_synthetic_source"

    def to_gold(self, event_id: str) -> dict:
        return {
            "event_id": event_id,
            "activity": {"description": self.description, "action": self.action},
            "execution": {
                "status": self.status,
                "assertion": self.assertion,
                "progress_percent": self.progress_percent,
            },
            "time": {"start": self.start, "end": self.end,
                     "time_certainty": self.time_certainty},
            "context": {
                "discipline": self.discipline,
                "location": self.location,
                "equipment": self.equipment,
                "line_number": self.line_number,
            },
            "quantity": {"completed": self.qty_completed, "total": self.qty_total,
                         "unit": self.qty_unit},
            "evidence": {"source_text": self.evidence},
            "warnings": sorted(set(self.warnings)),
        }


def _iso(d: date, hhmm: str) -> str:
    return f"{d.isoformat()}T{hhmm}:00"


def _time_pair(rng: random.Random) -> tuple[str, str, str, str]:
    """Return (start_hhmm, end_hhmm, phrase, style) for a work window."""
    sh = rng.choice([7, 8, 9, 10])
    eh = sh + rng.randint(4, 8)
    style = rng.choice(["24h", "ampm", "short", "mil"])
    if style == "24h":
        phrase = f"from {sh:02d}:00 to {eh:02d}:00"
    elif style == "ampm":
        e12 = eh - 12 if eh > 12 else eh
        phrase = f"from {sh} AM to {e12} {'PM' if eh > 12 else 'AM'}"
    elif style == "short":
        e12 = eh - 12 if eh > 12 else eh
        phrase = f"{sh} to {e12}"
    else:
        phrase = f"{sh:02d}00-{eh:02d}00"
    return f"{sh:02d}:00", f"{eh:02d}:00", phrase, style


# --------------------------------------------------------------------------
# report builders: each returns (raw_text, [EventPlan], source_type)
# --------------------------------------------------------------------------

def _base_plan(rng: random.Random, act: dict, rdate: date, work: dict) -> EventPlan:
    return EventPlan(
        activity=act,
        description=f"{_field_noun(work, act)} {work['action']}".strip(),
        action=work["action"],
        status="completed",
        discipline=act["discipline"],
        location=act["location"] or None,
        # identifiers default to null: only a builder that actually renders the
        # line/tag into the text may set them. Gold never leaks schedule fields
        # the report text does not contain.
        equipment=None,
        line_number=None,
        qty_unit=None,
        match_id=act["activity_id"],
    )


def _no_identifier(p: EventPlan) -> None:
    """Text carries no line/tag -> identifier fields stay null and are flagged."""
    p.line_number = None
    p.equipment = None
    p.warnings += ["missing_line_number", "ambiguous_activity"]


def build_normal(rng, act, rdate, work):
    p = _base_plan(rng, act, rdate, work)
    sh, eh, phrase, _ = _time_pair(rng)
    p.start, p.end = _iso(rdate, sh), _iso(rdate, eh)
    p.time_certainty = "explicit"
    p.progress_percent = 100
    noun = _field_noun(work, act)
    loc = act["location"]
    if act["line_number"]:
        ident_txt = f" Line {act['line_number']}"
        p.line_number = act["line_number"]
    elif act["equipment_tag"]:
        ident_txt = f" {act['equipment_tag']}"
        p.equipment = act["equipment_tag"]
    else:
        ident_txt = ""
        _no_identifier(p)
    text = (f"{act['discipline']} crew completed {_work_phrase(work, act)}"
            f"{ident_txt} at {loc} {phrase}.")
    p.evidence = text
    return text, [p], "daily_report"


def build_noisy(rng, act, rdate, work):
    """Abbreviated / typo'd / lowercase field-speak. May drop fields -> null gold."""
    from vocab import ACTION_ALIASES
    p = _base_plan(rng, act, rdate, work)
    noun = _field_noun(work, act).lower()
    alias = rng.choice(ACTION_ALIASES.get(work["action"], [work["action"]]))
    code = DISCIPLINES[act["discipline"]]["code"]
    variant = rng.choice(["abbrev", "informal", "typo"])

    drop_time = rng.random() < 0.3
    drop_loc = rng.random() < 0.25

    if drop_time:
        phrase = ""
        p.time_certainty = "missing"
        p.warnings.append("missing_time")
    else:
        sh, eh, phrase, _ = _time_pair(rng)
        p.start, p.end = _iso(rdate, sh), _iso(rdate, eh)
        p.time_certainty = "explicit"

    if drop_loc:
        loc_word = ""
        p.location = None
        p.warnings.append("missing_location")
    else:
        loc_word = _loc_alias(rng, act["location"])

    _no_identifier(p)   # noisy field-speak drops the identifier
    p.progress_percent = 100
    if variant == "abbrev":
        at = f" @ {loc_word}" if loc_word else ""
        text = f"{code} - {noun} {alias}{at}, {phrase}, complete."
    elif variant == "informal":
        at = f" at {loc_word}" if loc_word else ""
        text = f"{noun} {alias} done{at}, {phrase}."
    else:
        head = f"{loc_word} " if loc_word else ""
        text = f"{head}{noun} {alias} today {phrase} completd."
    text = " ".join(text.replace(" ,", ",").replace(",,", ",").split())
    p.evidence = text
    return text, [p], "daily_report"


def build_multi(rng, acts, rdate):
    """2-4 events in one report, mixed statuses.

    Real multi-discipline DPRs tag each line with the discipline, so the text
    carries it explicitly -- otherwise gold discipline would have to be null.
    """
    lines, plans = [], []
    for act in acts:
        work = _work_for(act["discipline"], act["activity_name"])
        p = _base_plan(rng, act, rdate, work)
        noun = _work_phrase(work, act)
        status = rng.choice(["completed", "completed", "started", "in_progress"])
        p.status = status
        _no_identifier(p)
        sh, eh, phrase, _ = _time_pair(rng)
        tag = act["discipline"]
        if status == "completed":
            p.start, p.end = _iso(rdate, sh), _iso(rdate, eh)
            p.time_certainty = "explicit"
            p.progress_percent = 100
            sent = f"{tag}: {noun} completed at {act['location']} {phrase}"
        elif status == "started":
            p.start = _iso(rdate, sh)
            p.time_certainty = "explicit"
            sent = f"{tag}: {noun} started at {act['location']} at {sh}"
        else:
            p.time_certainty = "missing"
            p.warnings.append("missing_time")
            sent = f"{tag}: {noun} in progress at {act['location']}"
        p.evidence = sent + "."
        lines.append(sent)
        plans.append(p)
    return ". ".join(lines) + ".", plans, "daily_report"


def build_partial(rng, act, rdate, work):
    """Explicit quantity progress -> progress_percent derived from quantities only."""
    p = _base_plan(rng, act, rdate, work)
    _no_identifier(p)
    total = max(int(act["planned_quantity"] or 10), 3)
    done = rng.randint(1, total - 1)
    p.status = "in_progress"
    p.qty_completed, p.qty_total, p.qty_unit = float(done), float(total), act["unit"]
    p.progress_percent = round(done * 100 / total)
    p.warnings.append("quantity_partial")
    sh, eh, phrase, _ = _time_pair(rng)
    p.start, p.end = _iso(rdate, sh), _iso(rdate, eh)
    p.time_certainty = "explicit"
    text = (f"Started {_work_phrase(work, act)} at {act['location']} {phrase}. "
            f"{done} of {total} {act['unit']} completed today.")
    p.evidence = text
    return text, [p], "daily_report"


def build_ambiguous(rng, act, rdate, work, sched):
    """Identifier stripped, and >=2 schedule activities fit -> gold match is null."""
    p = _base_plan(rng, act, rdate, work)
    _no_identifier(p)
    p.progress_percent = 100
    p.match_id = None
    p.match_reason = "multiple_close_candidates"

    drop_time = rng.random() < 0.6
    if drop_time:
        p.time_certainty = "missing"
        p.warnings.append("missing_time")
        tail = ""
    else:
        sh, eh, phrase, _ = _time_pair(rng)
        p.start, p.end = _iso(rdate, sh), _iso(rdate, eh)
        p.time_certainty = "explicit"
        tail = f" {phrase}"
    phrase_txt = _work_phrase(work, act)
    text = f"{phrase_txt[0].upper()}{phrase_txt[1:]} completed at {act['location']}{tail}."
    p.evidence = text
    return text, [p], "daily_report"


IRRELEVANT = [
    "Lunch was delayed today because the canteen was crowded.",
    "Toolbox talk conducted for 25 workers in the morning.",
    "Two new pickup vehicles arrived at the site gate.",
    "Site office printer is out of toner, replacement requested.",
    "Heavy rain from 14:00, all crews took shelter. No further updates.",
    "Visitor pass issued to third party inspection agency representative.",
    "Drinking water tanker refilled at labour colony.",
]


def build_irrelevant(rng):
    return rng.choice(IRRELEVANT), [], "daily_report"


def build_negative(rng, act, rdate, work):
    """Work did NOT happen. Must not become a completed activity."""
    p = _base_plan(rng, act, rdate, work)
    _no_identifier(p)
    p.assertion = "negated"
    p.status = rng.choice(["not_started", "suspended", "cancelled"])
    p.progress_percent = None
    p.time_certainty = "missing"
    p.warnings += ["negated_statement", "missing_time"]
    reason = rng.choice(["material not received", "crane unavailable",
                         "permit not issued", "heavy rain", "manpower shortage"])
    phrase_txt = _work_phrase(work, act)
    head = f"{phrase_txt[0].upper()}{phrase_txt[1:]} at {act['location']}"
    if p.status == "not_started":
        text = f"{head} could not be started today, {reason}."
    elif p.status == "suspended":
        text = f"{head} was stopped midway due to {reason}."
        p.assertion = "affirmed"          # the suspension itself is affirmed
        p.warnings.remove("negated_statement")
    else:
        text = f"{head} has been cancelled, {reason}."
    p.evidence = text
    return text, [p], "daily_report"


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def load_schedule() -> list[dict]:
    with SCHEDULE.open(encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["wbs_level"] == "L6"]


def _ambiguous_pool(l6: list[dict]) -> list[dict]:
    """L6 activities whose (discipline, work-head) has >=2 siblings."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in l6:
        head = r["activity_name"].split(" at ")[0].split(" - Part")[0]
        groups[(r["discipline"], head)].append(r)
    return [r for g in groups.values() if len(g) >= 2 for r in g]


def main() -> None:
    rng = random.Random(SEED)
    l6 = load_schedule()
    amb_pool = _ambiguous_pool(l6) or l6

    plan_kinds = [k for k, n in MIX.items() for _ in range(n)]
    rng.shuffle(plan_kinds)

    reports, golds, matches = [], [], []
    day0 = date(2026, 9, 1)

    for i, kind in enumerate(plan_kinds, start=1):
        rid = f"REP-{i:04d}"
        rdate = day0 + timedelta(days=rng.randint(0, 160))
        if rdate.weekday() == 6:
            rdate += timedelta(days=1)

        if kind == "multi":
            acts = rng.sample(l6, k=rng.randint(2, 4))
            text, plans, stype = build_multi(rng, acts, rdate)
        elif kind == "irrelevant_negative":
            if rng.random() < 0.5:
                text, plans, stype = build_irrelevant(rng)
            else:
                act = rng.choice(l6)
                text, plans, stype = build_negative(
                    rng, act, rdate, _work_for(act["discipline"], act["activity_name"]))
        else:
            act = rng.choice(amb_pool if kind == "ambiguous" else l6)
            work = _work_for(act["discipline"], act["activity_name"])
            builder = {"normal": build_normal, "noisy": build_noisy,
                       "partial": build_partial}.get(kind)
            if builder:
                text, plans, stype = builder(rng, act, rdate, work)
            else:
                text, plans, stype = build_ambiguous(rng, act, rdate, work, l6)

        disc = plans[0].discipline if len(plans) == 1 else None
        reports.append({
            "report_id": rid,
            "source_type": stype,
            "report_date": rdate.isoformat(),
            "discipline": disc,
            "raw_text": text,
        })

        events = []
        for j, p in enumerate(plans, start=1):
            eid = f"{rid}-EVT-{j:02d}"
            events.append(p.to_gold(eid))
            matches.append({
                "report_id": rid, "event_id": eid,
                "schedule_activity_id": p.match_id,
                "decision": "auto_match" if p.match_id else "needs_review",
                "reason": p.match_reason,
            })
        golds.append({
            "report_id": rid,
            "relevance": {
                "is_relevant": bool(plans),
                "reason": "contains project execution information" if plans
                          else "no project execution information",
            },
            "events": events,
            "case_kind": kind,
        })

    for path, rows in ((OUT_REPORTS, reports), (OUT_GOLD, golds), (OUT_MATCH, matches)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ev = sum(len(g["events"]) for g in golds)
    n_irr = sum(1 for g in golds if not g["events"])
    n_rev = sum(1 for m in matches if m["decision"] == "needs_review")
    print(f"{OUT_REPORTS}: {len(reports)} reports")
    print(f"{OUT_GOLD}: {n_ev} gold events ({n_irr} irrelevant reports, 0 events)")
    print(f"{OUT_MATCH}: {len(matches)} matches ({n_rev} needs_review)")


if __name__ == "__main__":
    main()
