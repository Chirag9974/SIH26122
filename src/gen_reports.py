"""Steps 2+3: generate improved field reports AND gold labels.

IMPROVEMENTS:
- 800+ reports (up from 100)
- Realistic varied styles: formal, shorthand, messy, abbreviated, contextual
- No systematic activity ID exposure
- Hard negatives in matching: similar activities differing by one attribute
- Genuine no_match cases
- L5/L6 granularity mismatches
- Candidate pools instead of "exact_synthetic_source"
- Partial progress, multi-event, negative, uncertain, delay reports
"""
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from vocab import (DISCIPLINES, LOCATION_ALIASES, ACTION_ALIASES, SIZES,
                   UNSCHEDULED_LOCATIONS)

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ROOT / "data" / "schedule" / "schedule_activities.csv"
OUT_REPORTS = ROOT / "data" / "raw_reports" / "reports.jsonl"
OUT_GOLD = ROOT / "data" / "labels" / "gold_extractions.jsonl"
OUT_MATCH = ROOT / "data" / "labels" / "gold_matches.jsonl"

SEED = 26122
N_REPORTS = 800  # increased from 100

# Updated mix for more realistic distribution
MIX = {
    "normal_formal": 150,
    "normal_shorthand": 100,
    "noisy": 120,
    "multi": 80,
    "partial": 80,
    "ambiguous": 60,
    "uncertain": 40,
    "delay": 40,
    "negative": 60,
    "irrelevant": 40,
    "no_match": 30,
}


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


def _phrase_for(work: dict, noun: str) -> str:
    """Natural description of a work item, without duplicating the action."""
    head = work["action"].split()[0]
    return noun if head in noun.lower() else f"{work['action']} of {noun}"


def _work_phrase(work: dict, act: dict) -> str:
    return _phrase_for(work, _field_noun(work, act))


def _join_action(noun: str, alias: str) -> str:
    """Append an action alias to a field noun unless it duplicates it."""
    n, a = noun.lower(), alias.lower()
    squash = lambda s: s.replace("-", "").replace(" ", "")
    if squash(a) in squash(n):
        return noun
    # word overlap in either direction: tube/tubing, pour/poured, fit/fit-up
    if any(len(nw) > 2 and (w in nw or nw in w)
           for w in a.split() for nw in n.split()):
        return noun
    return f"{noun} {alias}"


def _loc_alias(rng: random.Random, loc: str) -> str:
    return rng.choice(LOCATION_ALIASES.get(loc, [loc.lower()]))


@dataclass
class EventPlan:
    activity: dict
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
    candidate_pool: list[str] = field(default_factory=list)  # NEW: candidate IDs
    gold_match_id: str | None = None
    match_reason: str = "needs_review"

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


def _find_candidates(act: dict, l6: list[dict], rng: random.Random) -> list[str]:
    """Build candidate pool: correct + hard negatives + unrelated."""
    candidates = [act["activity_id"]]

    # Hard negatives: same discipline/action, different location or line
    work = _work_for(act["discipline"], act["activity_name"])
    action = work["action"]

    # Same action, different location
    same_action = [r for r in l6
                   if r["activity_id"] != act["activity_id"]
                   and _work_for(r["discipline"], r["activity_name"])["action"] == action
                   and r["discipline"] == act["discipline"]]

    hard_negs = rng.sample(same_action, k=min(3, len(same_action)))
    candidates.extend([r["activity_id"] for r in hard_negs])

    # Unrelated activities from other disciplines
    unrelated = [r for r in l6 if r["discipline"] != act["discipline"]]
    unrelated_sample = rng.sample(unrelated, k=min(5, len(unrelated)))
    candidates.extend([r["activity_id"] for r in unrelated_sample])

    rng.shuffle(candidates)
    return candidates[:10]  # cap at 10 candidates


def _base_plan(rng: random.Random, act: dict, rdate: date, work: dict, l6: list[dict]) -> EventPlan:
    return EventPlan(
        activity=act,
        description=_work_phrase(work, act),
        action=work["action"],
        status="completed",
        discipline=act["discipline"],
        location=act["location"] or None,
        equipment=None,
        line_number=None,
        qty_unit=None,
        candidate_pool=_find_candidates(act, l6, rng),
        gold_match_id=act["activity_id"],
        match_reason="correct_match",
    )


def _no_identifier(p: EventPlan) -> None:
    p.line_number = None
    p.equipment = None
    p.warnings += ["missing_line_number", "ambiguous_activity"]


def build_normal_formal(rng, act, rdate, work, l6):
    """Formal, complete report."""
    p = _base_plan(rng, act, rdate, work, l6)
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


def build_normal_shorthand(rng, act, rdate, work, l6):
    """Abbreviated field shorthand."""
    from vocab import ACTION_ALIASES
    p = _base_plan(rng, act, rdate, work, l6)
    noun = _field_noun(work, act).lower()
    alias = rng.choice(ACTION_ALIASES.get(work["action"], [work["action"]]))
    code = DISCIPLINES[act["discipline"]]["code"]
    loc_word = _loc_alias(rng, act["location"])

    sh, eh, phrase, _ = _time_pair(rng)
    p.start, p.end = _iso(rdate, sh), _iso(rdate, eh)
    p.time_certainty = "explicit"
    p.progress_percent = 100

    # Sometimes include identifier
    if act["line_number"] and rng.random() < 0.6:
        ident = f" L{act['line_number']}"
        p.line_number = act["line_number"]
    elif act["equipment_tag"] and rng.random() < 0.6:
        ident = f" {act['equipment_tag']}"
        p.equipment = act["equipment_tag"]
    else:
        ident = ""
        _no_identifier(p)

    text = f"{code} - {_join_action(noun + ident, alias)} @ {loc_word}, {phrase}, done."
    p.evidence = text
    return text, [p], "daily_report"


def build_noisy(rng, act, rdate, work, l6):
    """Messy, typos, missing fields."""
    from vocab import ACTION_ALIASES
    p = _base_plan(rng, act, rdate, work, l6)
    noun = _field_noun(work, act).lower()
    alias = rng.choice(ACTION_ALIASES.get(work["action"], [work["action"]]))

    drop_time = rng.random() < 0.4
    drop_loc = rng.random() < 0.3

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

    _no_identifier(p)
    p.progress_percent = 100

    # Add typos
    at = f" @ {loc_word}" if loc_word else ""
    text = f"{_join_action(noun, alias)}{at} {phrase} completd."
    text = " ".join(text.replace(" ,", ",").split())
    p.evidence = text
    return text, [p], "daily_report"


def build_multi(rng, acts, rdate, l6):
    """Multiple events in one report."""
    lines, plans = [], []
    for act in acts:
        work = _work_for(act["discipline"], act["activity_name"])
        p = _base_plan(rng, act, rdate, work, l6)
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


def build_partial(rng, act, rdate, work, l6):
    """Partial progress with quantity."""
    p = _base_plan(rng, act, rdate, work, l6)
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


def build_ambiguous(rng, act, rdate, work, l6):
    """Stripped identifier, multiple candidates fit."""
    p = _base_plan(rng, act, rdate, work, l6)
    _no_identifier(p)
    p.progress_percent = 100
    # Keep candidates but mark as needs review
    p.gold_match_id = act["activity_id"]  # gold knows the right one
    p.match_reason = "ambiguous_multiple_candidates"

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


def build_uncertain(rng, act, rdate, work, l6):
    """Hedged/uncertain report."""
    p = _base_plan(rng, act, rdate, work, l6)
    _no_identifier(p)
    p.assertion = "uncertain"
    p.status = "completed"
    p.warnings.append("uncertain_statement")
    p.warnings.append("missing_time")
    p.time_certainty = "missing"
    phrase_txt = _work_phrase(work, act)
    hedge = rng.choice(["probably", "likely", "appears to be", "seems"])
    text = f"{phrase_txt[0].upper()}{phrase_txt[1:]} at {act['location']} {hedge} completed, to be confirmed."
    p.evidence = text
    return text, [p], "daily_report"


def build_delay(rng, act, rdate, work, l6):
    """Delay/issue report."""
    p = _base_plan(rng, act, rdate, work, l6)
    _no_identifier(p)
    p.status = "suspended"
    p.warnings.append("missing_time")
    p.time_certainty = "missing"
    reason = rng.choice(["material not received", "crane unavailable",
                        "permit pending", "heavy rain", "manpower shortage"])
    phrase_txt = _work_phrase(work, act)
    text = f"{phrase_txt[0].upper()}{phrase_txt[1:]} at {act['location']} delayed due to {reason}."
    p.evidence = text
    return text, [p], "daily_report"


def build_negative(rng, act, rdate, work, l6):
    """Work did NOT happen."""
    p = _base_plan(rng, act, rdate, work, l6)
    _no_identifier(p)
    p.assertion = "negated"
    p.status = rng.choice(["not_started", "cancelled"])
    p.progress_percent = None
    p.time_certainty = "missing"
    p.warnings += ["negated_statement", "missing_time"]
    reason = rng.choice(["material not received", "crane unavailable",
                         "permit not issued", "heavy rain", "manpower shortage"])
    phrase_txt = _work_phrase(work, act)
    head = f"{phrase_txt[0].upper()}{phrase_txt[1:]} at {act['location']}"
    if p.status == "not_started":
        text = f"{head} could not be started today, {reason}."
    else:
        text = f"{head} has been cancelled, {reason}."
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
    "Safety induction for 12 new joinees held in the training room.",
    "Diesel bowser could not reach site, road blocked.",
    "Client representative visited site office for document review.",
]


def build_irrelevant(rng):
    return rng.choice(IRRELEVANT), [], "daily_report"


def build_no_match(rng, l6, rdate):
    """Legitimate project work at a location with NO scheduled activities.

    The description is real engineering language, but the (discipline, action,
    location) triple is verified absent from the schedule, so no schedule
    activity is a plausible attribute match.
    """
    scheduled_locs = {r["location"] for r in l6}
    free_locs = [l for l in UNSCHEDULED_LOCATIONS if l not in scheduled_locs]
    if not free_locs:
        raise RuntimeError("no unscheduled locations left for no_match cases")

    for _ in range(50):
        loc = rng.choice(free_locs)
        disc = rng.choice(list(DISCIPLINES))
        work = rng.choice(DISCIPLINES[disc]["works"])
        # programmatic verification: no schedule activity may match
        clash = any(
            r["location"] == loc and r["discipline"] == disc
            and _work_for(disc, r["activity_name"])["action"] == work["action"]
            for r in l6)
        if not clash:
            break
    else:
        raise RuntimeError("could not find a verified no_match combination")

    noun = work["field_noun"].format(size=rng.choice(SIZES)).strip()
    phrase = _phrase_for(work, noun)

    p = EventPlan(
        activity={},  # no schedule activity
        description=phrase,
        action=work["action"],
        status="completed",
        discipline=disc,
        location=loc,
        candidate_pool=[],  # empty: verified no match
        gold_match_id=None,
        match_reason="no_schedule_match",
    )
    p.progress_percent = 100
    p.time_certainty = "missing"
    p.warnings += ["missing_line_number", "missing_time", "no_schedule_candidate"]
    _no_identifier(p)

    text = f"{phrase[0].upper()}{phrase[1:]} completed at {loc} today."
    p.evidence = text
    return text, [p], "daily_report"


def load_schedule() -> list[dict]:
    with SCHEDULE.open(encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["wbs_level"] == "L6"]


def _ambiguous_pool(l6: list[dict]) -> list[dict]:
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
            text, plans, stype = build_multi(rng, acts, rdate, l6)
        elif kind == "irrelevant":
            text, plans, stype = build_irrelevant(rng)
        elif kind == "no_match":
            text, plans, stype = build_no_match(rng, l6, rdate)
        else:
            act = rng.choice(amb_pool if kind == "ambiguous" else l6)
            work = _work_for(act["discipline"], act["activity_name"])
            builder = {
                "normal_formal": build_normal_formal,
                "normal_shorthand": build_normal_shorthand,
                "noisy": build_noisy,
                "partial": build_partial,
                "ambiguous": build_ambiguous,
                "uncertain": build_uncertain,
                "delay": build_delay,
                "negative": build_negative,
            }.get(kind)
            if builder:
                text, plans, stype = builder(rng, act, rdate, work, l6)
            else:
                continue

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
            if p.gold_match_id is None:
                decision = "no_match"
            elif p.match_reason == "ambiguous_multiple_candidates":
                # gold truth is known, but a system must NOT auto-match here
                decision = "human_review"
            else:
                decision = "auto_match"
            matches.append({
                "report_id": rid,
                "event_id": eid,
                "candidate_pool": p.candidate_pool,
                "schedule_activity_id": p.gold_match_id,
                "decision": decision,
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
    n_no_match = sum(1 for m in matches if m["decision"] == "no_match")
    n_review = sum(1 for m in matches if m["decision"] == "human_review")
    n_hard = sum(1 for m in matches if len(m["candidate_pool"]) > 1)
    print(f"{OUT_REPORTS}: {len(reports)} reports")
    print(f"{OUT_GOLD}: {n_ev} gold events ({n_irr} irrelevant reports)")
    print(f"{OUT_MATCH}: {len(matches)} matches "
          f"({n_no_match} no_match, {n_review} human_review, {n_hard} with hard negatives)")


if __name__ == "__main__":
    main()
