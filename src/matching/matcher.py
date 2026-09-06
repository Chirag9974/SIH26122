"""Hybrid schedule matcher: extracted event -> L5/L6 schedule activity.

Pipeline: normalize -> hybrid candidate generation (semantic + lexical +
metadata) -> weighted scoring -> conservative decision.

    from matching.matcher import match_event
    result = match_event(event_dict)

Decisions, in priority order:
- no_match      : best score below the floor -- never force a match
- auto_match    : high score, clear margin over runner-up, no hard conflicts
- human_review  : everything in between (or conflicting evidence)

Metadata (discipline / line / equipment) acts as a soft prior plus a hard
veto on otherwise-weak identity signals, so a same-text event on a different
line never outranks the true line-specific activity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from matching.embeddings import DEFAULT_MODEL, ScheduleIndex
from matching.normalize import EventNorm, canon_action, normalize_event

_ROOT = Path(__file__).resolve().parents[2]
SCHEDULE_CSV = _ROOT / "data" / "schedule" / "schedule_activities.csv"

# --- ranking weights (sum to 1.0) ------------------------------------------
W_SEMANTIC = 0.50
W_LEXICAL = 0.20
W_ACTION = 0.15
W_LOCATION = 0.05
W_DISCIPLINE = 0.04
W_LINE = 0.03
W_EQUIPMENT = 0.02
W_HIERARCHY = 0.15  # L5 work-package rollups outrank only via metadata

# --- decision thresholds ----------------------------------------------------
NO_MATCH_FLOOR = 0.42   # below this: no_match, never force
AUTO_THRESHOLD = 0.56   # at/above this (with margin): auto_match
AUTO_MARGIN = 0.01      # runner-up must trail by at least this
# (operating point chosen by matching.sweep_thresholds on the dev gold:
#  accuracy 0.357 / auto-precision 0.93 / no-match accuracy 1.00)
STRONG_LINE_BONUS = 0.06    # exact line+discipline corroboration
UNIQUE_LINE_BONUS = 0.10    # the ONLY activity carrying the event's line
STRONG_LOC_BONUS = 0.04     # exact canonical location match
LOC_MISMATCH_PENALTY = 0.08   # both locations known and different
DISC_MISMATCH_PENALTY = 0.06  # both disciplines known and different


@dataclass
class Candidate:
    activity_id: str
    name: str
    score: float
    signals: dict = field(default_factory=dict)


@dataclass
class MatchResult:
    event_id: str | None
    decision: str  # auto_match | human_review | no_match
    schedule_activity_id: str | None
    matched_activity: str | None
    confidence: float
    candidates: list[Candidate]
    reasons: list[str]
    event_norm: EventNorm | None = None

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "decision": self.decision,
            "schedule_activity_id": self.schedule_activity_id,
            "matched_activity": self.matched_activity,
            "confidence": round(self.confidence, 4),
            "candidates": [
                {
                    "activity_id": c.activity_id,
                    "name": c.name,
                    "score": round(c.score, 4),
                    "signals": c.signals,
                }
                for c in self.candidates[:5]
            ],
            "reasons": self.reasons,
        }


class Matcher:
    def __init__(
        self,
        csv_path: str | Path = SCHEDULE_CSV,
        model_name: str = DEFAULT_MODEL,
        use_embeddings: bool = True,
    ):
        self.index = ScheduleIndex(
            csv_path, model_name=model_name, use_embeddings=use_embeddings
        )
        # locations the schedule actually works in; an event naming a
        # location outside this inventory cannot have a schedule match
        self.known_locations = {
            r.location for r in self.index.rows if r.location
        }

    # ------------------------------------------------------------------
    def _score_candidates(
        self, ev: EventNorm, query: str
    ) -> tuple[list[Candidate], list[str]]:
        idx = self.index
        reasons: list[str] = []

        qvec = idx.embed_query(query)
        sem = idx.semantic_scores(qvec)
        lex = {h.activity_id: h.score for h in idx.top_lexical(query, k=40)}

        ids = set(sem) | set(lex)
        # metadata-filtered recall: same discipline always stays in play
        for r in idx.rows:
            if ev.discipline and r.discipline == ev.discipline:
                ids.add(r.activity_id)
        # optional official candidate pool: used only to force visibility
        # (scoring still judges every candidate on its own merits)
        pool = getattr(ev, "_pool", None) or []
        if getattr(self, "restrict_to_pool", False) and pool:
            ids &= set(pool)
        else:
            ids.update(pool)

        # unique-line discrimination: exactly one candidate carrying the
        # event's line number makes that line near-decisive.
        line_rows = [r for r in idx.rows
                     if ev.line_number and r.lines_match(ev.line_number)]
        unique_line = len(line_rows) == 1

        cands: list[Candidate] = []
        for aid in ids:
            row = idx.by_id.get(aid)
            if row is None:
                continue
            s_sem = sem.get(aid, 0.0)
            s_lex = lex.get(aid, 0.0)
            row_action = canon_action(row.name_core)
            s_act = 1.0 if (
                ev.action
                and (row_action == ev.action
                     or ev.action in row.name_core.lower())
            ) else 0.0
            s_loc = 1.0 if (ev.location and row.location
                            and ev.location.lower() == row.location.lower()) else 0.0
            s_disc = 1.0 if (ev.discipline and row.discipline == ev.discipline) else 0.0
            s_line = 1.0 if (ev.line_number and row.lines_match(ev.line_number)) else 0.0
            s_eq = 1.0 if (ev.equipment and row.equipment_match(ev.equipment)) else 0.0
            s_hier = 0.0 if row.is_l6 else -1.0  # gold targets are L6 work items;
            # L5 'Work Package' rollups are the parent, not the leaf

            score = (
                W_SEMANTIC * s_sem
                + W_LEXICAL * s_lex
                + W_ACTION * s_act
                + W_LOCATION * s_loc
                + W_DISCIPLINE * s_disc
                + W_LINE * s_line
                + W_EQUIPMENT * s_eq
                + W_HIERARCHY * s_hier
            )
            signals = {
                "semantic": round(s_sem, 3),
                "lexical": round(s_lex, 3),
                "action": s_act,
                "location": s_loc,
                "discipline": s_disc,
                "line": s_line,
                "equipment": s_eq,
                "L6": s_hier >= 0,
            }
            if ev.line_number and s_line and s_disc:
                score += STRONG_LINE_BONUS
                signals["line_bonus"] = STRONG_LINE_BONUS
            if unique_line and s_line:
                score += UNIQUE_LINE_BONUS
                signals["unique_line"] = UNIQUE_LINE_BONUS
            if ev.location and s_loc and (ev.discipline is None or s_disc):
                score += STRONG_LOC_BONUS
                signals["loc_bonus"] = STRONG_LOC_BONUS
            # metadata conflict: evidence points away from this activity
            if (ev.location and row.location
                    and ev.location.lower() != row.location.lower()):
                score -= LOC_MISMATCH_PENALTY
                signals["loc_mismatch"] = -LOC_MISMATCH_PENALTY
            if (ev.discipline and row.discipline
                    and ev.discipline != row.discipline):
                score -= DISC_MISMATCH_PENALTY
                signals["disc_mismatch"] = -DISC_MISMATCH_PENALTY
            cands.append(
                Candidate(aid, row.name, min(score, 1.0), signals)
            )
        cands.sort(key=lambda c: -c.score)
        return cands, reasons

    def _decide(
        self, ev: EventNorm, cands: list[Candidate]
    ) -> tuple[str, str | None, float, list[str]]:
        reasons: list[str] = []
        if not cands:
            return "no_match", None, 0.0, ["no candidates generated"]

        best, second = cands[0], cands[1] if len(cands) > 1 else None
        # margin over the best *sibling* (same canonical action); full-pool
        # margin would ignore the real confusion: look-alike activities.
        best_act = best.signals.get("action")
        second = next(
            (c for c in cands[1:]
             if best_act is None or c.signals.get("action") == best_act),
            None,
        )
        margin = best.score - (second.score if second else 0.0)

        # hard conflict veto: strong metadata points at a different activity
        if ev.line_number and best.signals.get("line") == 0.0:
            conflict = [
                c
                for c in cands[1:6]
                if c.signals.get("line") == 1.0 and c.signals.get("discipline") == 1.0
            ]
            if conflict:
                return (
                    "human_review",
                    None,
                    0.4,
                    [f"line {ev.line_number} matches a different activity than top score"],
                )

        # indistinguishable twins: near-identical candidates the event's
        # fields cannot separate (e.g. 'Install Pump FT-346 - Part 1' vs
        # '- Part 2' at the same location with no tag in the event). Forcing
        # a pick would be guessing; route to review instead.
        TWIN_SIG = 0.03
        twins = [
            c for c in cands[1:6]
            if best.score - c.score <= TWIN_SIG
            and c.signals.get("action") == best.signals.get("action")
            and c.signals.get("location") == best.signals.get("location")
            and c.signals.get("discipline") == best.signals.get("discipline")
            and c.signals.get("line") == best.signals.get("line")
        ]
        if twins and best.signals.get("location") == 1.0 \
                and best.signals.get("action") == 1.0:
            return (
                "human_review",
                best.activity_id,
                best.score,
                [f"indistinguishable near-ties: "
                 f"{', '.join(c.activity_id for c in twins[:3])} "
                 f"({len(twins) + 1} candidates match action+location equally)"],
            )

        if best.score < NO_MATCH_FLOOR:
            return "no_match", None, best.score, [
                f"best score {best.score:.2f} below floor {NO_MATCH_FLOOR}"
            ]

        conf = min(best.score, 1.0)
        if ev.line_number and best.signals.get("line") == 1.0:
            conf = min(1.0, conf + STRONG_LINE_BONUS / 2)
            reasons.append(f"line number {ev.line_number} corroborates")

        auto_ok = (
            best.score >= AUTO_THRESHOLD
            and margin >= AUTO_MARGIN
            and (ev.action is None or best.signals.get("action") == 1.0 or
                 best.signals.get("semantic", 0) >= 0.5)
        )
        if auto_ok:
            reasons.append(
                f"score {best.score:.2f} >= {AUTO_THRESHOLD} with margin {margin:.2f}"
            )
            return "auto_match", best.activity_id, conf, reasons
        return "human_review", best.activity_id, conf, [
            f"score {best.score:.2f} / margin {margin:.2f} below auto bar"
        ]

    # ------------------------------------------------------------------
    @staticmethod
    def _query_text(ev: EventNorm) -> str:
        """Query text: activity phrase enriched with known metadata so the
        semantic view that carries it (name / name+loc / name+line) fires."""
        base = ev.description or ev.action or ""
        parts = [base]
        if ev.location and ev.location.lower() not in base.lower():
            parts.append(f"at {ev.location}")
        if ev.line_number:
            parts.append(f"Line {ev.line_number}")
        if ev.equipment:
            parts.append(ev.equipment)
        return " ".join(parts)

    def match(self, event: dict | EventNorm, candidate_pool=None) -> MatchResult:
        ev = event if isinstance(event, EventNorm) else normalize_event(event)
        if candidate_pool:
            setattr(ev, "_pool", list(candidate_pool))

        query = self._query_text(ev)
        if not query.strip():
            return MatchResult(
                ev.event_id, "no_match", None, None, 0.0, [],
                ["event has no action/description to match"], ev,
            )
        if ev.location and self.known_locations \
                and ev.location not in self.known_locations:
            return MatchResult(
                ev.event_id, "no_match", None, None, 0.0, [],
                [f"location '{ev.location}' not in schedule inventory"], ev,
            )
        cands, _ = self._score_candidates(ev, query)
        decision, aid, conf, reasons = self._decide(ev, cands)
        matched = self.index.by_id.get(aid) if aid else None
        return MatchResult(
            event_id=ev.event_id,
            decision=decision,
            schedule_activity_id=aid,
            matched_activity=matched.name if matched else None,
            confidence=conf,
            candidates=cands[:5],
            reasons=reasons,
            event_norm=ev,
        )


_DEFAULT: Matcher | None = None


def match_event(
    event: dict, candidate_pool: list[str] | None = None, use_embeddings: bool = True
) -> dict:
    """Stable API: extracted event dict -> match decision dict."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Matcher(use_embeddings=use_embeddings)
    return _DEFAULT.match(event, candidate_pool=candidate_pool).to_dict()
