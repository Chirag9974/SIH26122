"""Review/verification service for the vertical slice.

Decides what may touch the schedule:
- auto_match + clean event  -> auto update (the only automatic path)
- human_review / no_match   -> parked for a human; schedule untouched until
  accept / correct / mark-unmatched / reject
- reject / mark-unmatched   -> never update

The LLM/baseline extractor never writes the schedule directly; every write
goes through here and lands in one transaction (actuals + audit row).
"""
from __future__ import annotations

from dataclasses import dataclass

from pipeline.db import PipelineDB
from matching.matcher import MatchResult

# events carrying these extractor warnings need a human before auto-update
REVIEW_WARNINGS = {
    "conflicting_report", "unsupported_uncertainty", "contradiction_status_progress",
    "unsupported_value", "unsupported_evidence", "hallucination_guard",
}

# only actual work justifies writing actuals; a negated statement
# ('No welding was carried out') must not push not_started onto an activity
POSITIVE_STATUS = {"completed", "in_progress", "started", "suspended"}


@dataclass
class ReviewOutcome:
    event_id: str
    action: str          # accepted | corrected | rejected | marked_unmatched
    schedule_activity_id: str | None
    updated: bool
    audit_id: int | None
    detail: str


class ReviewService:
    def __init__(self, db: PipelineDB):
        self.db = db

    # ------------------------------------------------------------------
    def auto_gate(self, event: dict, match: dict) -> bool:
        """True only for a clean auto_match the pipeline may apply itself."""
        if match.get("decision") != "auto_match":
            return False
        if event.get("needs_review"):
            return False
        warnings = set(event.get("warnings") or [])
        if warnings & REVIEW_WARNINGS:
            return False
        exe = event.get("execution") or {}
        if exe.get("status") not in POSITIVE_STATUS:
            return False
        if exe.get("assertion") not in (None, "affirmed"):
            return False
        return True

    def auto_update(self, event: dict, match: dict,
                    action_source: str = "pipeline:auto") -> ReviewOutcome:
        """Apply an auto_match to the schedule (call only after auto_gate)."""
        new_values = _event_to_actual(event)
        audit_id = self.db.apply_update(
            event, match, new_values, action_source=action_source
        )
        return ReviewOutcome(
            event_id=event.get("event_id"),
            action="accepted",
            schedule_activity_id=match["schedule_activity_id"],
            updated=True,
            audit_id=audit_id,
            detail="auto_match applied automatically",
        )

    # ------------------------------------------------------------------
    def decide(
        self,
        event_id: str,
        action: str,                      # accept | correct | reject | mark_unmatched
        corrected_activity_id: str | None = None,
        reviewer: str = "human",
        note: str | None = None,
        overrides: dict | None = None,    # human may fix values while accepting
    ) -> ReviewOutcome:
        ev = self.db.get_event(event_id)
        if ev is None:
            raise KeyError(f"unknown event_id: {event_id}")
        event = ev["extraction"]
        match = ev["match"]
        if action not in ("accept", "correct", "reject", "mark_unmatched"):
            raise ValueError(f"unknown review action: {action}")
        if action == "correct" and not corrected_activity_id:
            raise ValueError("correct requires corrected_activity_id")

        report_id = ev.get("report_id")

        if action == "reject":
            self.db.record_review(event_id, "reject", None, reviewer, note, report_id)
            return ReviewOutcome(
                event_id=event_id, action="rejected",
                schedule_activity_id=None, updated=False, audit_id=None,
                detail="reviewer rejected the match; schedule untouched",
            )

        if action == "mark_unmatched":
            self.db.record_review(event_id, "mark_unmatched", None, reviewer,
                                  note, report_id)
            return ReviewOutcome(
                event_id=event_id, action="marked_unmatched",
                schedule_activity_id=None, updated=False, audit_id=None,
                detail="marked new/unmatched; no schedule activity applies",
            )

        if action == "correct":
            match = {**match, "schedule_activity_id": corrected_activity_id,
                     "decision": "human_review"}
            match["corrected_by"] = reviewer
            event = {**event, "_corrected": True}

        new_values = _event_to_actual(event)
        if overrides:
            new_values.update(overrides)
        new_values = {k: v for k, v in new_values.items() if v is not None}
        audit_id = self.db.apply_update(
            event, match, new_values, action_source=f"review:{reviewer}"
        )
        self.db.record_review(
            event_id, action, match["schedule_activity_id"], reviewer, note,
            report_id,
        )
        return ReviewOutcome(
            event_id=event_id,
            action="accepted" if action == "accept" else "corrected",
            schedule_activity_id=match["schedule_activity_id"],
            updated=True,
            audit_id=audit_id,
            detail=f"reviewer {action}d; schedule updated"
                   if action == "accept" else
                   f"corrected to {match['schedule_activity_id']}; schedule updated",
        )


def _event_to_actual(event: dict) -> dict:
    """Map extraction fields -> schedule actual fields (None = don't touch)."""
    exe = event.get("execution") or {}
    t = event.get("time") or {}
    return {
        "actual_start": t.get("start"),
        "actual_end": t.get("end"),
        "status": exe.get("status"),
        "progress_percent": exe.get("progress_percent"),
    }
