"""SQLite persistence for the vertical slice.

Simplest practical persistence: one file (default data/pipeline.db), stdlib
sqlite3, no ORM. Tables: schedule_actual (verified actuals), reports,
events (extraction + match payloads), reviews (human actions), audit_log
(every accepted/corrected schedule update).

The schedule master (planned data) stays in data/schedule/schedule_activities.csv
-- this DB stores only the *actual* state layered on top of it.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = _ROOT / "data" / "pipeline.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schedule_actual (
    activity_id     TEXT PRIMARY KEY,
    actual_start    TEXT,
    actual_end      TEXT,
    status          TEXT,
    progress_percent REAL,
    updated_at      TEXT,
    updated_by      TEXT
);
CREATE TABLE IF NOT EXISTS reports (
    report_id       TEXT PRIMARY KEY,
    source_type     TEXT,
    report_date     TEXT,
    raw_text        TEXT,
    relevance_json  TEXT,
    extraction_json TEXT,
    engine          TEXT,
    processed_at    TEXT
);
CREATE TABLE IF NOT EXISTS events (
    event_id            TEXT PRIMARY KEY,
    report_id           TEXT,
    extraction_json     TEXT,
    match_json          TEXT,
    decision            TEXT,
    schedule_activity_id TEXT,
    needs_review        INTEGER DEFAULT 0,
    updated             INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS reviews (
    review_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        TEXT,
    report_id       TEXT,
    action          TEXT,
    corrected_activity_id TEXT,
    reviewer        TEXT,
    note            TEXT,
    created_at      TEXT
);
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id       TEXT,
    event_id        TEXT,
    schedule_activity_id TEXT,
    old_values      TEXT,
    new_values      TEXT,
    confidence      REAL,
    decision        TEXT,
    evidence        TEXT,
    action_source   TEXT,
    timestamp       TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PipelineDB:
    def __init__(self, path: str | Path = DEFAULT_DB):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # -- writes ----------------------------------------------------------
    def upsert_report(self, report: dict, extraction: dict, engine: str) -> None:
        doc = extraction.get("document", {})
        self.conn.execute(
            "INSERT INTO reports (report_id, source_type, report_date, raw_text,"
            " relevance_json, extraction_json, engine, processed_at)"
            " VALUES (?,?,?,?,?,?,?,?)"
            " ON CONFLICT(report_id) DO UPDATE SET relevance_json=excluded.relevance_json,"
            " extraction_json=excluded.extraction_json, engine=excluded.engine,"
            " processed_at=excluded.processed_at",
            (
                report.get("report_id"), report.get("source_type"),
                report.get("report_date"), report.get("raw_text"),
                json.dumps(extraction.get("relevance")),
                json.dumps(extraction), engine, _now(),
            ),
        )
        self.conn.commit()

    def upsert_event(self, event: dict, match: dict) -> None:
        decision = match.get("decision")
        needs_review = bool(event.get("needs_review")) or decision != "auto_match"
        self.conn.execute(
            "INSERT INTO events (event_id, report_id, extraction_json, match_json,"
            " decision, schedule_activity_id, needs_review, updated)"
            " VALUES (?,?,?,?,?,?,?,0)"
            " ON CONFLICT(event_id) DO UPDATE SET match_json=excluded.match_json,"
            " decision=excluded.decision,"
            " schedule_activity_id=excluded.schedule_activity_id,"
            " needs_review=excluded.needs_review",
            (
                event.get("event_id"), event.get("report_id")
                or (event.get("event_id", "").rsplit("-EVT-", 1)[0] or None),
                json.dumps(event), json.dumps(match), decision,
                match.get("schedule_activity_id"), int(needs_review),
            ),
        )
        self.conn.commit()

    def get_event(self, event_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM events WHERE event_id = ?", (event_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["extraction"] = json.loads(d.pop("extraction_json") or "{}")
        d["match"] = json.loads(d.pop("match_json") or "{}")
        d["needs_review"] = bool(d["needs_review"])
        d["updated"] = bool(d["updated"])
        return d

    def pending_reviews(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT event_id, report_id, decision, schedule_activity_id,"
            " needs_review FROM events WHERE needs_review = 1"
            " ORDER BY event_id"
        ).fetchall()
        return [dict(r) for r in rows]

    # -- schedule update + audit (single transaction) ---------------------
    def apply_update(
        self,
        event: dict,
        match: dict,
        new_values: dict,
        action_source: str,
    ) -> int:
        """Write actuals + one audit row atomically. Returns audit_id."""
        aid = match["schedule_activity_id"]
        cur = self.conn.execute(
            "SELECT actual_start, actual_end, status, progress_percent,"
            " updated_at, updated_by FROM schedule_actual WHERE activity_id = ?",
            (aid,),
        ).fetchone()
        old = dict(cur) if cur else {}

        ts = _now()
        merged = {**old, **{k: v for k, v in new_values.items() if v is not None},
                  "updated_at": ts, "updated_by": action_source}
        self.conn.execute(
            "INSERT INTO schedule_actual (activity_id, actual_start, actual_end,"
            " status, progress_percent, updated_at, updated_by)"
            " VALUES (?,?,?,?,?,?,?)"
            " ON CONFLICT(activity_id) DO UPDATE SET actual_start=excluded.actual_start,"
            " actual_end=excluded.actual_end, status=excluded.status,"
            " progress_percent=excluded.progress_percent,"
            " updated_at=excluded.updated_at, updated_by=excluded.updated_by",
            (aid, merged.get("actual_start"), merged.get("actual_end"),
             merged.get("status"), merged.get("progress_percent"),
             ts, action_source),
        )
        cur2 = self.conn.execute(
            "INSERT INTO audit_log (report_id, event_id, schedule_activity_id,"
            " old_values, new_values, confidence, decision, evidence,"
            " action_source, timestamp) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                event.get("report_id"), event.get("event_id"), aid,
                json.dumps(old), json.dumps(new_values),
                match.get("confidence"), match.get("decision"),
                (event.get("evidence") or {}).get("source_text"),
                action_source, ts,
            ),
        )
        self.conn.execute(
            "UPDATE events SET updated = 1 WHERE event_id = ?",
            (event.get("event_id"),),
        )
        self.conn.commit()
        return cur2.lastrowid

    def record_review(
        self, event_id: str, action: str, corrected_activity_id: str | None,
        reviewer: str, note: str | None, report_id: str | None = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO reviews (event_id, report_id, action,"
            " corrected_activity_id, reviewer, note, created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (event_id, report_id, action, corrected_activity_id,
             reviewer, note, _now()),
        )
        self.conn.commit()
        return cur.lastrowid

    # -- reads -------------------------------------------------------------
    def activity_state(self, activity_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM schedule_actual WHERE activity_id = ?", (activity_id,)
        ).fetchone()
        actual = dict(row) if row else None
        audit_rows = self.conn.execute(
            "SELECT * FROM audit_log WHERE schedule_activity_id = ?"
            " ORDER BY audit_id DESC", (activity_id,),
        ).fetchall()
        audit = []
        for r in audit_rows:
            d = dict(r)
            d["old_values"] = json.loads(d.pop("old_values") or "{}")
            d["new_values"] = json.loads(d.pop("new_values") or "{}")
            audit.append(d)
        return {"activity_id": activity_id, "actual": actual, "audit": audit}

    def audit_for_event(self, event_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM audit_log WHERE event_id = ? ORDER BY audit_id",
            (event_id,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["old_values"] = json.loads(d.pop("old_values") or "{}")
            d["new_values"] = json.loads(d.pop("new_values") or "{}")
            out.append(d)
        return out
