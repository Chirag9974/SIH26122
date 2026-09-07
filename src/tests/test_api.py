"""API contract tests for the vertical slice backend.

Run:  python -m tests.test_api

Exercises the real FastAPI app with httpx (ASGI transport, no open port):
POST /reports, POST /reviews, GET /reviews/pending, GET /activities/{id},
and the PDF ingestion path using the attached test report as a fixture.
Uses a throwaway DB per app instance.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient  # noqa: E402

from pipeline.api import create_app  # noqa: E402

FIXTURE_PDF = (_P(__file__).resolve().parent / "fixtures"
               / "SIH26122_Test_Daily_Progress_Report.pdf")


def run() -> None:
    stats = {"ok": 0, "fail": 0}

    def check(name: str, cond: bool) -> None:
        if cond:
            stats["ok"] += 1
            print(f"  ok    {name}")
        else:
            stats["fail"] += 1
            print(f"  FAIL  {name}")

    app = create_app(tempfile.mktemp(suffix=".db"))
    with TestClient(app) as client:

        # ---- POST /reports -------------------------------------------
        r = client.post("/reports", json={
            "report_id": "API-T1", "report_date": "2026-09-07",
            "raw_text": "Piping crew completed erection of the 18in spool at "
                        "Rack C on Line 40-09. Work ran from 09:00 to 16:00.",
        })
        check("POST /reports -> 200", r.status_code == 200)
        body = r.json()
        check("POST /reports: event auto-matched + updated",
              body["events"] and body["events"][0]["decision"] == "auto_match"
              and body["events"][0]["auto_updated"])

        # ---- pending review queue ------------------------------------
        r = client.post("/reports", json={
            "report_id": "API-T2", "report_date": "2026-09-07",
            "raw_text": "Pump installation was reported complete by the "
                        "contractor. Supervisor confirmation is still pending.",
        })
        body2 = r.json()
        ev2 = body2["events"][0]
        r = client.get("/reviews/pending")
        check("GET /reviews/pending lists the ambiguous event",
              r.status_code == 200 and any(
                  p["event_id"] == ev2["event_id"] for p in r.json()["pending"]))

        # ---- POST /reviews accept ------------------------------------
        r = client.post("/reviews", json={
            "event_id": ev2["event_id"], "action": "accept",
            "reviewer": "supervisor1", "note": "verified on site",
        })
        check("POST /reviews accept -> 200 + updated",
              r.status_code == 200 and r.json()["updated"])
        accepted_id = r.json()["schedule_activity_id"]

        # ---- GET /activities/{id} ------------------------------------
        r = client.get(f"/activities/{accepted_id}")
        check("GET /activities/:id -> 200", r.status_code == 200)
        act = r.json()
        check("GET /activities/:id has planned + actual + audit",
              act["planned"] is not None and act["actual"] is not None
              and len(act["audit"]) >= 1)

        # ---- review validation errors --------------------------------
        r = client.post("/reviews", json={
            "event_id": "MISSING-EVT-01", "action": "accept"})
        check("POST /reviews unknown event -> 404", r.status_code == 404)
        r = client.post("/reviews", json={
            "event_id": ev2["event_id"], "action": "nonsense"})
        check("POST /reviews bad action -> 400", r.status_code == 400)

        # ---- reject leaves schedule untouched -------------------------
        r = client.post("/reviews", json={
            "event_id": ev2["event_id"], "action": "reject",
            "reviewer": "qa1", "note": "second look, wrong"})
        check("POST /reviews reject -> 200 not updated",
              r.status_code == 200 and not r.json()["updated"])

        # ---- PDF ingestion path ---------------------------------------
        if FIXTURE_PDF.exists():
            with open(FIXTURE_PDF, "rb") as fh:
                r = client.post("/reports/pdf", files={
                    "file": ("SIH26122_Test_Daily_Progress_Report.pdf", fh,
                             "application/pdf")})
            check("POST /reports/pdf -> 200", r.status_code == 200)
            pdf_body = r.json()
            check("PDF: report id parsed from header",
                  pdf_body.get("report_id") == "TEST-DPR-001")
            check("PDF: multiple events extracted",
                  len(pdf_body.get("events", [])) >= 3)
            check("PDF: some events auto-applied, some parked",
                  any(e["auto_updated"] for e in pdf_body["events"])
                  and any(e["needs_review"] for e in pdf_body["events"]))
        else:
            print("  skip  PDF fixture not present")

    total = stats["ok"] + stats["fail"]
    print(f"\n{stats['ok']}/{total} API tests passed")
    if stats["fail"]:
        print(f"{stats['fail']} FAILED")
        sys.exit(1)


if __name__ == "__main__":
    run()
