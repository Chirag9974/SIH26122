"""Backend API for the vertical slice.

    cd src && python -m pipeline.api          # serves on 127.0.0.1:8000

Endpoints:
- POST /reports            JSON report -> extract -> validate -> match -> store
- POST /reports/pdf        multipart PDF upload -> text -> same flow
- POST /reviews            accept | correct | reject | mark_unmatched
- GET  /reviews/pending    events waiting for a human
- GET  /activities/{id}    planned + latest actual + audit history
"""
from __future__ import annotations

from datetime import date

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from pipeline.db import PipelineDB, DEFAULT_DB
from pipeline.orchestrator import get_activity, ingest_pdf, process_report
from pipeline.review import ReviewService


class ReportIn(BaseModel):
    report_id: str | None = None
    report_date: str | None = None
    source_type: str = "daily_report"
    raw_text: str = Field(min_length=1)


class ReviewIn(BaseModel):
    event_id: str
    action: str                      # accept | correct | reject | mark_unmatched
    corrected_activity_id: str | None = None
    reviewer: str = "human"
    note: str | None = None
    overrides: dict | None = None    # optional field fixes while accepting


def create_app(db_path=DEFAULT_DB, use_embeddings: bool = True) -> FastAPI:
    app = FastAPI(title="SIH26122 vertical slice", version="0.1")
    db = PipelineDB(db_path)
    svc = ReviewService(db)
    state = {"use_embeddings": use_embeddings}

    @app.post("/reports")
    def post_report(body: ReportIn) -> dict:
        report = body.model_dump()
        report["report_date"] = report.get("report_date") \
            or date.today().isoformat()
        return process_report(
            report, db=db, engine="baseline",
            use_embeddings=state["use_embeddings"],
        )

    @app.post("/reports/pdf")
    def post_report_pdf(file: UploadFile = File(...),
                        report_id: str | None = None) -> dict:
        import tempfile

        data = file.file.read()
        if not data.startswith(b"%PDF"):
            raise HTTPException(400, "uploaded file is not a PDF")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            return ingest_pdf(tmp_path, report_id=report_id, db=db,
                              use_embeddings=state["use_embeddings"])
        finally:
            import os
            os.unlink(tmp_path)

    @app.post("/reviews")
    def post_review(body: ReviewIn) -> dict:
        try:
            outcome = svc.decide(
                body.event_id, body.action,
                corrected_activity_id=body.corrected_activity_id,
                reviewer=body.reviewer, note=body.note,
                overrides=body.overrides,
            )
        except KeyError as exc:
            raise HTTPException(404, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return outcome.__dict__

    @app.get("/reviews/pending")
    def get_pending() -> dict:
        return {"pending": db.pending_reviews()}

    @app.get("/activities/{activity_id}")
    def get_activity_state(activity_id: str) -> dict:
        result = get_activity(activity_id, db=db)
        if result["planned"] is None and result["actual"] is None:
            raise HTTPException(404, f"unknown activity: {activity_id}")
        return result

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
