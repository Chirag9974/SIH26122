# Vertical Slice Pipeline

REPORT → EXTRACT → MATCH → VERIFY/REVIEW → UPDATE SCHEDULE → AUDIT.

The extractor and matcher contracts are unchanged; this stage orchestrates
them and owns the only code path that may write schedule actuals. **The LLM
(or baseline) output never touches the schedule directly** — every write is
either a gated auto-match or a human review decision, and every write lands
in one transaction with an audit row.

## Components (src/pipeline/)

| File | Role |
|---|---|
| `db.py` | SQLite (`data/pipeline.db`, stdlib `sqlite3`, no ORM). Tables: `schedule_actual` (verified actuals only), `reports`, `events`, `reviews`, `audit_log`. Update + audit are a single transaction. |
| `review.py` | `ReviewService`: `auto_gate` (clean `auto_match` + positive affirmed status, no review warnings) and `decide` (`accept` / `correct` / `reject` / `mark_unmatched`). |
| `orchestrator.py` | `process_report` (extract → validate → match → store → auto-apply), `ingest_pdf` (pypdf + line-unwrap + header parsing), `get_activity` (planned + actual + audit). |
| `api.py` | FastAPI app (`create_app()`), `python -m pipeline.api` serves on 127.0.0.1:8000. |

## API

| Endpoint | Purpose |
|---|---|
| `POST /reports` | JSON `{report_id?, report_date?, source_type?, raw_text}` → extract → validate → match → store (+ auto-apply clean matches). Returns per-event decisions. |
| `POST /reports/pdf` | multipart PDF upload → text → same flow (`report_id`/`report_date` parsed from the header). |
| `GET /reviews/pending` | events parked for a human (`human_review`, `no_match`, extractor warnings). |
| `POST /reviews` | `{event_id, action: accept\|correct\|reject\|mark_unmatched, corrected_activity_id?, reviewer?, note?, overrides?}`. |
| `GET /activities/{id}` | planned row + latest actual state + full audit history. |

## Decision → schedule rules

- `auto_match` + no review warnings + positive affirmed status → **auto update** (`pipeline:auto`).
- `human_review` / `no_match` / negated or uncertain statements → parked; nothing is written.
- `accept` → update with reviewer as source. `correct` → same, against the corrected activity id.
- `reject` / `mark_unmatched` → **no schedule change** (still recorded in `reviews`).

## Audit record (one row per accepted/corrected update)

`report_id, event_id, schedule_activity_id, old_values, new_values, confidence, decision, evidence/source_text, action_source, timestamp`.

## Run

```bash
cd src
python -m pipeline.api                 # serve the API
python -m tests.test_pipeline          # 21 flow tests (throwaway DBs)
python -m tests.test_api               # 13 endpoint tests (httpx ASGI)
# one-shot E2E on the attached test PDF:
python - <<'PY'
import sys; sys.path.insert(0, '.')
from pipeline.orchestrator import ingest_pdf
from pipeline.db import PipelineDB
print(ingest_pdf(r"C:\path\to\SIH26122_Test_Daily_Progress_Report.pdf", db=PipelineDB()))
PY
```

New runtime deps: `fastapi`, `uvicorn`, `pypdf`, `python-multipart` (all
installed user-level). Dev-metric regression after the evidence-contiguity
extractor fix: event F1 0.904 (unchanged), baseline eval unchanged.
