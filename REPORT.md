# SIH26122 Dataset — Final Report

## Pipeline Status: All Steps PASS

| Step | Result |
|------|--------|
| Schedule master | 301 activities (195 L6, 106 L5) |
| Report generation | 1,020 reports |
| Terminology export | 257 alias rows, 82 abbreviations |
| Edge cases | 26 adversarial reports |
| Validation | **PASS** (0 errors) |
| Extractor tests | **17/17 passed** |

---

## Report & Event Counts

| Metric | Count |
|--------|-------|
| Raw reports | 1,020 |
| Gold events | 1,161 |
| Irrelevant reports (no_match) | 60 |
| Matches with candidate pools | 1,131 |
| no_match (verified unmatchable) | 30 |
| human_review (ambiguous) | 60 |
| auto_match | 1,091 |

### Case Distribution

| Case kind | Count |
|-----------|-------|
| normal_formal | 180 |
| normal_shorthand | 120 |
| noisy | 130 |
| multi | 100 |
| partial | 100 |
| ambiguous | 60 |
| negative | 60 |
| irrelevant | 60 |
| delay | 50 |
| uncertain | 50 |
| conflict | 40 |
| suspended | 40 |

---

## Split Sizes

| Split | Reports | Source |
|-------|---------|--------|
| train | 534 | — |
| dev | 178 | — |
| test | 178 | standard mix |
| test_hard | 130 | unseen case types (delay/uncertain/no_match/edge) |
| regression | 100 | frozen original benchmark |

No leakage between splits — verified.

---

## Extractor Metrics

### All splits (1,020 reports)

| Metric | Value |
|--------|-------|
| Relevance F1 | 0.985 (P=1.000, R=0.970) |
| Event F1 | 0.973 (tp=1,130, fp=32, fn=31) |
| Field accuracy | 0.972 |

### By field (all)

| Field | Accuracy |
|-------|----------|
| activity.action | 1.000 |
| context.discipline | 0.993 |
| context.location | 1.000 |
| execution.assertion | 0.985 |
| execution.progress_percent | 0.922 |
| execution.status | 0.905 |
| quantity.completed | 0.976 |
| quantity.total | 0.976 |
| time.time_certainty | 0.998 |
| date/time start | 0.998 |
| date/time end | 0.915 |

### Split-specific

| Split | Rel F1 | Event F1 | Field Acc |
|-------|--------|----------|-----------|
| test (178) | 0.972 | 0.954 | 0.976 |
| test_hard (130) | 1.000 | 1.000 | 0.994 |
| edge (26) | 1.000 | 1.000 | 1.000 |
| regression (100) | 1.000 | 1.000 | 1.000 (end=0.992) |

### Candidate Pool Quality

Pool recall 1.000 on all splits with candidate pools — the gold activity is always retrievable from a pool of ~9 candidates (correct + 3 hard negatives + 5 unrelated, shuffled, capped at 10).

---

## Remaining Weaknesses

1. **`normal_shorthand` incomplete**: 89/120 extracted — 31 reports produce no events. Shorthand templates use abbreviated wording the extractor occasionally misses. Gold labels are correct; extractor under-counts.

2. **Conflict quantity parsing**: Conflict reports embed "X remaining still open" phrasing. The extractor parses the completion quantity but misses the remaining count → progress_percent and quantity fields under-populated. Gold labels are correct.

3. **`test_hard` no_match discipline errors**: 8/10 no_match reports about Pipe Rack South get discipline wrong (pred=Piping for Civil/Instrumentation/Electrical/Structural activities). The extractor infers discipline from action verbs, and DISCIPLINE_ALIASES maps common shorthands broadly. These are genuine extractor false positives on unscheduled locations — the gold labels (verified absent from schedule) are correct.

4. **Regression set has no candidate pools**: `metrics_regression.json` shows 0% pool recall — the frozen regression set predates the candidate-pool schema. Extraction metrics still run at 1.0, but pool-quality cannot be measured. Documented in LIMITATIONS.md §2.

5. **Progress percent on `all`**: 0.922 — 8 reports with `partial`/`conflict` phrasing where the extractor can't always reconstruct the exact percentage from fragmented quantity text.

---

## Data Integrity

- 0 unresolved placeholders (`{size}`, `{id}`, etc.) in any text or gold
- 0 duplicated phrases in report text, gold evidence, or schedule names
- Candidate pool size: avg 8.96 (correct + 3 hard negatives + 5 unrelated, shuffled, capped at 10)
- no_match cases verified against schedule (no plausible activity exists at unscheduled locations)
- All 1,161 gold_match event_ids match gold_extraction event_ids
- Every matchable event's gold activity is present in its candidate pool
