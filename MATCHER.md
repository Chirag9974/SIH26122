# Schedule Matcher

Maps an extracted event (frozen extractor JSON) to its L5/L6 schedule
activity, returning a **decision** — never just an id.

```python
from matching.matcher import match_event
result = match_event(event_dict)                     # -> decision dict
result = match_event(event, candidate_pool=[...])    # optional pool hint
```

Output: `decision` (`auto_match | human_review | no_match`),
`schedule_activity_id`, `matched_activity`, `confidence`, top-5 `candidates`
(each with per-signal scores), and `reasons`. The extractor JSON is never
modified — matching is a separate stage.

## Architecture (src/matching/)

| File | Role |
|---|---|
| `normalize.py` | Canonical forms via the shared vocab alias tables: action/location/discipline (`rb-c` → `Rack C`, `erected` → `erection`), line/equipment canonicalization, plus name-embedded field recovery (`Line 40-09`, tag `LT-140` inside activity names) |
| `embeddings.py` | `ScheduleIndex`: sentence-transformers embeddings of every activity in **4 views** (name / name+location / name+line / name+both) — a metadata-bearing query matches its best-fitting view; numpy cosine (301 activities, no ANN needed); dependency-free token-set lexical fallback |
| `matcher.py` | Hybrid weighted ranking + conservative decision layer, `match_event` stable API |

## Signals and weights

semantic 0.50 · lexical 0.20 · action 0.15 · location 0.05 · discipline 0.04 ·
line 0.03 · equipment 0.02 · L6-hierarchy −0.15 (L5 "Work Package" rollups are
parents, not leaves — demoted).

**Bonuses/penalties**: unique-line +0.10 (only one activity carries the
event's line), line+discipline corroboration +0.06, exact location +0.04;
location mismatch −0.08, discipline mismatch −0.06.

## Decision rules (never force weak evidence)

1. **Location gate**: event names a location not in the schedule's location
   inventory → `no_match` (catches all synthetic no-match cases deterministically).
2. **Floor**: best score < 0.42 → `no_match`.
3. **Twin guard**: several candidates tie on action+location within 0.03
   (e.g. "Install Pump FT-346 - Part 1 vs Part 2", event has no tag) →
   `human_review` — the event cannot discriminate, so a pick would be guessing.
4. **Auto bar**: score ≥ 0.56 and sibling-margin ≥ 0.01 → `auto_match`.
5. Otherwise `human_review` (top pick still surfaced with confidence).

Operating point chosen by `sweep_thresholds.py` over floor × threshold ×
margin on the gold labels.

## Model

`sentence-transformers/all-MiniLM-L6-v2` (CPU-friendly, 80 MB; torch CPU
build is sufficient — embeddings are one-time per process).

## Run

```bash
cd src
python -m matching.evaluate_matcher --failures 15   # full eval + failures
python -m matching.sweep_thresholds                 # recalibrate operating point
python -m tests.test_matcher                        # 22 assertions
```

Quick smoke from the repo root:

```bash
cd src && python -c "
import sys; sys.path.insert(0, '.')
from matching.matcher import match_event
import json
ev = {'activity': {'action': 'erection', 'description': 'erection of 18in spool'},
      'context': {'discipline': 'Piping', 'location': 'Rack C', 'line_number': '40-09'}}
print(json.dumps(match_event(ev), indent=2))"
```

## Dev-set metrics (1,161 gold match rows)

| Metric | Value |
|---|---|
| Recall@1 / @3 / @5 | 0.547 / 0.895 / 0.973 |
| Final match accuracy | 0.357 |
| Auto-match precision | 0.931 |
| Auto-match recall | 0.550 |
| No-match accuracy | **1.000** (30/30, location gate) |
| Human-review recall | 0.750 |

## Known limits (top remaining failures)

1. **Underspecified events vs part-twins** (~45% of gold auto rows): the
   synthetic generator strips identifiers ("pump installation" with no tag)
   while the schedule splits work into "Part 1/Part 2" or per-tag activities.
   When several candidates tie on every signal the event carries, the matcher
   correctly refuses to guess → `human_review`. Lifting this needs richer
   reports (or accepting lower precision), not a better scorer.
2. **Sibling lines with no line number in the event** ("line hydrotest at
   Pipe Rack North" — 3 hydrotest lines at that location): top pick is often
   a sibling; honestly flagged for review.
3. Semantic-only losses (gold ranks top-5 but a paraphrase sibling edges it):
   a cross-encoder reranker (e.g. ms-marco-MiniLM) over the top-10 would
   likely add several points of R@1 — the natural next step.

## Layout note

`src/` is organized by work: `common/` (schema, vocab), `extraction/`
(baseline + LLM extractor, prompt, validators), `matching/` (this stage),
`evaluation/`, `generation/`, `quality/`, `tests/`. `run_all.py` runs the
full data + baseline pipeline via `python -m`.
