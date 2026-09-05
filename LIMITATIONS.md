# Limitations & Known Gaps

These are honest constraints of the current benchmark, not bugs to be filed.
Recorded so future work can scope against the real ceiling.

## 1. Extractor is overfit to generator vocabulary
`test_extractor.py` passes 17/17 and extraction F1 is ~1.0 on every split
because `extractor.py` was written from the same alias tables (`vocab.py`)
that `gen_reports.py` uses to render text. The ~0.6-point F1 gap on
`test` (~0.988) vs `all` (~0.994) is only the noisiest 20% of generated
phrasings. A genuinely hard generalization set needs **human-written**
daily reports; that is out of scope for this benchmark's synthetic design.
Upgrade ceiling: add ~150 hand-annotated real reports → measure the gap.

## 2. Regression set has no candidate pools
`data/regression/` is a frozen snapshot from commit `8983e17` and is never
regenerated. Its `gold_matches.jsonl` rows predate the candidate-pool
schema: `candidate_pool` is empty and `decision` is `needs_review`.
`eval.py` degrades gracefully (`.get()`), so regression extraction metrics
still run, but **candidate-pool quality cannot be reported for it**. The
regression set therefore measures extraction recall only, not matching
quality. Fix would require hand-labeling pools — explicitly avoided here.

## 3. Hard negatives are discipline-matched, not location-matched
Pool construction adds 3 hard negatives (same discipline + action, different
location/line) + 5 unrelated. A matcher that ignores *location* still
achieves pool recall 1.0 because every gold activity is in its own pool by
construction. There is no *same-location, different-action* confusable (e.g.
"painting" vs "hydrotest" at the same spool). Upgrade ceiling: extend
`gen_reports.build_ambiguous` / pool sampling to inject that axis.

## 4. No matcher is shipped
This repo deliberately ships **zero** matcher/ranker. `eval.py` reports
candidate-pool *retrieval* quality (pool_recall) only. There is no
no-match-detection accuracy and no final-ranking accuracy — those numbers
require a matcher that does not exist here. Adding one is a separate
feature, not a benchmark-data fix.

## 5. `val` split only exists in regression splits.json
The main `data/evaluation/splits.json` uses `dev` (not `val`). Running
`eval.py --split val` against the main data dir fails with `KeyError`; use
`--split dev`, or `--data-dir ../data/regression --split val`.
