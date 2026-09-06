# LLM Extractor v1 (SIH26122)

Implementation of `SIH26122_Extractor_Training_and_Build_Pipeline.pdf`: an LLM
extractor wrapped in deterministic controls. This is the model-based extractor
that complements the deterministic baseline in `src/extractor.py` — same public
contract, different engine.

## Architecture (PDF section 3)

```
raw report
  -> Ollama /api/chat, JSON-schema constrained (format=flat schema)
  -> tolerant JSON parse -> normalize_nested (repair model slips)
  -> parse_extraction (Pydantic v2, frozen contract)
  -> repair/retry (max 2, validation error fed back)
  -> validators.py deterministic checks (PDF section 10)
  -> final Pydantic validation
  -> conservative fallback (events=[], needs_review) if not safely resolvable
```

Fallbacks are never cached — a re-run retries them.

## Files

| File | Purpose |
|------|---------|
| `src/schema.py` | Pydantic contract + flat JSON schema for Ollama + normalizer |
| `src/prompt.py` | System prompt (PDF section 6 rules) + user/repair prompts |
| `src/extractor_llm.py` | Ollama client, retry loop, cache, `extract()` |
| `src/validators.py` | Deterministic consistency checks + safe repairs |
| `src/eval_extractor.py` | PDF section 12 metrics incl. unsafe-extraction rate |
| `src/error_analysis.py` | PDF section 13 failure taxonomy + recommendations |
| `src/test_extractor_llm.py` | Offline invariants (mocked LLM, no server needed) |

## Stable API (MVP)

```python
import sys
sys.path.insert(0, "src")
from extractor_llm import extract_report

out = extract_report(
    "Aaj welding kaam Rack A pe ho gaya 08:00 se 16:00 tak.",
    metadata={"report_date": "2026-09-01", "discipline": None},
)
# -> frozen extraction JSON; out["_meta"]["engine"] is "llm" or "baseline"
```

Chain: LLM structured output -> Pydantic -> deterministic validators. If the
LLM fails validation it retries **once**; if it still fails (or Ollama is
down), the deterministic baseline extractor (`src/extractor.py`) fills the
same frozen shape with `needs_review=true` and a `baseline_fallback` warning.
The API never raises to the caller.

## Run

```bash
# 0. one-time: model tag must match the local Ollama model
ollama list                     # e.g. qwen2.5:7b-instruct-q4_K_M

# 1. start the server (skip if Ollama is already running)
ollama serve

# 2. offline invariants (no model needed)
python src/test_extractor_llm.py

# 3. smoke: 30 dev reports, resumable per-report cache
python src/eval_extractor.py --split dev --sample 30

# 4. full split eval (resumes from cache; ~30-45 s per uncached report)
python src/eval_extractor.py --split dev

# 5. error analysis over the result
python src/error_analysis.py data/evaluation/metrics_llm_dev.json

# 6. re-score from cache without new LLM calls (after validator/prompt fixes)
python src/eval_extractor.py --split dev --no-cache   # NOT this one;
# use the rescore script instead (cache-only, seconds):
python src/rescore_dev.py
```

**Re-scoring note:** validator and prompt fixes do not invalidate the
generation cache (the same report still maps to the same raw LLM output);
only the post-processing changes. `rescore_dev.py` replays the cache through
the current validators and rewrites `metrics_llm_dev.json` in seconds.

Other splits: `--split train|test|test_hard`, or `--paired
data/evaluation/test_set.jsonl` for the hand-written edge cases. Force fresh
LLM calls with `--no-cache`. Results land in
`data/evaluation/metrics_llm_<split>.json`; raw generations cache to
`data/evaluation/llm_cache_<model>.jsonl`.

## Model substitution note

The PDF names **Qwen3-8B** as the first model to benchmark. This build ran with
the locally available **qwen2.5:7b-instruct-q4_K_M** (confirmed with the user;
zero download). Switching is one flag:

```bash
ollama pull qwen3:8b
python src/eval_extractor.py --split dev --model qwen3:8b
```

`extractor_llm._think_flag` disables qwen3's hybrid thinking mode automatically
(`think: false`) so output stays deterministic; qwen2.5 ignores it.

## Metrics (PDF section 12)

- Relevance / event precision-recall-F1
- Field accuracy per field (action, status, progress, time, context, quantity)
- Evidence coverage (claim text traceable to the report)
- Needs-review accuracy (recall: hedged/contradicted gold events flagged;
  specificity: clean events not flagged)
- **Unsafe-extraction rate** (primary): predicted a value where the gold label
  is null — a confident wrong fact is worse than a cautious null

## Deterministic repair layer (validators.py)

Between the raw generation and the final JSON, the validator deterministically
repairs or flags:

- **Timestamp resolution**: bare clock times ("08:00") resolve onto the
  report date (`relative_date_resolved`); unparseable times are nulled + review;
  glued junk timestamps ("...-log24T11:30:00") are salvaged when the clock part
  is clean; unsupported midnight-exact starts (a model placeholder) are dropped
- **Overnight rollover**: end < start rolls the end to the next day (capped at
  a 1-day gap; larger gaps are unsafe and nulled)
- **Action canonicalization**: status words ("done") are rejected; alias
  paraphrases map to canonical work types via `vocab.ACTION_ALIASES`;
  unrecognized actions fall back to the description, else null + review;
  a canonical action the text never names is swapped for the one it does
  (single-event docs)
- **Location + line-number canonicalization/recovery**: shorthand maps to
  canonical names ("u-300" -> "Unit 300"); glued ids ("log24" -> "24"); an
  omitted location/line is recovered only when exactly one occurs in the text;
  pseudo-values ("synced", "in Line 18") are nulled
- **Progress derivation** (gold conventions): completed+affirmed -> 100;
  quantity present -> round(completed/total*100); quantity with completed >
  total is nulled + review
- **Status inference/recovery**: partial quantity implies in_progress; a
  status-cue verb in the event's own clause ("have been completed", "started
  another spool") recovers a dropped status - clause-scoped so sibling
  clauses' cues do not leak in
- **Assertion repair** (hedge/negation judgement): hedging a clean fact is
  downgraded to affirmed; hedged-field scopes ("around 16:30") do not make the
  status uncertain; a real hedge in the sentence keeps uncertain + review;
  negation is asserted only when a cue sits within 3 tokens of the event's own
  activity term and not next to a reporting verb ("confirm nahi kiya" negates
  the report, not the work); "X except Y" exclusions are not negations
- **Conflicting-report coherence**: the model stamps `conflicting_report` on
  its own hedging; the validator rejects the warning when the final assertion
  is coherent, the event's clause names at most one activity, and the sentence
  contains no activity-less unfinished-work clause ("two joints unfinished")
- **Review flags are deterministic**: the model's self-declared
  needs_review is discarded; the validator's rules are the sole authority

## Safety properties

- Missing information stays `null`; never guessed (validator forces review
  when a status can't be resolved)
- Negation never becomes completion (validator nulls status/progress on
  `assertion=negated` + completion status)
- Uncertainty stays `uncertain` and always sets `needs_review=true` (and a
  hedge in the text always survives the downgrade repair)
- Evidence must exist in the raw report text; else `unsupported_evidence`
  warning + review (closed warning vocabulary in `vocab.py`)
- The extractor never emits `schedule_activity_id` and never sees gold labels
  at inference (PDF section 16)
