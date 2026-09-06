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
```

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
  report date (`relative_date_resolved`); unparseable times are nulled + review
- **Overnight rollover**: end < start rolls the end to the next day (capped at
  a 1-day gap; larger gaps are unsafe and nulled)
- **Action canonicalization**: status words ("done") are rejected; alias
  paraphrases map to canonical work types via `vocab.ACTION_ALIASES`;
  unrecognized actions fall back to the description, else null + review
- **Location canonicalization + recovery**: shorthand maps to canonical names
  ("u-300" -> "Unit 300"); an omitted location is recovered only when exactly
  one known location occurs in the text
- **Progress derivation** (gold conventions): completed+affirmed -> 100;
  quantity present -> round(completed/total*100); quantity with completed >
  total is nulled + review
- **Negation/uncertainty guards**: negation never coexists with a completion
  status; uncertainty always sets needs_review

## Safety properties

- Missing information stays `null`; never guessed (validator forces review
  when a status can't be resolved)
- Negation never becomes completion (validator nulls status/progress on
  `assertion=negated` + completion status)
- Uncertainty stays `uncertain` and always sets `needs_review=true`
- Evidence must exist in the raw report text; else `unsupported_evidence`
  warning + review (closed warning vocabulary in `vocab.py`)
- The extractor never emits `schedule_activity_id` and never sees gold labels
  at inference (PDF section 16)
