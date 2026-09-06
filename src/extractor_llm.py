"""LLM extractor: Ollama structured output + deterministic controls.

Architecture (PDF section 3):
  raw report -> LLM structured output (JSON schema constrained)
             -> normalize -> deterministic validators -> Pydantic
             -> safe fallback when not safely resolvable

Public API matches the deterministic baseline contract:
  extract(report, model=...) -> {"document": ..., "relevance": ..., "events": [...]}

Safety properties:
  - never crashes on model output: schema failure -> repair/retry -> fallback
  - fallback is conservative: events=[], flagged for review, never cached
  - never emits schedule_activity_id; never sees gold labels
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests

from prompt import SYSTEM_PROMPT, repair_prompt, user_prompt
from schema import Extraction, flat_schema, parse_extraction
from validators import validate_extraction

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "evaluation"
OLLAMA_URL = "http://127.0.0.1:11434"
RETRY_LIMIT = 2
TIMEOUT_S = 300


def _cache_path(model: str) -> Path:
    safe = model.replace(":", "_").replace("/", "_")
    return CACHE_DIR / f"llm_cache_{safe}.jsonl"


def _cache_key(report: dict, model: str) -> str:
    basis = json.dumps({"t": report["raw_text"],
                        "d": report.get("report_date"),
                        "m": model}, sort_keys=True)
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


class ResponseCache:
    """Append-only per-report cache so evals are resumable and re-runs free."""

    def __init__(self, model: str, use_cache: bool = True):
        self.path = _cache_path(model)
        self.enabled = use_cache
        self.hits = 0
        self._data: dict[str, dict] = {}
        if self.enabled and self.path.exists():
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        row = json.loads(line)
                        self._data[row["key"]] = row

    def get(self, key: str):
        if not self.enabled:
            return None
        row = self._data.get(key)
        if row:
            self.hits += 1
        return row

    def put(self, key: str, payload: dict) -> None:
        if not self.enabled:
            return
        self._data[key] = payload
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, **payload},
                                ensure_ascii=False) + chr(10))


def _chat(model: str, messages: list[dict], *, schema_json: dict | None,
          options: dict, stage: int = 0) -> str:
    """One Ollama /api/chat call; returns assistant content text."""
    body = {"model": model, "messages": messages, "stream": False,
            "options": options}
    body.update(_think_flag(model))
    if schema_json is not None:
        body["format"] = schema_json
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=body,
                      timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()["message"]["content"]


def _think_flag(model: str) -> dict:
    """qwen3 hybrid-thinking off-switch, merged into the request body
    (no-op for qwen2.5 models)."""
    if model.startswith("qwen3"):
        return {"think": False}
    return {}


def _json_loads_loose(text: str):
    """Parse model output tolerantly: raw JSON, fenced, or brace block."""
    text = text.strip()
    if text.startswith("```"):
        for p in text.split("```"):
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                text = p
                break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _fallback(report: dict, reason: str) -> dict:
    """Conservative output when the model cannot produce a valid extraction.

    PDF section 3: needs_review when not safely resolvable. Never guesses.
    """
    return {
        "document": {"report_id": report.get("report_id"),
                     "source_type": report.get("source_type", "daily_report"),
                     "report_date": report.get("report_date"),
                     "discipline": report.get("discipline"),
                     "raw_text": report["raw_text"]},
        "relevance": {"is_relevant": True, "confidence": 0.3,
                      "reason": f"extraction not safely resolvable: {reason}"},
        "events": [],
        "_meta": {"fallback": True, "fallback_reason": reason},
    }


def extract(report: dict, model: str = "qwen2.5:7b-instruct-q4_K_M", *,
            use_cache: bool = True, client=None) -> dict:
    """Extract execution events from one field report via the LLM.

    client: callable(model, messages, schema_json, options, stage)
      -> str content. Tests inject a mock here; production uses Ollama.
    """
    chat = client or _chat
    key = _cache_key(report, model)
    cache = ResponseCache(model, use_cache)
    cached = cache.get(key)

    if cached is not None:
        model_out = cached["doc"]
        meta = {"cached": True, "fallback": False, "validator_issues": []}
    else:
        schema_json = flat_schema()
        options = {"temperature": 0, "num_ctx": 8192, "num_predict": 2048}
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt(report)}]
        model_out, err = None, "?"
        for stage in range(RETRY_LIMIT + 1):
            try:
                content = chat(model, messages, schema_json=schema_json,
                               options=options, stage=stage)
            except TypeError:
                # client without a stage kwarg: retry loop is pointless
                content = chat(model, messages, schema_json=schema_json,
                               options=options)
                break
            try:
                parsed = _json_loads_loose(content)
            except json.JSONDecodeError as e:
                err = f"not valid JSON: {e}"
                continue
            candidate, verr = parse_extraction(
                parsed if isinstance(parsed, dict) else {})
            if verr:
                err = verr
                if stage < RETRY_LIMIT:
                    messages = messages[:1] + [{
                        "role": "user",
                        "content": repair_prompt(report, content, err)}]
                continue
            model_out = candidate.to_dict()
            break
        if model_out is None:
            fb = _fallback(report, f"schema failure after retries: {err}")
            return fb  # fallbacks are never cached: retry on the next run
        cache.put(key, {"raw": "", "doc": model_out})
        meta = {"cached": False, "fallback": False, "validator_issues": []}

    final, issues = validate_extraction(model_out, report)
    try:
        Extraction(**final)
    except Exception as e:
        return _fallback(report, f"final validation failed: {e}")
    meta["validator_issues"] = issues
    return {"document": {"report_id": report.get("report_id"),
                         "source_type": report.get("source_type",
                                                    "daily_report"),
                         "report_date": report.get("report_date"),
                         "discipline": report.get("discipline"),
                         "raw_text": report["raw_text"]},
            "relevance": final["relevance"],
            "events": final["events"],
            "_meta": meta}
