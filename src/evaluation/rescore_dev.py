"""Re-score cached LLM generations with the CURRENT post-processing.

Validator/prompt fixes change results but not raw generations, so the
cache stays valid. This script replays every cached dev report through
extract() (no new LLM calls) and rewrites metrics_llm_dev.json.
Reports without a cache entry are skipped and counted as pending.

Run:  python src/rescore_dev.py
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from evaluation.eval import _jsonl, align
from evaluation.eval_extractor import CONF_FIELDS, _eq_num, _prf, evaluate
from extraction.extractor_llm import _cache_key, ResponseCache, extract

ROOT = Path(__file__).resolve().parents[2]
MODEL = "qwen2.5:7b-instruct-q4_K_M"


def main() -> None:
    cache = ResponseCache(MODEL, use_cache=True)
    reports = {r["report_id"]: r
               for r in _jsonl(ROOT / "data/raw_reports/reports.jsonl")}
    golds = {g["report_id"]: g
             for g in _jsonl(ROOT / "data/labels/gold_extractions.jsonl")}
    splits = json.loads((ROOT / "data/evaluation/splits.json")
                        .read_text(encoding="utf-8"))
    ids = splits["dev"]

    merged: dict = defaultdict(Counter)
    examples: list[dict] = []
    per_report: list[dict] = []
    pending = 0
    t0 = time.time()
    for rid in ids:
        rep = reports[rid]
        if _cache_key(rep, MODEL) not in cache._data:
            pending += 1
            continue
        pred = extract(rep, model=MODEL, use_cache=True)
        gold = golds[rid]
        m, exs = evaluate(pred, gold, rep)
        for k, c in m.items():
            merged[k].update(c)
        examples.extend(exs)
        per_report.append({"report_id": rid,
                           "case_kind": gold.get("case_kind", "?"),
                           "fallback": pred.get("_meta", {})
                                       .get("fallback", False),
                           "events_pred": len(pred["events"]),
                           "events_gold": len(gold["events"])})

    ev, rel, rev = merged["ev"], merged["rel"], merged["review"]
    unsafe = merged["unsafe"]["n"]
    metrics = {
        "split": "dev (rescored from cache)",
        "model": MODEL,
        "n_reports": len(ids) - pending,
        "n_pending": pending,
        "rescore_seconds": round(time.time() - t0, 1),
        "relevance": _prf(rel["tp"], rel["fp"], rel["fn"])
                     | {"tn": rel["tn"]},
        "event_detection": _prf(ev["tp"], ev["fp"], ev["fn"]),
        "field_accuracy": {k.split("field:")[1]:
                           round(c["ok"] / max(1, c["ok"] + c["bad"]), 4)
                           for k, c in sorted(merged.items())
                           if k.startswith("field:")},
        "unsafe_extraction": {
            "count": unsafe,
            "rate_per_matched_event": round(unsafe / max(1, ev["tp"]), 4),
            "by_field": {k.split("unsafe_field:")[1]: c["n"]
                         for k, c in sorted(merged.items())
                         if k.startswith("unsafe_field:")},
        },
        "needs_review": {
            "recall": round(rev["tp"] / max(1, rev["tp"] + rev["fn"]), 4),
            "specificity": round(rev["tn"] / max(1, rev["tn"] + rev["fp"]), 4),
        },
        "evidence_coverage": round(merged["evid"]["covered"]
                                   / max(1, sum(merged["evid"].values())), 4),
        "fallback_reports": merged["flags"]["fallback_reports"],
        "case_type_metrics": {k.split("kind:", 1)[1]: {
                                  "n_events": c["tp"] + c["fn"],
                                  "event_prf": _prf(c["tp"], c["fp"],
                                                    c["fn"])}
                              for k, c in sorted(merged.items())
                              if k.startswith("kind:")},
        "field_errors_by_kind": {k.split("fielderr:")[1]: c["n"]
                                 for k, c in sorted(merged.items())
                                 if k.startswith("fielderr:")},
        "event_miss_by_kind": {k.split("miss_kind:")[1]:
                               {"fn": c["fn"], "fp": c["fp"]}
                               for k, c in sorted(merged.items())
                               if k.startswith("miss_kind:")
                               and (c["fn"] or c["fp"])},
        "n_error_examples": len(examples),
        "error_examples": examples[:14],
        "per_report": per_report,
    }
    out = ROOT / "data/evaluation/metrics_llm_dev.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    printable = {k: v for k, v in metrics.items()
                 if k not in ("per_report", "error_examples")}
    print(json.dumps(printable, indent=2))
    print("wrote", out, f"({pending} pending reports had no cache entry)")


if __name__ == "__main__":
    main()
