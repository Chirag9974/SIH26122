"""Evaluation harness for the LLM extractor (PDF section 12).

Metrics: relevance P/R/F1, event P/R/F1, field accuracy, datetime
accuracy, evidence coverage, needs-review accuracy, and the primary
reliability metric: unsafe-extraction rate (confident wrong facts where
the safe behavior was null or review).

Usage:
  python eval_extractor.py --split dev --sample 30
  python eval_extractor.py --split dev            # full dev (resumable cache)
  python eval_extractor.py --paired ../data/evaluation/test_set.jsonl
  python eval_extractor.py --split dev --no-cache  # force fresh LLM calls
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from eval import align, _jsonl
from extractor_llm import extract

ROOT = Path(__file__).resolve().parents[1]

CONF_FIELDS = [
    ("activity.action", lambda e: e["activity"]["action"]),
    ("execution.status", lambda e: e["execution"]["status"]),
    ("execution.progress_percent",
     lambda e: e["execution"]["progress_percent"]),
    ("time.start", lambda e: e["time"]["start"]),
    ("time.end", lambda e: e["time"]["end"]),
    ("context.location", lambda e: e["context"]["location"]),
    ("context.line_number", lambda e: e["context"]["line_number"]),
    ("context.equipment", lambda e: e["context"]["equipment"]),
    ("quantity.completed", lambda e: e["quantity"]["completed"]),
    ("quantity.total", lambda e: e["quantity"]["total"]),
]

NULLABLE_FIELDS = {name for name, _ in CONF_FIELDS} | {"activity.action"}


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}


def _eq_num(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-6
    return a == b


def evaluate(pred: dict, gold: dict, report: dict) -> tuple[dict, list]:
    """Per-report metric contributions + concrete error examples."""
    m: dict = defaultdict(Counter)
    examples: list[dict] = []
    rep_id = gold["report_id"]
    kind = gold.get("case_kind", "?")

    # relevance
    g_rel, p_rel = gold["relevance"]["is_relevant"], pred["relevance"]["is_relevant"]
    m["rel"]["tp" if g_rel and p_rel else "fp" if p_rel else
             "fn" if g_rel else "tn"] += 1

    fallback = pred.get("_meta", {}).get("fallback", False)
    if fallback:
        m["flags"]["fallback_reports"] += 1

    pairs = align(gold["events"], pred["events"])
    for ge, pe in pairs:
        m["kind:" + kind]["tp" if ge and pe else "fn" if ge else "fp"] += 1
        if ge and pe:
            m["ev"]["tp"] += 1
        elif ge:
            m["ev"]["fn"] += 1
            m["miss_kind:" + kind]["fn"] += 1
            examples.append({"report_id": rep_id, "kind": kind,
                             "issue": "event_missed", "field": None,
                             "gold": ge["activity"]["action"],
                             "pred": None,
                             "text": report["raw_text"][:110]})
        else:
            m["ev"]["fp"] += 1
            m["miss_kind:" + kind]["fp"] += 1
            examples.append({"report_id": rep_id, "kind": kind,
                             "issue": "event_extra", "field": None,
                             "gold": None,
                             "pred": pe["activity"]["action"],
                             "text": report["raw_text"][:110]})

        if ge and pe:
            for name, get in CONF_FIELDS:
                gv, pv = get(ge), get(pe)
                ok = _eq_num(gv, pv)
                m["field:" + name]["ok" if ok else "bad"] += 1
                if not ok:
                    m["fielderr:" + name + ":" + kind]["n"] += 1
                    examples.append({"report_id": rep_id, "kind": kind,
                                     "issue": "field_mismatch", "field": name,
                                     "gold": gv, "pred": pv,
                                     "text": report["raw_text"][:110]})

            # unsafe extraction: confident wrong value where gold is null
            for name, get in CONF_FIELDS:
                gv, pv = get(ge), get(pe)
                if gv is None and pv is not None:
                    conf = pred["events"][0].get("confidence", {})
                    m["unsafe"]["n"] += 1
                    m[f"unsafe_field:{name}"]["n"] += 1

            # needs-review accuracy: hedged/contradicted gold events must
            # carry needs_review, clean affirmed events should not
            g_needs = ("uncertain" == ge["execution"].get("assertion")
                       or "conflicting_report" in ge.get("warnings", []))
            p_needs = bool(pe.get("needs_review"))
            if g_needs:
                m["review"]["tp" if p_needs else "fn"] += 1
            else:
                m["review"]["tn" if not p_needs else "fp"] += 1

            src = pe["evidence"]["source_text"]
            m["evid"]["covered" if src and src.strip(".") in report["raw_text"]
                     else "uncovered"] += 1

    return m, examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "dev", "test", "test_hard"])
    ap.add_argument("--paired", type=Path,
                    help="paired {report, gold} JSONL (edge cases)")
    ap.add_argument("--sample", type=int, default=0,
                    help="only run the first N reports (smoke)")
    ap.add_argument("--model", default="qwen2.5:7b-instruct-q4_K_M")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-examples", type=int, default=14,
                    help="concrete error examples kept in the output")
    args = ap.parse_args()

    if args.paired:
        rows = _jsonl(args.paired)
        items = [(r["report"], r["gold"]) for r in rows]
        split_name = "paired_" + args.paired.stem
    else:
        reports = {r["report_id"]: r
                   for r in _jsonl(ROOT / "data/raw_reports/reports.jsonl")}
        golds = {g["report_id"]: g
                 for g in _jsonl(ROOT / "data/labels/gold_extractions.jsonl")}
        splits = json.loads((ROOT / "data/evaluation/splits.json")
                            .read_text(encoding="utf-8"))
        ids = splits[args.split]
        items = [(reports[i], golds[i]) for i in ids]
        split_name = args.split
    if args.sample:
        items = items[:args.sample]

    print("=== LLM extractor eval:", split_name, len(items), "reports,"
          "model =", args.model, "===", flush=True)

    merged: dict = defaultdict(Counter)
    per_report: list[dict] = []
    examples: list[dict] = []
    t_start = time.time()
    for i, (rep, gold) in enumerate(items, 1):
        t0 = time.time()
        pred = extract(rep, model=args.model, use_cache=not args.no_cache)
        dt = time.time() - t0
        fb = pred.get("_meta", {}).get("fallback", False)
        m, exs = evaluate(pred, gold, rep)
        examples.extend(exs)
        per_report.append({"report_id": rep["report_id"],
                           "case_kind": gold.get("case_kind", "?"),
                           "latency_s": round(dt, 1), "fallback": fb,
                           "events_pred": len(pred["events"]),
                           "events_gold": len(gold["events"])})
        for k, c in m.items():
            merged[k].update(c)
        tag = " FALLBACK" if fb else ""
        if i == 1 or i % 10 == 0 or fb:
            print("  [", i, "/", len(items), "]", rep["report_id"],
                  round(dt, 1), "s" + tag, flush=True)

    wall = time.time() - t_start
    rel, ev = merged["rel"], merged["ev"]
    unsafe = merged["unsafe"]["n"]
    rev = merged["review"]
    metrics = {
        "split": split_name,
        "model": args.model,
        "n_reports": len(items),
        "wall_time_s": round(wall, 1),
        "avg_latency_s": round(wall / max(1, len(items)), 1),
        "relevance": _prf(rel["tp"], rel["fp"], rel["fn"]) | {"tn": rel["tn"]},
        "event_detection": _prf(ev["tp"], ev["fp"], ev["fn"]),
        "field_accuracy": {},
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
        "event_miss_by_kind": {k.split("miss_kind:")[1]: {"fn": c["fn"],
                                  "fp": c["fp"]}
                               for k, c in sorted(merged.items())
                               if k.startswith("miss_kind:")
                               and (c["fn"] or c["fp"])},
        "field_errors_by_kind": {k.split("fielderr:")[1]: c["n"]
                                 for k, c in sorted(merged.items())
                                 if k.startswith("fielderr:")},
        "case_type_metrics": {k.split("kind:", 1)[1]: {
                                  "n_events": c["tp"] + c["fn"],
                                  "event_prf": _prf(c["tp"], c["fp"], c["fn"])}
                              for k, c in sorted(merged.items())
                              if k.startswith("kind:")},
        "per_report": per_report,
        "error_examples": examples[:args.max_examples],
        "n_error_examples": len(examples),
    }
    for k, c in sorted(merged.items()):
        if k.startswith("field:"):
            metrics["field_accuracy"][k.split("field:")[1]] = round(
                c["ok"] / max(1, c["ok"] + c["bad"]), 4)

    printable = {k: v for k, v in metrics.items() if k != "per_report"}
    print(json.dumps(printable, indent=2))
    out = args.out or (ROOT / "data/evaluation"
                       / ("metrics_llm_" + split_name + ".json"))
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
