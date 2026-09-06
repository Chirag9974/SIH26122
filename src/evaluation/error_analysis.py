"""Error-driven development tool (PDF section 13).

Reads an LLM eval metrics file and categorizes failures into the PDF
taxonomy: schema failure / wrong event / wrong field / unsupported guess /
ambiguity. Prints a prioritized summary for the improve stage of the
PDF section 8 loop. Read-only: never touches gold labels.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _field_bucket(name: str) -> str:
    """Map a field name to the PDF section 8 diagnosis group."""
    if name.startswith("execution.status"):
        return "status"
    if name.startswith("execution.progress") or name.startswith("quantity"):
        return "progress"
    if name.startswith("time."):
        return "time"
    if name.startswith("context."):
        return "context"
    return "activity"


def analyze(metrics_path: Path) -> dict:
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    ev = m["event_detection"]
    buckets: Counter = Counter()
    detail: Counter = Counter()

    # wrong events, grouped by case kind
    for kind, c in m.get("event_miss_by_kind", {}).items():
        buckets["wrong_event"] += c["fn"] + c["fp"]
        detail[f"wrong_event:{kind}"] += c["fn"] + c["fp"]

    # wrong fields, grouped into diagnosis buckets
    for combo, n in m.get("field_errors_by_kind", {}).items():
        field = combo.rsplit(":", 1)[0]
        bucket = _field_bucket(field)
        buckets["wrong_field:" + bucket] += n
        detail[f"wrong_field:{field}"] += n

    # unsupported guesses: predicted a value where gold is null
    unsafe = m.get("unsafe_extraction", {})
    for field, n in unsafe.get("by_field", {}).items():
        buckets["unsupported_guess"] += n
        detail[f"unsupported_guess:{field}"] += n

    # pipeline-level failures
    if unsafe.get("count"):
        pass
    if m.get("fallback_reports"):
        buckets["schema_failure_fallback"] += m["fallback_reports"]
    review = m.get("needs_review", {})
    if review.get("recall", 1.0) < 1.0:
        buckets["missed_review"] += int(round(
            (1 - review["recall"]) * 100))  # approx count per 100 events

    n_events = ev["tp"] + ev["fn"]
    return {
        "metrics_file": str(metrics_path),
        "split": m.get("split"),
        "n_reports": m.get("n_reports"),
        "n_gold_events": n_events,
        "buckets": dict(buckets.most_common()),
        "detail": dict(detail.most_common(20)),
        "primary_metric_unsafe_rate": unsafe.get("rate_per_matched_event"),
        "review_recall": review.get("recall"),
        "recommendations": _recommend(buckets, detail),
    }


def _recommend(buckets: Counter, detail: Counter) -> list[str]:
    recs = []
    if buckets.get("unsupported_guess"):
        recs.append("Strengthen the null/abstain rule: the model states values"
                    " the text does not support (PDF section 13).")
    for key, n in buckets.items():
        if key.startswith("wrong_field:time") and n:
            recs.append("Time parsing dominates: add more TIME_PHRASES"
                        " coverage and tighten the ISO timestamp rule.")
        if key.startswith("wrong_field:status") and n:
            recs.append("Status confusion: review negation/suspension cues"
                        " in the prompt and consider case-kind few-shots.")
        if key.startswith("wrong_field:progress") and n:
            recs.append("Progress/quantity errors: enforce completed/total"
                        " derivation in the validator.")
        if key.startswith("wrong_event:") and n >= 5:
            kind = key.split(":", 1)[1]
            recs.append(f"Event misses concentrate on case kind {kind!r}:"
                        " add structural examples of it to development data.")
    if buckets.get("schema_failure_fallback"):
        recs.append("Schema fallbacks occurred: inspect raw model output and"
                    " widen normalize_nested tolerance.")
    return recs or ["No dominant bucket; iterate on the largest detail row."]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", type=Path, nargs="?",
                    default=ROOT / "data/evaluation/metrics_llm_dev.json")
    args = ap.parse_args()
    out = analyze(args.metrics)
    print("=== Error analysis:", out["split"], "===")
    print("reports:", out["n_reports"], " gold events:", out["n_gold_events"])
    print("unsafe-extraction rate (primary):",
          out["primary_metric_unsafe_rate"])
    print("needs-review recall:", out["review_recall"])
    print("failure buckets:")
    for k, v in out["buckets"].items():
        print(f"  {k:<32} {v}")
    print("top detail rows:")
    for k, v in out["detail"].items():
        print(f"  {k:<44} {v}")
    print("recommendations:")
    for r in out["recommendations"]:
        print("  -", r)


if __name__ == "__main__":
    main()
