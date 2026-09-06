"""Threshold sweep for the matcher: load the model once, score once, then
re-decide across threshold grids in memory.

    python -m matching.sweep_thresholds
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matching.matcher as M  # noqa: E402
from matching.evaluate_matcher import load_event_bodies  # noqa: E402
from matching.matcher import Matcher  # noqa: E402
from matching.normalize import normalize_event  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    gold = [json.loads(l) for l in
            open(ROOT / "data" / "labels" / "gold_matches.jsonl", encoding="utf-8")]
    bodies = load_event_bodies(ROOT / "data" / "labels" / "gold_extractions.jsonl")

    matcher = Matcher()
    scored = []  # (gold_row, ev, candidates)
    for g in gold:
        ev = normalize_event(bodies.get(g["event_id"], {"event_id": g["event_id"]}))
        if getattr(matcher, "restrict_to_pool", False) and g.get("candidate_pool"):
            pass
        res = matcher.match(bodies.get(g["event_id"], {"event_id": g["event_id"]}),
                            candidate_pool=g.get("candidate_pool"))
        scored.append((g, res.event_norm, res.candidates))

    print(f"scored {len(scored)} events; sweeping thresholds...")
    best = None
    for floor, th, mg in itertools.product(
        (0.42, 0.46),
        (0.52, 0.54, 0.56, 0.58, 0.60, 0.62),
        (0.01, 0.02),
    ):
        M.NO_MATCH_FLOOR = floor
        M.AUTO_THRESHOLD = th
        M.AUTO_MARGIN = mg
        ok = auto_p_n = auto_p_d = 0
        for g, ev, cands in scored:
            dec, aid, _conf, _r = matcher._decide(ev, cands)
            gd = g["decision"]
            if gd == "auto_match":
                good = dec == "auto_match" and aid == g["schedule_activity_id"]
            else:
                good = dec == gd
            ok += good
            if dec == "auto_match":
                auto_p_d += 1
                auto_p_n += aid == g["schedule_activity_id"]
        acc = ok / len(scored)
        prec = auto_p_n / max(1, auto_p_d)
        nm_ok = sum(1 for g, _e, _c in scored if g['decision'] == 'no_match')
        if best is None or (acc, prec) > (best[0], best[1]):
            best = (acc, prec, floor, th, mg)
        print(f"floor={floor:.2f} th={th:.2f} mg={mg:.2f} "
              f"-> acc={acc:.4f} auto_prec={prec:.4f}")

    print("\nBEST:", best)


if __name__ == "__main__":
    main()
