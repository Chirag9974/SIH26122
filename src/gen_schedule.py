"""Step 1: generate the synthetic L5/L6 schedule master.

Writes data/schedule/schedule_activities.csv (~200 activities).

Design notes:
- Deliberate near-duplicates: same work at Rack A / Rack B, and same work split
  into Part 1 / Part 2 on the same line. Matching must not be artificially easy.
- L5 = parent work package, L6 = executable activity. Reports describe L6 work.
"""
from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

from vocab import DISCIPLINES, PROJECT_ID, SIZES

OUT = Path(__file__).resolve().parents[1] / "data" / "schedule" / "schedule_activities.csv"
SEED = 26122
TARGET = 210
PROJECT_START = date(2026, 9, 1)

FIELDS = [
    "project_id", "wbs_code", "activity_id", "activity_name", "wbs_level",
    "discipline", "location", "line_number", "equipment_tag",
    "planned_start", "planned_finish", "unit", "planned_quantity", "predecessor_ids",
]


def _ref(rng: random.Random, action: str) -> tuple[str, str | None, str | None]:
    """Return (ref token, line_number, equipment_tag) for a work item."""
    if action in {"erection", "welding", "fit-up", "hydrotest", "painting"}:
        line = str(rng.choice([10, 12, 14, 16, 20, 22, 24, 30, 32, 40]))
        return f"{line}-{rng.randint(1, 9):02d}", line, None
    if action in {"cable pulling", "termination"}:
        tag = f"C-{rng.randint(100, 899)}"
        return tag, None, tag
    if action in {"installation", "alignment", "grouting", "loop check", "calibration", "tubing"}:
        prefix = rng.choice(["P", "V", "PT", "FT", "LT", "TK"])
        tag = f"{prefix}-{rng.randint(101, 499)}"
        return tag, None, tag
    if action == "earthing":
        return f"E-{rng.randint(1, 12):02d}", None, None
    tag = f"F-{rng.randint(101, 499)}"
    return tag, None, tag


def _workday(d: date) -> date:
    """Skip Sundays -- oil & gas sites typically work 6 days."""
    return d + timedelta(days=1) if d.weekday() == 6 else d


def generate() -> list[dict]:
    rng = random.Random(SEED)
    rows: list[dict] = []
    seq = {code: 0 for code in (d["code"] for d in DISCIPLINES.values())}
    wbs_seq = 0

    disc_names = list(DISCIPLINES)
    while len(rows) < TARGET:
        disc = disc_names[len(rows) % len(disc_names)]
        meta = DISCIPLINES[disc]
        code = meta["code"]
        work = rng.choice(meta["works"])

        wbs_seq += 1
        wbs_l5 = f"{code}.{wbs_seq:03d}"
        ref, line, tag = _ref(rng, work["action"])
        size = rng.choice(SIZES)

        # L5 parent work package
        l5_start = _workday(PROJECT_START + timedelta(days=rng.randint(0, 150)))
        l5_finish = _workday(l5_start + timedelta(days=rng.randint(6, 20)))
        seq[code] += 1
        rows.append({
            "project_id": PROJECT_ID,
            "wbs_code": wbs_l5,
            "activity_id": f"{code}-L5-{seq[code]:04d}",
            "activity_name": f"{work['verb']} {work['obj'].format(size=size, ref=ref)} "
                             f"- Work Package",
            "wbs_level": "L5",
            "discipline": disc,
            "location": "",
            "line_number": line or "",
            "equipment_tag": tag or "",
            "planned_start": l5_start.isoformat(),
            "planned_finish": l5_finish.isoformat(),
            "unit": "",
            "planned_quantity": "",
            "predecessor_ids": "",
        })
        parent_id = rows[-1]["activity_id"]

        # 1-3 L6 children. Near-duplicate strategy: multiple locations, or Part 1/2.
        n_child = rng.choices([1, 2, 3], weights=[4, 4, 2])[0]
        locs = rng.sample(meta["locations"], k=min(n_child, len(meta["locations"])))
        split = n_child > 1 and rng.random() < 0.35  # Part 1/Part 2 on same location
        if split:
            locs = [locs[0]] * n_child

        prev_id = ""
        cursor = l5_start
        for i, loc in enumerate(locs):
            seq[code] += 1
            lo, hi = work["qty"]
            qty = rng.randint(lo, hi)
            dur = rng.randint(1, 5)
            start = _workday(cursor)
            finish = _workday(start + timedelta(days=dur - 1))
            cursor = finish + timedelta(days=1)

            name = f"{work['verb']} {work['obj'].format(size=size, ref=ref)}"
            if split:
                name += f" - Part {i + 1}"
            name += f" at {loc}"

            rows.append({
                "project_id": PROJECT_ID,
                "wbs_code": f"{wbs_l5}.{i + 1:02d}",
                "activity_id": f"{code}-L6-{seq[code]:04d}",
                "activity_name": name,
                "wbs_level": "L6",
                "discipline": disc,
                "location": loc,
                "line_number": line or "",
                "equipment_tag": tag or "",
                "planned_start": start.isoformat(),
                "planned_finish": finish.isoformat(),
                "unit": work["unit"],
                "planned_quantity": qty,
                "predecessor_ids": prev_id or parent_id,
            })
            prev_id = rows[-1]["activity_id"]
            if len(rows) >= TARGET:
                break

    return rows


def main() -> None:
    rows = generate()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    l6 = [r for r in rows if r["wbs_level"] == "L6"]
    print(f"{OUT}: {len(rows)} activities ({len(l6)} L6, {len(rows) - len(l6)} L5)")

    # near-duplicate audit: how many L6 share (discipline, action-ish name head, location)
    from collections import Counter
    heads = Counter((r["discipline"], r["activity_name"].split(" at ")[0].split(" - Part")[0])
                    for r in l6)
    dupes = sum(1 for c in heads.values() if c > 1)
    print(f"near-duplicate name groups (same work, >1 activity): {dupes}")


if __name__ == "__main__":
    main()
