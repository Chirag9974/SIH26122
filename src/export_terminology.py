"""Emit the terminology data files the spec's section 13 layout expects.

vocab.py is the single source of truth; these CSVs are its published form for
the backend team and for manual review by planners.
"""
from __future__ import annotations

import csv
from pathlib import Path

from vocab import (ACTION_ALIASES, DISCIPLINE_ALIASES, LOCATION_ALIASES,
                   STATUS_CUES, WARNINGS)

OUT = Path(__file__).resolve().parents[1] / "data" / "terminology"


def write(name: str, header: list[str], rows: list[tuple]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"{path}: {len(rows)} rows")


def main() -> None:
    aliases = [(kind, canon, surface)
               for kind, table in (("action", ACTION_ALIASES),
                                   ("discipline", DISCIPLINE_ALIASES),
                                   ("location", LOCATION_ALIASES),
                                   ("status", STATUS_CUES))
               for canon, surfaces in table.items()
               for surface in surfaces]
    write("aliases.csv", ["kind", "canonical", "surface_form"], aliases)

    # abbreviations = the short surface forms only (<= 6 chars, no spaces)
    abbrev = [(k, c, s) for k, c, s in aliases if len(s) <= 6 and " " not in s]
    write("abbreviations.csv", ["kind", "canonical", "abbreviation"], abbrev)

    write("warnings.csv", ["warning_code"], [(w,) for w in WARNINGS])


if __name__ == "__main__":
    main()
