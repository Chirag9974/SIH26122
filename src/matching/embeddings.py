"""Embedding + lexical indexes over the schedule activities.

Hybrid candidate generation:
- semantic: sentence-transformers embeddings (default all-MiniLM-L6-v2),
  cosine similarity via a plain numpy matrix (301 activities -- no ANN needed)
- lexical: fuzzy token-set ratio over activity names (no external dep)
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from matching.normalize import ActivityNorm, parse_activity

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class LexicalHit:
    activity_id: str
    score: float  # 0..1


class ScheduleIndex:
    """Loads schedule_activities.csv and builds both indexes."""

    def __init__(
        self,
        csv_path: str | Path,
        model_name: str = DEFAULT_MODEL,
        use_embeddings: bool = True,
    ):
        self.rows: list[ActivityNorm] = []
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                self.rows.append(parse_activity(row))
        self.by_id: dict[str, ActivityNorm] = {r.activity_id: r for r in self.rows}
        self.model = None
        self.emb: object | None = None
        # multi-view: each activity embedded as name, name+location,
        # name+line, name+both -- a metadata-bearing query matches its
        # best-fitting view instead of being averaged away.
        self._views = [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ]
        if use_embeddings:
            try:
                import numpy as np
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(model_name)
                mats = []
                for with_loc, with_line in self._views:
                    texts = []
                    for r in self.rows:
                        t = r.name_core
                        if with_loc and r.location:
                            t += f" at {r.location}"
                        if with_line and r.name_line:
                            t += f" Line {r.name_line}"
                        texts.append(t)
                    mats.append(self.model.encode(
                        texts, normalize_embeddings=True,
                        show_progress_bar=False,
                    ))
                self.emb = np.stack(mats)  # (V, N, d)
            except Exception as exc:  # degrade to lexical-only
                print(f"[matcher] embeddings unavailable ({exc}); lexical only")
                self.model = None
                self.emb = None

    # ------------------------------------------------------------------
    def embed_query(self, text: str) -> object | None:
        if self.model is None:
            return None
        return self.model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0]

    def semantic_scores(self, qvec) -> dict[str, float]:
        """Max cosine similarity across activity views for the query."""
        if qvec is None or self.emb is None:
            return {}
        import numpy as np

        sims = self.emb @ np.asarray(qvec)  # (V, N)
        best = sims.max(axis=0)
        return {r.activity_id: float(s) for r, s in zip(self.rows, best)}

    # ------------------------------------------------------------------
    @staticmethod
    def _fuzzy(a: str, b: str) -> float:
        """Token-set ratio in [0,1] without external deps."""
        ta, tb = set(a.lower().split()), set(b.lower().split())
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        return 2 * inter / (len(ta) + len(tb))

    def lexical_scores(self, text: str) -> list[LexicalHit]:
        core = text
        hits = [
            LexicalHit(r.activity_id, self._fuzzy(core, r.name_core))
            for r in self.rows
        ]
        hits.sort(key=lambda h: -h.score)
        return hits

    def top_lexical(self, text: str, k: int = 20) -> list[LexicalHit]:
        return self.lexical_scores(text)[:k]
