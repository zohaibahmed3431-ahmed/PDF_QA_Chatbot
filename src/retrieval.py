from __future__ import annotations

import re
from collections import Counter

import numpy as np

from .embeddings import GeminiEmbedder


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "why",
    "how", "where", "when", "which", "who", "and", "or", "to",
    "of", "in", "on", "for", "with", "this", "that", "it", "me",
}


def tokenize(text: str) -> list[str]:
    return [
        t for t in re.findall(r"[a-zA-Z0-9_]+", text.lower())
        if t not in STOPWORDS and len(t) > 1
    ]


class HybridRetriever:
    def __init__(self):
        self.records: list[dict] = []
        self.matrix: np.ndarray | None = None
        self.embedder: GeminiEmbedder | None = None

    def clear(self):
        self.records = []
        self.matrix = None

    def build(self, records: list[dict]):
        self.records = list(records)
        if not records:
            self.matrix = None
            return

        try:
            self.embedder = self.embedder or GeminiEmbedder()
            self.matrix = self.embedder.embed([r["text"] for r in records])
        except Exception:
            # App remains usable with lexical retrieval if embeddings fail.
            self.matrix = None

    def _lexical_score(self, query: str, text: str) -> float:
        q = Counter(tokenize(query))
        d = Counter(tokenize(text))
        if not q or not d:
            return 0.0
        overlap = sum(min(q[k], d[k]) for k in q)
        return overlap / max(1, sum(q.values()))

    def search(self, query: str, top_k: int = 8) -> list[dict]:
        if not self.records:
            return []

        lexical = np.array([
            self._lexical_score(query, r["text"]) for r in self.records
        ], dtype=np.float32)

        if self.matrix is not None and self.embedder is not None:
            try:
                qvec = self.embedder.embed([query])[0]
                semantic = self.matrix @ qvec
            except Exception:
                semantic = np.zeros(len(self.records), dtype=np.float32)
        else:
            semantic = np.zeros(len(self.records), dtype=np.float32)

        # Hybrid score: semantic retrieval + lexical evidence.
        combined = 0.72 * semantic + 0.28 * lexical

        # Lightweight reranking: exact phrase and query-token coverage.
        q_lower = query.lower().strip()
        reranked = []
        for i, record in enumerate(self.records):
            bonus = 0.0
            text_lower = record["text"].lower()
            if q_lower and q_lower in text_lower:
                bonus += 0.12
            if tokenize(query) and all(t in text_lower for t in tokenize(query)[:5]):
                bonus += 0.06
            reranked.append((float(combined[i] + bonus), i))

        reranked.sort(reverse=True)
        results = []
        for score, i in reranked[:top_k]:
            item = dict(self.records[i])
            item["score"] = score
            results.append(item)
        return results
