from __future__ import annotations

import os
import numpy as np
from google import genai
from google.genai import types


EMBEDDING_MODELS = [
    os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2"),
    "gemini-embedding-001",
]


class GeminiEmbedder:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        self.client = genai.Client(api_key=api_key)
        self.model = EMBEDDING_MODELS[0]

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        last_error = None
        for model in EMBEDDING_MODELS:
            try:
                response = self.client.models.embed_content(
                    model=model,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        output_dimensionality=768,
                        task_type="RETRIEVAL_DOCUMENT",
                    ),
                )
                vectors = [np.asarray(e.values, dtype=np.float32) for e in response.embeddings]
                matrix = np.vstack(vectors)
                self.model = model
                return self._normalize(matrix)
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"Embedding failed: {last_error}")

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms
