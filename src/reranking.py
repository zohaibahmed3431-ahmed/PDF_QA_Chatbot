# Reranking is currently integrated into HybridRetriever so retrieval remains
# lightweight for Streamlit Cloud. This module is kept as a clean extension
# point for a cross-encoder or Gemini reranker in a later production stage.

def rerank(query: str, results: list[dict], top_k: int = 8) -> list[dict]:
    return sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)[:top_k]
