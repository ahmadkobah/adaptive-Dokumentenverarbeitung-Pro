"""
Embeddings Service
SentenceTransformer — kept for semantic search vector index.
Gemini handles everything else; this is the only local model in the Pro tier.
"""

import logging
from typing import Dict, List

from sentence_transformers import SentenceTransformer

logger = logging.getLogger("embeddings")

_model: SentenceTransformer | None = None
_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {_MODEL_NAME}")
        _model = SentenceTransformer(_MODEL_NAME)
        logger.info("Embedding model ready")
    return _model


def embed(text: str) -> List[float]:
    """Return a 384-dim embedding vector for the given text."""
    if not text or not text.strip():
        return []
    model = get_model()
    vector = model.encode(text, convert_to_tensor=False)
    return vector.tolist()


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split *text* into overlapping character-window chunks.

    Character windows are used (rather than token windows) because the
    multilingual MiniLM tokenizer has a 128-token cap; aiming for ~1500
    characters keeps each chunk safely under that while preserving
    enough context for semantic search.
    """
    if not text:
        return []
    if chunk_size <= 0:
        return [text]
    overlap = max(0, min(overlap, chunk_size - 1))
    step = max(1, chunk_size - overlap)
    chunks: List[str] = []
    pos = 0
    n = len(text)
    while pos < n:
        chunks.append(text[pos : pos + chunk_size])
        pos += step
    return chunks


def embed_chunks(
    text: str,
    *,
    chunk_size: int = 1500,
    overlap: int = 200,
    max_chunks: int = 60,
    snippet_length: int = 200,
) -> List[Dict]:
    """
    Return per-chunk embeddings for a long document.

    Each list entry is a dict with ``idx``, ``text`` (a short snippet
    for display), ``char_start`` / ``char_end`` for citation, and
    ``embedding`` (a 384-d float list). The whole list is capped at
    ``max_chunks`` to bound memory and search latency.
    """
    if not text or not text.strip():
        return []
    raw_chunks = _chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not raw_chunks:
        return []
    raw_chunks = raw_chunks[: max(1, int(max_chunks))]

    model = get_model()
    vectors = model.encode(raw_chunks, convert_to_tensor=False, batch_size=8)

    out: List[Dict] = []
    cursor = 0
    step = max(1, chunk_size - max(0, overlap))
    for i, (chunk, vec) in enumerate(zip(raw_chunks, vectors)):
        snippet = chunk[:snippet_length]
        out.append(
            {
                "idx": i,
                "text": snippet,
                "char_start": cursor,
                "char_end": cursor + len(chunk),
                "embedding": vec.tolist() if hasattr(vec, "tolist") else list(vec),
            }
        )
        cursor += step
    return out


__all__ = ["embed", "embed_chunks", "get_model"]

