import hashlib
from functools import lru_cache
from math import sqrt
from typing import Iterable, List, Sequence

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK_DIM = 384


@lru_cache(maxsize=1)
def _model():
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None
    try:
        return SentenceTransformer(DEFAULT_MODEL)
    except Exception:
        return None


def _fallback_embedding(text: str) -> List[float]:
    vector = [0.0] * FALLBACK_DIM
    tokens = (text or "").lower().split()
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % FALLBACK_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def embed_texts(texts: Sequence[str]) -> List[List[float]]:
    cleaned = [text or "" for text in texts]
    if not cleaned:
        return []
    model = _model()
    if model is None:
        return [_fallback_embedding(text) for text in cleaned]
    vectors = model.encode(cleaned, normalize_embeddings=True)
    return [vector.astype(float).tolist() for vector in vectors]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def novelty_score(text: str, comparison_texts: Iterable[str]) -> float:
    comparisons = [item for item in comparison_texts if item]
    if not text or not comparisons:
        return 1.0 if text else 0.0
    vectors = embed_texts([text, *comparisons])
    target = vectors[0]
    best_similarity = max(cosine(target, other) for other in vectors[1:])
    return max(0.0, min(1.0, 1.0 - best_similarity))
