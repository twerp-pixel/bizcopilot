"""
Hybrid retrieval tool: BM25 (keyword) + dense embeddings (semantic),
fused with Reciprocal Rank Fusion (RRF).

Deliberately lightweight for an 8GB RAM machine:
  - embedding model: sentence-transformers/all-MiniLM-L6-v2 (~80MB, CPU-friendly)
  - similarity via plain numpy cosine sim (no faiss needed - corpus is small)
  - the sentence-transformers model is lazy-loaded on first use, and can be
    swapped for an API embedding call (see embed_texts_api below) if you'd
    rather not install torch at all.
"""
import os
import json
import glob
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(HERE, "data", "docs")
MANIFEST_PATH = os.path.join(HERE, "data", "docs_manifest.json")

_model = None  # lazy-loaded sentence-transformers model
_doc_cache = None
_bm25_cache = None
_embeddings_cache = None


def _load_docs():
    global _doc_cache
    if _doc_cache is not None:
        return _doc_cache
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    docs = []
    for entry in manifest:
        path = os.path.join(HERE, "data", entry["path"])
        with open(path) as f:
            text = f.read()
        docs.append({"doc_id": entry["doc_id"], "title": entry["title"], "text": text})
    _doc_cache = docs
    return docs


def _tokenize(text: str):
    return text.lower().replace("\n", " ").split()


def _get_bm25():
    global _bm25_cache
    if _bm25_cache is not None:
        return _bm25_cache
    docs = _load_docs()
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise RuntimeError(
            "Hybrid retrieval requires rank-bm25. Install project dependencies "
            "with: python -m pip install -r requirements.txt"
        ) from exc
    tokenized = [_tokenize(d["text"]) for d in docs]
    _bm25_cache = BM25Okapi(tokenized)
    return _bm25_cache


def _get_embed_model():
    """Local embedding model (lazy import so this file doesn't require torch unless used)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_texts_local(texts):
    model = _get_embed_model()
    return np.array(model.encode(texts, show_progress_bar=False))


def _get_embeddings():
    global _embeddings_cache
    if _embeddings_cache is not None:
        return _embeddings_cache
    docs = _load_docs()
    _embeddings_cache = embed_texts_local([d["text"] for d in docs])
    return _embeddings_cache


def _cosine_sim(query_vec, matrix):
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
    matrix_norm = matrix / norms
    return matrix_norm @ query_vec


def retrieve(query: str, k: int = 5):
    """
    Hybrid retrieval: BM25 rank + embedding rank fused via RRF.
    Returns list of {doc_id, title, text, score} sorted best-first.
    """
    docs = _load_docs()

    bm25 = _get_bm25()
    bm25_scores = bm25.get_scores(_tokenize(query))
    bm25_rank = np.argsort(-bm25_scores)

    embeddings = _get_embeddings()
    query_vec = embed_texts_local([query])[0]
    sem_scores = _cosine_sim(query_vec, embeddings)
    sem_rank = np.argsort(-sem_scores)

    # Reciprocal Rank Fusion
    RRF_K = 60
    fused = {}
    for rank, idx in enumerate(bm25_rank):
        fused[idx] = fused.get(idx, 0) + 1.0 / (RRF_K + rank + 1)
    for rank, idx in enumerate(sem_rank):
        fused[idx] = fused.get(idx, 0) + 1.0 / (RRF_K + rank + 1)

    ranked_idx = sorted(fused.keys(), key=lambda i: -fused[i])[:k]
    results = []
    for idx in ranked_idx:
        d = docs[idx]
        results.append({
            "doc_id": d["doc_id"],
            "title": d["title"],
            "text": d["text"],
            "score": round(fused[idx], 5),
        })
    return results


if __name__ == "__main__":
    for r in retrieve("why did sales drop in the Midwest", k=3):
        print(r["doc_id"], r["title"], r["score"])
