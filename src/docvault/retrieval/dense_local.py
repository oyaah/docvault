"""Local brute-force dense backend — offline alternative to Pinecone.

Stores chunk vectors in a single file under data/ and does exact cosine top-k
in NumPy. Embeddings are already unit-normalised by the embedder, so cosine is a
dot product. Fine for the policy-corpus scale (hundreds–low-thousands of chunks);
not meant to replace an ANN index at large scale.

Selected via DOCVAULT_DENSE_BACKEND=local. Mirrors the dense_pinecone interface.
"""

import pickle
import logging

import numpy as np

from docvault.config import settings

logger = logging.getLogger(__name__)

_store: dict[str, dict] | None = None  # chunk_id -> {"vec", "document_id", "section_path"}


def _path():
    return settings.data_dir / "dense_local.pkl"


def _load() -> dict:
    global _store
    if _store is None:
        p = _path()
        _store = pickle.loads(p.read_bytes()) if p.exists() else {}
    return _store


def _save():
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    _path().write_bytes(pickle.dumps(_store))


def index_chunks(chunk_ids, embeddings, metadata=None):
    store = _load()
    for i, cid in enumerate(chunk_ids):
        meta = metadata[i] if metadata else {}
        store[cid] = {
            "vec": np.asarray(embeddings[i], dtype=np.float32),
            "document_id": meta.get("document_id", ""),
            "section_path": meta.get("section_path", ""),
        }
    _save()
    logger.info(f"Indexed {len(chunk_ids)} chunks locally ({len(store)} total)")


def search(query_embedding, top_k=None):
    store = _load()
    if not store:
        return []
    k = top_k or settings.dense_top_k
    ids = list(store.keys())
    mat = np.stack([store[i]["vec"] for i in ids])
    q = np.asarray(query_embedding, dtype=np.float32)
    sims = mat @ q  # vectors are unit-normalised -> cosine
    top = np.argsort(-sims)[:k]
    return [
        {
            "chunk_id": ids[j],
            "score": float(sims[j]),
            "document_id": store[ids[j]]["document_id"],
            "section_path": store[ids[j]]["section_path"],
        }
        for j in top
    ]


def delete_by_document(document_id):
    store = _load()
    for cid in [c for c, v in store.items() if v["document_id"] == document_id]:
        del store[cid]
    _save()


def reset():
    global _store
    _store = {}
    p = _path()
    if p.exists():
        p.unlink()


def warmup():
    logger.info(f"Local dense backend: {len(_load())} vectors")
