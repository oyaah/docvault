"""Dense retrieval dispatcher.

Selects the backend from settings.dense_backend:
  - "pinecone" (default): managed serverless ANN (dense_pinecone)
  - "local": offline brute-force NumPy cosine (dense_local)

Keeping this as a thin dispatcher lets the rest of the pipeline import
`from docvault.retrieval import dense` and call dense.search / dense.index_chunks
without caring which backend is active — and lets tests / offline dev run with
no Pinecone account.
"""

from docvault.config import settings


def _backend():
    if settings.dense_backend == "local":
        from docvault.retrieval import dense_local
        return dense_local
    from docvault.retrieval import dense_pinecone
    return dense_pinecone


def index_chunks(chunk_ids, embeddings, metadata=None):
    return _backend().index_chunks(chunk_ids, embeddings, metadata=metadata)


def search(query_embedding, top_k=None):
    return _backend().search(query_embedding, top_k=top_k)


def delete_by_document(document_id):
    return _backend().delete_by_document(document_id)


def reset():
    return _backend().reset()


def warmup():
    return _backend().warmup()
