"""Dense retrieval via Pinecone managed vector DB.

Replaces local LanceDB — data persists across container restarts,
supports horizontal scaling, and requires zero infrastructure management.

Free tier: 1 index, 100K vectors — sufficient for company policy docs.
"""

import logging

from pinecone import Pinecone, ServerlessSpec
import numpy as np

from docvault.config import settings

logger = logging.getLogger(__name__)

_index = None


def _get_index():
    global _index
    if _index is None:
        pc = Pinecone(api_key=settings.pinecone_api_key)

        # Create index if it doesn't exist
        existing = [idx.name for idx in pc.list_indexes()]
        if settings.pinecone_index_name not in existing:
            pc.create_index(
                name=settings.pinecone_index_name,
                dimension=settings.embedding_dimensions,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.pinecone_cloud,
                    region=settings.pinecone_region,
                ),
            )
            logger.info(f"Created Pinecone index: {settings.pinecone_index_name}")

        _index = pc.Index(settings.pinecone_index_name)
    return _index


def index_chunks(
    chunk_ids: list[str],
    embeddings: np.ndarray,
    metadata: list[dict] | None = None,
):
    """Upsert chunk vectors into Pinecone."""
    index = _get_index()

    vectors = []
    for i, chunk_id in enumerate(chunk_ids):
        meta = {}
        if metadata:
            meta["document_id"] = metadata[i].get("document_id", "")
            meta["section_path"] = metadata[i].get("section_path", "")
        vectors.append({
            "id": chunk_id,
            "values": embeddings[i].tolist(),
            "metadata": meta,
        })

    # Pinecone upsert in batches of 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)

    logger.info(f"Indexed {len(vectors)} chunks in Pinecone")


def search(query_embedding: np.ndarray, top_k: int | None = None) -> list[dict]:
    """ANN search. Returns list of {chunk_id, score, document_id, section_path}."""
    index = _get_index()
    k = top_k or settings.dense_top_k

    results = index.query(
        vector=query_embedding.tolist(),
        top_k=k,
        include_metadata=True,
    )

    return [
        {
            "chunk_id": match.id,
            "score": match.score,  # cosine similarity [0, 1]
            "document_id": match.metadata.get("document_id", ""),
            "section_path": match.metadata.get("section_path", ""),
        }
        for match in results.matches
    ]


def delete_by_document(document_id: str):
    """Remove all vectors for a document."""
    index = _get_index()
    # Pinecone requires listing IDs by metadata filter, then deleting
    # For serverless indexes, use delete with filter
    try:
        index.delete(filter={"document_id": {"$eq": document_id}})
    except Exception as e:
        logger.warning(f"Failed to delete vectors for {document_id}: {e}")


def reset():
    """Delete all vectors in the index."""
    index = _get_index()
    try:
        index.delete(delete_all=True)
    except Exception:
        pass


def warmup():
    """Validate Pinecone connectivity."""
    index = _get_index()
    stats = index.describe_index_stats()
    logger.info(f"Pinecone connected: {stats.total_vector_count} vectors")
