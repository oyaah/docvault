"""Reciprocal Rank Fusion (RRF) — merges dense and sparse results."""

from docvault.config import settings


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    top_k: int | None = None,
    k: int = 60,  # RRF constant
) -> list[dict]:
    """Merge dense and sparse results using RRF.

    Args:
        dense_results: list of {chunk_id, score, ...}
        sparse_results: list of {chunk_id, score, ...}
        top_k: number of results to return
        k: RRF smoothing constant (standard = 60)

    Returns:
        Merged list sorted by RRF score, with source metadata.
    """
    fused_top_k = top_k or settings.fused_top_k
    scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    # Score from dense results
    for rank, result in enumerate(dense_results):
        cid = result["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        chunk_data[cid] = result

    # Score from sparse results
    for rank, result in enumerate(sparse_results):
        cid = result["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        if cid not in chunk_data:
            chunk_data[cid] = result

    # Sort by fused score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results = []
    for cid in sorted_ids[:fused_top_k]:
        entry = {**chunk_data[cid], "rrf_score": scores[cid]}
        results.append(entry)

    return results
