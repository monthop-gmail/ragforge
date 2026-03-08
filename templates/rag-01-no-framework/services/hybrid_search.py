from config import settings


def reciprocal_rank_fusion(
    vector_results: list[dict],
    keyword_results: list[dict],
    k: int | None = None,
) -> list[dict]:
    """Merge vector and keyword results using Reciprocal Rank Fusion (RRF).

    RRF score = sum(1 / (k + rank)) for each result list where the item appears.
    """
    rrf_k = k or settings.rrf_k
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for rank, result in enumerate(vector_results):
        rid = result["id"]
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (rrf_k + rank + 1)
        items[rid] = result

    for rank, result in enumerate(keyword_results):
        rid = result["id"]
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (rrf_k + rank + 1)
        if rid not in items:
            items[rid] = result

    sorted_ids = sorted(scores, key=lambda rid: scores[rid], reverse=True)

    merged = []
    for rid in sorted_ids:
        item = items[rid].copy()
        item["score"] = scores[rid]
        merged.append(item)

    return merged
