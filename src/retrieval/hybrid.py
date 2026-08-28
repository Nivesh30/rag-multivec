from typing import Dict, List, Sequence, Tuple


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60
) -> List[Tuple[str, float]]:
    """Combine multiple ranked id lists into one, via Reciprocal Rank Fusion.

    Each input ranking is a list of doc ids ordered best-first. RRF rewards ids
    that rank well across multiple sources without needing comparable scores.
    """
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
