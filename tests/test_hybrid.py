from src.retrieval.hybrid import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_rewards_agreement():
    dense = ["a", "b", "c"]
    sparse = ["b", "a", "d"]

    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    fused_ids = [doc_id for doc_id, _ in fused]

    # "a" and "b" appear near the top of both rankings, so they should lead.
    assert set(fused_ids[:2]) == {"a", "b"}
    assert "c" in fused_ids and "d" in fused_ids


def test_reciprocal_rank_fusion_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
