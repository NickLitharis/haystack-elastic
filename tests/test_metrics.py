"""Tests for the retrieval metrics.

Run with::

    python -m pytest tests -v
"""
import pytest

from metrics import (
    evaluate,
    mean_precision_at_k,
    mean_recall_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RETRIEVED = ["a", "b", "c", "d", "e"]
RELEVANT = {"c", "e"}  # c sits at rank 3, e at rank 5


def test_recall_at_k_counts_only_relevant_hits():
    assert recall_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(0.5)  # only c
    assert recall_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(1.0)  # c and e
    assert recall_at_k(RETRIEVED, RELEVANT, 1) == pytest.approx(0.0)  # a is irrelevant


def test_recall_at_k_larger_than_the_result_list_is_safe():
    assert recall_at_k(RETRIEVED, RELEVANT, 99) == pytest.approx(1.0)


def test_precision_at_k_divides_by_k_not_by_hits():
    assert precision_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(1 / 3)
    assert precision_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(2 / 5)


def test_reciprocal_rank_uses_the_first_hit():
    assert reciprocal_rank(["c", "a"], {"c"}) == pytest.approx(1.0)
    assert reciprocal_rank(["a", "c"], {"c"}) == pytest.approx(0.5)
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_when_nothing_relevant_was_returned():
    assert reciprocal_rank(["a", "b"], {"z"}) == pytest.approx(0.0)


def test_mean_reciprocal_rank_averages_across_queries():
    pairs = {"q1": ["x", "y"], "q2": ["y", "x"]}
    labels = {"q1": {"x"}, "q2": {"x"}}  # rank 1 then rank 2
    assert mean_reciprocal_rank(pairs, labels) == pytest.approx((1.0 + 0.5) / 2)


def test_mean_metrics_ignore_queries_that_have_no_labels():
    pairs = {"labelled": ["a"], "unlabelled": ["zzz"]}
    labels = {"labelled": {"a"}}
    assert mean_recall_at_k(pairs, labels, 1) == pytest.approx(1.0)
    assert mean_precision_at_k(pairs, labels, 1) == pytest.approx(1.0)


def test_evaluate_returns_the_whole_table():
    result = evaluate({"q1": ["a", "b"]}, {"q1": {"b"}}, ks=(1, 2))
    assert set(result) == {"mrr", "recall@1", "recall@2", "precision@1", "precision@2"}
    assert result["mrr"] == pytest.approx(0.5)
    assert result["recall@2"] == pytest.approx(1.0)


def test_non_positive_k_is_rejected():
    for bad_k in (0, -1):
        with pytest.raises(ValueError):
            recall_at_k(RETRIEVED, RELEVANT, bad_k)


def test_an_empty_relevant_set_is_rejected_rather_than_scored():
    """Better to fail loudly than to report 0.0 for an unlabelled query."""
    with pytest.raises(ValueError):
        recall_at_k(RETRIEVED, set(), 3)
    with pytest.raises(ValueError):
        reciprocal_rank(RETRIEVED, set())


def test_no_labelled_queries_at_all_is_an_error():
    with pytest.raises(ValueError):
        mean_recall_at_k({"a": ["x"]}, {}, 1)
