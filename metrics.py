"""Retrieval evaluation metrics.

Dependency-free helpers for scoring retrieval quality, so the keyword and
semantic pipelines in this repository can be compared on the same footing
instead of by eyeballing answers.

Terminology used throughout:

* a **run** is ``{query_id: [doc_id, ...]}`` — retrieved documents, best first
* **labels** is ``{query_id: {doc_id, ...}}`` — the truly relevant documents

Metrics assume one relevant-document set per query, which is what a small
hand-labelled gold set gives you.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "mean_recall_at_k",
    "mean_precision_at_k",
    "mean_reciprocal_rank",
    "evaluate",
]


def _top_k(retrieved: Sequence[str], k: int) -> Sequence[str]:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return retrieved[:k]


def _check_relevant(relevant: Iterable[str]) -> set[str]:
    relevant = set(relevant)
    if not relevant:
        raise ValueError(
            "a query with no relevant documents has no defined recall or "
            "reciprocal rank — drop it from the gold set instead"
        )
    return relevant


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the relevant documents that appear in the top ``k``."""
    relevant = _check_relevant(relevant)
    hits = set(_top_k(retrieved, k)) & relevant
    return len(hits) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top ``k`` results that are relevant."""
    relevant = set(relevant)
    top = _top_k(retrieved, k)
    hits = set(top) & relevant
    return len(hits) / len(top)


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """``1 / rank`` of the first relevant result, or ``0.0`` if there is none."""
    relevant = _check_relevant(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _mean(pairs: Mapping[str, Sequence[str]], labels: Mapping[str, Iterable[str]], fn) -> float:
    queries = [q for q in pairs if q in labels]
    if not queries:
        raise ValueError("no query in the run has labels, so no metric is defined")
    return sum(fn(pairs[q], labels[q]) for q in queries) / len(queries)


def mean_recall_at_k(
    pairs: Mapping[str, Sequence[str]], labels: Mapping[str, Iterable[str]], k: int
) -> float:
    """Mean ``recall@k`` across every labelled query."""
    return _mean(pairs, labels, lambda r, l: recall_at_k(r, l, k))


def mean_precision_at_k(
    pairs: Mapping[str, Sequence[str]], labels: Mapping[str, Iterable[str]], k: int
) -> float:
    """Mean ``precision@k`` across every labelled query."""
    return _mean(pairs, labels, lambda r, l: precision_at_k(r, l, k))


def mean_reciprocal_rank(
    pairs: Mapping[str, Sequence[str]], labels: Mapping[str, Iterable[str]]
) -> float:
    """Mean reciprocal rank across every labelled query."""
    return _mean(pairs, labels, reciprocal_rank)


def evaluate(
    pairs: Mapping[str, Sequence[str]],
    labels: Mapping[str, Iterable[str]],
    ks: Sequence[int] = (1, 3, 5),
) -> dict[str, float]:
    """Every metric at every ``k`` in one dict, for a results table."""
    results = {f"mrr": mean_reciprocal_rank(pairs, labels)}
    for k in ks:
        results[f"recall@{k}"] = mean_recall_at_k(pairs, labels, k)
        results[f"precision@{k}"] = mean_precision_at_k(pairs, labels, k)
    return results
