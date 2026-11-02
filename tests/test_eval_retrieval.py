"""Metric functions of scripts/eval_retrieval.py (no retrieval is run)."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_retrieval.py"
    spec = importlib.util.spec_from_file_location("eval_retrieval_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ev = _load()

RELS = {"a": 2, "b": 1, "c": 0, "d": 1}  # three relevant documents: a, b, d


def test_recall_counts_relevant_found_in_top_k():
    assert ev.recall_at_k(["x", "a", "y", "b"], RELS, 2) == pytest.approx(1 / 3)
    assert ev.recall_at_k(["x", "a", "y", "b"], RELS, 4) == pytest.approx(2 / 3)
    assert ev.recall_at_k(["a", "b", "d"], RELS, 10) == 1.0


def test_grade_zero_documents_do_not_count_as_relevant():
    assert ev.recall_at_k(["c"], RELS, 10) == 0.0


def test_queries_without_relevant_documents_are_skipped():
    empty = {"a": 0, "b": 0}
    assert ev.recall_at_k(["a"], empty, 5) is None
    assert ev.mrr_at_k(["a"], empty, 5) is None
    assert ev.ndcg_at_k(["a"], empty, 5) is None


def test_mrr_uses_first_relevant_rank_within_k():
    assert ev.mrr_at_k(["c", "x", "b"], RELS, 10) == pytest.approx(1 / 3)
    assert ev.mrr_at_k(["c", "x", "b"], RELS, 2) == 0.0
    assert ev.mrr_at_k(["a"], RELS, 10) == 1.0


def test_ndcg_is_one_for_the_ideal_order_and_lower_otherwise():
    assert ev.ndcg_at_k(["a", "b", "d"], RELS, 10) == pytest.approx(1.0)
    worse = ev.ndcg_at_k(["b", "a", "d"], RELS, 10)
    assert 0.0 < worse < 1.0


def test_ndcg_value_for_a_known_ranking():
    # gains 2^g - 1: a=3, b=1, d=1.  ranking [b, a]: dcg = 1/log2(2) + 3/log2(3)
    dcg = 1 / math.log2(2) + 3 / math.log2(3)
    ideal = 3 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ev.ndcg_at_k(["b", "a"], RELS, 10) == pytest.approx(dcg / ideal)


def test_precision_and_noise_are_complements():
    ranked = ["a", "c", "zzz", "b", "c"]
    assert ev.precision_at_k(ranked, RELS, 5) == pytest.approx(2 / 5)
    assert ev.noise_at_k(ranked, RELS, 5) == pytest.approx(3 / 5)  # grade 0 and unjudged both count


def test_empty_ranking_is_handled():
    assert ev.precision_at_k([], RELS, 5) == 0.0
    assert ev.noise_at_k([], RELS, 5) == 0.0
    assert ev.recall_at_k([], RELS, 5) == 0.0


def test_percentile_and_mean_helpers():
    assert ev.percentile([1, 2, 3, 4, 100], 95) == 100
    assert ev.percentile([], 95) == 0.0
    assert ev.mean_or_none([None, 1.0, 3.0]) == 2.0
    assert ev.mean_or_none([None]) is None


def test_score_slices_by_group_and_skips_unlabelled_queries():
    queries = [
        {"qid": "q1", "group": "zh", "kind": "concept", "query": "x"},
        {"qid": "q2", "group": "en", "kind": "concept", "query": "y"},
        {"qid": "q3", "group": "en", "kind": "concept", "query": "z"},
    ]
    qrels = {"q1": {"a": 2}, "q2": {"a": 1}, "q3": {"a": 0}}
    runs = {
        "A": {
            "q1": {"ranked": ["a"], "latency_s": 0.1},
            "q2": {"ranked": ["x", "a"], "latency_s": 0.3},
            "q3": {"ranked": ["a"], "latency_s": 0.2},
        }
    }

    scores = ev.score(runs, queries, qrels)["A"]

    assert scores["全部"]["n"] == 3 and scores["全部"]["n_scored"] == 2
    assert scores["全部"]["MRR@10"] == pytest.approx((1.0 + 0.5) / 2)
    assert scores["英文"]["n_scored"] == 1 and scores["英文"]["MRR@10"] == 0.5
    assert scores["全部"]["latency_mean_ms"] == pytest.approx(200.0)


def test_checked_in_dataset_is_consistent():
    queries = ev.load_jsonl(ev.DEFAULT_QUERIES)
    pool = {row["qid"]: row for row in ev.load_jsonl(ev.DEFAULT_POOL)}
    qrels = ev.load_qrels(ev.DEFAULT_QRELS)

    assert 40 <= len(queries) <= 60
    assert len({q["qid"] for q in queries}) == len(queries)
    assert {q["group"] for q in queries} == {"zh", "en", "id"}
    assert {q["qid"] for q in queries} == set(pool) == set(qrels)
    for qid, row in pool.items():
        keys = {c["key"] for c in row["candidates"]}
        assert keys == set(qrels[qid]), f"{qid}: every pooled candidate needs exactly one judgment"
        assert set(qrels[qid].values()) <= {0, 1, 2}
        assert any(g >= 1 for g in qrels[qid].values()), f"{qid} has no relevant document"
