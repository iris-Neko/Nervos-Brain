#!/usr/bin/env python3
"""Offline retrieval evaluation: compare hash vectors, no vectors and real embeddings.

Configurations (all run against the same SQLite archives, three corpora merged
the way the online runtime merges them):

  A  today's behaviour: 64-dim SHA-256 pseudo vectors + BM25 + exact + fuzzy
  B  vector path off:   BM25 + exact + fuzzy
  C  real embedding:    vector + BM25 + exact + fuzzy (RRF)
  D  real embedding, vector path only

The LLM-written regex path is not evaluated (it needs an LLM call per query).

Workflow::

    # 1. build the three Qdrant directory sets once (see evaluation/README.md)
    # 2. dump the judging pool: union of every configuration's top-k per query
    python scripts/eval_retrieval.py pool
    # 3. label the pool (evaluation/retrieval_qrels.jsonl, grades 0/1/2)
    # 4. score
    python scripts/eval_retrieval.py run
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_QUERIES = ROOT / "evaluation" / "retrieval_queries.jsonl"
DEFAULT_POOL = ROOT / "evaluation" / "retrieval_pool.jsonl"
DEFAULT_QRELS = ROOT / "evaluation" / "retrieval_qrels.jsonl"
REPORT_DIR = ROOT / "evaluation" / "reports"

GROUP_NAMES = {"zh": "中文", "en": "英文", "id": "标识符/报错"}
POOL_TEXT_CHARS = 1200


@dataclass(frozen=True)
class EvalConfig:
    name: str
    label: str
    qdrant_set: str  # "hash" or "real"
    overrides: dict[str, Any] = field(default_factory=dict)


CONFIGS: dict[str, EvalConfig] = {
    "A": EvalConfig(
        "A",
        "A 现状：哈希向量 + BM25/exact/fuzzy",
        "hash",
        {"embedding_backend": "hash", "embedding_allow_hash": True},
    ),
    "B": EvalConfig("B", "B 关向量路：BM25/exact/fuzzy", "hash", {"enable_vector": False}),
    "C": EvalConfig("C", "C 真实向量 + BM25/exact/fuzzy", "real", {}),
    "D": EvalConfig(
        "D",
        "D 只用真实向量",
        "real",
        {"enable_bm25": False, "enable_exact": False, "enable_fuzzy": False},
    ),
}


# ── metrics ────────────────────────────────────────────────────────────────


def recall_at_k(ranked: Sequence[str], rels: dict[str, int], k: int) -> float | None:
    """Share of the query's relevant documents (grade >= 1) found in the top k."""
    relevant = {key for key, grade in rels.items() if grade >= 1}
    if not relevant:
        return None
    return len(relevant & set(ranked[:k])) / len(relevant)


def precision_at_k(ranked: Sequence[str], rels: dict[str, int], k: int) -> float:
    top = list(ranked[:k])
    if not top:
        return 0.0
    return sum(1 for key in top if rels.get(key, 0) >= 1) / len(top)


def noise_at_k(ranked: Sequence[str], rels: dict[str, int], k: int) -> float:
    """Share of the top k that is irrelevant (grade 0 or unjudged)."""
    top = list(ranked[:k])
    if not top:
        return 0.0
    return sum(1 for key in top if rels.get(key, 0) < 1) / len(top)


def mrr_at_k(ranked: Sequence[str], rels: dict[str, int], k: int) -> float | None:
    if not any(grade >= 1 for grade in rels.values()):
        return None
    for rank, key in enumerate(ranked[:k], start=1):
        if rels.get(key, 0) >= 1:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], rels: dict[str, int], k: int) -> float | None:
    """nDCG with gain 2^grade - 1 and a log2 rank discount."""
    ideal_grades = sorted((g for g in rels.values() if g >= 1), reverse=True)[:k]
    if not ideal_grades:
        return None

    def dcg(grades: Iterable[int]) -> float:
        return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades))

    actual = dcg(rels.get(key, 0) for key in ranked[:k])
    return actual / dcg(ideal_grades)


def mean_or_none(values: Iterable[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(pct / 100 * len(ordered)) - 1))
    return ordered[index]


# ── retriever construction ─────────────────────────────────────────────────


@dataclass
class RetrieverSet:
    """Backends opened once per Qdrant directory set (local mode locks the folder)."""

    composite: Any
    backends: list[Any]
    base_configs: dict[str, Any]


def build_set(qdrant_root: Path, base_overrides: dict[str, Any]) -> RetrieverSet:
    from migrate_qdrant_server_from_archive import load_public_backend_configs
    from nervos_brain.retrieval import ArchiveStore, MultiRetriever, QdrantStore
    from nervos_brain.retrieval.composite_retriever import CompositeRetriever, RetrievalBackend

    backends = []
    base_configs = {}
    for name, cfg in load_public_backend_configs():
        path = qdrant_root / Path(cfg.qdrant_path).name
        cfg = replace(cfg, qdrant_url="", qdrant_path=str(path), **base_overrides)
        qdrant = QdrantStore(config=cfg, qdrant_location=cfg.qdrant_path)
        archive = ArchiveStore(db_path=cfg.archive_db, config=cfg)
        retriever = MultiRetriever(qdrant_store=qdrant, archive_store=archive, config=cfg)
        backends.append(
            RetrievalBackend(name=name, config=cfg, qdrant_store=qdrant, archive_store=archive, retriever=retriever)
        )
        base_configs[name] = cfg
    composite = CompositeRetriever(backends)
    started = time.perf_counter()
    size = composite.rebuild_bm25()
    print(f"  built lexical indexes for {size} records in {time.perf_counter() - started:.1f}s", flush=True)
    return RetrieverSet(composite, backends, base_configs)


def apply_config(rset: RetrieverSet, overrides: dict[str, Any]) -> list[tuple[str, bool, str]]:
    """Point every backend at ``base config + overrides`` and re-check its vector path."""
    states = []
    for backend in rset.backends:
        cfg = replace(rset.base_configs[backend.name], **overrides)
        backend.retriever._cfg = cfg
        backend.qdrant_store.config = cfg
        enabled, reason = backend.retriever.check_vector_path(warmup=bool(cfg.enable_vector))
        states.append((backend.name, enabled, reason))
    rset.composite._cfg = replace(rset.base_configs[rset.backends[0].name], **overrides)
    return states


def evidence_key(row: dict[str, Any]) -> str:
    payload = row.get("payload") or {}
    return f"{payload.get('backend', row.get('source', ''))}::{row.get('anchor', '')}"


def run_queries(
    rset: RetrieverSet, queries: list[dict[str, Any]], top_k: int
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    rset.composite.search(queries[0]["query"], top_k=top_k)  # warm-up (model, caches)
    for item in queries:
        started = time.perf_counter()
        rows = rset.composite.search(item["query"], top_k=top_k)
        elapsed = time.perf_counter() - started
        results[item["qid"]] = {
            "ranked": [evidence_key(row) for row in rows],
            "titles": {evidence_key(row): row.get("title", "") for row in rows},
            "paths": {
                evidence_key(row): sorted(
                    k[len("contrib_") :] for k in (row.get("payload") or {}) if k.startswith("contrib_")
                )
                for row in rows
            },
            "latency_s": elapsed,
        }
    return results


def execute(config_names: Sequence[str], queries: list[dict[str, Any]], top_k: int, real_root: Path, hash_root: Path):
    """Run each configuration; returns {config: {qid: result}} and vector-path states."""
    runs: dict[str, dict[str, dict[str, Any]]] = {}
    states: dict[str, list[tuple[str, bool, str]]] = {}
    for set_name, root in (("hash", hash_root), ("real", real_root)):
        wanted = [CONFIGS[n] for n in config_names if CONFIGS[n].qdrant_set == set_name]
        if not wanted:
            continue
        print(f"opening {set_name} Qdrant set at {root}", flush=True)
        base = _real_overrides() if set_name == "real" else {"vector_size": 64}
        rset = build_set(root, base)
        for conf in wanted:
            overrides = {**base, **conf.overrides}
            states[conf.name] = apply_config(rset, overrides)
            print(f"running {conf.label}  vector path: {states[conf.name]}", flush=True)
            runs[conf.name] = run_queries(rset, queries, top_k)
        for backend in rset.backends:
            backend.qdrant_store.client.close()
    return runs, states


def _real_overrides() -> dict[str, Any]:
    return {
        "embedding_backend": "local",
        "embedding_model": "BAAI/bge-m3",
        "vector_size": 1024,
        "embedding_batch_size": 64,
    }


# ── pool / qrels io ────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for row in load_jsonl(path):
        qrels[row["qid"]] = {key: int(grade) for key, grade in row["judgments"].items()}
    return qrels


def archive_lookup(rset_backends: Iterable[Any]) -> dict[str, Any]:
    return {backend.name: backend.archive_store for backend in rset_backends}


def build_pool(runs: dict[str, dict[str, dict[str, Any]]], queries: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Union of every configuration's ranked list; shuffled and blind to the config."""
    from migrate_qdrant_server_from_archive import load_public_backend_configs
    from nervos_brain.retrieval import ArchiveStore

    stores = {name: ArchiveStore(db_path=cfg.archive_db, config=cfg) for name, cfg in load_public_backend_configs()}
    rng = random.Random(seed)
    pool = []
    for item in queries:
        keys: dict[str, None] = {}
        titles: dict[str, str] = {}
        for per_config in runs.values():
            result = per_config[item["qid"]]
            for key in result["ranked"]:
                keys.setdefault(key, None)
                titles.setdefault(key, result["titles"].get(key, ""))
        candidates = []
        for key in keys:
            backend, anchor = key.split("::", 1)
            record = stores[backend].get_by_anchor(anchor)
            candidates.append(
                {
                    "key": key,
                    "title": record.title if record else titles.get(key, ""),
                    "text": (record.raw_text if record else "")[:POOL_TEXT_CHARS],
                }
            )
        rng.shuffle(candidates)
        pool.append({"qid": item["qid"], "query": item["query"], "candidates": candidates})
    return pool


# ── scoring and report ─────────────────────────────────────────────────────

METRICS = [
    ("Recall@5", lambda r, q: recall_at_k(r, q, 5)),
    ("Recall@10", lambda r, q: recall_at_k(r, q, 10)),
    ("MRR@10", lambda r, q: mrr_at_k(r, q, 10)),
    ("nDCG@10", lambda r, q: ndcg_at_k(r, q, 10)),
    ("Precision@5", lambda r, q: precision_at_k(r, q, 5)),
    ("Noise@5", lambda r, q: noise_at_k(r, q, 5)),
]


def score(
    runs: dict[str, dict[str, dict[str, Any]]],
    queries: list[dict[str, Any]],
    qrels: dict[str, dict[str, int]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """{config: {slice: {metric: value, 'latency_*': ..., 'n': ...}}}."""
    slices: dict[str, list[dict[str, Any]]] = {"全部": queries}
    for group, name in GROUP_NAMES.items():
        slices[name] = [q for q in queries if q["group"] == group]
    for kind, name in (("forum", "论坛讨论"), ("concept", "概念/教程"), ("case", "样例问题"), ("function", "函数名"), ("error", "报错名")):
        subset = [q for q in queries if q["kind"] == kind]
        if subset:
            slices[f"· {name}"] = subset

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for conf, per_query in runs.items():
        out[conf] = {}
        for slice_name, subset in slices.items():
            row: dict[str, Any] = {}
            labelled = [q for q in subset if any(g >= 1 for g in qrels.get(q["qid"], {}).values())]
            for metric, fn in METRICS:
                row[metric] = mean_or_none(
                    fn(per_query[q["qid"]]["ranked"], qrels.get(q["qid"], {})) for q in labelled
                )
            latencies = [per_query[q["qid"]]["latency_s"] * 1000 for q in subset]
            row["latency_mean_ms"] = statistics.fmean(latencies) if latencies else 0.0
            row["latency_p95_ms"] = percentile(latencies, 95)
            row["n"] = len(subset)
            row["n_scored"] = len(labelled)
            out[conf][slice_name] = row
    return out


def _fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def render_report(
    scores: dict[str, dict[str, dict[str, Any]]],
    runs: dict[str, dict[str, dict[str, Any]]],
    queries: list[dict[str, Any]],
    qrels: dict[str, dict[str, int]],
    states: dict[str, list[tuple[str, bool, str]]],
    top_k: int,
) -> str:
    names = list(scores)
    slice_names = list(next(iter(scores.values())))
    unjudged = sum(
        1
        for conf in runs.values()
        for qid, result in conf.items()
        for key in result["ranked"]
        if key not in qrels.get(qid, {})
    )
    no_rel = [q["qid"] for q in queries if not any(g >= 1 for g in qrels.get(q["qid"], {}).values())]

    lines = [f"# 检索评测报告（自动生成 {date.today().isoformat()}）", ""]
    lines += [
        "## 设置",
        "",
        f"- 查询 {len(queries)} 条，有相关文档的 {len(queries) - len(no_rel)} 条参与打分（无相关文档的：{', '.join(no_rel) or '无'}）。",
        f"- 每条查询取融合后的前 {top_k} 条；三个库（docs / 论坛 / 代码）按线上方式合并。",
        f"- 未标注的结果数：{unjudged}（按不相关计）。",
        "- 正则路依赖 LLM 生成正则，评测里不开。",
        "",
        "| 配置 | 说明 | 向量路状态 |",
        "|---|---|---|",
    ]
    for name in names:
        flags = ", ".join(f"{b}:{'开' if e else '关'}" for b, e, _ in states.get(name, []))
        lines.append(f"| {name} | {CONFIGS[name].label} | {flags} |")
    lines.append("")

    def table(title: str, metric_keys: list[tuple[str, str]], slice_filter: list[str]):
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| 切片 | 配置 | " + " | ".join(label for _, label in metric_keys) + " |")
        lines.append("|---|---|" + "|".join("---" for _ in metric_keys) + "|")
        for slice_name in slice_filter:
            for name in names:
                row = scores[name][slice_name]
                cells = []
                for key, _ in metric_keys:
                    cells.append(_fmt(row.get(key), 0 if key.startswith("latency") else 3))
                lines.append(f"| {slice_name}（n={row['n_scored']}） | {name} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## 结果")
    lines.append("")
    quality = [(m, m) for m, _ in METRICS]
    table("检索质量（按类别）", quality, slice_names[:4])
    table("检索质量（按查询类型细分）", quality, slice_names[4:])
    table("耗时（毫秒/次查询，含三个库的合并，不含 LLM）", [("latency_mean_ms", "平均"), ("latency_p95_ms", "P95")], ["全部"])

    if len(names) > 1:
        base = names[0]
        lines.append("### 相对 A 的变化（全部查询）")
        lines.append("")
        lines.append("| 配置 | " + " | ".join(m for m, _ in METRICS) + " |")
        lines.append("|---|" + "|".join("---" for _ in METRICS) + "|")
        for name in names[1:]:
            cells = []
            for metric, _ in METRICS:
                a, b = scores[base]["全部"][metric], scores[name]["全部"][metric]
                cells.append("-" if a is None or b is None else f"{b - a:+.3f}")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


# ── commands ───────────────────────────────────────────────────────────────


def cmd_pool(args: argparse.Namespace) -> int:
    queries = load_jsonl(Path(args.queries))
    runs, _ = execute(args.configs.split(","), queries, args.top_k, Path(args.real_root), Path(args.hash_root))
    pool = build_pool(runs, queries, seed=args.seed)
    write_jsonl(Path(args.out), pool)
    sizes = [len(row["candidates"]) for row in pool]
    print(f"wrote {args.out}: {len(pool)} queries, {sum(sizes)} candidates (mean {statistics.fmean(sizes):.1f})")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    queries = load_jsonl(Path(args.queries))
    qrels = load_qrels(Path(args.qrels))
    names = args.configs.split(",")
    runs, states = execute(names, queries, args.top_k, Path(args.real_root), Path(args.hash_root))
    scores = score(runs, queries, qrels)
    report = render_report(scores, runs, queries, qrels, states, args.top_k)
    stem = args.report or str(REPORT_DIR / f"retrieval_eval_{date.today().isoformat()}")
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    Path(stem + ".md").write_text(report, encoding="utf-8", newline="\n")
    Path(stem + ".json").write_text(
        json.dumps({"scores": scores, "runs": runs}, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
    )
    print(report)
    print(f"wrote {stem}.md / .json")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("pool", cmd_pool), ("run", cmd_run)):
        p = sub.add_parser(name)
        p.add_argument("--queries", default=str(DEFAULT_QUERIES))
        p.add_argument("--configs", default="A,B,C,D")
        p.add_argument("--top-k", type=int, default=10)
        p.add_argument("--real-root", default=str(ROOT / "data" / "qdrant_eval"))
        p.add_argument("--hash-root", default=str(ROOT / "data" / "qdrant_eval_hash"))
        p.set_defaults(func=fn)
        if name == "pool":
            p.add_argument("--out", default=str(DEFAULT_POOL))
            p.add_argument("--seed", type=int, default=20261102)
        else:
            p.add_argument("--qrels", default=str(DEFAULT_QRELS))
            p.add_argument("--report", default="", help="output path without extension")
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
