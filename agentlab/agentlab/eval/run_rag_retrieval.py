"""P0/S1-0 检索评测运行器。

评测只读产物，不修改 Vault 或线上索引。默认本地路是
``vault_search + memory_query``，名称明确为 ``local_combined``；RRF
在未传 ``--p2-index`` 时显式记为 unavailable，传入 P2 index 后才运行
hybrid shadow 对照，避免把空路由当成质量基线。

用法：
  python -m agentlab.eval.run_rag_retrieval \
    --vault-root E:/peik1_books [--tasks PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import re
import statistics
import subprocess
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path


RUNNER_VERSION = "p4-provenance-v5"
RRF_UNAVAILABLE = (
    "brain_config_not_injected: current build_rag_tools/RAGRecall path is not "
    "wired for an evaluator-quality RRF baseline"
)


def _p2_store_from_index(index_path: str | Path, vault_root: str | Path, embedder=None):
    """Open a derived P2 index using its own metadata contract."""
    from agentlab.rag.chunker import make_chunker_v2
    from agentlab.rag.index_store import RagIndexStore

    path = Path(index_path)
    meta: dict[str, str] = {}
    try:
        conn = sqlite3.connect(path)
        try:
            rows = conn.execute("SELECT key,value FROM index_meta").fetchall()
            meta = {str(key): str(value) for key, value in rows}
        finally:
            conn.close()
    except (OSError, sqlite3.Error):
        meta = {}
    strategy = meta.get("chunk_strategy_version") or meta.get("parser_version") or "markdown-structure-v1"
    chunker = None
    # Both the original runner and the v2 shadow runner have existed in the
    # wild; accept the historical ``min64`` spelling and the canonical
    # ``min-64`` spelling when reconstructing the chunker.
    match = re.search(r"markdown-structure-v2-min-?(32|64|80)$", strategy)
    if match:
        chunker = make_chunker_v2(int(match.group(1)))
    try:
        scope = list(json.loads(meta.get("scope_prefixes", "[]")))
    except (TypeError, ValueError, json.JSONDecodeError):
        scope = []
    return RagIndexStore(
        path,
        embedder,
        vault_root=vault_root,
        embedding_model=meta.get("embedding_model") or None,
        index_version=meta.get("index_version") or "s1-p2-v1",
        parser_version=meta.get("parser_version") or strategy,
        chunker=chunker,
        chunk_strategy_version=strategy,
        include_prefixes=scope,
    )


def _load_tasks(path: str, limit: int | None) -> list[dict]:
    tasks = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            tasks.append(json.loads(line))
    # S1 schema fields are additive and migrated here for direct evaluator
    # callers that do not run the standalone schema CLI first.
    from agentlab.eval.rag_task_schema import migrate_v1_task, validate_tasks
    tasks = [migrate_v1_task(task) for task in tasks]
    audit = validate_tasks(tasks)
    if not audit["valid"]:
        raise ValueError("invalid retrieval task set: " + "; ".join(audit["errors"][:8]))
    return tasks[:limit] if limit else tasks


def _norm_ref(ref: str, *, keep_anchor: bool = True) -> str:
    """规范化引用；默认保留 ``#mem-id``，避免桶条目退化成文件级。"""
    text = str(ref or "").strip().replace("\\", "/")
    if not keep_anchor:
        text = text.split("#", 1)[0]
    return text


def _file_ref(ref: str) -> str:
    return _norm_ref(ref, keep_anchor=False)


def _entry_ref(ref: str) -> str:
    return _norm_ref(ref, keep_anchor=True)


def _relative_ref(ref: str, cfg: dict) -> str:
    """把 brain/向量路可能返回的绝对路径归一到 Vault 相对路径。"""
    text = _norm_ref(ref)
    path_part, anchor = (text.split("#", 1) + [""])[:2]
    root = Path(str(cfg.get("vault_path") or ".")).resolve()
    candidate = Path(path_part)
    if candidate.is_absolute():
        try:
            path_part = candidate.resolve().relative_to(root).as_posix()
        except ValueError:
            path_part = candidate.as_posix()
    else:
        path_part = _norm_ref(path_part)
    return f"{path_part}#{anchor}" if anchor else path_part


def _canonical_memory_ref(mem_id: str, cfg: dict, store=None) -> str:
    if not mem_id:
        return ""
    if store is None:
        return f"memory#{mem_id}"
    try:
        fp = store._find_memory_file(mem_id)
    except Exception:
        fp = None
    if fp is None and store is not None:
        # MarkdownStore intentionally does not resolve a single entry inside a
        # bucket. The evaluator still needs a stable path#mem-id reference, so
        # scan bucket headers read-only without changing runtime semantics.
        try:
            for candidate in (Path(store.memory_root)).rglob("*.md"):
                if "archive" in candidate.parts:
                    continue
                text = candidate.read_text(encoding="utf-8", errors="ignore")
                if re.search(rf"^## {re.escape(mem_id)}\s*$", text, flags=re.M):
                    fp = candidate
                    break
        except OSError:
            fp = None
    if fp is None:
        return f"memory#{mem_id}"
    return f"{_relative_ref(str(fp), cfg)}#{mem_id}"


def _keyword_search(cfg, query: str, k: int) -> tuple[list[str], float]:
    from agentlab.tools.connectors.brain_tools import _add_brain_path
    _add_brain_path()
    t0 = time.perf_counter()
    import tools_vault
    result = tools_vault.vault_search(cfg, query, limit=k)
    elapsed = time.perf_counter() - t0
    rows = result.get("results", []) if isinstance(result, dict) else []
    return [_relative_ref(row.get("path", ""), cfg) for row in rows], elapsed


def _memory_search(cfg, query: str, k: int) -> tuple[list[str], float]:
    """Run the Markdown memory query without mutating source metadata."""
    from agentlab.memory.markdown_store import MemoryMarkdownStore

    store = MemoryMarkdownStore(str(cfg["vault_path"]), create_dirs=False)
    t0 = time.perf_counter()
    rows = store.query(
        query,
        limit=k,
        project_id=cfg.get("project_id"),
        track_access=False,
    )
    elapsed = time.perf_counter() - t0
    refs = []
    for item in rows:
        refs.append(_canonical_memory_ref(str(item.get("id", "")), cfg, store))
    return [r for r in refs if r], elapsed


def _rag_retrieve(
    cfg,
    query: str,
    k: int,
    required_groups: list[list[str]] | None = None,
) -> tuple[list[str], float, str | None]:
    """Run the optional P2 hybrid route, otherwise stay explicitly unavailable.

    The baseline remains unchanged unless the caller injects a
    ``HybridRetriever``.  This keeps old reports honest while allowing P4
    shadow evaluation against the versioned P2 index.
    """
    retriever = cfg.get("_hybrid_retriever")
    if cfg.get("_task_has_independent_vault"):
        return [], 0.0, "not_applicable: independent_vault_fixture"
    if retriever is not None:
        store_root = getattr(getattr(retriever, "store", None), "vault_root", None)
        task_root = cfg.get("vault_path")
        if store_root and task_root:
            try:
                if Path(store_root).resolve() != Path(task_root).resolve():
                    return [], 0.0, "not_applicable: index_vault_scope_mismatch"
            except OSError:
                return [], 0.0, "not_applicable: index_vault_scope_unknown"
        try:
            record = retriever.shadow_record(
                query,
                limit=k,
                project_id=cfg.get("project_id"),
                statuses=cfg.get("statuses"),
                required_groups=required_groups,
            )
            cfg["_last_hybrid_shadow"] = record
            return list(record.get("hybrid_refs", [])), float(record.get("latency_ms", 0.0)) / 1000, None
        except Exception as exc:
            return [], 0.0, f"p2_hybrid_error: {type(exc).__name__}: {exc}"
    return [], 0.0, RRF_UNAVAILABLE


def _ref_matches(expected: str, got: str) -> bool:
    expected = _entry_ref(expected)
    got = _entry_ref(got)
    if "#" in expected:
        # Gold qrels are normally entry-level (``#mem-id`` or a heading),
        # while retrieval returns a concrete chunk ref
        # (``#mem-id:ch<content-hash>``).  Match within the same entry without
        # collapsing two different bucket entries or two different files.
        return got == expected or got.startswith(expected + ":ch")
    return _file_ref(got) == expected


def _expectation_sets(task: dict) -> list[set[str]]:
    """把评测条目的 all/any/groups 策略转成可评分的条件集合。"""
    raw_refs = [_entry_ref(ref) for ref in task.get("expected_refs", []) if ref]
    groups = task.get("expected_groups")
    if groups:
        return [{_entry_ref(ref) for ref in group if ref} for group in groups if group]
    if task.get("expected_policy", "all") == "any":
        return [set(raw_refs)] if raw_refs else []
    return [{ref} for ref in raw_refs]


def _recall_at(expectations: list[set[str]], got: list[str], k: int) -> float:
    if not expectations:
        return 1.0 if not got[:k] else 0.0
    top = got[:k]
    satisfied = sum(
        1 for group in expectations
        if any(_ref_matches(expected, actual) for expected in group for actual in top)
    )
    return satisfied / len(expectations)


def _mrr(expectations: list[set[str]], got: list[str]) -> float:
    if not expectations:
        return 0.0
    for index, actual in enumerate(got, 1):
        if any(_ref_matches(expected, actual) for group in expectations for expected in group):
            return 1.0 / index
    return 0.0


def _qrels(task: dict) -> dict[str, int]:
    """Return normalized graded relevance labels for a task.

    ``qrels`` is an optional list of ``{"ref": ..., "relevance": 0..2}``.
    Legacy expected_refs remain binary relevance-1 labels so historical reports
    stay comparable.  Duplicate refs keep the highest supplied grade.
    """
    raw = task.get("qrels")
    labels: dict[str, int] = {}
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict) or not item.get("ref"):
                continue
            try:
                grade = max(0, int(item.get("relevance", 0)))
            except (TypeError, ValueError):
                grade = 0
            ref = _entry_ref(str(item["ref"]))
            labels[ref] = max(labels.get(ref, 0), grade)
    if labels:
        return labels
    for group in _expectation_sets(task):
        for ref in group:
            labels[_entry_ref(ref)] = max(labels.get(_entry_ref(ref), 0), 1)
    return labels


def _graded_score(expected: str, got: str) -> bool:
    return _ref_matches(expected, got)


def _ndcg_at(task: dict, got: list[str], k: int) -> float:
    labels = _qrels(task)
    if not labels:
        return 0.0
    used: set[str] = set()
    retrieved: list[int] = []
    for actual in got[:k]:
        matches = [
            (ref, grade) for ref, grade in labels.items()
            if ref not in used and grade > 0 and _graded_score(ref, actual)
        ]
        if not matches:
            retrieved.append(0)
            continue
        ref, grade = max(matches, key=lambda item: item[1])
        used.add(ref)
        retrieved.append(grade)
    dcg = sum(
        (2 ** grade - 1) / math.log2(index + 2)
        for index, grade in enumerate(retrieved)
        if grade > 0
    )
    ideal = sorted(labels.values(), reverse=True)[:k]
    idcg = sum(
        (2 ** grade - 1) / math.log2(index + 2)
        for index, grade in enumerate(ideal)
        if grade > 0
    )
    return dcg / idcg if idcg else 0.0


def _map_at(task: dict, got: list[str], k: int) -> float:
    labels = _qrels(task)
    relevant = {ref for ref, grade in labels.items() if grade > 0}
    if not relevant:
        return 0.0
    seen: set[str] = set()
    hits = 0
    precision_sum = 0.0
    for index, actual in enumerate(got[:k], 1):
        matched = next((ref for ref in relevant if _graded_score(ref, actual)), None)
        if matched is None or matched in seen:
            continue
        seen.add(matched)
        hits += 1
        precision_sum += hits / index
    return precision_sum / len(relevant)


def _merge_refs(*lists: list[str]) -> list[str]:
    return list(dict.fromkeys(ref for values in lists for ref in values if ref))


def _bucket_dup_ratio(refs: list[str]) -> float | None:
    bucket_files = [_file_ref(ref) for ref in refs if "/sessions/" in _file_ref(ref)]
    if not bucket_files:
        return None
    counts = Counter(bucket_files)
    duplicated = sum(count - 1 for count in counts.values() if count > 1)
    return round(duplicated / len(bucket_files), 3)


def _route_for_task(cfg: dict, task: dict, query: str, k: int) -> tuple[str, list[str], list[str], float, dict]:
    route = task.get("route", "local_combined")
    if route == "memory_only":
        refs, elapsed = _memory_search(cfg, query, k)
        return "memory_keyword", refs, refs, elapsed, {"memory_keyword": (refs, elapsed)}
    if route == "vault_only":
        refs, elapsed = _keyword_search(cfg, query, k)
        return "vault_keyword", refs, refs, elapsed, {"vault_keyword": (refs, elapsed)}
    refs_vault, dt_vault = _keyword_search(cfg, query, k)
    refs_memory, dt_memory = _memory_search(cfg, query, k)
    raw = refs_vault + refs_memory
    return (
        "local_combined",
        _merge_refs(refs_vault, refs_memory),
        raw,
        dt_vault + dt_memory,
        {
            "vault_keyword": (refs_vault, dt_vault),
            "memory_keyword": (refs_memory, dt_memory),
        },
    )


def _freshness_probe(cfg: dict, task: dict) -> dict | None:
    spec = task.get("freshness")
    if not spec:
        return None
    root = Path(str(cfg.get("vault_path") or ".")).resolve()
    threshold = dt.datetime.fromisoformat(spec["modified_after"].replace("Z", "+00:00"))
    observations = []
    for ref in task.get("expected_refs", []):
        path = root / _file_ref(ref)
        if not path.exists():
            observations.append({"ref": _file_ref(ref), "exists": False, "mtime_ok": False})
            continue
        modified = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
        observations.append({
            "ref": _file_ref(ref),
            "exists": True,
            "mtime": modified.isoformat(),
            "mtime_ok": modified >= threshold.astimezone(dt.timezone.utc),
        })
    result = {
        "status": "source_mtime_only",
        "modified_after": threshold.isoformat(),
        "max_index_lag_seconds": spec.get("max_index_lag_seconds"),
        "index_mtime": None,
        "observations": observations,
    }
    store = cfg.get("_p2_store")
    db_path = getattr(store, "db_path", None) if store is not None else None
    if store is None or db_path is None or not Path(db_path).exists():
        result["index"] = {
            "status": "unavailable",
            "reason": "p2_index_not_supplied",
        }
    else:
        store_root = getattr(store, "vault_root", None)
        try:
            same_root = store_root is None or Path(store_root).resolve() == root
        except OSError:
            same_root = False
        if not same_root:
            result["index"] = {
                "status": "not_applicable",
                "reason": "index_vault_scope_mismatch",
            }
        else:
            try:
                status = store.index_status(root)
                result["index"] = {
                    "status": "available",
                    "coverage": status.get("coverage"),
                    "fresh_files": status.get("fresh_files"),
                    "stale_files": status.get("stale_files"),
                    "index_version": status.get("index_version"),
                }
            except Exception as exc:
                result["index"] = {
                    "status": "unavailable",
                    "reason": f"index_status_error: {type(exc).__name__}",
                }
    return result


def _task_cfg(base_cfg: dict, task: dict) -> dict:
    cfg = dict(base_cfg)
    cfg["_task_has_independent_vault"] = bool(task.get("vault_root"))
    task_root = Path(str(base_cfg.get("_tasks_root") or "."))
    task_vault = task.get("vault_root")
    if task_vault:
        path = Path(str(task_vault))
        if not path.is_absolute():
            path = (task_root / path).resolve()
        cfg["vault_path"] = str(path)
    if task.get("project_id"):
        cfg["project_id"] = task["project_id"]
    cfg.update(task.get("filters") or {})
    return cfg


def _score_route(
    row: dict,
    field: str,
    expectations: list[set[str]],
    refs: list[str],
    elapsed: float,
    *,
    status: str = "available",
    reason: str = "",
) -> None:
    if not expectations:
        r5 = 1.0 if not refs[:5] else 0.0
        r10 = 1.0 if not refs[:10] else 0.0
    else:
        r5 = _recall_at(expectations, refs, 5)
        r10 = _recall_at(expectations, refs, 10)
    row[field] = {
        "status": status,
        "refs": refs,
        "recall@5": r5,
        "recall@10": r10,
        "mrr": _mrr(expectations, refs),
        "ndcg@5": _ndcg_at(row.get("_task", {}), refs, 5),
        "ndcg@10": _ndcg_at(row.get("_task", {}), refs, 10),
        "map@10": _map_at(row.get("_task", {}), refs, 10),
        "latency_ms": round(elapsed * 1000, 1),
    }
    if reason:
        row[field]["reason"] = reason


def evaluate(
    cfg: dict,
    tasks: list[dict],
    k: int = 10,
    *,
    use_eval_coverage_contract: bool = True,
) -> dict:
    from agentlab.tools.connectors.brain_tools import _add_brain_path
    _add_brain_path()
    per_query = []
    for task in tasks:
        query = task["query"]
        expectations = _expectation_sets(task)
        cfg_for_task = _task_cfg(cfg, task)
        route_name, refs, raw_refs, elapsed, components = _route_for_task(
            cfg_for_task, task, query, k
        )
        row = {
            "id": task["id"],
            "query_type": task.get("query_type", ""),
            "target_scope": task.get("target_scope", "file"),
            "route": route_name,
            "expected_policy": task.get("expected_policy", "all"),
            "expected_count": len(task.get("expected_refs", [])),
            "expected_conditions": len(expectations),
            "vault_root": cfg_for_task.get("vault_path"),
            "eval_coverage_contract": (
                "expected_refs" if use_eval_coverage_contract else "disabled"
            ),
            "_task": task,
        }
        if task.get("negative_kind"):
            row["negative_kind"] = task["negative_kind"]
        row["route_statuses"] = {}
        _score_route(row, route_name, expectations, refs[:k], elapsed)
        row["route_statuses"][route_name] = "available"
        for component, (component_refs, component_elapsed) in components.items():
            _score_route(row, component, expectations, component_refs[:k], component_elapsed)
            row["route_statuses"][component] = "available"
        dup = _bucket_dup_ratio(raw_refs)
        if dup is not None:
            row["bucket_dup_ratio"] = dup

        forbidden = [_entry_ref(ref) for ref in task.get("forbidden_refs", []) if ref]
        if forbidden:
            row["forbidden_hits@5"] = sum(
                1 for got in refs[:5]
                if any(_ref_matches(expected, got) for expected in forbidden)
            )
            row["forbidden_refs"] = forbidden
        freshness = _freshness_probe(cfg_for_task, task)
        if freshness is not None:
            row["freshness"] = freshness

        cfg_for_task.pop("_last_hybrid_shadow", None)
        rrf_refs, rrf_elapsed, rrf_status = _rag_retrieve(
            cfg_for_task,
            query,
            k,
            required_groups=expectations if use_eval_coverage_contract else None,
        )
        if rrf_status:
            status = "not_applicable" if rrf_status.startswith("not_applicable:") else "unavailable"
            row["rrf"] = {"status": status, "reason": rrf_status}
            row["route_statuses"]["rrf"] = status
            if status == "not_applicable":
                row["route_statuses"]["p2_lexical"] = "not_applicable"
                row["route_statuses"]["p2_vector"] = "not_applicable"
        else:
            if cfg_for_task.get("_last_hybrid_shadow"):
                shadow = cfg_for_task["_last_hybrid_shadow"]
                row["hybrid_shadow"] = shadow
                if shadow.get("hybrid_status") == "available":
                    _score_route(
                        row, "rrf", expectations, rrf_refs[:k], rrf_elapsed,
                    )
                    row["route_statuses"]["rrf"] = "available"
                else:
                    row["rrf"] = {
                        "status": "unavailable",
                        "reason": shadow.get("hybrid_status"),
                        "refs": rrf_refs[:k],
                    }
                    row["route_statuses"]["rrf"] = "unavailable"
                for route_name in ("lexical", "vector"):
                    route = shadow.get("routes", {}).get(route_name)
                    metric_name = f"p2_{route_name}"
                    if route is not None:
                        route_status = str(route.get("status") or "unavailable")
                        row["route_statuses"][metric_name] = route_status
                    if route is not None and route.get("status") == "available":
                        _score_route(
                            row,
                            metric_name,
                            expectations,
                            route.get("refs", [])[:k],
                            float(route.get("latency_ms", 0.0)) / 1000,
                        )
                    elif route is not None and route.get("status") not in {"unavailable", "not_applicable"}:
                        row["route_statuses"][metric_name] = "unavailable"
        row["vector"] = "deferred_to_p2"
        row.pop("_task", None)
        per_query.append(row)

    def _status_counts(mode: str) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for row in per_query:
            metric = row.get(mode)
            if isinstance(metric, dict):
                counts[str(metric.get("status") or "available")] += 1
                continue
            status = row.get("route_statuses", {}).get(mode)
            if status:
                counts[str(status)] += 1
        return dict(sorted(counts.items()))

    def _overall_status(counts: dict[str, int]) -> str:
        if not counts:
            return "unavailable"
        if len(counts) == 1:
            return next(iter(counts))
        return "mixed"

    def _metric_summary(mode: str) -> dict:
        status_counts = _status_counts(mode)
        rows = [
            row for row in per_query
            if row.get("expected_count", 0) > 0
            and isinstance(row.get(mode), dict)
            and row[mode].get("status", "available") == "available"
            and all(field in row[mode] for field in (
                "recall@5", "recall@10", "mrr", "ndcg@5", "ndcg@10", "map@10"
            ))
        ]
        result = {
            "tasks": len(rows),
            "status": _overall_status(status_counts),
            "status_counts": status_counts,
        }
        if not rows:
            return result
        vals = {field: [row[mode][field] for row in rows] for field in (
            "recall@5", "recall@10", "mrr", "ndcg@5", "ndcg@10", "map@10"
        )}
        p50, p95 = _p50_p95([row[mode]["latency_ms"] for row in rows])
        result.update({
            "recall@5": round(statistics.mean(vals["recall@5"]), 4),
            "recall@10": round(statistics.mean(vals["recall@10"]), 4),
            "mrr": round(statistics.mean(vals["mrr"]), 4),
            "ndcg@5": round(statistics.mean(vals["ndcg@5"]), 4),
            "ndcg@10": round(statistics.mean(vals["ndcg@10"]), 4),
            "map@10": round(statistics.mean(vals["map@10"]), 4),
            "p50_ms": p50,
            "p95_ms": p95,
        })
        return result

    negative_rows = [row for row, task in zip(per_query, tasks) if not task.get("expected_refs")]
    route_modes = sorted({
        mode
        for row in per_query
        for mode in set(row.get("route_statuses", {})) | {
            key for key, value in row.items() if isinstance(value, dict) and "refs" in value
        }
    })
    negative_by_route = {}
    for mode in route_modes:
        status_counts = _status_counts(mode)
        rows = [
            row for row in negative_rows
            if isinstance(row.get(mode), dict)
            and row[mode].get("status", "available") == "available"
        ]
        result = {
            "tasks": len(rows),
            "status": _overall_status(status_counts),
            "status_counts": status_counts,
        }
        if rows:
            result.update({
                "clean@5": round(sum(not row[mode]["refs"][:5] for row in rows) / len(rows), 4),
                "clean@10": round(sum(not row[mode]["refs"][:10] for row in rows) / len(rows), 4),
            })
        negative_by_route[mode] = result

    forbidden_rows = [row for row in per_query if "forbidden_refs" in row]
    forbidden_hits = sum(int(row.get("forbidden_hits@5", 0)) for row in forbidden_rows)
    duplicate_rows = [row["bucket_dup_ratio"] for row in per_query if row.get("bucket_dup_ratio") is not None]

    return {
        "schema": "rag-retrieval-baseline-v2",
        "summary": {
            "local_combined": _metric_summary("local_combined"),
            "vault_keyword": _metric_summary("vault_keyword"),
            "memory_keyword": _metric_summary("memory_keyword"),
            "p2_lexical": _metric_summary("p2_lexical"),
            "p2_vector": _metric_summary("p2_vector"),
            "rrf": _metric_summary("rrf"),
            "vector": "deferred_to_p2",
            "negative": negative_by_route,
            "route_statuses": {
                mode: {
                    "status": _overall_status(_status_counts(mode)),
                    "status_counts": _status_counts(mode),
                }
                for mode in route_modes
            },
            "forbidden": {
                "tasks": len(forbidden_rows),
                "hits@5": forbidden_hits,
                "clean@5": round(
                    sum(int(row.get("forbidden_hits@5", 0)) == 0 for row in forbidden_rows)
                    / len(forbidden_rows), 4
                ) if forbidden_rows else None,
            },
            "bucket_duplicates": {
                "tasks": len(duplicate_rows),
                "mean_ratio": round(statistics.mean(duplicate_rows), 4) if duplicate_rows else None,
                "max_ratio": max(duplicate_rows) if duplicate_rows else None,
            },
            "scope": {
                "total_tasks": len(tasks),
                "positive_tasks": sum(bool(task.get("expected_refs")) for task in tasks),
                "negative_tasks": sum(not bool(task.get("expected_refs")) for task in tasks),
                "fixture_tasks": sum(bool(task.get("vault_root")) for task in tasks),
            },
        },
        "per_query": per_query,
    }


def _p50_p95(vals: list[float]) -> tuple[float | None, float | None]:
    if not vals:
        return None, None
    values = sorted(vals)
    return (
        round(values[int(0.5 * (len(values) - 1))], 1),
        round(values[min(len(values) - 1, int(0.95 * (len(values) - 1)))], 1),
    )


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip())
    except Exception:
        return None


def _sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="S1-0 rag retrieval baseline runner")
    parser.add_argument("--vault-root", required=True)
    parser.add_argument("--tasks", default=".ai/evals/rag_retrieval.jsonl")
    parser.add_argument("--out", help="write JSON report to this path")
    parser.add_argument("--inventory", help="inventory JSON snapshot to embed in metadata")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--p2-index", help="optional P2 SQLite index for hybrid shadow evaluation")
    parser.add_argument("--config", help="agentlab config.json for optional embedding provider")
    parser.add_argument("--shadow-log", help="optional bounded JSONL path for hybrid shadow metadata")
    parser.add_argument("--vector-mode", choices=("off", "shadow", "on"), default="shadow")
    parser.add_argument("--lexical-mode", choices=("off", "on"), default="on")
    parser.add_argument("--hybrid-candidate-k", type=int, default=40)
    parser.add_argument(
        "--vector-min-score", type=float, default=0.0,
        help="offline vector cosine floor; 0 keeps all positive scores",
    )
    parser.add_argument(
        "--lexical-min-coverage", type=float, default=0.0,
        help="offline lexical evidence floor; 0 keeps the legacy candidate behavior",
    )
    parser.add_argument("--small-to-big-mode", choices=("off", "shadow", "on"), default="shadow")
    parser.add_argument("--small-to-big-neighbors", type=int, default=1)
    parser.add_argument("--small-to-big-max-chars", type=int, default=2400)
    parser.add_argument("--vector-fallback-mode", choices=("off", "on"), default="off",
                        help="guarded semantic vector display while vector_mode=shadow; default off")
    parser.add_argument("--dedupe-by", choices=("entry", "file", "ref"), default="entry")
    parser.add_argument(
        "--disable-eval-coverage-contract",
        action="store_true",
        help="do not pass expected refs into HybridRetriever; required for production-shaped evidence",
    )
    args = parser.parse_args()

    tasks_path = Path(args.tasks).resolve()
    cfg = {"vault_path": args.vault_root, "_tasks_root": str(tasks_path.parent)}
    if args.p2_index:
        from agentlab.rag.hybrid import HybridRetriever, ShadowLogWriter
        from agentlab.rag.index_store import RagIndexStore

        embedder = None
        loaded = None
        if args.config:
            from agentlab.runtime.config import load_config
            loaded = load_config(args.config)
            if loaded.rag.embed_base_url:
                from agentlab.rag.embed import OpenAIEmbedder
                embedder = OpenAIEmbedder(
                    loaded.rag.embed_base_url, loaded.rag.embed_model,
                    api_key=loaded.rag.effective_key(), timeout=loaded.rag.embed_timeout,
                )
        store = _p2_store_from_index(args.p2_index, args.vault_root, embedder)
        cfg["_p2_store"] = store
        cfg["_p2_index_path"] = str(Path(args.p2_index).resolve())
        shadow_log_path = args.shadow_log
        if not shadow_log_path and loaded is not None:
            shadow_log_path = loaded.rag.shadow_log_path
        shadow_logger = ShadowLogWriter(shadow_log_path) if shadow_log_path else None
        cfg["_hybrid_retriever"] = HybridRetriever(
            store,
            vector_mode=args.vector_mode,
            lexical_mode=args.lexical_mode,
            candidate_k=args.hybrid_candidate_k,
            dedupe_by=args.dedupe_by,
            vector_min_score=args.vector_min_score,
            lexical_min_coverage=args.lexical_min_coverage,
            small_to_big_mode=args.small_to_big_mode,
            small_to_big_neighbors=args.small_to_big_neighbors,
            small_to_big_max_chars=args.small_to_big_max_chars,
            vector_fallback_mode=args.vector_fallback_mode,
            shadow_logger=shadow_logger,
        )
    tasks = _load_tasks(str(tasks_path), args.limit)
    report = evaluate(
        cfg,
        tasks,
        use_eval_coverage_contract=not args.disable_eval_coverage_contract,
    )
    inventory = None
    if args.inventory:
        inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    index_provenance = None
    if args.p2_index:
        store = cfg.get("_p2_store")
        index_path = Path(args.p2_index).resolve()
        if store is not None and index_path.exists():
            try:
                index_provenance = store.index_status(args.vault_root)
                if not index_provenance.get("metadata_valid", False):
                    index_provenance = {
                        "status": "unavailable",
                        "path": str(index_path),
                        "reason": "index_metadata_missing_or_invalid",
                        **index_provenance,
                    }
                else:
                    index_provenance = {
                        "status": "available",
                        "path": str(index_path),
                        **index_provenance,
                    }
            except Exception as exc:
                index_provenance = {
                    "status": "unavailable",
                    "path": str(index_path),
                    "reason": f"index_status_error: {type(exc).__name__}",
                }
        else:
            index_provenance = {
                "status": "unavailable",
                "path": str(index_path),
                "reason": "index_missing",
            }
    if index_provenance and index_provenance.get("status") != "unavailable":
        chunking = {
            "strategy": index_provenance.get("chunk_strategy_version", ""),
            "parser_version": index_provenance.get("parser_version", ""),
            "index_version": index_provenance.get("index_version", ""),
            "source": "p2_index",
        }
        embedding_model = index_provenance.get("embedding_model") or None
    else:
        chunking = {
            "strategy": "legacy-600-estimate",
            "chunk_chars": 600,
            "paragraph_split": "blank-line",
            "hard_split": True,
            "source": "local_baseline",
        }
        embedding_model = None
    report["meta"] = {
        "runner_version": RUNNER_VERSION,
        "tasks_path": str(tasks_path),
        "vault_root": str(Path(args.vault_root).resolve()),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "snapshot_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": sys.argv,
        "hardware": {"platform": platform.platform(), "python": platform.python_version()},
        "chunking": chunking,
        "embedding_model": embedding_model,
        "index": index_provenance,
        "inventory_path": str(Path(args.inventory).resolve()) if args.inventory else None,
        "vault_snapshot": {
            "status": "available" if inventory is not None else "unavailable",
            "generated_at": inventory.get("generated_at") if inventory else None,
            "scanner_version": inventory.get("scanner_version") if inventory else None,
        },
        "input_hashes": {
            "runner_sha256": _sha256_file(__file__),
            "tasks_sha256": _sha256_file(tasks_path),
            "inventory_sha256": _sha256_file(args.inventory),
            "index_sha256": _sha256_file(args.p2_index),
        },
        "inventory": inventory,
        "notes": {
            "local_combined": "vault_search + memory_query; positive metrics exclude negative tasks",
            "negative": "clean@k is strict candidate-list emptiness, not answer-level abstention",
            "rrf": (
                "p2 hybrid shadow route"
                if args.p2_index else f"unavailable: {RRF_UNAVAILABLE}"
            ),
            "freshness": "source mtime is checked where declared; index lag remains unavailable until an index is supplied",
            "refs": "file refs match by path; entry refs require exact path#mem-id",
            "eval_coverage_contract": (
                "expected refs are reserved into display candidates; diagnostic only"
                if not args.disable_eval_coverage_contract else
                "disabled: expected refs are not passed to HybridRetriever; eligible for production-shaped comparison"
            ),
        },
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"[BASELINE] 已写入 {args.out}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
