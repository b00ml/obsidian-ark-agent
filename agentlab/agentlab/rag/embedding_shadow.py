"""Controlled P4.5-C embedding shadow runner.

The runner is deliberately separate from the production RAG tools.  It can
build a v2 shadow index, but it is a no-op plan unless both ``--execute`` and
``--allow-remote`` are supplied with an explicit embedding provider.  Plans
contain counts and hashes only; they never persist Markdown content or API
credentials.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from agentlab.rag.chunker import Chunk, make_chunker_v2
from agentlab.rag.embed import Embedder, OpenAIEmbedder
from agentlab.rag.index_store import RagIndexStore
from agentlab.runtime.config import load_config


SCHEMA = "rag-embedding-shadow-v1"
DEFAULT_INDEX_VERSION = "s1-p4.5c-v2-min64"
_EXCLUDE_DIRS = {".obsidian", ".agent-brain", ".trash", ".tmp", "node_modules"}
_ARCHIVE_REL = "ark/memory/archive/"


def _normalise_prefixes(prefixes: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize an optional Vault-relative directory allowlist."""
    values = []
    for value in prefixes or ():
        text = str(value or "").replace("\\", "/").strip().strip("/")
        if text and text not in values:
            values.append(text)
    return tuple(sorted(values))


class BudgetExceeded(RuntimeError):
    """Raised before a request would exceed the approved embedding budget."""

    budget_stop = True


@dataclass
class EmbeddingBudget:
    """Count-only wrapper around an embedder with a hard pre-call budget."""

    inner: Embedder
    max_items: int
    price_per_1k_tokens: float = 0.0
    max_cost: float = 0.0
    requests: int = 0
    items: int = 0
    input_chars: int = 0
    estimated_tokens: int = 0
    dimensions: set[int] = field(default_factory=set)
    cache_hits: int = 0
    blocked_reason: str = ""

    @staticmethod
    def estimate_tokens(texts: Sequence[str]) -> int:
        # A conservative planning approximation; provider usage remains the
        # accounting authority in a real invoice.
        return sum(max(1, math.ceil(len(text or "") / 4)) for text in texts)

    def _project(self, texts: Sequence[str]) -> tuple[int, int, float]:
        items = self.items + len(texts)
        tokens = self.estimated_tokens + self.estimate_tokens(texts)
        cost = tokens / 1000.0 * max(0.0, float(self.price_per_1k_tokens))
        return items, tokens, cost

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        values = list(texts)
        if not values:
            return []
        items, tokens, cost = self._project(values)
        if self.max_items > 0 and items > self.max_items:
            self.blocked_reason = f"embedding item budget exceeded: {items}>{self.max_items}"
            raise BudgetExceeded(self.blocked_reason)
        if self.max_cost > 0 and cost > self.max_cost:
            self.blocked_reason = f"embedding cost budget exceeded: {cost:.6f}>{self.max_cost:.6f}"
            raise BudgetExceeded(self.blocked_reason)
        started = time.perf_counter()
        vectors = self.inner.embed(values)
        elapsed = time.perf_counter() - started
        if len(vectors) != len(values):
            raise RuntimeError(f"embedding 数量不匹配: 期望 {len(values)}，得到 {len(vectors)}")
        dimensions = {len(vector) for vector in vectors if vector}
        if len(dimensions) > 1 or any(not vector for vector in vectors):
            raise RuntimeError("embedding 返回空向量或同批维度不一致")
        self.requests += 1
        self.items = items
        self.input_chars += sum(len(text or "") for text in values)
        self.estimated_tokens = tokens
        self.dimensions.update(dimensions)
        return [list(map(float, vector)) for vector in vectors]

    def summary(self) -> dict:
        estimated_cost = self.estimated_tokens / 1000.0 * max(0.0, self.price_per_1k_tokens)
        return {
            "requests": self.requests,
            "items": self.items,
            "input_chars": self.input_chars,
            "estimated_tokens": self.estimated_tokens,
            "price_per_1k_tokens": self.price_per_1k_tokens,
            "estimated_cost": round(estimated_cost, 8),
            "dimensions": sorted(self.dimensions),
            "cache_hits": self.cache_hits,
            "blocked_reason": self.blocked_reason,
        }


def _normalise_rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _iter_markdown(
    root: Path, *, include_prefixes: Sequence[str] | None = None,
) -> Iterable[tuple[str, str]]:
    allowlist = _normalise_prefixes(include_prefixes)
    for path in sorted(root.rglob("*.md")):
        rel = _normalise_rel(path, root)
        if any(part in _EXCLUDE_DIRS for part in path.relative_to(root).parts):
            continue
        if rel.startswith(_ARCHIVE_REL):
            continue
        if allowlist and not any(rel == prefix or rel.startswith(prefix + "/")
                                 for prefix in allowlist):
            continue
        try:
            yield rel, path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue


def _category(chunk: Chunk) -> str:
    if chunk.mem_id:
        return "bucket"
    if chunk.split_reason in {"fence", "fence_split"}:
        return "fence"
    if chunk.split_reason in {"table", "table_split", "table_row_hard_cut", "table_header_hard_cut"}:
        return "table"
    if chunk.split_reason in {"list", "list_split", "list_item_split", "list_item_hard_cut"}:
        return "list"
    if chunk.heading_path:
        return "heading"
    return "prose"


def collect_chunks(
    root: str | Path, *, min_chars: int = 64,
    include_prefixes: Sequence[str] | None = None,
) -> list[tuple[str, Chunk]]:
    """Parse the Vault with the candidate strategy without creating an index."""
    vault = Path(root).resolve()
    chunker = make_chunker_v2(min_chars)
    chunks: list[tuple[str, Chunk]] = []
    for rel, text in _iter_markdown(vault, include_prefixes=include_prefixes):
        parsed = chunker(text, rel)
        chunks.extend((rel, chunk) for chunk in parsed.chunks)
    return chunks


def _embedding_text(chunk: Chunk) -> str:
    """Mirror the P2 index input contract for planning/cost estimation."""
    heading = " / ".join(chunk.heading_path or [])
    tags = " ".join(chunk.tags or [])
    return f"title: {chunk.title or ''}\nheading: {heading}\ntags: {tags}\ncontent:\n{chunk.content}".strip()


def select_sample(chunks: Sequence[tuple[str, Chunk]], limit: int) -> dict:
    """Choose whole files in a deterministic, lightly stratified sample."""
    if limit <= 0:
        return {"files": [], "chunks": 0, "categories": {}, "skipped_large_files": 0}
    by_file: dict[str, list[Chunk]] = {}
    for rel, chunk in chunks:
        by_file.setdefault(rel, []).append(chunk)
    categories: dict[str, list[str]] = {}
    for rel, items in by_file.items():
        observed = {_category(item) for item in items}
        for category in observed:
            categories.setdefault(category, []).append(rel)
    for paths in categories.values():
        paths.sort(key=lambda path: (len(by_file[path]), path))

    selected: list[str] = []
    selected_set: set[str] = set()
    remaining = int(limit)
    # First take one small file from each observed structural category.
    for category in sorted(categories):
        for rel in categories[category]:
            if rel in selected_set:
                continue
            size = len(by_file[rel])
            if size <= remaining:
                selected.append(rel)
                selected_set.add(rel)
                remaining -= size
                break
    # Then fill by increasing file size, avoiding a single large file consuming
    # the entire sample.  A file is always atomic so its parser boundaries stay
    # intact during the sample run.
    for rel in sorted(by_file, key=lambda path: (len(by_file[path]), path)):
        if rel in selected_set:
            continue
        size = len(by_file[rel])
        if size <= remaining:
            selected.append(rel)
            selected_set.add(rel)
            remaining -= size
        if remaining <= 0:
            break
    chosen_chunks = [item for item in chunks if item[0] in selected_set]
    counts: dict[str, int] = {}
    for _, chunk in chosen_chunks:
        key = _category(chunk)
        counts[key] = counts.get(key, 0) + 1
    return {
        "files": sorted(selected),
        "chunks": len(chosen_chunks),
        "categories": counts,
        "skipped_large_files": sum(
            1 for rel, items in by_file.items() if rel not in selected_set and len(items) > limit
        ),
    }


def _estimate(chunks: Sequence[tuple[str, Chunk]]) -> dict:
    texts = [_embedding_text(chunk) for _, chunk in chunks]
    chars = sum(len(text) for text in texts)
    tokens = sum(max(1, math.ceil(len(text) / 4)) for text in texts)
    return {"chunks": len(chunks), "input_chars": chars, "estimated_tokens": tokens}


def _manifest(root: str | Path, *, include_prefixes: Sequence[str] | None = None) -> dict:
    """Return a content fingerprint without putting source text in reports."""
    vault = Path(root).resolve()
    rows: list[str] = []
    files = 0
    for rel, text in _iter_markdown(vault, include_prefixes=include_prefixes):
        files += 1
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        rows.append(f"{rel}\0{digest}")
    payload = "\n".join(rows).encode("utf-8")
    return {
        "files": files,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def build_plan(
    vault_root: str | Path,
    *,
    sample_limit: int = 200,
    min_chars: int = 64,
    model: str = "",
    index_version: str = DEFAULT_INDEX_VERSION,
    price_per_1k_tokens: float = 0.0,
    include_prefixes: Sequence[str] | None = None,
) -> dict:
    prefixes = _normalise_prefixes(include_prefixes)
    chunks = collect_chunks(vault_root, min_chars=min_chars,
                            include_prefixes=prefixes)
    sample = select_sample(chunks, sample_limit)
    chosen = [item for item in chunks if item[0] in set(sample["files"])]
    total = _estimate(chunks)
    sample_estimate = _estimate(chosen)
    for estimate in (total, sample_estimate):
        estimate["estimated_cost"] = round(
            estimate["estimated_tokens"] / 1000.0 * max(0.0, price_per_1k_tokens), 8
        )
    return {
        "schema": SCHEMA,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "vault_root": str(Path(vault_root).resolve()),
        "vault_manifest": _manifest(vault_root, include_prefixes=prefixes),
        "strategy": f"markdown-structure-v2-min{min_chars}",
        "model": model,
        "index_version": index_version,
        "files": len({rel for rel, _ in chunks}),
        "total": total,
        "sample": {**sample, **sample_estimate},
        "price_per_1k_tokens": price_per_1k_tokens,
        "include_prefixes": list(prefixes),
        "remote_execution": "disabled",
    }


def _make_store(args: argparse.Namespace, vault: Path, budget: EmbeddingBudget) -> RagIndexStore:
    strategy = f"markdown-structure-v2-min{args.min_chars}"
    return RagIndexStore(
        Path(args.index), budget,
        vault_root=vault,
        embedding_model=args.model,
        index_version=args.index_version,
        parser_version=strategy,
        chunker=make_chunker_v2(args.min_chars),
        chunk_strategy_version=strategy,
        include_prefixes=getattr(args, "include_prefix", []),
        batch_size=args.batch_size,
    )


def execute(args: argparse.Namespace, plan: dict, embedder: Embedder) -> dict:
    vault = Path(args.vault).resolve()
    budget = EmbeddingBudget(
        embedder,
        max_items=args.max_items,
        price_per_1k_tokens=args.price_per_1k_tokens,
        max_cost=args.max_cost,
    )
    store = _make_store(args, vault, budget)
    started = time.perf_counter()
    sample_files = set(plan["sample"]["files"])
    if args.phase == "sample":
        paths = sorted(sample_files)
        failed = 0
        embedded = 0
        cache_hits = 0
        processed = 0
        indexed_chunks = 0
        for rel in paths:
            try:
                path = vault / rel
                result = store.upsert_document(
                    rel, path.stat().st_mtime,
                    path.read_text(encoding="utf-8", errors="ignore"),
                )
                failed += int(result.get("failed", 0))
                embedded += int(result.get("embedded", 0))
                cache_hits += int(result.get("cache_hits", 0))
                indexed_chunks += int(result.get("chunks", 0))
                processed += 1
            except (OSError, RuntimeError) as exc:
                failed += 1
                # The index store records embedding failures where possible;
                # this top-level marker makes a sample budget stop explicit.
                if isinstance(exc, BudgetExceeded):
                    break
            if budget.blocked_reason:
                break
        run = {
            "phase": "sample", "files": processed,
            "chunks": indexed_chunks, "planned_chunks": plan["sample"]["chunks"],
            "failed": failed,
            "embedded": embedded, "cache_hits": cache_hits,
        }
    else:
        # Process one file per sync call so a budget stop prevents later files
        # from entering the provider path.  ``RagIndexStore`` deliberately
        # records provider failures per chunk; this loop adds the runner-level
        # stop semantics without changing the production index store.
        aggregate = {
            "total": 0, "updated": 0, "unchanged": 0, "attempted": 0,
            "deferred": 0, "removed": 0, "failed": 0, "chunks": 0,
            "new_chunks": 0, "embedded": 0, "cache_hits": 0,
            "failures": 0, "compatible": True,
        }
        latest: dict = {}
        rounds = 0
        while True:
            if budget.blocked_reason:
                break
            if budget.max_items > 0 and budget.items >= budget.max_items:
                budget.blocked_reason = (
                    f"embedding item budget reached: {budget.items}>={budget.max_items}"
                )
                break
            if budget.max_cost > 0:
                projected = budget.estimated_tokens / 1000.0 * max(
                    0.0, budget.price_per_1k_tokens
                )
                if projected >= budget.max_cost:
                    budget.blocked_reason = (
                        f"embedding cost budget reached: {projected:.6f}>={budget.max_cost:.6f}"
                    )
                    break
            latest = store.sync_vault(vault, max_files=1)
            rounds += 1
            aggregate["total"] = latest.get("total", aggregate["total"])
            aggregate["updated"] += int(latest.get("updated", 0) or 0)
            aggregate["attempted"] += int(latest.get("attempted", 0) or 0)
            aggregate["removed"] += int(latest.get("removed", 0) or 0)
            aggregate["failed"] += int(latest.get("failed", 0) or 0)
            aggregate["chunks"] = int(latest.get("chunks", aggregate["chunks"]) or 0)
            aggregate["new_chunks"] += int(latest.get("new_chunks", 0) or 0)
            aggregate["embedded"] += int(latest.get("embedded", 0) or 0)
            aggregate["cache_hits"] += int(latest.get("cache_hits", 0) or 0)
            aggregate["failures"] = int(latest.get("failures", aggregate["failures"]) or 0)
            aggregate["compatible"] = bool(latest.get("compatible", True))
            if budget.blocked_reason or int(latest.get("attempted", 0) or 0) == 0:
                break
            # A bounded guard protects against an unexpected store contract
            # returning attempted work without changing its candidate set.
            if rounds > max(1, aggregate["total"] + 1):
                break
        aggregate["deferred"] = int(latest.get("deferred", 0) or 0)
        aggregate["unchanged"] = max(
            0,
            aggregate["total"] - aggregate["attempted"] - aggregate["deferred"],
        )
        run = {"phase": "full", **aggregate}
    status = "completed"
    if int(run.get("failed", 0) or 0) or budget.blocked_reason:
        status = "partial"
    return {
        "status": status,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "run": run,
        "budget": {
            **budget.summary(),
            "cache_hits": int(run.get("cache_hits", 0) or 0),
        },
        "index": store.index_status(vault),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P4.5-C controlled embedding shadow")
    parser.add_argument("--vault", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--config", help="agentlab config.json with explicit embedding provider")
    parser.add_argument("--model", default="", help="model override; otherwise config.rag.embed_model")
    parser.add_argument("--index-version", default=DEFAULT_INDEX_VERSION)
    parser.add_argument("--min-chars", type=int, default=64)
    parser.add_argument("--phase", choices=("sample", "full"), default="sample")
    parser.add_argument("--sample-limit", type=int, default=200)
    parser.add_argument("--max-items", type=int, default=300)
    parser.add_argument("--max-cost", type=float, default=0.0)
    parser.add_argument("--price-per-1k-tokens", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--include-prefix", action="append", default=[],
        help="仅处理 Vault 下指定目录前缀；可重复，空值表示不增加白名单",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅输出计划；默认也是只计划")
    parser.add_argument("--execute", action="store_true", help="允许创建 shadow index 并调用 provider")
    parser.add_argument("--allow-remote", action="store_true", help="明确允许向配置的 embedding provider 发送正文")
    parser.add_argument("--out", help="写入 JSON 计划/结果")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_chars not in (32, 64, 80):
        raise SystemExit("--min-chars must be 32, 64, or 80")
    if args.max_items < 0 or args.max_cost < 0 or args.price_per_1k_tokens < 0:
        raise SystemExit("--max-items/--max-cost/--price-per-1k-tokens must be non-negative")
    if args.sample_limit < 0 or args.batch_size < 1:
        raise SystemExit("--sample-limit must be non-negative and --batch-size must be positive")
    cfg = load_config(args.config) if args.config else None
    model = args.model or (cfg.rag.embed_model if cfg else "")
    plan = build_plan(
        args.vault,
        sample_limit=args.sample_limit,
        min_chars=args.min_chars,
        model=model,
        index_version=args.index_version,
        price_per_1k_tokens=args.price_per_1k_tokens,
        include_prefixes=args.include_prefix,
    )
    plan["requested_phase"] = args.phase
    plan["requested_execute"] = bool(args.execute)
    plan["requested_allow_remote"] = bool(args.allow_remote)
    output: dict = {"plan": plan}
    can_execute = bool(args.execute and args.allow_remote and cfg and cfg.rag.embed_base_url and model)
    if args.execute and not can_execute:
        output["status"] = "blocked"
        output["reason"] = "execute requires --allow-remote, config.rag.embed_base_url, and model"
    elif args.execute and args.dry_run:
        output["status"] = "planned"
        output["reason"] = "--dry-run prevents provider calls even when execution flags are present"
    elif args.execute and args.phase == "sample" and args.max_items > 0 and plan["sample"]["chunks"] > args.max_items:
        output["status"] = "blocked"
        output["reason"] = (
            "sample estimate exceeds --max-items; raise the cap or reduce --sample-limit "
            "before allowing remote execution"
        )
    elif args.execute and args.phase == "full" and args.max_items > 0 and plan["total"]["chunks"] > args.max_items:
        output["status"] = "blocked"
        output["reason"] = (
            "full estimate exceeds --max-items; use --max-items 0 only with an approved "
            "unlimited item budget or set a cap at least as large as the estimate"
        )
    elif args.execute and args.max_cost > 0 and plan["sample" if args.phase == "sample" else "total"]["estimated_cost"] > args.max_cost:
        output["status"] = "blocked"
        estimate = plan["sample" if args.phase == "sample" else "total"]["estimated_cost"]
        output["reason"] = (
            f"{args.phase} estimate exceeds --max-cost; estimated={estimate:.8f} "
            f"limit={args.max_cost:.8f}"
        )
    elif not args.execute or args.dry_run:
        output["status"] = "planned"
        output["reason"] = "no provider call; pass --execute --allow-remote with explicit config to run"
    else:
        assert cfg is not None
        # Keep the resolved model in the namespace used by ``execute`` so the
        # index metadata cannot accidentally be created with an empty model.
        args.model = model
        embedder = OpenAIEmbedder(
            cfg.rag.embed_base_url,
            model,
            api_key=cfg.rag.effective_key(),
            timeout=cfg.rag.embed_timeout,
            batch_size=args.batch_size,
        )
        output["execution"] = execute(args, plan, embedder)
        output["status"] = output["execution"]["status"]
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return {"planned": 0, "completed": 0, "partial": 1, "blocked": 2}.get(
        output["status"], 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
