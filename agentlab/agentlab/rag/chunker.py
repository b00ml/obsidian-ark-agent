"""Markdown structure-aware chunking (P1.1, ``markdown-structure-v1``).

The module is independent from the legacy ``vector_index`` chunker. It
produces stable, inspectable chunk objects for the shadow index; production
indexing is not switched to this strategy until P2 evaluation.
"""
from __future__ import annotations

import hashlib
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

STRATEGY_V1 = "markdown-structure-v1"
STRATEGY_V2 = "markdown-structure-v2"
TARGET_CHARS = 500
HARD_CHARS = 800
OVERLAP_CHARS = 80
V2_MIN_CHARS = (32, 64, 80)

_MEM_HEAD_RE = re.compile(r"^##\s+(mem-[0-9a-f]+)\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")
_CALLOUT_RE = re.compile(r"^\s*>\s*\[!")
_SENT_CUT_RE = re.compile(r"(?<=[。！？!?])|(?<=\.)(?=\s|$)")


@dataclass
class Chunk:
    content: str
    chunk_id: str
    source_ref: str
    entry_ref: str
    mem_id: str | None
    heading_path: list[str]
    anchor: str
    chunk_index: int
    split_reason: str
    entry_index: int = 0
    title: str = ""
    tags: list[str] = field(default_factory=list)
    start_offset: int = 0
    end_offset: int = 0
    meta: dict = field(default_factory=dict)
    diagnostic: bool = False
    # Appended after the legacy defaults so positional construction remains
    # compatible with the pre-diagnostics Chunk contract.
    context_header: str = ""


@dataclass
class ParseResult:
    chunks: list[Chunk] = field(default_factory=list)
    frontmatter: dict = field(default_factory=dict)
    duplicate_mem_ids: list[str] = field(default_factory=list)
    orphan_entry_count: int = 0
    entry_count: int = 0
    coverage_ratio: float = 1.0
    zero_chunk_files: int = 0
    source_chars: int = 0
    covered_chars: int = 0
    bucket_mode: bool = False
    invalid_entry_count: int = 0
    strategy_version: str = STRATEGY_V1
    validation: "ChunkValidation" = field(default_factory=lambda: ChunkValidation())


@dataclass(frozen=True)
class ChunkValidation:
    """Deterministic acceptance/diagnostic result for a chunk set.

    This is deliberately a pure data contract.  It does not pick a fallback
    strategy or mutate the source, which keeps preview and indexing free to
    share the same parser while letting callers decide whether a warning is
    acceptable for their route.
    """

    ok: bool = False
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    chunk_count: int = 0
    min_chars: int = 0
    max_chars: int = 0
    avg_chars: float = 0.0
    coverage_ratio: float = 1.0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "chunk_count": self.chunk_count,
            "min_chars": self.min_chars,
            "max_chars": self.max_chars,
            "avg_chars": self.avg_chars,
            "coverage_ratio": self.coverage_ratio,
        }


def validate_chunks(
    chunks: list[Chunk] | tuple[Chunk, ...],
    total_chars: int,
    *,
    coverage_ratio: float = 1.0,
    target_chars: int = TARGET_CHARS,
    hard_chars: int = HARD_CHARS,
) -> ChunkValidation:
    """Validate parser output using stable, explainable heuristics.

    The checks mirror WeKnora's validator boundary without copying its Go
    implementation: broken output is reported as a reason, while benign
    source diagnostics remain warnings.  No fallback is performed here.
    """
    values = [len(item.content) for item in chunks if item and item.content]
    if not values:
        return ChunkValidation(ok=False, reasons=("no_chunks",), coverage_ratio=coverage_ratio)

    target = max(1, int(target_chars))
    hard = max(target, int(hard_chars))
    count = len(values)
    minimum, maximum = min(values), max(values)
    average = round(sum(values) / count, 1)
    reasons: list[str] = []
    warnings: list[str] = []
    source_chars = max(0, int(total_chars))

    if count == 1 and source_chars > max(hard, target * 2):
        reasons.append("single_chunk_for_large_document")
    if maximum > hard:
        reasons.append("chunk_exceeds_hard_bound")
    tiny = sum(1 for value in values[:-1] if value < 50)
    if tiny > max(2, count // 4):
        reasons.append("too_many_tiny_chunks")
    if source_chars > target and maximum < max(1, target // 4):
        reasons.append("all_chunks_far_below_target")
    if coverage_ratio < 0.95:
        warnings.append("coverage_below_95_percent")

    return ChunkValidation(
        ok=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        warnings=tuple(dict.fromkeys(warnings)),
        chunk_count=count,
        min_chars=minimum,
        max_chars=maximum,
        avg_chars=average,
        coverage_ratio=round(float(coverage_ratio), 3),
    )


def _set_validation(result: ParseResult) -> ParseResult:
    result.validation = validate_chunks(
        result.chunks,
        result.source_chars,
        coverage_ratio=result.coverage_ratio,
    )
    return result


def _hash8(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _content_chars(text: str) -> int:
    """Count indexable characters while ignoring Markdown whitespace gaps."""
    return sum(1 for char in text if not char.isspace())


def _strip_frontmatter(text: str) -> tuple[dict, str]:
    """Strip only a YAML block anchored at the beginning of the document."""
    match = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)", text, re.S)
    if not match:
        return {}, text
    meta: dict = {}
    tags: list[str] = []
    in_tags = False
    for raw in match.group(1).splitlines():
        line = raw.strip()
        if not line:
            continue
        if in_tags and line.startswith("-"):
            tags.append(line[1:].strip(" '\""))
            continue
        if ":" not in line:
            in_tags = False
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "tags":
            in_tags = not value
            if value.startswith("[") and value.endswith("]"):
                tags = [item.strip(" '\"") for item in value[1:-1].split(",") if item.strip()]
                in_tags = False
            elif value:
                tags = [value.strip(" '\"")]
                in_tags = False
            meta[key] = tags
        else:
            in_tags = False
            meta[key] = value
    if tags:
        meta["tags"] = tags
    return meta, text[match.end():]


def _slug(content: str) -> str:
    first = content.strip().splitlines()[0].lstrip("# ") if content.strip() else ""
    clean = re.sub(r'[\\/:*?"<>|#\[\]]', "", first).strip()
    return re.sub(r"\s+", "-", clean)[:40] or "section"


def _heading_slug(heading_path: list[str]) -> str:
    return _slug(" ".join(heading_path)) if heading_path else "doc"


def _fence_marker(line: str) -> tuple[str, int] | None:
    match = _FENCE_RE.match(line)
    if not match:
        return None
    marker = match.group(1)
    return marker[0], len(marker)


def _is_fence_close(line: str, marker: tuple[str, int]) -> bool:
    stripped = line.strip()
    char, width = marker
    return bool(re.match(rf"^{re.escape(char)}{{{width},}}\s*$", stripped))


def _mem_headers_outside_fences(text: str) -> list[tuple[str, int, int]]:
    headers: list[tuple[str, int, int]] = []
    marker: tuple[str, int] | None = None
    cursor = 0
    for raw in text.splitlines(keepends=True):
        line = raw.rstrip("\r\n")
        if marker is not None:
            if _is_fence_close(line, marker):
                marker = None
            cursor += len(raw)
            continue
        opened = _fence_marker(line)
        if opened:
            marker = opened
            cursor += len(raw)
            continue
        match = _MEM_HEAD_RE.match(line)
        if match:
            headers.append((match.group(1), cursor, cursor + len(raw)))
        cursor += len(raw)
    return headers


def _is_bucket(text: str, meta: dict) -> bool:
    """Strict bucket detection: frontmatter marker or two real mem headings."""
    if str(meta.get("bucket", "")).lower() == "true":
        return True
    return len(_mem_headers_outside_fences(text)) >= 2


def _split_sentences(paragraph: str) -> list[str]:
    parts = [part for part in _SENT_CUT_RE.split(paragraph) if part.strip()]
    return parts or [paragraph]


def _slice_long(block: str, hard: int, reason: str) -> list[tuple[str, str, int, int]]:
    out: list[tuple[str, str, int, int]] = []
    cursor = 0
    for sentence in _split_sentences(block):
        leading = len(sentence) - len(sentence.lstrip())
        sentence = sentence.strip()
        cursor += leading
        while len(sentence) > hard:
            piece = sentence[:hard]
            out.append((piece, reason + "+hard_cut", cursor, cursor + len(piece)))
            cursor += len(piece)
            sentence = sentence[hard:].lstrip()
        if sentence:
            out.append((sentence, reason, cursor, cursor + len(sentence)))
            cursor += len(sentence)
    return out


def _parse_blocks(body: str) -> list[dict]:
    """Parse body lines into structural blocks with source offsets."""
    blocks: list[dict] = []
    heading: list[str] = []
    para: list[dict] = []
    fence: list[dict] = []
    table: list[dict] = []
    list_block: list[dict] = []
    callout: list[dict] = []
    fence_marker: tuple[str, int] | None = None

    def emit(kind: str, lines: list[dict]) -> None:
        if not lines or not any(item["text"].strip() for item in lines):
            return
        blocks.append({
            "kind": kind,
            "lines": [item["text"] for item in lines],
            "heading": list(heading),
            "start": lines[0]["start"],
            "end": lines[-1]["end"],
        })

    def flush_para() -> None:
        nonlocal para
        emit("paragraph", para)
        para = []

    def flush_table() -> None:
        nonlocal table
        emit("table", table)
        table = []

    def flush_list() -> None:
        nonlocal list_block
        emit("list", list_block)
        list_block = []

    def flush_callout() -> None:
        nonlocal callout
        emit("callout", callout)
        callout = []

    cursor = 0
    for raw in body.splitlines(keepends=True):
        line = raw.rstrip("\r\n")
        item = {"text": line, "start": cursor, "end": cursor + len(raw)}
        cursor += len(raw)

        if fence_marker is not None:
            fence.append(item)
            if _is_fence_close(line, fence_marker):
                emit("fence", fence)
                fence = []
                fence_marker = None
            continue
        opened = _fence_marker(line)
        if opened:
            flush_para(); flush_table(); flush_list(); flush_callout()
            fence = [item]
            fence_marker = opened
            continue

        if table:
            if _TABLE_RE.match(line):
                table.append(item)
                continue
            flush_table()
        if list_block:
            if _LIST_RE.match(line) or (line.strip() and line.startswith((" ", "\t"))):
                list_block.append(item)
                continue
            flush_list()
        if callout:
            if line.strip().startswith(">"):
                callout.append(item)
                continue
            flush_callout()

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            flush_para()
            level = len(heading_match.group(1))
            heading = heading[: level - 1] + [heading_match.group(2).strip()]
            emit("heading", [item])
            continue
        if _TABLE_RE.match(line) and "|" in line.strip()[1:]:
            flush_para(); flush_list(); flush_callout()
            table = [item]
            continue
        if _CALLOUT_RE.match(line):
            flush_para(); flush_list()
            callout = [item]
            continue
        if _LIST_RE.match(line):
            flush_para(); flush_callout()
            list_block = [item]
            continue
        if not line.strip():
            flush_para(); flush_table(); flush_list(); flush_callout()
            continue
        flush_table(); flush_list(); flush_callout()
        para.append(item)

    if fence:
        emit("fence", fence)
    flush_para(); flush_table(); flush_list(); flush_callout()
    return blocks


def _entry_sections(body: str) -> tuple[list[tuple[str, str, int, int]], str, int]:
    headers = _mem_headers_outside_fences(body)
    if not headers:
        return [], body, 0
    sections: list[tuple[str, str, int, int]] = []
    for index, (mem_id, _, body_start) in enumerate(headers):
        body_end = headers[index + 1][1] if index + 1 < len(headers) else len(body)
        sections.append((mem_id, body[body_start:body_end], body_start, body_end))
    preamble = body[: headers[0][1]]
    invalid = sum(1 for line in preamble.splitlines() if line.strip().startswith("## "))
    return sections, preamble, invalid


def _is_entry_meta_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith(">") and any(
        key in stripped for key in ("importance=", "tags=", "created=", "project=", "source=", "updated=")
    )


def _entry_content(entry_body: str) -> tuple[str, int]:
    """Remove the optional generated metadata line before an entry body.

    Memory bucket entries normally have a blank line, a ``> importance=...``
    line, and another blank line before the actual text.  The parser should
    keep that metadata in the chunk contract, not index it as user content.
    """
    lines = entry_body.splitlines(keepends=True)
    cursor = 0
    index = 0
    while index < len(lines) and not lines[index].strip():
        cursor += len(lines[index])
        index += 1
    while index < len(lines) and _is_entry_meta_line(lines[index]):
        cursor += len(lines[index])
        index += 1
        while index < len(lines) and not lines[index].strip():
            cursor += len(lines[index])
            index += 1
    remainder = entry_body[cursor:]
    # Leading indentation may be meaningful for nested lists/code; only the
    # blank lines above are discarded, never whitespace on the first content
    # line itself.
    content = remainder.rstrip()
    return content, cursor


def _entry_tags(entry_body: str) -> list[str]:
    """Extract tags from a bucket entry's compact metadata line.

    Bucket entries do not have individual YAML frontmatter, so previously the
    entry tags were lost before lexical/vector indexing.  Keeping these tags
    on the child chunks gives topic and parent-entry queries a deterministic
    anchor without changing the source Markdown format.
    """
    for line in entry_body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not _is_entry_meta_line(stripped):
            break
        match = re.search(r"(?:^|\|)\s*tags=([^|]+)", stripped)
        if not match:
            continue
        return [item.strip() for item in re.split(r"[,，、;；]", match.group(1))
                if item.strip()]
    return []


def _pack_blocks(
    blocks: list[dict],
    anchor: str,
    rel_path: str,
    heading_path: list[str],
    entry_index: int,
    mem_id: str | None,
    title: str,
    tags: list[str],
    meta: dict,
    body_offset: int = 0,
    diagnostic: bool = False,
) -> list[Chunk]:
    """Pack structural blocks while preserving metadata and source spans."""
    chunks: list[Chunk] = []
    next_index: dict[str, int] = {}
    used_hashes: set[str] = set()
    overlap_tail = ""
    overlap_anchor = ""

    def emit(content: str, reason: str, block: dict, local_anchor: str, start: int, end: int) -> None:
        nonlocal overlap_tail, overlap_anchor
        if block.get("kind") in {"list", "fence", "table", "callout"}:
            content = content.strip("\r\n")
        else:
            content = content.strip()
        if not content:
            return
        body = content
        if local_anchor == overlap_anchor and reason in {"paragraph", "sentence", "tail", "target_reached"} and overlap_tail:
            room = max(0, HARD_CHARS - len(content) - 1)
            prefix = overlap_tail[-min(OVERLAP_CHARS, room):]
            body = f"{prefix}\n{content}" if prefix else content
        index = next_index.get(local_anchor, 0)
        next_index[local_anchor] = index + 1
        digest = _hash8(f"{local_anchor}|{content}")
        duplicate = 1
        while digest in used_hashes:
            digest = _hash8(f"{local_anchor}|{content}|duplicate-{duplicate}")
            duplicate += 1
        used_hashes.add(digest)
        cid = f"{rel_path}#{local_anchor}:ch{digest}"
        entry_ref = f"{rel_path}#{mem_id}" if mem_id else rel_path
        chunks.append(Chunk(
            content=body,
            chunk_id=cid,
            source_ref=cid,
            entry_ref=entry_ref,
            mem_id=mem_id,
            heading_path=list(block.get("heading") or heading_path),
            anchor=local_anchor,
            chunk_index=index,
            split_reason=reason,
            context_header="\n".join(block.get("heading") or heading_path),
            entry_index=entry_index,
            title=title,
            tags=list(tags),
            start_offset=body_offset + max(0, start),
            end_offset=body_offset + max(start, end),
            meta=dict(meta),
            diagnostic=diagnostic,
        ))
        if reason in {"paragraph", "sentence", "tail", "target_reached"}:
            overlap_tail = content[-OVERLAP_CHARS:]
            overlap_anchor = local_anchor
        else:
            overlap_tail = ""
            overlap_anchor = ""

    def pack_group(group: list[dict], local_anchor: str) -> None:
        buffer: list[dict] = []
        buf_len = 0

        def flush(reason: str) -> None:
            nonlocal buffer, buf_len
            if not buffer:
                return
            text = "\n".join(
                line for item in buffer for line in item["lines"]
            ).strip()
            emit(text, reason, buffer[0], local_anchor, buffer[0]["start"], buffer[-1]["end"])
            buffer = []
            buf_len = 0

        for block in group:
            kind = block["kind"]
            text = "\n".join(block["lines"]).strip()
            if not text:
                continue
            if kind == "heading":
                flush("heading_boundary")
                emit(text, "heading", block, _heading_slug(block.get("heading") or heading_path), block["start"], block["end"])
                continue
            if kind == "fence":
                flush("fence_boundary")
                lines = block["lines"]
                marker = _fence_marker(lines[0])
                marker_text = lines[0].strip()[: marker[1] if marker else 3] if marker else "```"
                close = marker_text[0] * (len(marker_text) if marker_text else 3)
                closed = bool(marker and _is_fence_close(lines[-1], marker))
                if len(text) <= HARD_CHARS and closed:
                    emit("\n".join(lines), "fence", block, local_anchor, block["start"], block["end"])
                else:
                    inner = lines[1:-1] if closed else lines[1:]
                    language = lines[0].strip()[len(marker_text):].strip() if marker_text else ""
                    opening = f"{marker_text}{language}"
                    max_inner = max(1, HARD_CHARS - len(opening) - len(close) - 2)
                    piece: list[str] = []
                    piece_len = 0
                    for line in inner:
                        # Keep complete code lines whenever possible. A single
                        # pathological line is split only as a last resort.
                        if len(line) > max_inner:
                            if piece:
                                wrapped = opening + "\n" + "\n".join(piece) + f"\n{close}"
                                emit(wrapped, "fence_split", block, local_anchor, block["start"], block["end"])
                                piece, piece_len = [], 0
                            for offset in range(0, len(line), max_inner):
                                fragment = line[offset: offset + max_inner]
                                wrapped = opening + "\n" + fragment + f"\n{close}"
                                emit(wrapped, "fence_split", block, local_anchor, block["start"], block["end"])
                            continue
                        candidate_len = piece_len + len(line) + (1 if piece else 0)
                        if piece and candidate_len > max_inner:
                            wrapped = opening + "\n" + "\n".join(piece) + f"\n{close}"
                            emit(wrapped, "fence_split", block, local_anchor, block["start"], block["end"])
                            piece, piece_len = [], 0
                        piece.append(line)
                        piece_len += len(line) + (1 if piece_len else 0)
                    if piece:
                        wrapped = opening + "\n" + "\n".join(piece) + f"\n{close}"
                        emit(wrapped, "fence_split", block, local_anchor, block["start"], block["end"])
                continue
            if kind == "table":
                flush("table_boundary")
                lines = block["lines"]
                if len(text) <= HARD_CHARS:
                    emit(text, "table", block, local_anchor, block["start"], block["end"])
                else:
                    header = lines[0]
                    separator = lines[1] if len(lines) > 1 and set(lines[1].replace("|", "").replace("-", "").replace(":", "").strip()) <= {""} else ""
                    rows = lines[2:] if separator else lines[1:]
                    base = [header, separator] if separator else [header]
                    base_text = "\n".join(base)
                    if len(base_text) >= HARD_CHARS:
                        # A pathological header cannot be kept with data rows;
                        # retain it in bounded table-header fragments.
                        for offset in range(0, len(base_text), HARD_CHARS):
                            emit(base_text[offset: offset + HARD_CHARS], "table_header_hard_cut", block, local_anchor, block["start"], block["end"])
                        continue
                    piece = list(base)
                    for row in rows:
                        if len("\n".join(piece + [row])) > HARD_CHARS:
                            if len(piece) > len(base):
                                emit("\n".join(piece), "table_split", block, local_anchor, block["start"], block["end"])
                                piece = list(base)
                            if len("\n".join(piece + [row])) > HARD_CHARS:
                                max_row = max(1, HARD_CHARS - len(base_text) - 1)
                                for offset in range(0, len(row), max_row):
                                    fragment = row[offset: offset + max_row]
                                    emit(base_text + "\n" + fragment, "table_row_hard_cut", block, local_anchor, block["start"], block["end"])
                                continue
                        piece.append(row)
                    if len(piece) > len(base):
                        emit("\n".join(piece), "table_split", block, local_anchor, block["start"], block["end"])
                continue
            if kind == "list":
                flush("list_boundary")
                lines = block["lines"]
                if len(text) <= HARD_CHARS:
                    emit("\n".join(lines), "list", block, local_anchor, block["start"], block["end"])
                else:
                    items: list[list[str]] = []
                    current: list[str] = []
                    current_indent: int | None = None
                    for line in lines:
                        marker_match = _LIST_RE.match(line)
                        indent = len(line) - len(line.lstrip()) if marker_match else None
                        starts_item = bool(marker_match and (current_indent is None or indent <= current_indent))
                        if starts_item and current:
                            items.append(current)
                            current = []
                        if starts_item:
                            current_indent = indent
                        current.append(line)
                    if current:
                        items.append(current)

                    packed_items: list[str] = []
                    packed_len = 0

                    def flush_items(reason: str = "list_split") -> None:
                        nonlocal packed_items, packed_len
                        if packed_items:
                            emit("\n".join(packed_items), reason, block, local_anchor, block["start"], block["end"])
                            packed_items, packed_len = [], 0

                    for item in items:
                        item_text = "\n".join(item)
                        if len(item_text) <= HARD_CHARS:
                            if packed_items and packed_len + len(item_text) + 1 > HARD_CHARS:
                                flush_items()
                            packed_items.append(item_text)
                            packed_len += len(item_text) + 1
                            continue
                        flush_items()
                        item_piece: list[str] = []
                        item_piece_len = 0
                        for line in item:
                            prefix_match = _LIST_RE.match(line)
                            prefix = line[:prefix_match.end()] if prefix_match else line[: len(line) - len(line.lstrip())]
                            payload = line[len(prefix):] if prefix else line
                            if len(line) <= HARD_CHARS and item_piece and item_piece_len + len(line) + 1 <= HARD_CHARS:
                                item_piece.append(line)
                                item_piece_len += len(line) + 1
                                continue
                            if len(line) > HARD_CHARS:
                                if item_piece:
                                    emit("\n".join(item_piece), "list_item_split", block, local_anchor, block["start"], block["end"])
                                    item_piece, item_piece_len = [], 0
                                max_payload = max(1, HARD_CHARS - len(prefix))
                                for part_index in range(0, len(payload), max_payload):
                                    fragment = payload[part_index: part_index + max_payload]
                                    continuation = (" " * len(prefix)) + fragment if part_index and prefix else prefix + fragment
                                    emit(continuation, "list_item_hard_cut", block, local_anchor, block["start"], block["end"])
                                continue
                            if item_piece:
                                emit("\n".join(item_piece), "list_item_split", block, local_anchor, block["start"], block["end"])
                            item_piece, item_piece_len = [line], len(line) + 1
                        if item_piece:
                            emit("\n".join(item_piece), "list_item_split", block, local_anchor, block["start"], block["end"])
                    flush_items()
                continue
            if kind == "callout":
                flush("callout_boundary")
                if len(text) <= HARD_CHARS:
                    emit("\n".join(block["lines"]), "callout", block, local_anchor, block["start"], block["end"])
                else:
                    piece: list[str] = []
                    piece_len = 0
                    for line in block["lines"]:
                        if len(line) > HARD_CHARS:
                            if piece:
                                emit("\n".join(piece), "callout_split", block, local_anchor, block["start"], block["end"])
                                piece, piece_len = [], 0
                            prefix_match = re.match(r"^(\s*>\s?)", line)
                            prefix = prefix_match.group(1) if prefix_match else ""
                            payload = line[len(prefix):] if prefix else line
                            max_payload = max(1, HARD_CHARS - len(prefix))
                            for offset in range(0, len(payload), max_payload):
                                fragment = payload[offset: offset + max_payload]
                                emit(prefix + fragment, "callout_split", block, local_anchor, block["start"], block["end"])
                            continue
                        candidate_len = piece_len + len(line) + (1 if piece else 0)
                        if piece and candidate_len > HARD_CHARS:
                            emit("\n".join(piece), "callout_split", block, local_anchor, block["start"], block["end"])
                            piece, piece_len = [], 0
                        piece.append(line)
                        piece_len += len(line) + (1 if piece_len else 0)
                    if piece:
                        emit("\n".join(piece), "callout_split", block, local_anchor, block["start"], block["end"])
                continue
            if len(text) > HARD_CHARS:
                flush("size")
                for piece, reason, start, end in _slice_long(text, HARD_CHARS, "paragraph"):
                    emit(piece, "sentence" if reason == "paragraph" else reason, block, local_anchor, block["start"] + start, block["start"] + end)
                continue
            if buffer and buf_len + len(text) + 1 > TARGET_CHARS:
                flush("target_reached")
            buffer.append(block)
            buf_len += len(text) + 1
        flush("tail")

    group: list[dict] = []
    group_anchor = anchor
    group_heading_key: tuple[str, ...] | None = None
    for block in blocks:
        block_heading = tuple(block.get("heading") or heading_path)
        block_anchor = _heading_slug(list(block_heading)) if mem_id is None else anchor
        if group and block_heading != group_heading_key:
            pack_group(group, group_anchor)
            group = []
        group_anchor = block_anchor
        group_heading_key = block_heading
        group.append(block)
    if group:
        pack_group(group, group_anchor)
    return chunks


def _metadata_chunk(rel_path: str, meta: dict, title: str, tags: list[str], source_end: int = 0) -> Chunk:
    text = "\n".join(f"{key}: {value}" for key, value in meta.items() if value)
    digest = _hash8(text)
    return Chunk(
        content=f"[frontmatter-only]\n{text}",
        chunk_id=f"{rel_path}#meta:ch{digest}",
        source_ref=f"{rel_path}#meta:ch{digest}",
        entry_ref=rel_path,
        mem_id=None,
        heading_path=[],
        anchor="meta",
        chunk_index=0,
        split_reason="frontmatter_only",
        title=title,
        tags=tags,
        start_offset=0,
        end_offset=source_end or len(text),
        meta=dict(meta),
    )


def chunk_markdown_v1(text: str, rel_path: str) -> ParseResult:
    rel_path = rel_path.replace("\\", "/")
    meta, body = _strip_frontmatter(text)
    body_offset = len(text) - len(body)
    title = str(meta.get("title", "") or "")
    tags = [item for item in meta.get("tags", []) if isinstance(item, str)]
    bucket_mode = _is_bucket(text, meta)
    result = ParseResult(frontmatter=dict(meta), bucket_mode=bucket_mode)
    chunks: list[Chunk] = []
    indexable_spans: list[tuple[int, int]] = []
    if body.strip():
        body_start = len(body) - len(body.lstrip())
        body_end = len(body.rstrip())
        indexable_spans.append((body_start, body_end))

    if bucket_mode:
        sections, preamble, invalid = _entry_sections(body)
        result.invalid_entry_count = invalid
        if any(line.strip() and not line.strip().startswith(("#", ">")) for line in preamble.splitlines()):
            result.orphan_entry_count += 1
        if sections:
            seen: set[str] = set()
            duplicate_occurrences: dict[str, int] = {}
            indexable_spans = []
            for entry_index, (mem_id, entry_body, start, _) in enumerate(sections):
                entry_text, entry_text_offset = _entry_content(entry_body)
                entry_tags = _entry_tags(entry_body)
                duplicate = mem_id in seen
                seen.add(mem_id)
                if duplicate:
                    duplicate_occurrences[mem_id] = duplicate_occurrences.get(mem_id, 0) + 1
                    result.duplicate_mem_ids.append(mem_id)
                if not entry_text:
                    continue
                indexable_spans.append((start + entry_text_offset, start + entry_text_offset + len(entry_text)))
                entry_anchor = mem_id if not duplicate else f"{mem_id}:d{duplicate_occurrences[mem_id]}"
                chunks.extend(_pack_blocks(
                    _parse_blocks(entry_text), entry_anchor, rel_path, [], entry_index,
                    mem_id, title, list(dict.fromkeys(tags + entry_tags)), meta,
                    body_offset=body_offset + start + entry_text_offset,
                    diagnostic=duplicate,
                ))
            result.entry_count = len(sections)
        else:
            result.orphan_entry_count += 1
            chunks.extend(_pack_blocks(
                _parse_blocks(body), "doc", rel_path, [], 0, None, title, tags, meta,
                body_offset=body_offset,
            ))
            result.entry_count = 0
    else:
        chunks = _pack_blocks(
            _parse_blocks(body), "doc", rel_path, [], 0, None, title, tags, meta,
            body_offset=body_offset,
        )
        result.entry_count = 1

    indexable_source_chars = sum(_content_chars(body[start:end]) for start, end in indexable_spans)

    if not chunks and meta:
        chunks = [_metadata_chunk(rel_path, meta, title, tags, source_end=body_offset)]
        result.entry_count = max(1, result.entry_count)

    result.chunks = chunks
    result.source_chars = indexable_source_chars
    metadata_only = bool(chunks) and all(chunk.split_reason == "frontmatter_only" for chunk in chunks)
    if not result.source_chars and meta:
        result.source_chars = sum(len(str(value)) for value in meta.values())
    spans = sorted(
        (chunk.start_offset - body_offset, chunk.end_offset - body_offset)
        for chunk in chunks
        if chunk.end_offset > chunk.start_offset
    )
    if metadata_only:
        result.covered_chars = result.source_chars
        result.coverage_ratio = 1.0
    elif indexable_spans:
        covered = 0
        for expected_start, expected_end in indexable_spans:
            local = []
            for start, end in spans:
                start = max(start, expected_start)
                end = min(end, expected_end)
                if end > start:
                    local.append((start, end))
            cursor = None
            for start, end in sorted(local):
                if cursor is None:
                    cursor = [start, end]
                elif start <= cursor[1]:
                    cursor[1] = max(cursor[1], end)
                else:
                    covered += _content_chars(body[cursor[0]:cursor[1]])
                    cursor = [start, end]
            if cursor is not None:
                covered += _content_chars(body[cursor[0]:cursor[1]])
        result.covered_chars = min(result.source_chars, covered)
        result.coverage_ratio = round(result.covered_chars / result.source_chars, 3) if result.source_chars else 0.0
    else:
        result.covered_chars = 1 if chunks else 0
        result.coverage_ratio = 1.0 if chunks else 0.0
    result.zero_chunk_files = 0 if chunks else 1
    return _set_validation(result)


_V2_PROSE_REASONS = {
    "paragraph", "paragraph+hard_cut", "sentence", "tail", "target_reached",
    "short_merge", "heading_prefix",
}


def _v2_structure_kind(reason: str) -> str:
    """Map parser reasons to merge-safe structural classes."""
    if reason in _V2_PROSE_REASONS:
        return "prose"
    if reason.startswith("list"):
        return "list"
    if reason.startswith("table"):
        return "table"
    if reason.startswith("fence"):
        return "fence"
    if reason.startswith("callout"):
        return "callout"
    if reason.startswith("heading"):
        return "heading"
    return reason


def _v2_same_scope(left: Chunk, right: Chunk) -> bool:
    return (
        left.entry_index == right.entry_index
        and left.mem_id == right.mem_id
        and left.anchor == right.anchor
    )


def _v2_join_content(parts: list[Chunk]) -> str:
    return "\n".join(item.content.strip("\r\n") for item in parts if item.content.strip("\r\n"))


def _v2_rebuild_chunks(
    parts: list[tuple[list[Chunk], str]],
    rel_path: str,
    *,
    min_chars: int,
) -> list[Chunk]:
    """Rebuild refs after v2 merges without using a global chunk ordinal.

    ``parts`` carries the source chunks and the reason for the merge.  Content
    hashes make refs stable when unrelated content is inserted before a chunk;
    the per-anchor index is retained only as a display/debug field.
    """
    result: list[Chunk] = []
    next_index: dict[str, int] = {}
    used_hashes: set[str] = set()
    for source_parts, reason in parts:
        if not source_parts:
            continue
        first = source_parts[0]
        content = _v2_join_content(source_parts)
        if not content:
            continue
        anchor = first.anchor
        chunk_index = next_index.get(anchor, 0)
        meta = dict(first.meta)
        meta.update({
            "chunk_strategy": STRATEGY_V2,
            "v2_min_chars": min_chars,
            "v2_merge_reason": reason,
        })
        end = max(item.end_offset for item in source_parts)
        start = min(item.start_offset for item in source_parts)
        entry_ref = f"{rel_path}#{first.mem_id}" if first.mem_id else rel_path
        # A pathological heading can exceed the hard bound before it reaches
        # the normal v1 packing path.  Preserve it as deterministic bounded
        # slices rather than allowing one oversized v2 chunk through.
        content_parts = [content[i:i + HARD_CHARS] for i in range(0, len(content), HARD_CHARS)]
        for part_index, part in enumerate(content_parts):
            part_reason = reason if len(content_parts) == 1 else "heading_hard_cut"
            part_digest = _hash8(f"{anchor}|{part}")
            duplicate = 1
            while part_digest in used_hashes:
                part_digest = _hash8(f"{anchor}|{part}|duplicate-{duplicate}")
                duplicate += 1
            used_hashes.add(part_digest)
            part_meta = dict(meta)
            if len(content_parts) > 1:
                part_meta["v2_hard_cut_part"] = part_index
            result.append(Chunk(
                content=part,
                chunk_id=f"{rel_path}#{anchor}:ch{part_digest}" if anchor else f"{rel_path}:ch{part_digest}",
                source_ref=f"{rel_path}#{anchor}:ch{part_digest}" if anchor else f"{rel_path}:ch{part_digest}",
                entry_ref=entry_ref,
                mem_id=first.mem_id,
                heading_path=list(first.heading_path),
                anchor=anchor,
                chunk_index=chunk_index + part_index,
                split_reason=part_reason,
                context_header=first.context_header,
                entry_index=first.entry_index,
                title=first.title,
                tags=list(first.tags),
                start_offset=start,
                end_offset=end,
                meta=part_meta,
                diagnostic=any(item.diagnostic for item in source_parts),
            ))
        next_index[anchor] = chunk_index + len(content_parts)
    return result


def chunk_markdown_v2(
    text: str,
    rel_path: str,
    *,
    min_chars: int = 64,
) -> ParseResult:
    """Chunk Markdown with heading-prefix and short-prose consolidation.

    v2 is a shadow-only strategy.  It reuses v1's state machine and source
    spans, then removes the most common non-semantic fragments: a heading with
    following content is emitted as one chunk, and adjacent short prose chunks
    are merged only within the same heading, entry and prose class.  Lists,
    tables, fences, callouts and mem-id boundaries never cross a merge.
    """
    if int(min_chars) < 0:
        raise ValueError("min_chars must be non-negative")
    min_chars = int(min_chars)
    parsed = chunk_markdown_v1(text, rel_path)
    parsed.strategy_version = STRATEGY_V2
    if not parsed.chunks:
        return parsed

    chunks = parsed.chunks
    merged: list[tuple[list[Chunk], str]] = []
    index = 0
    while index < len(chunks):
        current = chunks[index]
        if (
            current.split_reason == "heading"
            and index + 1 < len(chunks)
            and _v2_same_scope(current, chunks[index + 1])
            and chunks[index + 1].split_reason != "heading"
            and len(current.content) + len(chunks[index + 1].content) + 1 <= HARD_CHARS
        ):
            merged.append(([current, chunks[index + 1]], "heading_prefix"))
            index += 2
            continue
        if current.split_reason == "heading":
            merged.append(([current], "heading_only"))
        else:
            merged.append(([current], current.split_reason))
        index += 1

    consolidated: list[tuple[list[Chunk], str]] = []
    for source_parts, reason in merged:
        if (
            consolidated
            and min_chars > 0
            and reason in _V2_PROSE_REASONS
            and consolidated[-1][1] in _V2_PROSE_REASONS
            and _v2_same_scope(consolidated[-1][0][-1], source_parts[0])
            and _v2_structure_kind(consolidated[-1][1]) == "prose"
            and _v2_structure_kind(reason) == "prose"
        ):
            previous = consolidated[-1][0]
            combined_length = len(_v2_join_content(previous + source_parts))
            if (
                (len(_v2_join_content(previous)) < min_chars
                 or len(_v2_join_content(source_parts)) < min_chars)
                and combined_length <= HARD_CHARS
            ):
                consolidated[-1] = (
                    previous + source_parts,
                    "short_merge",
                )
                continue
        consolidated.append((source_parts, reason))

    parsed.chunks = _v2_rebuild_chunks(consolidated, rel_path.replace("\\", "/"), min_chars=min_chars)
    # v2 only replaces chunks; it never removes source characters.  Keep the
    # v1 coverage accounting and expose the chosen variant in every chunk.
    parsed.strategy_version = STRATEGY_V2
    return _set_validation(parsed)


def make_chunker_v2(min_chars: int = 64):
    """Return a two-argument chunker suitable for ``RagIndexStore`` shadowing."""
    min_chars = int(min_chars)

    def chunker(text: str, rel_path: str) -> ParseResult:
        return chunk_markdown_v2(text, rel_path, min_chars=min_chars)

    return chunker


def build_shadow_report(
    vault_root: str,
    exclude_dirs: set[str] | None = None,
    archive_rel: str = "ark/memory/archive/",
    limit: int | None = None,
) -> dict:
    """Read-only legacy/v1 comparison with coverage and diagnostics."""
    from agentlab.rag.vector_index import chunk_text

    root = Path(vault_root)
    excluded = exclude_dirs or {".obsidian", ".agent-brain", ".trash", ".tmp", "node_modules"}
    rows: list[dict] = []
    total_dup = total_orphan = total_invalid = 0
    zero_chunk: list[str] = []
    empty_files: list[str] = []
    files = sorted(root.rglob("*.md"))
    if limit:
        files = files[:limit]
    for path in files:
        rel = path.relative_to(root).as_posix()
        if any(part in excluded for part in path.parts) or rel.startswith(archive_rel):
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        legacy = chunk_text(source)
        parsed = chunk_markdown_v1(source, rel)
        total_dup += len(parsed.duplicate_mem_ids)
        total_orphan += parsed.orphan_entry_count
        total_invalid += parsed.invalid_entry_count
        if parsed.zero_chunk_files:
            if source.strip():
                zero_chunk.append(rel)
            else:
                empty_files.append(rel)
        legacy_lengths = [len(item) for item in legacy]
        v1_lengths = [len(item.content) for item in parsed.chunks]
        rows.append({
            "path": rel,
            "legacy_chunks": len(legacy),
            "v1_chunks": len(parsed.chunks),
            "legacy_avg": round(sum(legacy_lengths) / len(legacy_lengths), 1) if legacy_lengths else 0,
            "v1_avg": round(sum(v1_lengths) / len(v1_lengths), 1) if v1_lengths else 0,
            "v1_over_hard": sum(length > HARD_CHARS for length in v1_lengths),
            "coverage": parsed.coverage_ratio,
            "empty": not bool(source.strip()),
            "is_bucket": parsed.bucket_mode,
            "dup_mem_ids": len(parsed.duplicate_mem_ids),
            "invalid_entries": parsed.invalid_entry_count,
            "orphan": parsed.orphan_entry_count,
        })
    return {
        "schema": "chunk-shadow-v1",
        "strategy": STRATEGY_V1,
        "files": len(rows),
        "legacy_total_chunks": sum(row["legacy_chunks"] for row in rows),
        "v1_total_chunks": sum(row["v1_chunks"] for row in rows),
        "zero_chunk_files": zero_chunk,
        "empty_files": empty_files,
        "coverage_avg": round(statistics.mean([row["coverage"] for row in rows if not row["empty"]]), 3)
        if any(not row["empty"] for row in rows) else 1.0,
        "coverage_min": round(min((row["coverage"] for row in rows if not row["empty"]), default=1.0), 3),
        "coverage_below_95": [row["path"] for row in rows if not row["empty"] and row["coverage"] < 0.95],
        "diagnostics": {
            "duplicate_mem_ids": total_dup,
            "invalid_entry_count": total_invalid,
            "orphan_entry_count": total_orphan,
        },
        "rows": rows,
    }


def build_v2_shadow_report(
    vault_root: str,
    *,
    min_chars: int = 64,
    exclude_dirs: set[str] | None = None,
    archive_rel: str = "ark/memory/archive/",
    limit: int | None = None,
) -> dict:
    """Compare v2 fragment statistics without writing an index or Vault file."""
    from agentlab.rag.vector_index import chunk_text

    root = Path(vault_root)
    excluded = exclude_dirs or {".obsidian", ".agent-brain", ".trash", ".tmp", "node_modules"}
    rows: list[dict] = []
    total_dup = total_orphan = total_invalid = 0
    files = sorted(root.rglob("*.md"))
    if limit:
        files = files[:limit]
    for path in files:
        rel = path.relative_to(root).as_posix()
        if any(part in excluded for part in path.parts) or rel.startswith(archive_rel):
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        legacy = chunk_text(source)
        parsed = chunk_markdown_v2(source, rel, min_chars=min_chars)
        lengths = [len(item.content) for item in parsed.chunks]
        total_dup += len(parsed.duplicate_mem_ids)
        total_orphan += parsed.orphan_entry_count
        total_invalid += parsed.invalid_entry_count
        rows.append({
            "path": rel,
            "legacy_chunks": len(legacy),
            "v1_chunks": len(chunk_markdown_v1(source, rel).chunks),
            "v2_chunks": len(parsed.chunks),
            "v2_short_20": sum(length <= 20 for length in lengths),
            "v2_short_50": sum(length <= 50 for length in lengths),
            "v2_max_chars": max(lengths, default=0),
            "v2_over_hard": sum(length > HARD_CHARS for length in lengths),
            "coverage": parsed.coverage_ratio,
            "empty": not bool(source.strip()),
            "is_bucket": parsed.bucket_mode,
            "heading_only": sum(item.split_reason == "heading_only" for item in parsed.chunks),
            "merged": sum(item.split_reason in {"heading_prefix", "short_merge"} for item in parsed.chunks),
            "dup_mem_ids": len(parsed.duplicate_mem_ids),
            "invalid_entries": parsed.invalid_entry_count,
            "orphan": parsed.orphan_entry_count,
        })
    nonempty = [row for row in rows if not row["empty"]]
    total_chunks = sum(row["v2_chunks"] for row in rows)
    return {
        "schema": "chunk-shadow-v2",
        "strategy": STRATEGY_V2,
        "min_chars": int(min_chars),
        "files": len(rows),
        "legacy_total_chunks": sum(row["legacy_chunks"] for row in rows),
        "v1_total_chunks": sum(row["v1_chunks"] for row in rows),
        "v2_total_chunks": total_chunks,
        "short_fragments": {
            "le20": sum(row["v2_short_20"] for row in rows),
            "le50": sum(row["v2_short_50"] for row in rows),
            "le20_ratio": round(sum(row["v2_short_20"] for row in rows) / total_chunks, 4)
            if total_chunks else 0.0,
            "le50_ratio": round(sum(row["v2_short_50"] for row in rows) / total_chunks, 4)
            if total_chunks else 0.0,
        },
        "max_chars": max((row["v2_max_chars"] for row in rows), default=0),
        "over_hard": sum(row["v2_over_hard"] for row in rows),
        "coverage_avg": round(statistics.mean([row["coverage"] for row in nonempty]), 3)
        if nonempty else 1.0,
        "coverage_min": round(min((row["coverage"] for row in nonempty), default=1.0), 3),
        "coverage_below_95": [row["path"] for row in nonempty if row["coverage"] < 0.95],
        "heading_only": sum(row["heading_only"] for row in rows),
        "merged": sum(row["merged"] for row in rows),
        "diagnostics": {
            "duplicate_mem_ids": total_dup,
            "invalid_entry_count": total_invalid,
            "orphan_entry_count": total_orphan,
        },
        "rows": rows,
    }
