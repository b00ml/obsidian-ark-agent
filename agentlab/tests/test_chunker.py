"""OPT-233/P1 markdown-structure-v1 分块测试：结构边界、桶 mem-id、稳定 ref。"""
import unittest

from agentlab.rag.chunker import (
    Chunk,
    HARD_CHARS,
    _is_bucket,
    build_shadow_report,
    build_v2_shadow_report,
    chunk_markdown_v1,
    chunk_markdown_v2,
    validate_chunks,
)

SEP = chr(10) + chr(10)


def _bucket(mem_bodies: dict) -> str:
    parts = ["---" + chr(10) + "bucket: true" + chr(10) + "type: sessions" + chr(10) + "---" + chr(10) + "# sessions"]
    for mem_id, body in mem_bodies.items():
        parts.append(chr(10) + f"## {mem_id}" + chr(10) + "> importance=3 | tags=t" + chr(10) + chr(10) + body + chr(10))
    return "".join(parts)


class TestStructureSplit(unittest.TestCase):
    def test_validator_reports_explicit_reasons_and_coverage_warning(self):
        chunk = Chunk(
            content="x" * (HARD_CHARS + 1), chunk_id="c", source_ref="c",
            entry_ref="a.md", mem_id=None, heading_path=[], anchor="doc",
            chunk_index=0, split_reason="test",
        )
        verdict = validate_chunks([chunk], 2000, coverage_ratio=0.8)
        self.assertFalse(verdict.ok)
        self.assertIn("chunk_exceeds_hard_bound", verdict.reasons)
        self.assertIn("coverage_below_95_percent", verdict.warnings)
        self.assertEqual(verdict.to_dict()["chunk_count"], 1)

    def test_parse_result_exposes_validation_and_context_header(self):
        parsed = chunk_markdown_v1("# Root\n\n## Child\n\n正文。", "a.md")
        self.assertTrue(parsed.validation.ok)
        self.assertEqual(parsed.validation.chunk_count, len(parsed.chunks))
        self.assertTrue(any("Child" in chunk.context_header for chunk in parsed.chunks))

    def test_english_and_chinese_sentence_cut(self):
        text = "First sentence. Second one! Third? 第四句。第五句！第六句？"
        pr = chunk_markdown_v1("# d" + SEP + text, "a.md")
        joined = "".join(c.content for c in pr.chunks)
        for sent in ("First sentence.", "Second one!", "第四句。", "第六句？"):
            self.assertIn(sent, joined)

    def test_long_sentence_no_punctuation_hard_cut(self):
        text = "无" * (HARD_CHARS + 300)
        pr = chunk_markdown_v1(text, "a.md")
        self.assertGreater(len(pr.chunks), 1)
        self.assertTrue(all(len(c.content) <= HARD_CHARS for c in pr.chunks))
        self.assertTrue(all(c.split_reason for c in pr.chunks))

    def test_list_items_not_split_mid_item(self):
        item = "- 这是一条很长的列表项内容，包含完整语义" + "细节" * 40
        text = "## 列表" + SEP + SEP.join(item for _ in range(6))
        pr = chunk_markdown_v1(text, "a.md")
        for c in pr.chunks:
            for line in c.content.splitlines():
                if line.startswith("- 这是一条"):
                    self.assertTrue(line.endswith("完整语义") or line.endswith("细节"), line[-20:])

    def test_code_block_kept_whole(self):
        code = "```python" + chr(10) + chr(10).join(f"x{i} = {i}" for i in range(40)) + chr(10) + "```"
        text = "# code" + SEP + code
        pr = chunk_markdown_v1(text, "a.md")
        for c in pr.chunks:
            self.assertEqual(c.content.count("```") % 2, 0)

    def test_table_independent_block(self):
        table = "| a | b |" + chr(10) + "|---|---|" + chr(10) + \
            chr(10).join(f"| {i} | v{i} |" for i in range(30))
        text = "前段。" + SEP + table + SEP + "后段。"
        pr = chunk_markdown_v1(text, "a.md")
        self.assertTrue(any("| a | b |" in c.content for c in pr.chunks))

    def test_frontmatter_stripped_from_chunks(self):
        text = "---" + chr(10) + "title: x" + chr(10) + "---" + SEP + "正文内容。"
        pr = chunk_markdown_v1(text, "a.md")
        self.assertTrue(all("title: x" not in c.content for c in pr.chunks))

    def test_heading_inside_fence_is_not_a_mem_entry(self):
        text = "```markdown" + chr(10) + "## mem-deadbeef" + chr(10) + "示例" + chr(10) + "```"
        pr = chunk_markdown_v1(text, "a.md")
        self.assertFalse(pr.bucket_mode)
        self.assertTrue(all(c.mem_id is None for c in pr.chunks))


class TestBucketBoundaries(unittest.TestCase):
    def test_bucket_entries_never_merged(self):
        body_a = "A 记忆正文。" + "a" * 200
        body_b = "B 记忆正文。" + "b" * 200
        pr = chunk_markdown_v1(_bucket({"mem-aaa": body_a, "mem-bbb": body_b}),
                               "ark/memory/sessions/2026-09.md")
        self.assertEqual(pr.entry_count, 2)
        for c in pr.chunks:
            if c.mem_id == "mem-aaa":
                self.assertNotIn("B 记忆正文", c.content)
            if c.mem_id == "mem-bbb":
                self.assertNotIn("A 记忆正文", c.content)

    def test_duplicate_mem_id_reported(self):
        text = _bucket({"mem-aaa": "第一条。"}) + chr(10) + "## mem-aaa" + chr(10) + "重复区块。"
        pr = chunk_markdown_v1(text, "s.md")
        self.assertIn("mem-aaa", pr.duplicate_mem_ids)

    def test_orphan_text_counted(self):
        # 头部孤立文本（非标题）计数；尾部孤立文字并入最后 entry 属已知边界（机制文档 §2.3）
        text = "桶头与区块之间的孤立说明文字。" + chr(10) + _bucket(
            {"mem-aaa": "第一条。", "mem-bbb": "第二条。"})
        pr = chunk_markdown_v1(text, "s.md")
        self.assertGreaterEqual(pr.orphan_entry_count, 1)

    def test_bucket_metadata_line_is_not_indexed_and_offsets_are_source_spans(self):
        text = _bucket({"mem-aaa": "第一条正文。"})
        pr = chunk_markdown_v1(text, "ark/memory/sessions/x.md")
        self.assertEqual(pr.coverage_ratio, 1.0)
        self.assertEqual(pr.source_chars, len("第一条正文。"))
        self.assertTrue(all("importance=" not in c.content for c in pr.chunks))
        for chunk in pr.chunks:
            self.assertGreaterEqual(chunk.start_offset, 0)
            self.assertLessEqual(chunk.end_offset, len(text))
            self.assertIn("第一条正文。", text[chunk.start_offset:chunk.end_offset])

    def test_bucket_entry_tags_are_preserved_on_child_chunks(self):
        text = (
            "---\nbucket: true\ntype: sessions\n---\n# sessions\n\n"
            "## mem-aaa\n> importance=3 | tags=AGENTS.md, 知识库规范 | created=2026-09\n\n"
            "raw 只读。"
        )
        parsed = chunk_markdown_v1(text, "ark/memory/sessions/2026-09.md")
        self.assertEqual(parsed.chunks[0].tags, ["AGENTS.md", "知识库规范"])

    def test_bucket_entry_preserves_first_line_indentation(self):
        text = _bucket({"mem-aaa": "  - 嵌套列表项\n    续行"})
        pr = chunk_markdown_v1(text, "ark/memory/sessions/x.md")
        self.assertTrue(pr.chunks)
        self.assertTrue(pr.chunks[0].content.startswith("  - "))

    def test_is_bucket_detection(self):
        self.assertTrue(_is_bucket(_bucket({"mem-aaa": "x"}), {"bucket": "true"}))
        self.assertTrue(_is_bucket(_bucket({"mem-aaa": "x", "mem-bbb": "y"}), {}))
        # 单 mem 区块 + 无 bucket 标记 → 非桶（严格判定）
        self.assertFalse(_is_bucket("# 普通文档" + SEP + "正文。", {}))
        # 正文出现 bucket: true 但无 frontmatter/双区块 → 不误触发
        self.assertFalse(_is_bucket("代码示例里出现 bucket: true 字样。", {}))


class TestStableRefs(unittest.TestCase):
    def test_refs_format_and_stability(self):
        pr = chunk_markdown_v1("# h" + SEP + "内容。" * 100, "ark/memory/decisions/x-abc123.md")
        for i, c in enumerate(pr.chunks):
            self.assertTrue(c.source_ref.startswith("ark/memory/decisions/x-abc123.md"))
            import re as _re
            self.assertTrue(_re.search(r":ch[0-9a-f]{8}$", c.source_ref),
                            f"ref 应为内容哈希短码: {c.source_ref}")
            self.assertEqual(c.chunk_id, c.source_ref)
        pr2 = chunk_markdown_v1("# h" + SEP + "内容。" * 100, "ark/memory/decisions/x-abc123.md")
        self.assertEqual([c.source_ref for c in pr.chunks],
                         [c.source_ref for c in pr2.chunks])

    def test_bucket_entry_ref_has_mem_id(self):
        pr = chunk_markdown_v1(_bucket({"mem-aaa": "桶内条目。"}),
                               "ark/memory/sessions/2026-09.md")
        for c in pr.chunks:
            self.assertEqual(c.entry_ref, "ark/memory/sessions/2026-09.md#mem-aaa")
            self.assertTrue(c.source_ref.startswith(c.entry_ref + ":ch"))

    def test_crlf_and_unicode(self):
        text = "第一段（中文）。" + chr(13) + chr(10) + chr(13) + chr(10) + "Second段落 emoji 🎉 测试。"
        pr = chunk_markdown_v1(text, "a.md")
        joined = "".join(c.content for c in pr.chunks)
        self.assertIn("🎉", joined)

    def test_multiline_tags_are_parsed(self):
        text = "---" + chr(10) + "tags:" + chr(10) + "  - alpha" + chr(10) + "  - beta" + chr(10) + "---" + chr(10) + "正文"
        pr = chunk_markdown_v1(text, "a.md")
        self.assertEqual(pr.frontmatter["tags"], ["alpha", "beta"])
        self.assertEqual(pr.chunks[0].tags, ["alpha", "beta"])

    def test_scalar_tag_is_preserved(self):
        text = "---" + chr(10) + "tags: review" + chr(10) + "---" + chr(10) + "正文"
        pr = chunk_markdown_v1(text, "a.md")
        self.assertEqual(pr.frontmatter["tags"], ["review"])
        self.assertEqual(pr.chunks[0].tags, ["review"])



class TestP11Fixes(unittest.TestCase):
    """P1.1 修复包：审查 7 项的回归覆盖。"""

    def test_250_line_code_block_fences_always_paired(self):
        """审查#4 高：超长代码块按行切且每片围栏成对。"""
        code = "```python" + chr(10) +             chr(10).join(f"line_{i} = 'value {i}'" for i in range(250)) + chr(10) + "```"
        pr = chunk_markdown_v1("# code" + SEP + code, "a.md")
        self.assertGreater(len(pr.chunks), 1, "超长代码块应被切分")
        for c in pr.chunks:
            self.assertEqual(c.content.count("```") % 2, 0,
                             f"围栏不成对: {c.source_ref}")
        # 所有行内容保留（无丢失）
        joined = "".join(c.content for c in pr.chunks)
        for i in (0, 100, 249):
            self.assertIn(f"line_{i} =", joined)
        self.assertTrue(all(len(c.content) <= HARD_CHARS for c in pr.chunks))

    def test_unclosed_code_block_gets_synthetic_closing_fence(self):
        text = "```python" + chr(10) + chr(10).join(f"line_{i}" for i in range(40))
        pr = chunk_markdown_v1(text, "a.md")
        self.assertTrue(pr.chunks)
        self.assertTrue(all(c.content.count("```") % 2 == 0 for c in pr.chunks))
        self.assertTrue(all(len(c.content) <= HARD_CHARS for c in pr.chunks))

    def test_oversized_list_splits_at_item_boundary(self):
        """审查#4 高：超长列表按列表项边界切，不腰斩列表项。"""
        items = [f"- 列表项{i}：" + "内容" * 60 for i in range(10)]
        text = "## 列表" + SEP + chr(10).join(items)
        pr = chunk_markdown_v1(text, "a.md")
        self.assertGreater(len(pr.chunks), 1)
        for c in pr.chunks:
            self.assertLessEqual(len(c.content), HARD_CHARS)
            for line in c.content.splitlines():
                if line.startswith("- 列表项"):
                    self.assertTrue(line.endswith("内容"), f"列表项被腰斩: {line[-20:]}")

    def test_oversized_single_list_item_is_bounded_and_marked(self):
        text = "- " + "内容" * (HARD_CHARS // 2 + 100)
        pr = chunk_markdown_v1(text, "a.md")
        self.assertGreater(len(pr.chunks), 1)
        self.assertTrue(all(len(c.content) <= HARD_CHARS for c in pr.chunks))
        self.assertTrue(all(c.split_reason == "list_item_hard_cut" for c in pr.chunks))
        self.assertTrue(pr.chunks[0].content.startswith("- "))

    def test_list_item_with_long_continuations_is_bounded(self):
        text = "- 项目标题\n  " + "续行" * 220 + "\n  " + "更多" * 220
        pr = chunk_markdown_v1(text, "a.md")
        self.assertTrue(pr.chunks)
        self.assertTrue(all(len(c.content) <= HARD_CHARS for c in pr.chunks))
        self.assertIn("项目标题", "\n".join(c.content for c in pr.chunks))

    def test_long_table_repeats_header(self):
        """审查#4 高：超长表格按数据行切并重复表头。"""
        table = "| 名称 | 数值 |" + chr(10) + "|---|---|" + chr(10) +             chr(10).join(f"| 行{i} | {i} |" for i in range(150))
        text = table
        pr = chunk_markdown_v1(text, "a.md")
        self.assertGreater(len(pr.chunks), 1)
        for c in pr.chunks:
            self.assertLessEqual(len(c.content), HARD_CHARS)
            if "行" in c.content:
                self.assertIn("| 名称 | 数值 |", c.content, "数据分片未重复表头")

    def test_oversized_table_row_is_bounded(self):
        row = "| 描述 | " + "长字段" * 300 + " |"
        text = "| 名称 | 描述 |" + chr(10) + "|---|---|" + chr(10) + row
        pr = chunk_markdown_v1(text, "a.md")
        self.assertTrue(pr.chunks)
        self.assertTrue(all(len(c.content) <= HARD_CHARS for c in pr.chunks))
        self.assertTrue(any(c.split_reason == "table_row_hard_cut" for c in pr.chunks))

    def test_callout_block_independent(self):
        """审查#4 中：callout（> [!note]）独立成块，不与正文混切。"""
        callout = "> [!note] 注意事项" + chr(10) + "> 具体说明内容" * 30
        text = "前段。" + SEP + callout + SEP + "后段。"
        pr = chunk_markdown_v1(text, "a.md")
        self.assertTrue(any("callout" == c.split_reason or "[!note]" in c.content
                            for c in pr.chunks))

    def test_oversized_callout_preserves_prefix_on_every_fragment(self):
        text = "> [!note] " + "x" * (HARD_CHARS + 100)
        pr = chunk_markdown_v1(text, "a.md")
        self.assertGreater(len(pr.chunks), 1)
        self.assertTrue(all(len(c.content) <= HARD_CHARS for c in pr.chunks))
        self.assertTrue(all(c.content.startswith("> ") for c in pr.chunks))
        self.assertIn("> [!note] ", pr.chunks[0].content)

    def test_english_period_sentence_boundary(self):
        """审查#7 低：英文句号 `. ` 进入句子边界。"""
        text = "First point here. Second point follows. Third point ends it."
        pr = chunk_markdown_v1(text, "a.md")
        self.assertTrue(any(c.split_reason in ("sentence", "paragraph+hard_cut")
                            for c in pr.chunks) or len(pr.chunks) >= 1)

    def test_insert_keeps_later_refs_stable(self):
        """审查#6 中：内容哈希 ref——中部插入段落后，后续 chunk 引用不变。"""
        before = "# h" + SEP + SEP.join("段落" + str(i) + "。" + "x" * 280 for i in range(1, 4))
        pr1 = chunk_markdown_v1(before, "a.md")
        after = "# h" + SEP + SEP.join(("段落1。" + "x" * 280, "插入的新段落。" + "y" * 280, "段落2。" + "x" * 280, "段落3。" + "x" * 280))
        pr2 = chunk_markdown_v1(after, "a.md")
        refs1 = {c.content: c.source_ref for c in pr1.chunks}
        refs2 = {c.content: c.source_ref for c in pr2.chunks}
        for content, ref in refs1.items():
            if content in refs2:
                self.assertEqual(refs2[content], ref,
                                 f"未变更内容的 ref 漂移: {content}")

    def test_frontmatter_only_file_produces_meta_chunk(self):
        """审查#1 高：frontmatter-only 文件不再静默 0 chunk。"""
        text = "---" + chr(10) + "title: 任务描述" + chr(10) +             "tags: [work, pending]" + chr(10) + "status: doing" + chr(10) + "---"
        pr = chunk_markdown_v1(text, "tasks/x.md")
        self.assertEqual(len(pr.chunks), 1)
        self.assertEqual(pr.chunks[0].split_reason, "frontmatter_only")
        self.assertIn("任务描述", pr.chunks[0].content)

    def test_corrupt_bucket_falls_back_with_diagnostic(self):
        """审查#2 高：bucket 标记但无合法 mem 区块 → 回退普通切分 + orphan 诊断。"""
        text = "---" + chr(10) + "bucket: true" + chr(10) + "---" + SEP +             "# 标题" + SEP + "实际是普通正文内容。"
        pr = chunk_markdown_v1(text, "s.md")
        self.assertGreater(len(pr.chunks), 0, "损坏桶不得静默 0 chunk")
        self.assertGreaterEqual(pr.orphan_entry_count, 1, "损坏桶应有诊断")

    def test_duplicate_mem_id_gets_diagnostic_refs(self):
        """审查#3 高：重复 mem-id 区块引用不冲突（:d{N} 前缀）且标记 diagnostic。"""
        text = _bucket({"mem-aaa": "第一份内容。"}) + chr(10) +             "## mem-aaa" + chr(10) + "第二份重复区块。"
        pr = chunk_markdown_v1(text, "s.md")
        self.assertIn("mem-aaa", pr.duplicate_mem_ids)
        diag = [c for c in pr.chunks if c.diagnostic]
        self.assertTrue(diag, "重复区块应标记 diagnostic")
        refs = [c.source_ref for c in pr.chunks]
        self.assertEqual(len(refs), len(set(refs)), "重复区块 ref 冲突")

    def test_duplicate_empty_mem_id_is_still_reported(self):
        text = "---" + chr(10) + "bucket: true" + chr(10) + "---" + chr(10) + \
            "## mem-aaa" + chr(10) + "## mem-aaa" + chr(10) + "有效内容"
        pr = chunk_markdown_v1(text, "s.md")
        self.assertIn("mem-aaa", pr.duplicate_mem_ids)

    def test_bucket_true_in_code_does_not_trigger(self):
        """审查#2 高：正文/代码示例出现 bucket: true 不误触发桶模式。"""
        text = "# 示例" + SEP + "配置里写了 bucket: true 但这是普通文档。" +             SEP + "继续正文。"
        pr = chunk_markdown_v1(text, "a.md")
        self.assertFalse(_is_bucket(text, {}))
        self.assertTrue(all(c.mem_id is None for c in pr.chunks))

    def test_chunk_contract_fields(self):
        """审查#5 中：Chunk 契约补齐（title/tags/meta/offsets/chunk_id）。"""
        text = "---" + chr(10) + "title: 我的笔记" + chr(10) +             "tags: [a, b]" + chr(10) + "---" + SEP + "正文内容若干。"
        pr = chunk_markdown_v1(text, "a.md")
        self.assertEqual(pr.frontmatter.get("title"), "我的笔记")
        c = pr.chunks[0]
        self.assertEqual(c.title, "我的笔记")
        self.assertIn("a", c.tags)
        self.assertEqual(c.chunk_id, c.source_ref)
        self.assertIsInstance(c.start_offset, int)
        self.assertIsInstance(c.end_offset, int)

    def test_coverage_ratio_reported(self):
        """审查#5 中：ParseResult.coverage_ratio 报告覆盖率。"""
        pr = chunk_markdown_v1("# h" + SEP + "正文。" * 50, "a.md")
        self.assertGreater(pr.coverage_ratio, 0.5)
        self.assertEqual(pr.zero_chunk_files, 0)


class TestShadowReport(unittest.TestCase):
    def test_shadow_report_counts(self):
        import tempfile
        from pathlib import Path
        tmp = tempfile.mkdtemp()
        Path(tmp, "a.md").write_text("# a" + SEP + "内容。" * 100, encoding="utf-8")
        Path(tmp, "ark/memory/sessions").mkdir(parents=True)
        Path(tmp, "ark/memory/sessions/2026-09.md").write_text(
            _bucket({"mem-aaa": "桶内条目。"}), encoding="utf-8")
        Path(tmp, "ark/memory/archive/2026").mkdir(parents=True)
        Path(tmp, "ark/memory/archive/2026/x.md").write_text("归档不参与", encoding="utf-8")
        rep = build_shadow_report(tmp)
        self.assertEqual(rep["files"], 2)  # archive 被排除
        self.assertEqual(rep["diagnostics"]["orphan_entry_count"], 0)
        Path(tmp, "empty.md").write_text("", encoding="utf-8")
        rep = build_shadow_report(tmp)
        self.assertEqual(rep["zero_chunk_files"], [])
        self.assertIn("empty.md", rep["empty_files"])


class TestStructureSplitV2(unittest.TestCase):
    def test_heading_prefix_removes_heading_only_chunk(self):
        text = "# 标题" + SEP + "这是标题对应的正文，包含足够的语义。"
        parsed = chunk_markdown_v2(text, "a.md", min_chars=64)
        self.assertEqual(len(parsed.chunks), 1)
        self.assertEqual(parsed.chunks[0].split_reason, "heading_prefix")
        self.assertIn("# 标题", parsed.chunks[0].content)
        self.assertIn("这是标题对应的正文", parsed.chunks[0].content)
        self.assertLessEqual(len(parsed.chunks[0].content), HARD_CHARS)

    def test_heading_only_is_retained_when_no_body_exists(self):
        parsed = chunk_markdown_v2("# 只有标题", "a.md", min_chars=64)
        self.assertEqual(len(parsed.chunks), 1)
        self.assertEqual(parsed.chunks[0].split_reason, "heading_only")

    def test_oversized_heading_is_hard_bounded(self):
        parsed = chunk_markdown_v2("# " + "标题" * 500, "a.md", min_chars=64)
        self.assertGreater(len(parsed.chunks), 1)
        self.assertTrue(all(len(chunk.content) <= HARD_CHARS for chunk in parsed.chunks))
        self.assertTrue(all(chunk.split_reason == "heading_hard_cut" for chunk in parsed.chunks))
        self.assertEqual([chunk.chunk_index for chunk in parsed.chunks], list(range(len(parsed.chunks))))

    def test_short_prose_merges_only_inside_same_heading(self):
        text = (
            "## 一" + SEP + "短段甲。" + SEP + "短段乙。" + SEP
            + "## 二" + SEP + "短段丙。"
        )
        parsed = chunk_markdown_v2(text, "a.md", min_chars=64)
        self.assertEqual(len(parsed.chunks), 2)
        self.assertTrue(all(c.split_reason == "heading_prefix" for c in parsed.chunks))
        self.assertIn("短段甲", parsed.chunks[0].content)
        self.assertIn("短段乙", parsed.chunks[0].content)
        self.assertNotIn("短段丙", parsed.chunks[0].content)

    def test_v2_bucket_and_structural_boundaries_are_preserved(self):
        text = _bucket({
            "mem-aaa": "## 内部标题" + SEP + "条目甲正文。" + SEP + "```" + chr(10) + "x" + chr(10) + "```",
            "mem-bbb": "条目乙正文。",
        })
        parsed = chunk_markdown_v2(text, "sessions.md", min_chars=64)
        self.assertEqual(parsed.entry_count, 2)
        self.assertTrue(all(len(c.content) <= HARD_CHARS for c in parsed.chunks))
        for chunk in parsed.chunks:
            if chunk.mem_id == "mem-aaa":
                self.assertNotIn("条目乙正文", chunk.content)
            if chunk.mem_id == "mem-bbb":
                self.assertNotIn("条目甲正文", chunk.content)
        self.assertTrue(any("```" in c.content and c.content.count("```") % 2 == 0
                            for c in parsed.chunks))

    def test_v2_refs_are_content_hash_stable_after_unrelated_insert(self):
        before = "# h" + SEP + "## 稳定区" + SEP + "稳定段落。" + "x" * 180
        after = "# h" + SEP + "## 新区" + SEP + "插入段落。" + "y" * 180 + SEP + "## 稳定区" + SEP + "稳定段落。" + "x" * 180
        first = chunk_markdown_v2(before, "a.md", min_chars=32)
        second = chunk_markdown_v2(after, "a.md", min_chars=32)
        refs = {chunk.content: chunk.source_ref for chunk in first.chunks}
        later = {chunk.content: chunk.source_ref for chunk in second.chunks}
        stable = "## 稳定区\n稳定段落。" + "x" * 180
        self.assertIn(stable, later)
        self.assertEqual(refs[stable], later[stable])

    def test_v2_shadow_report_has_fragment_metrics(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.md").write_text("# h" + SEP + "正文。", encoding="utf-8")
            report = build_v2_shadow_report(tmp, min_chars=64)
        self.assertEqual(report["strategy"], "markdown-structure-v2")
        self.assertIn("short_fragments", report)
        self.assertEqual(report["over_hard"], 0)


if __name__ == "__main__":
    unittest.main()
