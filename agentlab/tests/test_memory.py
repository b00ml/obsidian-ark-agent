import asyncio
import unittest
from agentlab.memory.working import WorkingMemory, _side_effects, _pin_directive, _mask_observations
from agentlab.core.message import Message, TokenUsage, ToolCall, ToolCallFunction
from agentlab.core.llm import LLMProvider, LLMResponse


def _user(s: str) -> Message:
    return Message(role="user", content=s)


def _asm(s: str) -> Message:
    return Message(role="assistant", content=s)


class FakeSummarizer(LLMProvider):
    """固定摘要 + 记录收到的 prompt，用于断言 .st 模板/增量语义。"""

    def __init__(self, text="## Goal\n目标\n## Key Decisions\n决策X"):
        self.prompts: list[str] = []
        self.text = text

    async def chat(self, messages, tools=None, **kw):
        self.prompts.append(messages[0].content)
        return LLMResponse(content=self.text, tool_calls=[],
                           usage=TokenUsage(input_tokens=1, output_tokens=1),
                           stop_reason="stop")


class TestWorkingMemory(unittest.TestCase):
    def test_cut_point_not_in_middle_of_tool_round(self):
        # 一个"回合" = user -> assistant(tool_calls) -> tool。切点不落在 assistant/tool 中段
        wm = WorkingMemory(context_window=1000, reserve_tokens=10, keep_recent_tokens=0)
        wm.add(_user("第一步目标"))
        wm.add(_asm("{\"tool\":\"x\"}"))
        wm.add(Message(role="tool", content="result"))
        wm.add(_user("继续"))
        cut = wm.find_cut_point()
        # 应回到第一个 user 那里（0），不切在中段的 assistant/tool
        self.assertEqual(cut, 0)

    def test_over_budget(self):
        wm = WorkingMemory(context_window=100, reserve_tokens=90)  # 阈值 10 token
        wm.add(_user("x" * 100))
        self.assertTrue(wm.over_budget)

    def test_compact_keeps_last_user(self):
        wm = WorkingMemory(context_window=100, reserve_tokens=0, keep_recent_tokens=0)
        wm.add(_user("旧内容"))
        wm.add(_user("新问题"))
        cut = wm.find_cut_point()
        # 最后一条 user 必须保留 → 切点 <= 1
        self.assertLessEqual(cut, 1)

    def test_compact_produces_checkpoint_via_st_template(self):
        fake = FakeSummarizer()
        wm = WorkingMemory(context_window=100, reserve_tokens=0,
                           keep_recent_tokens=0, summarizer=fake)
        # 足够长触发 over_budget（estimate≈chars/4，阈值=100-0）
        wm.add(_user("任务开始" + "很" * 200))
        wm.add(_asm("做了若干事" + "很" * 200))
        wm.add(_user("中途进展" + "很" * 200))
        wm.add(_user("最新问题"))
        import asyncio
        asyncio.run(wm.compact())
        # 压缩后应有检查点（摘要 + 保留段）
        caps = [m for m in wm.messages() if m.role == "system"]
        self.assertTrue(any("[工作记忆检查点]" in (m.content or "") for m in caps))
        # 摘要 prompt 来自 summarize-user.st 模板
        self.assertTrue(fake.prompts, "应调用一次 summarizer")
        self.assertIn("工作记忆检查点摘要", fake.prompts[0])
        self.assertIn("旧检查点", fake.prompts[0])
        self.assertIn("新增对话", fake.prompts[0])

    def test_condense_incremental_uses_prev_checkpoint(self):
        fake = FakeSummarizer()
        wm = WorkingMemory(summarizer=fake)
        seq = [
            Message(role="system", content="[工作记忆检查点]\nGoal: 旧目标"),
            _user("继续推进"),
            _asm("新做了Y"),
            Message(role="tool", content="ok"),
            _user("最新问题"),
        ]
        import asyncio
        asyncio.run(wm.condense(seq, keep_recent_tokens=0))
        # 增量：第二次摘要 prompt 应带入旧检查点摘要正文
        self.assertTrue(any("旧目标" in p for p in fake.prompts))

    def test_condense_drops_when_no_summarizer(self):
        wm = WorkingMemory(keep_recent_tokens=0)  # 无 summarizer
        seq = [_user("a" * 40), _user("b" * 40), _user("c" * 40), _user("最新")]
        import asyncio
        out = asyncio.run(wm.condense(seq, keep_recent_tokens=0))
        # 兜底：纯丢弃前段，仅保留近端（含最后一个 user 输入）
        self.assertLess(len(out), len(seq))
        self.assertEqual(out[-1].content, "最新")

    def test_condense_pins_latest_user_directive_even_when_folded(self):
        # OPT-087：前段被折叠时，"用户最近一条待办指令"必须逐字钉回保留段，
        # 否则干到一半就会忘了用户到底要什么（tau 展示历史≠回放上下文）。
        wm = WorkingMemory(keep_recent_tokens=0)  # 无 summarizer，走纯丢弃前段兜底
        seq = [
            _user("旧背景很" + "长" * 40),
            _user("把两份笔记改成按内容命名，不要按 BV 号"),
            _user("行程安排很" + "长" * 40),
            _user("完成了吗"),
        ]
        import asyncio
        out = asyncio.run(wm.condense(seq, keep_recent_tokens=0))
        # 折叠仍在（丢弃了更早的前段），但"待办指令"被逐字钉住
        self.assertLess(len(out), len(seq))
        self.assertTrue(any("按内容命名" in (m.content or "") for m in out))
        self.assertEqual(out[0].content.startswith("[待办指令]"), True)

    def test_pin_directive_falls_back_to_first_user_when_collapsed_has_no_user(self):
        # OPT-088 / tau「first user message always preserved」：折叠段内无 user 时，
        # 回退钉首条 user 原文（原始任务锚），不丢任务来源。
        entries = [
            _asm("{\"tool\":\"read\"}"),
            Message(role="tool", content="result", name="vault_read"),
            _user("原始任务：把笔记按内容命名"),
            _user("完成了吗"),
        ]
        pin = _pin_directive(entries, cut=2)  # collapsed[0:2]=[assistant, tool]，无 user
        self.assertEqual(pin, "原始任务：把笔记按内容命名")

    def test_mask_observations_masks_old_tool_result_keeps_name(self):
        # OPT-088 / tau Tier-2 观察遮盖：较旧 tool 结果 → [output from X omitted]，
        # 但保留工具名（与 assistant tool_calls 配对不破坏），且不污染原存储。
        old = "o" * 12000   # 很旧的工具结果（远超预算，会被遮盖）
        recent = "r" * 3000  # 较新的工具结果（落在保留预算内，保持原样）
        entries = [
            Message(role="tool", content=old, name="vault_read"),
            Message(role="tool", content=recent, name="vault_search"),
            _user("最新"),
        ]
        out = _mask_observations(entries, keep_recent_tokens=500)  # 预算只盖住较新范围
        # 较旧的 tool 被遮盖、保留工具名与 call_id；较新的 tool 保持原样；user 原样
        self.assertIn("omitted", out[0].content)
        self.assertEqual(out[0].name, "vault_read")
        self.assertEqual(out[0].tool_call_id, entries[0].tool_call_id)
        self.assertEqual(out[1].content, recent)
        self.assertEqual(out[2].content, "最新")
        # 原存储不被污染（视图裁剪，盘全量）
        self.assertEqual(entries[0].content, old)
        self.assertEqual(entries[1].content, recent)

    def test_side_effects_extracted_from_collapsed(self):
        tc = ToolCall(id="t", function=ToolCallFunction(
            name="vault_write", arguments='{"path":"raw/x.jpg"}'))
        msgs = [
            _user("任务"),
            Message(role="assistant", content=None, tool_calls=[tc]),
        ]
        effects = _side_effects(msgs)
        self.assertIn("raw/x.jpg", effects)


class TestAnchoredCondense(unittest.TestCase):
    """L9/OPT-104 锚定分段压缩：messages[:anchor] 冻结前缀逐字节保留（对象级不变），
    只折叠 [anchor, cut)，摘要插在 anchor 位——provider 前缀缓存跨压缩命中。"""

    def _wm(self, summarizer=None):
        return WorkingMemory(keep_recent_tokens=0, summarizer=summarizer)

    def test_frozen_prefix_preserved_verbatim(self):
        fake = FakeSummarizer(text="CP1")
        wm = self._wm(fake)
        sys_m = Message(role="system", content="sys prompt")
        seq = [sys_m, _user("u1"), _asm("a1"), _user("u2"), _asm("a2"), _user("u3")]
        out = asyncio.run(wm.condense(seq, keep_recent_tokens=0, anchor=1))
        # _find_cut 落在 u2（index 3）：折叠 [1,3)=[u1,a1]，保留 [u2,a2,u3]
        self.assertIs(out[0], seq[0])  # 冻结前缀 = 原对象原字节
        self.assertEqual(out[1].role, "system")
        self.assertIn("[工作记忆检查点]", out[1].content)
        self.assertIn("CP1", out[1].content)
        # 钉住指令取折叠区最近 user（u1），不越进保留段
        self.assertEqual(out[2].content, "[待办指令] u1")
        self.assertIs(out[3], seq[3])
        self.assertIs(out[4], seq[4])
        self.assertIs(out[5], seq[5])
        self.assertEqual(wm.last_condense_head, 2)

    def test_no_progress_when_cut_le_anchor(self):
        wm = self._wm()
        seq = [_user("a" * 40), _user("b" * 40), _user("c" * 40), _user("d" * 40)]
        out = asyncio.run(wm.condense(seq, keep_recent_tokens=0, anchor=10))
        self.assertIs(out, seq)  # 折叠区为空 → 原样放行，冻结前缀绝不动
        self.assertEqual(wm.last_condense_head, 0)

    def test_no_repin_from_frozen_region(self):
        # 冻结区里已钉过的指令不得重复钉；折叠区无 user → 不新增 pin
        wm = self._wm()
        sys_cp = Message(role="system", content="[工作记忆检查点]\nGoal: 旧")
        pin = _user("[待办指令] 把笔记改成按内容命名")
        seq = [sys_cp, pin, _asm("a1"), Message(role="tool", content="ok"),
               _user("u_mid"), _asm("a2"), _user("u_final")]
        out = asyncio.run(wm.condense(seq, keep_recent_tokens=0, anchor=2))
        self.assertIs(out[0], seq[0])
        self.assertIs(out[1], seq[1])
        joined = "\n".join((m.content or "") for m in out)
        self.assertEqual(joined.count("[待办指令]"), 1)

    def test_anchor_zero_matches_legacy_layout(self):
        fake = FakeSummarizer(text="CP")
        wm = self._wm(fake)
        seq = [_user("u1"), _asm("a1"), _user("u2"), _asm("a2"),
               _user("u3"), _asm("a4"), _user("u5")]
        out = asyncio.run(wm.condense(seq, keep_recent_tokens=0))
        # anchor=0（默认）与旧全量折叠同布局：检查点最前 → 钉指令 → 保留段
        self.assertEqual(out[0].role, "system")
        self.assertIn("[工作记忆检查点]", out[0].content)
        self.assertTrue(out[1].content.startswith("[待办指令]"))


if __name__ == "__main__":
    unittest.main()

class TestFoldSlice(unittest.TestCase):
    """OPT-110 四期：折叠分片——单次折叠区段 ≤ max_fold_tokens（小步多次，学 pi 早折勤折）。"""

    def _seq(self):
        return [_user("u" * 200), _asm("a" * 200), _user("u" * 200), _asm("a" * 200)]

    def test_find_forward_cut_slices_at_boundary(self):
        from agentlab.memory.working import _find_forward_cut
        seq = self._seq()
        cut = _find_forward_cut(seq, 0, 120, roles=frozenset({"user", "assistant"}))
        self.assertEqual(cut, 2)  # 累计 ~150 ≥120 → 切在第 2 条边界（区段 ~100 ≤ 120）
        self.assertIsNone(_find_forward_cut(seq, 0, 10000))  # 区段不足预算 → None

    def test_condense_slice_caps_fold(self):
        wm = WorkingMemory(keep_recent_tokens=0, summarizer=FakeSummarizer(text="CP"))
        seq = self._seq()
        out = asyncio.run(wm.condense(seq, keep_recent_tokens=0, anchor=0, max_fold_tokens=120))
        # 折叠区被切片：检查点只覆盖前半，后半保留原文
        self.assertTrue(any(m.role == "system" and "[工作记忆检查点]" in (m.content or "") for m in out))
        self.assertEqual(out[-1], seq[-1])
