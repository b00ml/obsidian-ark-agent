import unittest

from agentlab.eval.eval import evaluate_case, load_golden
from agentlab.eval.metrics import (
    answer_relevancy,
    chunk_boundary_coherence,
    context_precision,
    faithfulness,
)

_GOLDEN = {
    "id": "q1",
    "question": "对比 X 与 Y 笔记的结论差异",
    "expected_refs": ["X.md", "Y.md"],
    "expected_contexts": ["X.md 采用向量检索", "Y.md 采用关键词检索"],
}


class TestMetrics(unittest.TestCase):
    def test_answer_relevancy_ranks_oriented_answer_higher(self):
        q = "对比 X 与 Y 笔记的结论差异"
        # 覆盖了问题核心词（结论/差异/对比）的回答，相关性高于无关回答
        oriented = answer_relevancy("结论的差异在于向量 vs 关键词 对比", q)
        irrelevant = answer_relevancy("今天天气不错", q)
        self.assertGreater(oriented, irrelevant)
        self.assertLess(irrelevant, 0.4)

    def test_faithfulness_penalizes_hallucination(self):
        ctx = _GOLDEN["expected_contexts"]
        # 忠实：答案事实都在 ctx
        self.assertAlmostEqual(faithfulness("采用向量检索", ctx, _GOLDEN["question"]), 1.0)
        # 编造：token"哈工大"不在 ctx
        self.assertLess(faithfulness("哈工大模型效果最佳", ctx, _GOLDEN["question"]), 1.0)

    def test_faithfulness_repeating_question_is_no_hallucination(self):
        # 只复述问题词 = 无新增事实 = 无编造
        self.assertEqual(faithfulness("对比结论差异", _GOLDEN["expected_contexts"], _GOLDEN["question"]), 1.0)

    def test_context_precision_hit_ratio(self):
        self.assertEqual(context_precision(["X.md", "Y.md"], ["<X.md>", "<Y.md>"]), 1.0)
        self.assertEqual(context_precision(["X.md", "Y.md"], ["<X.md>", "<Z.md>"]), 0.5)
        self.assertEqual(context_precision(["X.md"], []), 0.0)

    def test_chunk_boundary_coherence_sentence_aware(self):
        # 句感知：每个块都以句子终止符收尾 → 边界完好 1.0
        good = ["RAG 不可被长上下文替代。", "先用 BM25 初筛，再重排；这样省钱。"]
        self.assertEqual(chunk_boundary_coherence(good), 1.0)
        # 硬切：块尾停在句中（无终止符）→ 被判为边界破坏（1/2 破碎）
        broken = ["RAG 不可被长上下文替", "代，先用 BM25 初筛再重排。"]
        self.assertEqual(chunk_boundary_coherence(broken), 0.5)
        # 空输入
        self.assertEqual(chunk_boundary_coherence([]), 0.0)


class TestEval(unittest.TestCase):
    def test_evaluate_case_offline(self):
        m = evaluate_case(_GOLDEN, answer="采用向量检索和关键词检索", retrieved=["<X.md>", "<Y.md>"])
        self.assertEqual(m.context_precision, 1.0)
        self.assertGreaterEqual(m.faithfulness, 0.8)

    def test_golden_loads(self):
        cases = load_golden()
        self.assertGreaterEqual(len(cases), 3)
        for c in cases:
            self.assertIn("question", c)
            self.assertIn("expected_refs", c)


if __name__ == "__main__":
    unittest.main()