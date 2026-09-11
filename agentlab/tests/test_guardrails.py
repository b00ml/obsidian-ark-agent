import unittest
from agentlab.core.errors import AgentError
from agentlab.core.guardrails import extract_json, validate_output, ensure_structured


class TestGuardrails(unittest.TestCase):
    def test_plain_dict(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_fenced_json(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_text_surrounding(self):
        self.assertEqual(extract_json('答案是 {"x": 2} 完毕'), {"x": 2})

    def test_array(self):
        self.assertEqual(extract_json('步骤：[1, 2, 3]'), [1, 2, 3])

    def test_extra_tail_text(self):
        # 括号配对：模型末尾追加说明文字也能剥出完整 JSON
        obj = extract_json('{"a": 1, "b": "ok"} 以上为结果')
        self.assertEqual(obj, {"a": 1, "b": "ok"})

    def test_invalid_raises(self):
        with self.assertRaises(AgentError) as e:
            extract_json("完全没有 JSON")
        self.assertEqual(e.exception.code, "AGENT_GUARDRAIL")

    def test_validate_output(self):
        self.assertTrue(validate_output("ok"))
        self.assertFalse(validate_output(""))
        self.assertFalse(validate_output(None))

    def test_ensure_structured_predicates(self):
        out = '{"name": "n", "age": 3}'
        self.assertTrue(ensure_structured(out, [lambda o: o.get("age", 0) > 0]))
        self.assertFalse(ensure_structured(out, [lambda o: o.get("age", 0) > 5]))


if __name__ == "__main__":
    unittest.main()