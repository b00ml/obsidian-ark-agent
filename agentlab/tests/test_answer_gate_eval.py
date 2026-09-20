import json
import tempfile
import unittest
from pathlib import Path

from agentlab.eval.answer_gate_eval import evaluate_answer_gate_report, load_external_probes
from agentlab.eval.rag_task_schema import migrate_v1_task


def _retrieval(rows, **extra):
    return {"per_query": rows, **extra}


class TestAnswerGateEval(unittest.TestCase):
    def test_migration_adds_answer_level_defaults(self):
        task = migrate_v1_task({
            "id": "positive", "query": "cache", "expected_refs": ["wiki/cache.md"],
        })
        self.assertEqual(task["expected_decision"], "answerable")
        self.assertFalse(task["expected_abstention"])
        self.assertEqual(task["allowed_refs"], [])

    def test_real_answerable_and_absent_probes_score_safely(self):
        tasks = [
            {
                "id": "positive", "query": "cache", "expected_refs": ["wiki/cache.md"],
                "answer_probes": [{
                    "source": "human", "answer": "依据 [[wiki/cache.md]] 可使用缓存。",
                    "expected_decision": "answerable", "expected_abstention": False,
                }],
            },
            {
                "id": "absent", "query": "unknown", "expected_refs": [],
                "answerability": "absent",
                "answer_probes": [{
                    "source": "human", "answer": "当前资料不足，无法确认。",
                    "expected_decision": "insufficient", "expected_abstention": True,
                }],
            },
        ]
        report = evaluate_answer_gate_report(_retrieval([
            {"id": "positive", "rrf": {"refs": ["wiki/cache.md#intro:ch1"]}},
            {"id": "absent", "rrf": {"refs": ["wiki/nearby.md"]}},
        ]), tasks)
        summary = report["summary"]["answer_gate"]
        self.assertEqual(summary["evaluation_mode"], "real")
        self.assertEqual(summary["real_cases"], 2)
        self.assertEqual(summary["false_answer_rate"], 0.0)
        self.assertEqual(summary["false_refusal_rate"], 0.0)
        self.assertEqual(summary["candidate_hit_but_refused_cases"], 0)
        self.assertEqual(summary["candidate_miss_refused_cases"], 0)
        self.assertEqual(summary["candidate_hit_out_of_scope_cases"], 0)
        self.assertEqual(summary["by_query_type"]["unknown"]["cases"], 2)
        self.assertEqual(summary["by_query_type"]["unknown"]["candidate_expected_ref_hit_rate"], 1.0)

    def test_unsafe_answer_and_outside_citation_are_visible(self):
        task = {
            "id": "unsafe", "query": "unknown", "expected_refs": [],
            "answerability": "absent",
            "answer_probes": [{
                "source": "llm", "answer": "结论见 [[wiki/nearby.md]]。",
                "expected_decision": "insufficient", "expected_abstention": True,
                "answerability": "answerable", "assessment": "answer",
            }],
        }
        report = evaluate_answer_gate_report(
            _retrieval([{"id": "unsafe", "rrf": {"refs": ["wiki/nearby.md"]}}]), [task],
        )
        summary = report["summary"]["answer_gate"]
        self.assertEqual(summary["false_answer_rate"], 1.0)
        self.assertEqual(summary["citation_out_of_scope_rate"], 0.0,
                         "absent-task decision takes precedence over citation scoring")

    def test_synthetic_probe_is_explicitly_not_real_evidence(self):
        task = {"id": "synthetic", "query": "cache", "expected_refs": ["wiki/cache.md"]}
        report = evaluate_answer_gate_report(
            _retrieval([{"id": "synthetic", "rrf": {"refs": ["wiki/cache.md"]}}]), [task],
        )
        summary = report["summary"]["answer_gate"]
        self.assertEqual(summary["evaluation_mode"], "synthetic")
        self.assertEqual(summary["real_cases"], 0)
        self.assertEqual(summary["synthetic_cases"], 1)
        self.assertIn("unknown", summary["by_query_type"])

    def test_fixture_replay_is_not_real_even_when_source_says_llm(self):
        """Transport labels must not turn deterministic fixture output into evidence."""
        task = {
            "id": "fixture-replay", "query": "cache",
            "expected_refs": ["wiki/cache.md"],
            "answer_probes": [{
                "source": "llm",
                "answer": "依据 [[wiki/cache.md]] 可回答。",
                "provenance": {
                    "kind": "fixture_replay",
                    "model": "deterministic-fixture-replay",
                },
            }],
        }
        report = evaluate_answer_gate_report(
            _retrieval([{
                "id": "fixture-replay",
                "rrf": {"refs": ["wiki/cache.md"]},
            }]),
            [task],
        )
        summary = report["summary"]["answer_gate"]
        case = report["cases"][0]
        self.assertEqual(case["source"], "llm")
        self.assertFalse(case["real_probe"])
        self.assertEqual(summary["real_cases"], 0)
        self.assertEqual(summary["real_probe_cases"], 0)
        self.assertEqual(summary["eligible_real_cases"], 0)
        self.assertEqual(summary["evaluation_mode"], "synthetic")
        self.assertLess(summary["metrics_cases"], summary["cases"])

    def test_live_overlay_uses_real_metrics_not_synthetic_fallbacks(self):
        tasks = [
            {"id": "live", "query": "cache", "expected_refs": ["wiki/cache.md"]},
            {"id": "fallback", "query": "missing", "expected_refs": ["wiki/missing.md"]},
        ]
        report = evaluate_answer_gate_report(
            _retrieval([
                {"id": "live", "rrf": {"refs": ["wiki/cache.md#h:ch1"]}},
                {"id": "fallback", "rrf": {"refs": []}},
            ]),
            tasks,
            external_probes=[{
                "task_id": "live", "source": "llm",
                "answer": "依据 [[wiki/cache.md]] 可以回答。",
                "candidate_refs": ["wiki/cache.md#h:ch1"],
                # An execution artifact cannot change the frozen expected
                # label. This value is intentionally ignored by the runner.
                "expected_decision": "insufficient",
            }],
        )
        summary = report["summary"]["answer_gate"]
        self.assertEqual(summary["evaluation_mode"], "mixed")
        self.assertEqual(summary["metrics_source"], "real")
        self.assertEqual(summary["metrics_cases"], 1)
        self.assertEqual(summary["false_refusal_rate"], 0.0)
        case = next(item for item in report["cases"] if item["task_id"] == "live")
        self.assertEqual(case["expected_decision"], "answerable")

    def test_external_probe_loader_accepts_collection_report(self):
        path = Path(tempfile.mkdtemp()) / "probes.json"
        path.write_text(json.dumps({"probes": [{
            "task_id": "live", "source": "llm", "answer": "回答",
        }]}, ensure_ascii=False), encoding="utf-8")
        self.assertEqual(load_external_probes(path)[0]["task_id"], "live")

    def test_route_falls_back_for_not_applicable_rrf_fixture(self):
        task = {
            "id": "isolation", "query": "inbox", "route": "memory_keyword",
            "expected_refs": ["ark/memory/core/mem-shared-default.md#mem-shared-default"],
            "answerability": "answerable",
            "answer_probes": [{
                "source": "human",
                "answer": "依据 [[ark/memory/core/mem-shared-default.md#mem-shared-default]] 可回答。",
            }],
        }
        report = evaluate_answer_gate_report(_retrieval([{
            "id": "isolation",
            "rrf": {"status": "not_applicable", "refs": []},
            "memory_keyword": {
                "status": "available",
                "refs": ["ark/memory/core/mem-shared-default.md#mem-shared-default"],
            },
        }]), [task], mode="rrf")
        case = report["cases"][0]
        self.assertEqual(case["retrieval_mode"], "memory_keyword")
        self.assertFalse(case["predicted_abstention"])

    def test_scope_mismatched_live_probe_is_excluded_from_real_metrics(self):
        task = {
            "id": "fixture", "query": "q", "vault_root": "E:/fixture-vault",
            "expected_refs": ["fixture.md"],
            "answer_probes": [{
                "source": "llm", "answer": "依据 [[fixture.md]] 回答。",
            }],
        }
        report = evaluate_answer_gate_report(
            _retrieval([{"id": "fixture", "rrf": {"refs": ["fixture.md"]}}]),
            [task],
            external_probes=[{
                "task_id": "fixture", "source": "llm",
                "answer": "依据 [[fixture.md]] 回答。",
                "provenance": {"kind": "live_agent", "vault_root": "E:/real-vault"},
            }],
        )
        summary = report["summary"]["answer_gate"]
        self.assertEqual(summary["real_cases"], 1)
        self.assertEqual(summary["eligible_real_cases"], 0)
        self.assertEqual(summary["scope_mismatch_cases"], 1)
        self.assertEqual(summary["metrics_source"], "synthetic")

    def test_report_meta_scope_is_used_for_non_fixture_tasks(self):
        task = {
            "id": "ordinary", "query": "q", "expected_refs": ["note.md"],
            "expected_decision": "answerable", "expected_abstention": False,
        }
        report = evaluate_answer_gate_report(
            _retrieval(
                [{"id": "ordinary", "rrf": {
                    "status": "available", "refs": ["note.md"]
                }}],
                meta={"vault_root": "E:/vault"},
            ),
            [task],
            external_probes=[{
                "task_id": "ordinary", "source": "llm",
                "answer": "依据 [[note.md]]。",
                "provenance": {"vault_root": "e:/vault"},
            }],
        )
        case = report["cases"][0]
        self.assertTrue(case["scope_match"])
        self.assertFalse(case["excluded_from_metrics"])
        self.assertEqual(report["summary"]["answer_gate"]["eligible_real_cases"], 1)

    def test_live_probe_from_different_index_is_excluded(self):
        task = {
            "id": "versioned", "query": "q", "expected_refs": ["note.md"],
            "expected_decision": "answerable", "expected_abstention": False,
        }
        report = evaluate_answer_gate_report(
            _retrieval(
                [{"id": "versioned", "rrf": {
                    "status": "available", "refs": ["note.md"],
                }}],
                meta={
                    "vault_root": "E:/vault",
                    "chunking": {
                        "index_version": "s1-p4.5c-v2-min64",
                        "parser_version": "markdown-structure-v2-min64",
                        "strategy": "markdown-structure-v2-min64",
                    },
                    "index": {"embedding_model": "text-embedding-v4"},
                },
            ),
            [task],
            external_probes=[{
                "task_id": "versioned", "source": "llm",
                "answer": "依据 [[note.md]]。",
                "provenance": {
                    "vault_root": "E:/vault",
                    "index_version": "s1-p2-v1",
                    "parser_version": "markdown-structure-v1",
                    "chunk_strategy_version": "markdown-structure-v1",
                    "embedding_model": "text-embedding-v4",
                },
            }],
        )
        case = report["cases"][0]
        self.assertFalse(case["provenance_match"])
        self.assertTrue(case["excluded_from_metrics"])
        self.assertEqual(report["summary"]["answer_gate"]["eligible_real_cases"], 0)
        self.assertEqual(report["summary"]["answer_gate"]["provenance_mismatch_cases"], 1)

    def test_bounded_partial_uses_the_same_evidence_ledger_as_direct_gate(self):
        task = {
            "id": "partial", "query": "q", "expected_refs": ["note.md"],
            "expected_decision": "answerable", "expected_abstention": False,
            "answer_probes": [{
                "source": "llm",
                "answer": "依据 [[note.md]]，可确认已记录的事实；其余背景资料未覆盖。",
                "assessment": "insufficient",
                "answerability": "answerable",
            }],
        }
        report = evaluate_answer_gate_report(
            _retrieval([{"id": "partial", "rrf": {
                "status": "available", "refs": ["note.md"],
            }}]),
            [task], allow_bounded_partial=True,
        )
        case = report["cases"][0]
        self.assertTrue(case["post_gate"]["decision"] == "answerable")
        self.assertIn("bounded_partial", case["post_gate"]["reasons"])
        self.assertFalse(case["ledger_mismatch"])

    def test_bounded_partial_never_allows_out_of_scope_citation(self):
        task = {
            "id": "partial-outside", "query": "q", "expected_refs": ["note.md"],
            "expected_decision": "answerable", "expected_abstention": False,
            "answer_probes": [{
                "source": "llm",
                "answer": "依据 [[other.md]]，可确认已记录的事实；其余背景资料未覆盖。",
                "assessment": "insufficient",
                "answerability": "answerable",
            }],
        }
        report = evaluate_answer_gate_report(
            _retrieval([{"id": "partial-outside", "rrf": {
                "status": "available", "refs": ["note.md"],
            }}]),
            [task], allow_bounded_partial=True,
        )
        case = report["cases"][0]
        self.assertFalse(case["post_gate"]["decision"] == "answerable")
        self.assertIn("citation_out_of_scope", case["post_gate"]["reasons"])
        self.assertFalse(case["ledger_mismatch"])

    def test_coverage_and_error_attribution_distinguish_retrieval_and_assessor(self):
        tasks = [
            {"id": "retrieve-miss", "query": "q1", "expected_refs": ["a.md"],
             "expected_decision": "answerable", "expected_abstention": False,
             "answer_probes": [{"source": "human", "answer": "资料不足，无法确认。"}]},
            {"id": "partial-all", "query": "q2", "expected_refs": ["a.md", "b.md"],
             "expected_policy": "all", "expected_decision": "answerable",
             "expected_abstention": False,
             "answer_probes": [{"source": "human", "answer": "资料不足，无法确认。"}]},
            {"id": "assessor-miss", "query": "q3", "expected_refs": ["a.md"],
             "expected_decision": "answerable", "expected_abstention": False,
             "answer_probes": [{"source": "human", "assessment": "insufficient",
                                 "answerability": "answerable",
                                 "answer": "资料不足，无法确认。"}]},
        ]
        report = evaluate_answer_gate_report(_retrieval([
            {"id": "retrieve-miss", "rrf": {"refs": []}},
            {"id": "partial-all", "rrf": {"refs": ["a.md"]}},
            {"id": "assessor-miss", "rrf": {"refs": ["a.md"]}},
        ]), tasks)
        cases = {row["task_id"]: row for row in report["cases"]}
        self.assertEqual(cases["retrieve-miss"]["error_attribution"], "retrieval_miss")
        self.assertTrue(cases["partial-all"]["candidate_expected_ref_any"])
        self.assertFalse(cases["partial-all"]["candidate_expected_ref_all"])
        self.assertEqual(cases["partial-all"]["error_attribution"], "required_ref_missing")
        self.assertEqual(cases["assessor-miss"]["error_attribution"], "assessor_false_negative")
        self.assertEqual(
            report["summary"]["answer_gate"]["error_attribution"]["retrieval_miss"], 1
        )

    def test_cited_non_supporting_proposition_is_not_false_refusal(self):
        task = {
            "id": "qualified", "query": "是否互斥", "expected_refs": ["decision.md"],
            "expected_decision": "answerable", "expected_abstention": False,
        }
        report = evaluate_answer_gate_report(
            _retrieval([{"id": "qualified", "rrf": {"refs": ["decision.md"]}}]),
            [task],
            external_probes=[{
                "task_id": "qualified", "source": "llm",
                "assessment": "answer", "answerability": "answerable",
                "candidate_refs": ["decision.md"],
                "answer": (
                    "资料并不支持把两种定义当作互斥选项"
                    "（ref: decision.md）。"
                ),
            }],
        )
        case = report["cases"][0]
        self.assertFalse(case["predicted_abstention"])
        self.assertEqual(case["error_attribution"], "none")

    def test_heading_only_expected_chunk_is_content_gap_not_assessor_failure(self):
        task = {
            "id": "thin", "query": "q", "expected_refs": ["note.md"],
            "expected_decision": "answerable", "expected_abstention": False,
        }
        report = evaluate_answer_gate_report(
            _retrieval([{"id": "thin", "rrf": {"refs": ["note.md#title:ch1"]}}]),
            [task],
            external_probes=[{
                "task_id": "thin", "source": "llm",
                "candidate_refs": ["note.md#title:ch1"],
                "candidate_evidence": [{
                    "ref": "note.md#title:ch1",
                    "content_chars": 12,
                    "content_non_heading_chars": 0,
                }],
                "assessment": "insufficient",
                "answerability": "answerable",
                "answer": "资料不足，无法确认。",
            }],
        )
        case = report["cases"][0]
        self.assertEqual(case["error_attribution"], "retrieval_content_gap")
        self.assertFalse(case["candidate_expected_ref_contentful"])

    def test_bounded_partial_does_not_override_insufficient_answerability(self):
        task = {
            "id": "partial-negative", "query": "q", "expected_refs": [],
            "expected_decision": "insufficient", "expected_abstention": True,
            "answerability": "absent",
            "answer_probes": [{
                "source": "llm",
                "answer": "依据 [[nearby.md]]，下面只复述命中的片段。",
                "assessment": "insufficient",
                "answerability": "insufficient",
            }],
        }
        report = evaluate_answer_gate_report(
            _retrieval([{"id": "partial-negative", "rrf": {
                "status": "available", "refs": ["nearby.md"],
            }}]),
            [task], allow_bounded_partial=True,
        )
        case = report["cases"][0]
        self.assertFalse(case["post_gate"]["decision"] == "answerable")
        self.assertIn("answerability_insufficient", case["post_gate"]["reasons"])
        self.assertFalse(case["ledger_mismatch"])


if __name__ == "__main__":
    unittest.main()
