import unittest

from agentlab.eval.rrf_sweep import sweep_rrf, weighted_rrf


class TestRrfSweep(unittest.TestCase):
    def test_vector_weight_promotes_vector_only_entry(self):
        routes = {
            "lexical": [{"ref": "lexical.md"}] * 3,
            "vector": [{"ref": "vector.md"}],
        }
        equal = weighted_rrf(routes, weights={"lexical": 1.0, "vector": 1.0}, k=1)
        boosted = weighted_rrf(routes, weights={"lexical": 1.0, "vector": 2.0}, k=1)
        self.assertEqual(equal, ["lexical.md"])
        self.assertEqual(boosted, ["vector.md"])

    def test_sweep_keeps_not_applicable_fixture_out_of_metrics(self):
        report = sweep_rrf({
            "meta": {"vault_root": "E:/vault"},
            "per_query": [
                {"id": "positive", "rrf": {"status": "available"},
                 "hybrid_shadow": {"routes": {
                     "lexical": {"refs": ["bad.md"]},
                     "vector": {"refs": ["gold.md#h:ch1"]},
                 }}},
                {"id": "fixture", "rrf": {"status": "not_applicable"}},
            ],
        }, [
            {"id": "positive", "query": "q", "expected_refs": ["gold.md"]},
            {"id": "fixture", "query": "q", "expected_refs": ["fixture.md"],
             "corpus_scope": "fixture", "vault_root": "fixtures/x"},
        ])
        equal = report["cases"][0]["variants"]["equal"]
        boosted = report["cases"][0]["variants"]["vector_2"]
        self.assertEqual(equal["recall@5"], 1.0)
        self.assertEqual(boosted["recall@5"], 1.0)
        summary = report["summary"]["by_query_type"]["unknown"]["equal"]
        self.assertEqual(summary["positive_tasks"], 1)
        self.assertEqual(report["cases"][1]["available"], False)

    def test_healthy_empty_negative_is_available_and_counted_as_clean(self):
        report = sweep_rrf({
            "per_query": [{
                "id": "negative",
                "rrf": {"status": "available"},
                "route_statuses": {
                    "p2_lexical": "available", "p2_vector": "available",
                },
                "hybrid_shadow": {
                    "routes": {
                        "lexical": {"status": "available", "refs": []},
                        "vector": {"status": "available", "refs": []},
                    },
                },
            }],
        }, [{
            "id": "negative", "query": "absent", "expected_refs": [],
            "query_type": "negative",
        }])
        case = report["cases"][0]
        self.assertTrue(case["available"])
        self.assertTrue(case["variants"]["equal"]["available"])
        summary = report["summary"]["by_query_type"]["negative"]["equal"]
        self.assertEqual(summary["negative_tasks"], 1)
        self.assertEqual(summary["clean@5"], 1.0)

    def test_query_type_policy_reports_single_route_and_concurrent_latency(self):
        report = sweep_rrf({
            "meta": {"vault_root": "E:/vault"},
            "per_query": [{
                "id": "exact", "route_statuses": {
                    "p2_lexical": "available", "p2_vector": "available",
                },
                "hybrid_shadow": {
                    "project_id": "default",
                    "routes": {
                        "lexical": {"status": "available", "refs": ["gold.md"], "latency_ms": 40},
                        "vector": {"status": "available", "refs": ["other.md"], "latency_ms": 90},
                    },
                },
            }],
        }, [{
            "id": "exact", "query": "gold", "expected_refs": ["gold.md"],
            "query_type": "exact_title", "scope": {"project_id": "default"},
        }])
        case = report["cases"][0]
        self.assertEqual(case["policy"]["variant"], "lexical")
        self.assertTrue(case["policy"]["available"])
        self.assertEqual(case["policy"]["recall@5"], 1.0)
        self.assertEqual(case["policy"]["estimated_latency_ms"], 40.0)
        group = report["summary"]["query_type_policy"]["exact_title"]
        self.assertEqual(group["latency"]["p50_ms"], 40.0)
        self.assertEqual(report["query_type_policy"]["exact_title"], "lexical")

    def test_unavailable_policy_route_is_not_counted_as_zero_recall(self):
        report = sweep_rrf({
            "per_query": [{
                "id": "semantic", "route_statuses": {
                    "p2_lexical": "available", "p2_vector": "unavailable",
                },
                "hybrid_shadow": {
                    "project_id": "default",
                    "routes": {
                        "lexical": {"status": "available", "refs": ["wrong.md"]},
                        "vector": {"status": "unavailable", "refs": ["gold.md"]},
                    },
                },
            }],
        }, [{
            "id": "semantic", "query": "meaning", "expected_refs": ["gold.md"],
            "query_type": "episodic_semantic", "scope": {"project_id": "default"},
        }])
        policy = report["cases"][0]["policy"]
        self.assertFalse(policy["available"])
        self.assertIsNone(policy["recall@5"])
        self.assertEqual(
            report["summary"]["query_type_policy"]["episodic_semantic"]["available_tasks"],
            0,
        )

    def test_policy_safety_carries_forbidden_and_scope_signals(self):
        report = sweep_rrf({
            "per_query": [{
                "id": "q", "route_statuses": {
                    "p2_lexical": "available", "p2_vector": "available",
                },
                "hybrid_shadow": {
                    "project_id": "other",
                    "routes": {
                        "lexical": {"status": "available", "refs": ["secret.md"]},
                        "vector": {"status": "available", "refs": ["secret.md"]},
                    },
                },
            }],
        }, [{
            "id": "q", "query": "q", "expected_refs": ["secret.md"],
            "query_type": "exact_title", "forbidden_refs": ["secret.md"],
            "scope": {"project_id": "default"},
        }])
        policy = report["cases"][0]["policy"]
        self.assertEqual(policy["forbidden_hits@5"], 1)
        self.assertEqual(policy["scope_status"], "mismatch")
        self.assertEqual(report["summary"]["policy_safety"]["forbidden_hits@5"], 1)
        self.assertEqual(report["summary"]["policy_safety"]["scope"]["mismatch"], 1)

    def test_policy_summary_attributes_non_clean_negative_candidates(self):
        report = sweep_rrf({
            "per_query": [{
                "id": "negative", "negative_kind": "plausible_absent",
                "route_statuses": {"p2_lexical": "available"},
                "hybrid_shadow": {
                    "project_id": "default",
                    "routes": {"lexical": {"status": "available", "refs": [
                        "ark/memory/core/preference.md#mem-1",
                        "Inbox/draft.md#heading",
                        "wiki/note.md#heading",
                    ]}},
                },
            }],
        }, [{
            "id": "negative", "query": "absent", "expected_refs": [],
            "query_type": "negative", "scope": {"project_id": "default"},
        }])
        case = report["cases"][0]
        self.assertEqual(case["negative_kind"], "plausible_absent")
        self.assertEqual(case["policy"]["source_types"], {
            "inbox": 1, "memory": 1, "vault": 1,
        })
        policy = report["summary"]["policy"]
        self.assertEqual(policy["negative_tasks"], 1)
        self.assertEqual(policy["clean@5"], 0.0)
        negative = policy["negative_by_kind"]["plausible_absent"]
        self.assertEqual(negative["non_clean_case_ids"], ["negative"])
        self.assertEqual(negative["non_clean_source_types"], {
            "inbox": 1, "memory": 1, "vault": 1,
        })

    def test_runtime_guarded_fallback_replays_semantic_classifier(self):
        report = sweep_rrf({
            "per_query": [
                {
                    "id": "semantic",
                    "rrf": {"status": "available"},
                    "route_statuses": {
                        "p2_lexical": "available", "p2_vector": "available",
                    },
                    "hybrid_shadow": {"routes": {
                        "lexical": {"status": "available", "refs": ["other.md"]},
                        "vector": {"status": "available", "refs": ["gold.md"]},
                    }},
                },
                {
                    "id": "negative",
                    "negative_kind": "plausible_absent",
                    "rrf": {"status": "available"},
                    "route_statuses": {
                        "p2_lexical": "available", "p2_vector": "available",
                    },
                    "hybrid_shadow": {"routes": {
                        "lexical": {"status": "available", "refs": []},
                        "vector": {"status": "available", "refs": ["nearby.md"]},
                    }},
                },
            ],
        }, [
            {
                "id": "semantic", "query": "如何处理这个问题", "expected_refs": ["gold.md"],
                "query_type": "paraphrase", "scope": {"project_id": "default"},
            },
            {
                "id": "negative", "query": "没有这个资料", "expected_refs": [],
                "query_type": "negative", "scope": {"project_id": "default"},
            },
        ], runtime_guarded_fallback=True)
        semantic, negative = report["cases"]
        self.assertTrue(semantic["runtime_guarded_fallback"]["enabled"])
        self.assertEqual(semantic["runtime_guarded_fallback"]["variant"], "equal")
        self.assertFalse(negative["runtime_guarded_fallback"]["enabled"])
        self.assertEqual(negative["runtime_guarded_fallback"]["variant"], "lexical")
        runtime = report["summary"]["runtime_guarded_fallback"]
        self.assertEqual(runtime["recall@5"], 1.0)
        self.assertEqual(runtime["clean@5"], 1.0)

    def test_runtime_display_uses_recorded_memory_filtered_refs(self):
        report = sweep_rrf({
            "per_query": [{
                "id": "memory", "rrf": {"status": "available"},
                "route_statuses": {
                    "p2_lexical": "available", "p2_vector": "available",
                },
                "hybrid_shadow": {
                    "hybrid_status": "available",
                    "display_mode": "memory_focused",
                    "display_refs": ["ark/memory/core/gold.md#mem-1"],
                    "latency_ms": 84.2,
                    "coverage_contract": {"required_groups": [["ark/memory/core/gold.md"]]},
                    "routes": {
                        "lexical": {"status": "available", "refs": ["other.md"]},
                        "vector": {"status": "available", "refs": ["gold.md"]},
                    },
                },
            }],
        }, [{
            "id": "memory", "query": "用户偏好", "expected_refs": ["ark/memory/core/gold.md"],
            "query_type": "user_preference", "scope": {"project_id": "default"},
        }])
        display = report["cases"][0]["runtime_display"]
        self.assertEqual(display["mode"], "memory_focused")
        self.assertEqual(display["recall@5"], 1.0)
        self.assertEqual(display["source_types"], {"memory": 1})
        self.assertTrue(display["eval_coverage_forced"])
        summary = report["summary"]["runtime_display"]
        self.assertEqual(summary["recall@5"], 1.0)
        self.assertEqual(summary["latency"]["p95_ms"], 84.2)
        self.assertFalse(summary["production_shaped"])

    def test_custom_variant_set_has_deterministic_policy_fallback(self):
        report = sweep_rrf({
            "per_query": [{
                "id": "custom", "route_statuses": {"p2_vector": "available"},
                "hybrid_shadow": {
                    "project_id": "default",
                    "routes": {"vector": {"status": "available", "refs": ["gold.md"]}},
                },
            }],
        }, [{
            "id": "custom", "query": "meaning", "expected_refs": ["gold.md"],
            "query_type": "unknown",
        }], variants={"vector_only": {"lexical": 0.0, "vector": 1.0}})
        policy = report["cases"][0]["policy"]
        self.assertEqual(policy["requested_variant"], "equal")
        self.assertEqual(policy["variant"], "vector_only")
        self.assertFalse(policy["configured"])
        self.assertTrue(policy["fallback"])
        self.assertTrue(policy["available"])
        self.assertEqual(policy["recall@5"], 1.0)

    def test_answer_gate_evidence_is_kept_without_overwriting_summary(self):
        report = sweep_rrf({
            "per_query": [{
                "id": "q", "route_statuses": {"p2_lexical": "available"},
                "hybrid_shadow": {
                    "project_id": "default",
                    "routes": {"lexical": {"status": "available", "refs": ["gold.md"]}},
                },
            }],
        }, [{"id": "q", "query": "q", "expected_refs": ["gold.md"]}],
            answer_gate_report={
                "schema": "answer-gate-test-v1", "generated_at": "2026-09-16T00:00:00Z",
                "mode": "mixed", "cases": [{"id": "q"}],
                "summary": {"answer_gate": {"status": "available", "false_answer_rate": 0.0}},
            })
        gate = report["answer_gate"]
        self.assertEqual(gate["status"], "available")
        self.assertEqual(gate["false_answer_rate"], 0.0)
        self.assertEqual(gate["evidence"]["schema"], "answer-gate-test-v1")
        self.assertEqual(gate["evidence"]["cases"], 1)
        self.assertEqual(len(gate["evidence"]["report_sha256"]), 64)

    def test_scope_non_default_filter_is_unknown_when_legacy_shadow_omits_it(self):
        report = sweep_rrf({
            "per_query": [{
                "id": "scoped", "route_statuses": {"p2_lexical": "available"},
                "hybrid_shadow": {
                    "project_id": "default", "session_id": None,
                    "routes": {"lexical": {"status": "available", "refs": ["gold.md"]}},
                },
            }],
        }, [{
            "id": "scoped", "query": "q", "expected_refs": ["gold.md"],
            "scope": {"project_id": "default", "session_id": "s-1", "statuses": [], "include_archive": False},
        }])
        self.assertEqual(report["cases"][0]["policy"]["scope_status"], "unknown")


if __name__ == "__main__":
    unittest.main()
