import assert from "node:assert/strict";
import { test } from "node:test";

import { memoryLifecycleActionArgs, memoryLifecycleListArgs, memoryReviewApplyArgs, memoryReviewListArgs, parseMemoryLifecycleReport, parseMemoryReviewReport } from "../../src/memory-review";

test("memory review list uses an explicit vault argument", () => {
  assert.deepEqual(memoryReviewListArgs("E:/vault with spaces"), [
    "-m", "agentlab.eval.memory_review_due", "--vault", "E:/vault with spaces",
  ]);
});

test("memory lifecycle list is read-only and parses governed rows", () => {
  assert.deepEqual(memoryLifecycleListArgs("E:/vault"), ["-m", "agentlab.eval.memory_lifecycle", "--vault", "E:/vault"]);
  assert.deepEqual(parseMemoryLifecycleReport(JSON.stringify({ items: [{
    id: "mem-1", status: "candidate", content_hash: "a".repeat(64),
  }] })), [{ id: "mem-1", status: "candidate", content_hash: "a".repeat(64) }]);
});

test("memory review apply preserves the hash-bound decision contract", () => {
  const args = memoryReviewApplyArgs("E:/vault", {
    id: "mem-123", content_hash: "a".repeat(64),
  }, "alice", "checked", "defer", "2026-10-01T00:00:00+00:00");
  assert.ok(args.includes("--apply"));
  assert.ok(args.includes("--expected-content-hash"));
  assert.ok(args.includes("a".repeat(64)));
  assert.ok(args.includes("--defer-until"));
});

test("memory review parser rejects malformed or failed reports", () => {
  assert.deepEqual(parseMemoryReviewReport("not json"), []);
  assert.deepEqual(parseMemoryReviewReport(JSON.stringify({ passed: false, error: "bad" })), []);
  assert.deepEqual(parseMemoryReviewReport(JSON.stringify({
    items: [{ id: "mem-1", content_hash: "a".repeat(64) }],
  })), [{ id: "mem-1", content_hash: "a".repeat(64) }]);
});

test("lifecycle actions are explicit and preserve reviewed hash and scope", () => {
  const args = memoryLifecycleActionArgs("E:/vault", {
    id: "mem-1", content_hash: "b".repeat(64), project_id: "project-a", session_id: "session-a",
  }, "alice", "confirmed", "promote");
  assert.deepEqual(args, [
    "-m", "agentlab.eval.memory_lifecycle_action", "--vault", "E:/vault", "--apply",
    "--memory-id", "mem-1", "--action", "promote", "--expected-content-hash", "b".repeat(64),
    "--reviewer", "alice", "--reason", "confirmed", "--project-id", "project-a",
    "--session-id", "session-a",
  ]);
});
