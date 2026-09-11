import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agentlab.runtime.state_store import TaskStore


class TestTaskStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = TaskStore(Path(self.tmp.name) / "state.db", default_lease_seconds=1)

    def tearDown(self):
        self.tmp.cleanup()

    def test_dedupe_is_idempotent(self):
        a = self.store.create_task("bili", {"bvid": "BV1"}, dedupe_key="bili:BV1")
        b = self.store.create_task("bili", {"bvid": "BV1", "changed": True}, dedupe_key="bili:BV1")
        self.assertEqual(a["id"], b["id"])
        self.assertEqual(b["payload"], {"bvid": "BV1"})

    def test_claim_complete_survives_new_store_instance(self):
        task = self.store.create_task("demo", {"x": 1}, dedupe_key="demo:1")
        claimed = self.store.claim(task["id"])
        self.assertEqual(claimed["status"], "running")
        self.store.complete(task["id"], {"ok": True})
        reopened = TaskStore(self.store.path)
        got = reopened.get(task["id"])
        self.assertEqual(got["status"], "succeeded")
        self.assertEqual(got["result"], {"ok": True})

    def test_retry_then_dead_letter(self):
        task = self.store.create_task("demo", {}, dedupe_key="demo:retry", max_attempts=2)
        self.store.claim(task["id"])
        first = self.store.fail(task["id"], "TEMP", "first", retryable=True)
        self.assertEqual(first["status"], "pending")
        self.store.claim(task["id"])
        last = self.store.fail(task["id"], "TEMP", "second", retryable=True)
        self.assertEqual(last["status"], "dead_letter")
        self.assertEqual(last["error_code"], "TEMP")

    def test_retry_delay_blocks_claim_until_due(self):
        task = self.store.create_task("demo", {}, dedupe_key="demo:delay", max_attempts=2)
        self.store.claim(task["id"])
        pending = self.store.fail(task["id"], "TEMP", "wait", retryable=True,
                                  retry_delay=0.05)
        self.assertEqual(pending["status"], "pending")
        self.assertIsNone(self.store.claim(task["id"]))
        time.sleep(0.07)
        self.assertEqual(self.store.claim(task["id"])["attempt"], 2)

    def test_concurrent_claim_has_one_winner(self):
        task = self.store.create_task("demo", {}, dedupe_key="demo:race")
        barrier = threading.Barrier(2)

        def claim():
            barrier.wait()
            return self.store.claim(task["id"])

        with ThreadPoolExecutor(max_workers=2) as pool:
            claims = list(pool.map(lambda _: claim(), range(2)))
        self.assertEqual(sum(item is not None for item in claims), 1)
        self.assertEqual(len(self.store.events(task["id"])), 1)

    def test_non_retryable_failure_is_failed(self):
        task = self.store.create_task("demo", {}, dedupe_key="demo:failed")
        self.store.claim(task["id"])
        failed = self.store.fail(task["id"], "BAD_INPUT", "no", retryable=False)
        self.assertEqual(failed["status"], "failed")

    def test_cancel_is_idempotent_and_completion_is_discarded(self):
        task = self.store.create_task("demo", {}, dedupe_key="demo:cancel")
        self.store.claim(task["id"])
        self.assertEqual(self.store.cancel(task["id"])["status"], "cancelled")
        self.assertEqual(self.store.cancel(task["id"])["status"], "cancelled")
        self.assertEqual(self.store.complete(task["id"], {"late": True})["status"], "cancelled")

    def test_expired_lease_returns_to_pending(self):
        task = self.store.create_task("demo", {}, dedupe_key="demo:lease", max_attempts=2)
        self.store.claim(task["id"], lease_seconds=0.01)
        time.sleep(0.03)
        self.assertEqual(self.store.recover_expired(), 1)
        self.assertEqual(self.store.get(task["id"])["status"], "pending")

    def test_expired_lease_dead_letters_after_max_attempts(self):
        task = self.store.create_task("demo", {}, dedupe_key="demo:lease-dead", max_attempts=1)
        self.store.claim(task["id"], lease_seconds=0.01)
        time.sleep(0.03)
        self.assertEqual(self.store.get(task["id"])["status"], "dead_letter")
        self.assertEqual(self.store.get(task["id"])["error_code"], "LEASE_EXPIRED")

    def test_run_lifecycle(self):
        task = self.store.create_task("demo", {}, dedupe_key="demo:run")
        run = self.store.create_run(task["id"], session_id="s1", project_id="p1")
        self.assertEqual(run["status"], "running")
        done = self.store.finish_run(run["id"], "succeeded")
        self.assertEqual(done["status"], "succeeded")


if __name__ == "__main__":
    unittest.main()
