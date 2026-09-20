import json
import unittest

from agentlab.runtime.operation_verifier import verify_operation
from agentlab.runtime.remote_operations import query_remote_operation


class _Response:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def read(self, _limit):
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


class TestRemoteOperations(unittest.TestCase):
    def test_terminal_status_is_normalised_and_operation_id_is_escaped(self):
        seen = {}

        def opener(request, timeout):
            seen["url"] = request.full_url
            seen["timeout"] = timeout
            return _Response({"state": "succeeded", "result_id": "remote-result"})

        result = query_remote_operation(
            "remote/op 1",
            {"base_url": "https://ops.example.test", "status_path": "/v1/{operation_id}"},
            opener=opener,
        )
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["result_ref"], "remote-result")
        self.assertIn("remote%2Fop%201", seen["url"])

    def test_non_terminal_or_bad_payload_fails_closed(self):
        self.assertIsNone(query_remote_operation(
            "op-1", {"base_url": "https://ops.example.test"},
            opener=lambda *_args, **_kwargs: _Response({"status": "running"}),
        ))
        self.assertIsNone(query_remote_operation(
            "op-1", {"base_url": "https://ops.example.test"},
            opener=lambda *_args, **_kwargs: _Response(["succeeded"]),
        ))

    def test_verifier_uses_remote_adapter_without_vault_root(self):
        operation = {"verification": {
            "kind": "remote_operation", "remote_operation_id": "op-1"
        }}
        self.assertIsNone(verify_operation(operation, remote_config={}))


if __name__ == "__main__":
    unittest.main()
