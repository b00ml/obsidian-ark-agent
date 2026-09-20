"""OpenAIEmbedder（OPT-264 连接复用/超时治理）单元测试。

用本机 ThreadingHTTPServer 充当 OpenAI 兼容 /embeddings 端点，验证：
连接复用（同一主机第二次请求不再新建 TCP 连接）、网络级瞬时错误有界重试
一次、HTTP 错误与结构错误不重试并抛 RuntimeError、非法 base_url 拒绝构造。
全程无外网请求。
"""
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agentlab.rag.embed import OpenAIEmbedder


class CountingHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    connections = 0
    requests = 0
    behavior = "ok"  # ok | drop_first | http500 | bad_json | short_vec

    def __init__(self, *args, **kwargs):
        type(self).connections += 1
        super().__init__(*args, **kwargs)

    def do_POST(self):
        type(self).requests += 1
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        if type(self).behavior == "drop_first" and type(self).requests == 1:
            self.wfile.flush()
            self.connection.close()
            return
        if type(self).behavior == "http500":
            body = b'{"error":"upstream busy"}'
            self.send_response(500)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        try:
            n = len(json.loads(body.decode("utf-8") or "{}").get("input") or [])
        except ValueError:
            n = 1
        if type(self).behavior == "short_vec":
            n = max(0, n - 1)
        data = [{"embedding": [1.0, 0.0, 0.0] if i % 2 == 0 else [0.0, 1.0, 0.0]}
                for i in range(n)]
        body = json.dumps({"data": data}).encode("utf-8")
        if type(self).behavior == "bad_json":
            body = b"this is not json"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class TestOpenAIEmbedder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), CountingHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        CountingHandler.connections = 0
        CountingHandler.requests = 0
        CountingHandler.behavior = "ok"

    def _client(self, **kwargs):
        client = OpenAIEmbedder(self.base_url, "fake-v3", timeout=5.0, **kwargs)
        self.addCleanup(client._drop_connection)
        return client

    def test_reuses_single_connection_across_requests(self):
        client = self._client(batch_size=2)
        first = client.embed(["alpha", "beta"])
        second = client.embed(["gamma", "delta"])
        self.assertEqual(first, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertEqual(second, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertEqual(CountingHandler.requests, 2)
        self.assertEqual(CountingHandler.connections, 1, "keep-alive 应复用连接")

    def test_batch_splits_into_multiple_requests(self):
        client = self._client(batch_size=2)
        vecs = client.embed(["a", "b", "c", "d", "e"])
        self.assertEqual(len(vecs), 5)
        self.assertEqual(CountingHandler.requests, 3)

    def test_retries_once_on_transient_connection_drop(self):
        CountingHandler.behavior = "drop_first"
        client = self._client(batch_size=2, max_retries=1)
        vecs = client.embed(["a", "b"])
        self.assertEqual(len(vecs), 2, "断连后应重试一次并成功")
        self.assertEqual(CountingHandler.requests, 2)
        self.assertEqual(CountingHandler.connections, 2)

    def test_no_retry_on_http_error(self):
        CountingHandler.behavior = "http500"
        client = self._client(max_retries=2)
        with self.assertRaisesRegex(RuntimeError, "HTTP 500"):
            client.embed(["a"])
        self.assertEqual(CountingHandler.requests, 1, "HTTP 错误不触发重试")

    def test_no_retry_on_malformed_json(self):
        CountingHandler.behavior = "bad_json"
        client = self._client(max_retries=2)
        with self.assertRaisesRegex(RuntimeError, "非 JSON"):
            client.embed(["a"])
        self.assertEqual(CountingHandler.requests, 1)

    def test_vector_count_mismatch_raises(self):
        CountingHandler.behavior = "short_vec"
        client = self._client()
        with self.assertRaisesRegex(RuntimeError, "期望 2 条向量"):
            client.embed(["a", "b"])

    def test_invalid_base_url_rejected(self):
        with self.assertRaises(ValueError):
            OpenAIEmbedder("not-a-url", "fake-v3")

    def test_empty_batch_returns_without_network(self):
        client = self._client()
        self.assertEqual(client.embed([]), [])
        self.assertEqual(CountingHandler.requests, 0)


if __name__ == "__main__":
    unittest.main()