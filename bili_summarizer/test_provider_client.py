import unittest
import requests

from provider_client import ProviderError, TextModelClient


class _Response:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class TestProviderClient(unittest.TestCase):
    def test_success_and_trace_are_structured(self):
        traces = []
        calls = []

        def transport(url, **kwargs):
            calls.append((url, kwargs))
            return _Response(payload={"choices": [{"message": {"content": "ok"}}],
                                     "usage": {"total_tokens": 3}})

        result = TextModelClient(
            api_base="https://example.test/v1/", api_key="key", model="m",
            transport=transport, trace=traces.append,
        ).chat("hello", max_tokens=9)
        self.assertEqual(result.content, "ok")
        self.assertEqual(calls[0][0], "https://example.test/v1/chat/completions")
        self.assertEqual(traces[0]["trace_id"], result.trace_id)
        self.assertEqual(result.usage["total_tokens"], 3)

    def test_http_and_malformed_errors_are_classified(self):
        def rate(*args, **kwargs):
            return _Response(429, payload={"error": "rate"})

        with self.assertRaises(ProviderError) as ctx:
            TextModelClient(
                api_base="x", api_key="k", model="m", transport=rate
            ).chat("x", max_tokens=1)
        self.assertEqual(ctx.exception.code, "RATE_LIMIT")
        self.assertTrue(ctx.exception.retryable)

        def malformed(*args, **kwargs):
            return _Response(payload={"choices": []})

        with self.assertRaises(ProviderError) as ctx:
            TextModelClient(
                api_base="x", api_key="k", model="m", transport=malformed
            ).chat("x", max_tokens=1)
        self.assertEqual(ctx.exception.code, "MALFORMED_RESPONSE")
        self.assertFalse(ctx.exception.retryable)

    def test_auth_error_is_not_retryable(self):
        with self.assertRaises(ProviderError) as ctx:
            TextModelClient(api_base="x", api_key="", model="m")
        self.assertEqual(ctx.exception.code, "AUTH_MISSING")
        self.assertFalse(ctx.exception.retryable)

    def test_requests_timeout_is_classified(self):
        def timeout(*args, **kwargs):
            raise requests.exceptions.Timeout("slow")

        with self.assertRaises(ProviderError) as ctx:
            TextModelClient(
                api_base="x", api_key="k", model="m", transport=timeout
            ).chat("x", max_tokens=1)
        self.assertEqual(ctx.exception.code, "TIMEOUT")
        self.assertTrue(ctx.exception.retryable)

    def test_http_error_from_raise_for_status_keeps_status_semantics(self):
        class ResponseWithoutStatus:
            text = "bad key"

            def raise_for_status(self):
                error = requests.exceptions.HTTPError("401")
                error.response = _Response(401, text="bad key")
                raise error

        with self.assertRaises(ProviderError) as ctx:
            TextModelClient(
                api_base="x", api_key="k", model="m", transport=lambda *a, **k: ResponseWithoutStatus()
            ).chat("x", max_tokens=1)
        self.assertEqual(ctx.exception.code, "HTTP_4XX")
        self.assertFalse(ctx.exception.retryable)


if __name__ == "__main__":
    unittest.main()
