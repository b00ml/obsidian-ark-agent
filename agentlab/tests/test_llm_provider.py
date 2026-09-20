"""LLM provider lifecycle and connection-pool regressions."""
import asyncio
import unittest
from unittest.mock import patch

from agentlab.core.errors import AgentError
from agentlab.core.llm import OpenAICompatProvider
from agentlab.core.message import Message


class _Response:
    status_code = 200
    text = ""

    def json(self):
        return {
            "choices": [{
                "message": {"content": "ok", "tool_calls": []},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }


class _AsyncClient:
    def __init__(self):
        self.posts = 0
        self.closed = 0

    async def post(self, *args, **kwargs):
        self.posts += 1
        return _Response()

    async def aclose(self):
        self.closed += 1


class _MalformedResponse(_Response):
    def json(self):
        return {"choices": []}


class TestOpenAICompatProvider(unittest.TestCase):
    def test_owned_client_is_reused_and_closed_explicitly(self):
        client = _AsyncClient()
        with patch("httpx.AsyncClient", return_value=client) as factory:
            provider = OpenAICompatProvider("http://localhost/v1", "key", "model")

            async def run():
                await provider.chat([Message(role="user", content="a")])
                await provider.chat([Message(role="user", content="b")])
                await provider.aclose()
                await provider.aclose()

            asyncio.run(run())

        factory.assert_called_once_with(timeout=60.0)
        self.assertEqual(client.posts, 2)
        self.assertEqual(client.closed, 1)

    def test_injected_client_remains_caller_owned(self):
        client = _AsyncClient()
        provider = OpenAICompatProvider("http://localhost/v1", "key", "model", httpx_client=client)
        asyncio.run(provider.chat([Message(role="user", content="a")]))
        asyncio.run(provider.aclose())
        self.assertEqual(client.posts, 1)
        self.assertEqual(client.closed, 0)

    def test_malformed_response_is_a_non_retryable_schema_error(self):
        client = _AsyncClient()
        client.post = lambda *args, **kwargs: None

        async def post(*args, **kwargs):
            return _MalformedResponse()

        client.post = post
        provider = OpenAICompatProvider("http://localhost/v1", "key", "model",
                                        httpx_client=client)
        with self.assertRaisesRegex(AgentError, "AGENT_LLM_SCHEMA"):
            asyncio.run(provider.chat([Message(role="user", content="a")]))


if __name__ == "__main__":
    unittest.main()
