import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from llm_control_plane.providers import GrokProvider
from llm_control_plane.types import ChatMessage, CompletionRequest, ProviderError


class FakeChatHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        assert payload["model"] == "grok-2-latest"
        body = {
            "model": "grok-2-latest",
            "choices": [{"message": {"role": "assistant", "content": "hello from grok"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        }
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args):
        return


def test_grok_provider_uses_openai_compatible_chat_api():
    server = HTTPServer(("127.0.0.1", 0), FakeChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        provider = GrokProvider(
            api_key="test-key",
            default_model="grok-2-latest",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
        )
        response = provider.complete(
            CompletionRequest(
                tenant_id="tenant-a",
                model="grok-2-latest",
                messages=[ChatMessage(role="user", content="hi")],
            )
        )
        assert response.provider == "grok"
        assert response.text == "hello from grok"
        assert response.usage.total_tokens == 7
    finally:
        server.shutdown()


def _server_with_status(status_code):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            self.send_response(status_code)
            self.end_headers()

        def log_message(self, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_grok_provider_5xx_is_retryable():
    server = _server_with_status(503)
    try:
        provider = GrokProvider(
            api_key="test-key",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
        )
        with pytest.raises(ProviderError) as exc:
            provider.complete(
                CompletionRequest(
                    tenant_id="tenant-a",
                    model="grok-2-latest",
                    messages=[ChatMessage(role="user", content="hi")],
                )
            )
        assert exc.value.retryable is True
        assert exc.value.status_code == 503
    finally:
        server.shutdown()


def test_grok_provider_4xx_is_not_retryable():
    server = _server_with_status(400)
    try:
        provider = GrokProvider(
            api_key="test-key",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
        )
        with pytest.raises(ProviderError) as exc:
            provider.complete(
                CompletionRequest(
                    tenant_id="tenant-a",
                    model="grok-2-latest",
                    messages=[ChatMessage(role="user", content="hi")],
                )
            )
        assert exc.value.retryable is False
    finally:
        server.shutdown()


def test_grok_provider_connection_error_is_retryable():
    provider = GrokProvider(api_key="test-key", base_url="http://127.0.0.1:1/v1")
    with pytest.raises(ProviderError) as exc:
        provider.complete(
            CompletionRequest(
                tenant_id="tenant-a",
                model="grok-2-latest",
                messages=[ChatMessage(role="user", content="hi")],
            )
        )
    assert exc.value.retryable is True
