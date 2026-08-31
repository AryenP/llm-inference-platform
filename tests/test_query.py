import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app

SSE = (
    'data: {"choices":[{"text":"paged "}]}\n\n'
    'data: {"choices":[{"text":"attention"}]}\n\n'
    "data: [DONE]\n\n"
)


def mock_http(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://vllm/v1")


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_query_streams_and_times(client):
    def handler(request):
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, content=SSE.encode())

    client.app.state.http = mock_http(handler)
    r = client.post("/query", json={"prompt": "what is", "max_tokens": 8})

    assert r.status_code == 200
    got = r.json()
    assert got["text"] == "paged attention"
    assert got["n_chunks"] == 2
    # ttft is measured at the first chunk, so it can never exceed the full-response time
    assert 0 < got["ttft_ms"] <= got["total_ms"]


def test_query_surfaces_vllm_errors(client):
    client.app.state.http = mock_http(lambda _: httpx.Response(500, content=b"engine dead"))
    r = client.post("/query", json={"prompt": "x"})

    assert r.status_code == 502
    assert "engine dead" in r.json()["detail"]


def test_query_rejects_empty_completion(client):
    client.app.state.http = mock_http(lambda _: httpx.Response(200, content=b"data: [DONE]\n\n"))
    r = client.post("/query", json={"prompt": "x"})

    assert r.status_code == 502


def test_health_reports_loaded_models(client):
    body = {"data": [{"id": "/workspace/models/qwen3-8b"}]}
    client.app.state.http = mock_http(lambda _: httpx.Response(200, json=body))
    r = client.get("/health")

    assert r.status_code == 200
    assert r.json()["models"] == ["/workspace/models/qwen3-8b"]


def test_health_503_when_vllm_down(client):
    def handler(_):
        raise httpx.ConnectError("refused")

    client.app.state.http = mock_http(handler)
    assert client.get("/health").status_code == 503
