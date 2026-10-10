"""FastAPI middleware integration tests."""

import json

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from asha.integrations.fastapi.middleware import ASHAMiddleware


def _make_app(**middleware_kwargs):
    app = FastAPI()
    app.add_middleware(ASHAMiddleware, **middleware_kwargs)

    @app.post("/chat/completions")
    async def chat(body: dict):
        return {"received": body}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def test_middleware_skips_non_configured_endpoints():
    app = _make_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_middleware_processes_chat_completions_json_body():
    app = _make_app(debug_mode=True)
    client = TestClient(app)
    body = {"messages": [{"role": "user", "content": "Email john@company.com please"}]}
    response = client.post("/chat/completions", json=body)
    assert response.status_code == 200
    assert response.headers.get("X-ASHA-Processed") == "true"


def test_middleware_fail_safe_on_process_error(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("process failed")

    import asha.integrations.fastapi.middleware as mw

    monkeypatch.setattr(mw, "process", boom)
    app = _make_app()
    client = TestClient(app)
    body = {"messages": [{"role": "user", "content": "keep@company.com"}]}
    response = client.post("/chat/completions", json=body)
    assert response.status_code == 200
    assert "keep@company.com" in response.json()["received"]["messages"][0]["content"]


def test_middleware_skips_empty_body():
    app = _make_app()
    client = TestClient(app)
    response = client.post("/chat/completions", content=b"")
    assert response.status_code in (200, 422)
