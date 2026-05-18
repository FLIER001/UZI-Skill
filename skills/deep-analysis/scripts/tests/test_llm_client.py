"""lib.llm_panel.client · HTTP 客户端单测（fake requests，无网络）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))


class _FakeResp:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


class _FakeRequests:
    """记录调用次数 · 按 scripted 序列返回。"""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self._responses.pop(0)


def _cfg():
    from lib.llm_panel.config import LLMConfig
    return LLMConfig(api_key="k", base_url="https://x/v1", model="m",
                     temperature=0.4, timeout=10, max_wall_seconds=300)


def _ok_resp(content: str):
    return _FakeResp(200, {"choices": [{"message": {"content": content}}]})


def test_happy_path_parses_json(monkeypatch):
    from lib.llm_panel import client as mod
    fake = _FakeRequests([_ok_resp('{"votes": [], "ok": true}')])
    monkeypatch.setattr(mod, "requests", fake)
    c = mod.LLMClient(_cfg())
    out = c.chat_json("sys", "usr")
    assert out == {"votes": [], "ok": True}
    assert fake.calls[0]["url"] == "https://x/v1/chat/completions"
    assert fake.calls[0]["headers"]["Authorization"] == "Bearer k"
    assert fake.calls[0]["json"]["response_format"] == {"type": "json_object"}


def test_strips_code_fence(monkeypatch):
    from lib.llm_panel import client as mod
    fake = _FakeRequests([_ok_resp('```json\n{"a": 1}\n```')])
    monkeypatch.setattr(mod, "requests", fake)
    assert mod.LLMClient(_cfg()).chat_json("s", "u") == {"a": 1}


def test_retries_then_succeeds(monkeypatch):
    from lib.llm_panel import client as mod
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
    fake = _FakeRequests([_FakeResp(500, text="boom"), _ok_resp('{"a": 2}')])
    monkeypatch.setattr(mod, "requests", fake)
    assert mod.LLMClient(_cfg()).chat_json("s", "u") == {"a": 2}
    assert len(fake.calls) == 2


def test_auth_error_no_retry(monkeypatch):
    from lib.llm_panel import client as mod
    fake = _FakeRequests([_FakeResp(401, text="bad key")])
    monkeypatch.setattr(mod, "requests", fake)
    with pytest.raises(mod.LLMError):
        mod.LLMClient(_cfg()).chat_json("s", "u")
    assert len(fake.calls) == 1  # 401 不重试


def test_non_json_raises(monkeypatch):
    from lib.llm_panel import client as mod
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
    fake = _FakeRequests([_ok_resp("totally not json"),
                          _ok_resp("still not json"),
                          _ok_resp("nope")])
    monkeypatch.setattr(mod, "requests", fake)
    with pytest.raises(mod.LLMError):
        mod.LLMClient(_cfg()).chat_json("s", "u")
