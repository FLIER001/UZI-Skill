"""lib.llm_panel.config · env 解析单测。"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))


def _clear(monkeypatch):
    for k in ("UZI_LLM_API_KEY", "UZI_LLM_MODEL", "UZI_LLM_BASE_URL",
              "UZI_LLM_TEMPERATURE", "UZI_LLM_TIMEOUT",
              "UZI_LLM_MAX_WALL_SECONDS", "UZI_NO_LLM"):
        monkeypatch.delenv(k, raising=False)


def test_no_key_returns_none(monkeypatch):
    _clear(monkeypatch)
    from lib.llm_panel.config import load_config
    assert load_config() is None


def test_key_without_model_returns_none(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("UZI_LLM_API_KEY", "sk-abc")
    from lib.llm_panel.config import load_config
    assert load_config() is None


def test_full_config(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("UZI_LLM_API_KEY", "sk-abc")
    monkeypatch.setenv("UZI_LLM_MODEL", "gpt-5.5")
    from lib.llm_panel.config import load_config
    cfg = load_config()
    assert cfg is not None
    assert cfg.api_key == "sk-abc"
    assert cfg.model == "gpt-5.5"
    assert cfg.base_url == "https://api.openai.com/v1"  # default
    assert cfg.temperature == 0.4                       # default
    assert cfg.timeout == 90                            # default
    assert cfg.max_wall_seconds == 300                  # default


def test_overrides_and_trailing_slash(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("UZI_LLM_API_KEY", "k")
    monkeypatch.setenv("UZI_LLM_MODEL", "deepseek-chat")
    monkeypatch.setenv("UZI_LLM_BASE_URL", "https://api.deepseek.com/v1/")
    monkeypatch.setenv("UZI_LLM_TEMPERATURE", "0.7")
    monkeypatch.setenv("UZI_LLM_TIMEOUT", "120")
    from lib.llm_panel.config import load_config
    cfg = load_config()
    assert cfg.base_url == "https://api.deepseek.com/v1"  # trailing / stripped
    assert cfg.temperature == 0.7
    assert cfg.timeout == 120


def test_kill_switch(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("UZI_LLM_API_KEY", "k")
    monkeypatch.setenv("UZI_LLM_MODEL", "gpt-5.5")
    monkeypatch.setenv("UZI_NO_LLM", "1")
    from lib.llm_panel.config import load_config
    assert load_config() is None


def test_bad_numeric_falls_back_to_default(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("UZI_LLM_API_KEY", "k")
    monkeypatch.setenv("UZI_LLM_MODEL", "m")
    monkeypatch.setenv("UZI_LLM_TEMPERATURE", "not-a-number")
    from lib.llm_panel.config import load_config
    assert load_config().temperature == 0.4
