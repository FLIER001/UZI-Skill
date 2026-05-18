"""lib.llm_panel.config · 读取 UZI_LLM_* 环境变量。

纯函数、无网络、无副作用。UZI_LLM_API_KEY 或 UZI_LLM_MODEL 缺失 →
load_config() 返回 None（功能整体关闭，CLI 行为与今天完全一致）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float
    timeout: int
    max_wall_seconds: int


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_config() -> LLMConfig | None:
    """Return LLMConfig when UZI_LLM_API_KEY + UZI_LLM_MODEL set and
    UZI_NO_LLM != '1'; otherwise None (feature off)."""
    if os.environ.get("UZI_NO_LLM") == "1":
        return None
    api_key = (os.environ.get("UZI_LLM_API_KEY") or "").strip()
    model = (os.environ.get("UZI_LLM_MODEL") or "").strip()
    if not api_key or not model:
        return None
    base_url = (os.environ.get("UZI_LLM_BASE_URL")
                or "https://api.openai.com/v1").strip().rstrip("/")
    return LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=_get_float("UZI_LLM_TEMPERATURE", 0.4),
        timeout=_get_int("UZI_LLM_TIMEOUT", 90),
        max_wall_seconds=_get_int("UZI_LLM_MAX_WALL_SECONDS", 300),
    )
