"""lib.llm_panel.client · 极薄 OpenAI 兼容 chat/completions 客户端。

零新依赖（用 requests，与 lib/mx_api.py 一致）。只负责 HTTP + 重试 +
取出 message.content 并 json.loads。业务逻辑全在 runner.py。
"""
from __future__ import annotations

import json
import time

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

from .config import LLMConfig


class LLMError(RuntimeError):
    """LLM 调用在重试后仍失败，或返回无法解析的 JSON。"""


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.url = f"{cfg.base_url}/chat/completions"

    def chat_json(self, system: str, user: str, attempts: int = 3) -> dict:
        """JSON mode 调用 chat/completions，返回解析后的 dict。

        网络失败（重试后）或非 JSON 内容 → 抛 LLMError。
        """
        if requests is None:
            raise LLMError("requests library missing")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.api_key}",
        }
        body = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.cfg.temperature,
            "response_format": {"type": "json_object"},
        }
        last_err = None
        drop_json_mode = False
        for i in range(attempts):
            payload = dict(body)
            if drop_json_mode:
                payload.pop("response_format", None)
            try:
                r = requests.post(self.url, headers=headers,
                                  json=payload, timeout=self.cfg.timeout)
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    if r.status_code in (401, 403):
                        break  # auth — 不重试
                    # 某些 OpenAI 兼容端点不支持 response_format → 去掉重试
                    if r.status_code == 400 and not drop_json_mode:
                        drop_json_mode = True
                        continue
                    time.sleep(1.0 * (i + 1))
                    continue
                content = r.json()["choices"][0]["message"]["content"]
                return _parse_json_content(content)
            except LLMError:
                raise
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:200]}"
                time.sleep(1.0 * (i + 1))
        raise LLMError(last_err or "unknown LLM error")


def _parse_json_content(content: str) -> dict:
    """容忍 ```json 围栏 / 前置散文，取出 JSON object。"""
    text = (content or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                raise LLMError(f"non-JSON content: {content[:200]}")
        else:
            raise LLMError(f"non-JSON content: {content[:200]}")
    if not isinstance(obj, dict):
        raise LLMError(f"expected JSON object, got {type(obj).__name__}")
    return obj
