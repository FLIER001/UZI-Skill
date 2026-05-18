# CLI 内置 OpenAI 兼容模型 · 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `python run.py <ticker>` 在 `.env` 配置了 OpenAI 兼容模型后，自动 role-play 51 评委并生成合规 `agent_analysis.json`，无需 Claude/Codex 介入即可跑完整流程。

**Architecture:** 新增自包含包 `lib/llm_panel/`（config / client / prompts / runner / `__init__`）。唯一调用点是 `run_real_test.stage2()` 函数体第一行的 `maybe_run_llm_review(ticker)`——pipeline 与全部 legacy 路径都经此汇合。复用 `lib/personas.py` 的 prefix-stable system message 和 persona 加载。复用 `lib/agent_analysis_validator.py` 做 schema 校验。失败永不阻塞报告。

**Tech Stack:** Python 3.9+ · `requests`（零新依赖，镜像 `lib/mx_api.py`）· pytest · OpenAI 兼容 `/v1/chat/completions` JSON mode。

**Spec:** `docs/superpowers/specs/2026-05-18-llm-panel-cli-integration-design.md`

**工作目录约定:** 所有命令在 `skills/deep-analysis/scripts/` 下运行（与现有 pytest 一致）。除非特别说明，`pytest` / `python` 都在该目录执行。

---

### Task 1: 包骨架 + config.py（环境变量解析）

**Files:**
- Create: `skills/deep-analysis/scripts/lib/llm_panel/__init__.py`
- Create: `skills/deep-analysis/scripts/lib/llm_panel/config.py`
- Test: `skills/deep-analysis/scripts/tests/test_llm_config.py`

- [ ] **Step 1: 写失败测试**

创建 `skills/deep-analysis/scripts/tests/test_llm_config.py`：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_llm_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lib.llm_panel'`

- [ ] **Step 3: 写最小实现**

创建 `skills/deep-analysis/scripts/lib/llm_panel/__init__.py`（先留空，Task 6 再填 `maybe_run_llm_review`）：

```python
"""lib.llm_panel · CLI 内置 OpenAI 兼容模型 · 自动生成 agent_analysis.json。

唯一对外入口：maybe_run_llm_review（Task 6 实现并在此 re-export）。
"""
from __future__ import annotations
```

创建 `skills/deep-analysis/scripts/lib/llm_panel/config.py`：

```python
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


def load_config() -> "LLMConfig | None":
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_llm_config.py -v`
Expected: PASS（6 passed）

- [ ] **Step 5: 提交**

```bash
git add skills/deep-analysis/scripts/lib/llm_panel/__init__.py \
        skills/deep-analysis/scripts/lib/llm_panel/config.py \
        skills/deep-analysis/scripts/tests/test_llm_config.py
git commit -m "feat(llm_panel): config loader for UZI_LLM_* env vars"
```

---

### Task 2: client.py（OpenAI 兼容 HTTP 客户端）

**Files:**
- Create: `skills/deep-analysis/scripts/lib/llm_panel/client.py`
- Test: `skills/deep-analysis/scripts/tests/test_llm_client.py`

- [ ] **Step 1: 写失败测试**

创建 `skills/deep-analysis/scripts/tests/test_llm_client.py`：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_llm_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lib.llm_panel.client'`

- [ ] **Step 3: 写最小实现**

创建 `skills/deep-analysis/scripts/lib/llm_panel/client.py`：

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_llm_client.py -v`
Expected: PASS（5 passed）

- [ ] **Step 5: 提交**

```bash
git add skills/deep-analysis/scripts/lib/llm_panel/client.py \
        skills/deep-analysis/scripts/tests/test_llm_client.py
git commit -m "feat(llm_panel): OpenAI-compatible chat/completions client"
```

---

### Task 3: prompts.py（snapshot + 分组 + 综合 prompt）

**Files:**
- Create: `skills/deep-analysis/scripts/lib/llm_panel/prompts.py`
- Test: `skills/deep-analysis/scripts/tests/test_llm_prompts.py`

复用 `lib.personas.build_system_message`（已 prefix-stable）+ `lib.personas.load_persona().to_prompt_block()`。

- [ ] **Step 1: 写失败测试**

创建 `skills/deep-analysis/scripts/tests/test_llm_prompts.py`：

```python
"""lib.llm_panel.prompts · 单测。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))


_PANEL = {
    "ticker": "600519.SH",
    "panel_consensus": 4.8,
    "vote_distribution": {"strong_buy": 0, "buy": 1, "hold": 10, "avoid": 40},
    "signal_distribution": {"bullish": 2, "neutral": 19, "bearish": 26, "skip": 4},
    "school_scores": {
        "A": {"label": "经典价值派", "desc": "巴菲特 一脉", "n_members": 1},
    },
    "investors": [
        {"investor_id": "buffett", "name": "巴菲特", "group": "A",
         "headline": "看空核心：ROE 5 年最低 -23%",
         "score": 20, "signal": "bearish",
         "pass": [{"name": "PE 在 5 年中位数以下", "msg": "PE -16.96", "weight": 3}],
         "fail": [{"name": "ROE 连续 5 年 > 15%", "msg": "ROE 5 年最低 -23.0%", "weight": 5}]},
    ],
}
_DIMS = {"fundamental_score": 31}


def test_market_snapshot_is_compact_json_with_key_numbers():
    from lib.llm_panel.prompts import build_market_snapshot
    snap = build_market_snapshot("600519.SH", _DIMS, _PANEL)
    obj = json.loads(snap)  # 必须是合法 JSON
    assert obj["ticker"] == "600519.SH"
    assert obj["fundamental_score"] == 31
    assert obj["panel_consensus"] == 4.8
    assert obj["signal_distribution"]["bearish"] == 26


def test_group_prompt_includes_persona_and_skeleton():
    from lib.llm_panel.prompts import build_group_prompt
    p = build_group_prompt("A", "经典价值派",
                            _PANEL["investors"], _DIMS)
    # flagship persona 的 key_metrics 必须注入（buffett.yaml 含 "ROE 连续 10 年"）
    assert "ROE" in p
    assert "巴菲特" in p
    # 骨架分必须带进去
    assert "buffett" in p
    assert "ROE 5 年最低 -23.0%" in p          # 来自 skeleton fail.msg
    # 必须要求 JSON 数组 + dim_commentary
    assert "votes" in p and "dim_commentary" in p


def test_system_message_is_byte_stable_across_calls():
    from lib.llm_panel.prompts import build_market_snapshot, build_system
    snap = build_market_snapshot("600519.SH", _DIMS, _PANEL)
    a = build_system(snap)
    b = build_system(snap)
    assert a == b and len(a) > 50


def test_synthesis_prompt_has_required_output_keys():
    from lib.llm_panel.prompts import build_synthesis_prompt
    votes = [{"investor_id": "buffett", "signal": "bearish", "score": 20}]
    s = build_synthesis_prompt(votes, {"0_basic": "x"}, _DIMS, _PANEL)
    for key in ("panel_insights", "great_divide_override",
                "narrative_override", "buy_zones", "value", "youzi"):
        assert key in s
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_llm_prompts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lib.llm_panel.prompts'`

- [ ] **Step 3: 写最小实现**

创建 `skills/deep-analysis/scripts/lib/llm_panel/prompts.py`：

```python
"""lib.llm_panel.prompts · 构建 snapshot + 分组 + 综合 prompt。

复用 lib.personas：
- build_system(snapshot) → lib.personas.build_system_message（prefix-stable）
- 组内每个 persona 用 load_persona(id).to_prompt_block()
"""
from __future__ import annotations

import json
from typing import Any

from lib.personas import build_system_message, load_persona


def build_market_snapshot(ticker: str, dims: dict, panel: dict) -> str:
    """全体 persona 共享的紧凑 JSON 快照（放进 prefix-stable system）。"""
    snap = {
        "ticker": ticker,
        "fundamental_score": dims.get("fundamental_score"),
        "panel_consensus": panel.get("panel_consensus"),
        "vote_distribution": panel.get("vote_distribution"),
        "signal_distribution": panel.get("signal_distribution"),
        "school_scores": {
            g: {"label": v.get("label"), "verdict": v.get("verdict"),
                "avg_score": v.get("avg_score")}
            for g, v in (panel.get("school_scores") or {}).items()
        },
    }
    return json.dumps(snap, ensure_ascii=False, sort_keys=True, indent=2)


def build_system(snapshot_json: str) -> str:
    """复用 personas 的 prefix-stable system message（prompt cache 优化）。"""
    return build_system_message(snapshot_json, lang="zh")


def _skeleton_block(inv: dict) -> str:
    """单个评委的规则引擎骨架分（带具体数字，喂给模型当 anchor）。"""
    def _rules(items):
        return "；".join(
            f"[权{it.get('weight', '?')}] {it.get('name', '')}：{it.get('msg', '')}"
            for it in (items or [])
        ) or "（无）"
    return (
        f"### {inv.get('name', '')}（{inv.get('investor_id', '')}）\n"
        f"- 规则骨架 signal={inv.get('signal')} score={inv.get('score')}\n"
        f"- headline 骨架：{inv.get('headline', '')}\n"
        f"- 命中规则：{_rules(inv.get('pass'))}\n"
        f"- 未达规则：{_rules(inv.get('fail'))}"
    )


def build_group_prompt(group_key: str, group_label: str,
                        investors: list, dims: dict) -> str:
    """一次调用 role-play 整组评委，要求返回 JSON。"""
    blocks = []
    for inv in investors:
        iid = inv.get("investor_id", "")
        persona = load_persona(iid)
        persona_txt = persona.to_prompt_block() if persona else f"(无 persona 档案：{iid})"
        blocks.append(persona_txt + "\n\n" + _skeleton_block(inv))
    joined = "\n\n========\n\n".join(blocks)
    ids = [inv.get("investor_id", "") for inv in investors]
    return f"""# 分组 role-play 任务 · Group {group_key} · {group_label}

下面是本组 {len(investors)} 位投资者的 persona 档案 + 规则引擎骨架分。
请逐位 in-character 给出判断。flagship persona（非 stub）的历史立场优先于
规则骨架；stub persona 以规则骨架为准、YAML 仅补语气。

每条 headline / reasoning 必须引用具体数字（PE/ROE/营收/市值等）和该 persona
的 key_metrics，禁止"基本面良好/值得关注/估值合理"这类空话。

{joined}

========

严格返回如下 JSON（不要任何解释文字、不要 markdown 围栏）：
{{
  "votes": [
    {{
      "investor_id": "{ids[0] if ids else 'xxx'}",
      "signal": "bullish|neutral|bearish|skip",
      "score": 0-100,
      "verdict": "强烈买入|买入|关注|观望|回避|不适合",
      "headline": "<80字强结论",
      "reasoning": "2-3段 in-voice，引用数据+key_metrics",
      "persona_used": "flagship|stub"
    }}
    // 本组每位投资者各一条，investor_id 必须是：{", ".join(ids)}
  ],
  "dim_commentary": {{
    "<dim_key如0_basic/1_financials/10_valuation>": "本组视角下该维度的定性评语（≥20字，引用数字）"
  }}
}}"""


def build_synthesis_prompt(all_votes: list, dim_commentary: dict,
                           dims: dict, panel: dict) -> str:
    """综合调用：产出 panel_insights / great_divide_override / narrative_override。"""
    tally: dict[str, int] = {}
    for v in all_votes:
        s = v.get("signal", "?")
        tally[s] = tally.get(s, 0) + 1
    votes_summary = json.dumps(
        [{"id": v.get("investor_id"), "signal": v.get("signal"),
          "score": v.get("score"), "headline": v.get("headline", "")[:60]}
         for v in all_votes], ensure_ascii=False)
    school = json.dumps(panel.get("school_scores") or {}, ensure_ascii=False)
    return f"""# 综合研判任务

这是 51 评委 role-play 后的全部投票（signal 计票：{tally}）：
{votes_summary}

各派系骨架共识：{school}
基本面分：{dims.get('fundamental_score')} · panel_consensus：{panel.get('panel_consensus')}

请综合给出最终定论。严格返回如下 JSON（无解释、无围栏）：
{{
  "panel_insights": "≥30字：投票分布 + 多空主要分歧分析",
  "great_divide_override": {{
    "punchline": "≥10字冲突金句，含具体数字",
    "bull_say_rounds": ["看多第1轮", "看多第2轮", "看多第3轮"],
    "bear_say_rounds": ["看空第1轮", "看空第2轮", "看空第3轮"]
  }},
  "narrative_override": {{
    "core_conclusion": "≥20字综合定论：名称+评分+关键证据",
    "risks": ["风险1", "风险2", "风险3"],
    "buy_zones": {{
      "value":     {{"price": <数值>, "rationale": "≥5字"}},
      "growth":    {{"price": <数值>, "rationale": "≥5字"}},
      "technical": {{"price": <数值>, "rationale": "≥5字"}},
      "youzi":     {{"price": <数值>, "rationale": "≥5字"}}
    }}
  }}
}}"""
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_llm_prompts.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add skills/deep-analysis/scripts/lib/llm_panel/prompts.py \
        skills/deep-analysis/scripts/tests/test_llm_prompts.py
git commit -m "feat(llm_panel): snapshot/group/synthesis prompt builders (reuse lib.personas)"
```

---

### Task 4: runner.py — 编排 happy path（写 agent_analysis.json + 回写 panel）

**Files:**
- Create: `skills/deep-analysis/scripts/lib/llm_panel/runner.py`
- Test: `skills/deep-analysis/scripts/tests/test_llm_runner.py`

- [ ] **Step 1: 写失败测试**

创建 `skills/deep-analysis/scripts/tests/test_llm_runner.py`：

```python
"""lib.llm_panel.runner · 编排单测（fake client，无网络）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

TICKER = "000001.SZ"  # 有效 A 股代码格式（parse_ticker 纯解析，不联网）


def _panel():
    return {
        "ticker": TICKER,
        "panel_consensus": 30.0,
        "vote_distribution": {"buy": 1, "hold": 1},
        "signal_distribution": {"bullish": 1, "neutral": 1, "bearish": 0, "skip": 0},
        "school_scores": {
            "A": {"label": "经典价值派", "verdict": "回避", "avg_score": 25},
            "F": {"label": "A股游资", "verdict": "回避", "avg_score": 0},
        },
        "investors": [
            {"investor_id": "buffett", "name": "巴菲特", "group": "A",
             "headline": "skel-A", "score": 20, "signal": "bearish",
             "pass": [], "fail": [{"name": "ROE", "msg": "ROE 低", "weight": 5}]},
            {"investor_id": "bj_cj", "name": "北京炒家", "group": "F",
             "headline": "skel-F", "score": 0, "signal": "bearish",
             "pass": [], "fail": []},
        ],
    }


def _seed_cache(tmp_path, monkeypatch):
    """把 .cache 指到 tmp 并写 raw/dims/panel。"""
    from lib import cache as cache_mod
    monkeypatch.setattr(cache_mod, "CACHE_ROOT", tmp_path / ".cache")
    cache_mod.write_task_output(TICKER, "raw_data", {"ticker": TICKER, "dimensions": {}})
    cache_mod.write_task_output(TICKER, "dimensions", {"fundamental_score": 31})
    cache_mod.write_task_output(TICKER, "panel", _panel())
    return cache_mod


class _FakeClient:
    """按 system+user 内容判断该返回 group 结果还是 synthesis 结果。"""
    def __init__(self):
        self.calls = 0

    def chat_json(self, system, user, attempts=3):
        self.calls += 1
        if "综合研判任务" in user:
            return {
                "panel_insights": "51 评委里看空为主，价值派因 ROE 过低集体回避，分歧集中在估值。",
                "great_divide_override": {
                    "punchline": "ROE 仅 5% 却要 25 倍 PE，多空在为故事定价",
                    "bull_say_rounds": ["重组预期", "PB 历史底", "讲通就翻倍"],
                    "bear_say_rounds": ["ROE 连降", "毛利天花板", "低质资产"],
                },
                "narrative_override": {
                    "core_conclusion": "测试标的 · 35 分 · 回避，ROE 过低无安全边际。",
                    "risks": ["ROE 持续下滑", "回款风险", "行业竞争"],
                    "buy_zones": {
                        "value": {"price": 3.8, "rationale": "PB 0.8x 底部"},
                        "growth": {"price": 4.1, "rationale": "重组博弈价"},
                        "technical": {"price": 4.2, "rationale": "MA120 支撑"},
                        "youzi": {"price": 4.5, "rationale": "板块联动切入"},
                    },
                },
            }
        # group call
        ids = [ln for ln in user.splitlines()]
        return {
            "votes": [{
                "investor_id": "buffett" if "巴菲特" in user else "bj_cj",
                "signal": "bearish", "score": 18, "verdict": "回避",
                "headline": "LLM 覆盖：ROE 5% 远低于 15% 红线，回避",
                "reasoning": "以巴菲特视角，ROE 连续低于 15%，无安全边际。" * 2,
                "persona_used": "flagship" if "巴菲特" in user else "stub",
            }],
            "dim_commentary": {
                "1_financials": "ROE 仅 5%，连续 3 年下滑，现金流波动大，回款承压明显。",
                "0_basic": "小市值基建股，主营低毛利，营收稳但利润极薄约 1.2%。",
            },
        }


def test_happy_path_writes_valid_agent_analysis(tmp_path, monkeypatch):
    cache_mod = _seed_cache(tmp_path, monkeypatch)
    from lib.llm_panel.config import LLMConfig
    from lib.llm_panel.runner import run_llm_review
    cfg = LLMConfig(api_key="k", base_url="b", model="gpt-5.5",
                    temperature=0.4, timeout=10, max_wall_seconds=300)
    ok = run_llm_review(TICKER, cfg=cfg, client=_FakeClient())
    assert ok is True
    aa = cache_mod.read_task_output(TICKER, "agent_analysis")
    assert aa["agent_reviewed"] is True
    assert aa["_llm_generated"] is True
    assert aa["_llm_model"] == "gpt-5.5"
    assert len(aa["dim_commentary"]) >= 2
    assert "panel_insights" in aa
    assert len(aa["narrative_override"]["risks"]) >= 3
    # schema 校验干净
    from lib.agent_analysis_validator import validate
    errs = [i for i in validate(aa) if i.severity == "error"]
    assert errs == []


def test_panel_investors_overwritten_from_votes(tmp_path, monkeypatch):
    cache_mod = _seed_cache(tmp_path, monkeypatch)
    from lib.llm_panel.config import LLMConfig
    from lib.llm_panel.runner import run_llm_review
    cfg = LLMConfig("k", "b", "m", 0.4, 10, 300)
    run_llm_review(TICKER, cfg=cfg, client=_FakeClient())
    panel = cache_mod.read_task_output(TICKER, "panel")
    buf = [i for i in panel["investors"] if i["investor_id"] == "buffett"][0]
    assert buf["headline"].startswith("LLM 覆盖")
    assert buf["score"] == 18
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_llm_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lib.llm_panel.runner'`

- [ ] **Step 3: 写最小实现**

创建 `skills/deep-analysis/scripts/lib/llm_panel/runner.py`：

```python
"""lib.llm_panel.runner · 编排：分组 role-play → 综合 → 组装 → 校验 → 写盘。

run_llm_review 是核心。所有失败在此吞掉，绝不向 stage2 抛出（Task 5 加固）。
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from lib.cache import read_task_output, write_task_output
from lib.market_router import parse_ticker

from .client import LLMClient, LLMError
from .config import LLMConfig
from .prompts import (build_group_prompt, build_market_snapshot,
                       build_synthesis_prompt, build_system)


def _group_investors(panel: dict) -> dict:
    groups: dict[str, list] = {}
    for inv in panel.get("investors") or []:
        g = inv.get("group") or "?"
        groups.setdefault(g, []).append(inv)
    return groups


def _run_one_group(client, system, gkey, glabel, investors, dims):
    user = build_group_prompt(gkey, glabel, investors, dims)
    return gkey, client.chat_json(system, user)


def run_llm_review(ticker: str, cfg: LLMConfig,
                   client=None, *, resume: bool = True) -> bool:
    """对 ticker 跑 LLM 评审，写 .cache/{ticker}/agent_analysis.json。

    返回 True 表示已写出（含降级写出）；False 表示完全跳过。
    """
    ti = parse_ticker(ticker)
    full = ti.full

    panel = read_task_output(full, "panel")
    dims = read_task_output(full, "dimensions")
    if not panel or not dims:
        print(f"   ⚠️ LLM 评审跳过：缺 panel/dimensions（{full}）")
        return False

    existing = read_task_output(full, "agent_analysis")
    if resume and existing and existing.get("agent_reviewed"):
        print(f"   ♻️  已有 agent_reviewed 的 agent_analysis.json · 跳过 LLM（--no-resume 强制重生成）")
        return True

    client = client or LLMClient(cfg)
    snapshot = build_market_snapshot(full, dims, panel)
    system = build_system(snapshot)
    groups = _group_investors(panel)
    school_scores = panel.get("school_scores") or {}

    t0 = time.time()
    all_votes: list = []
    dim_commentary: dict = {}

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {}
        for gkey, investors in groups.items():
            glabel = (school_scores.get(gkey) or {}).get("label", gkey)
            futs[ex.submit(_run_one_group, client, system,
                           gkey, glabel, investors, dims)] = gkey
        for fut in as_completed(futs):
            gkey = futs[fut]
            try:
                _, res = fut.result()
                for v in res.get("votes") or []:
                    all_votes.append(v)
                for k, txt in (res.get("dim_commentary") or {}).items():
                    if isinstance(txt, str) and len(txt.strip()) >= len(
                            str(dim_commentary.get(k, ""))):
                        dim_commentary[k] = txt
                print(f"   [Group {gkey}] {len((res.get('votes') or []))} 人 ✓")
            except LLMError as e:
                print(f"   [Group {gkey}] ✗ LLM 失败，保留规则骨架: {e}")

    agent_analysis = {
        "agent_reviewed": True,
        "_llm_generated": True,
        "_llm_model": cfg.model,
        "dim_commentary": dim_commentary,
    }

    # 综合调用（墙钟预算内）
    if time.time() - t0 < cfg.max_wall_seconds:
        try:
            syn_user = build_synthesis_prompt(all_votes, dim_commentary, dims, panel)
            syn = client.chat_json(system, syn_user)
            for k in ("panel_insights", "great_divide_override", "narrative_override"):
                if k in syn:
                    agent_analysis[k] = syn[k]
            print("   综合研判 ✓")
        except LLMError as e:
            print(f"   综合研判 ✗（仅写分组结果）: {e}")
    else:
        print("   ⏱ 超墙钟预算，跳过综合研判")

    _overwrite_panel(full, panel, all_votes)
    write_task_output(full, "agent_analysis", agent_analysis)
    print("✅ agent_analysis.json 已生成")
    return True


def _overwrite_panel(full: str, panel: dict, votes: list) -> None:
    """用 LLM votes 覆盖 panel.json 的 headline/reasoning/score/signal/verdict。"""
    by_id = {v.get("investor_id"): v for v in votes if v.get("investor_id")}
    changed = False
    for inv in panel.get("investors") or []:
        v = by_id.get(inv.get("investor_id"))
        if not v:
            continue
        for field, vkey in (("headline", "headline"), ("reasoning", "reasoning"),
                            ("score", "score"), ("signal", "signal"),
                            ("verdict", "verdict")):
            if v.get(vkey) not in (None, ""):
                inv[field] = v[vkey]
        changed = True
    if changed:
        write_task_output(full, "panel", panel)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_llm_runner.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add skills/deep-analysis/scripts/lib/llm_panel/runner.py \
        skills/deep-analysis/scripts/tests/test_llm_runner.py
git commit -m "feat(llm_panel): runner orchestration (grouped role-play + synthesis)"
```

---

### Task 5: runner.py — 校验自纠重试 + 优雅降级 + 幂等

**Files:**
- Modify: `skills/deep-analysis/scripts/lib/llm_panel/runner.py`
- Test: `skills/deep-analysis/scripts/tests/test_llm_runner.py`（追加用例）

- [ ] **Step 1: 追加失败测试**

在 `tests/test_llm_runner.py` 末尾追加：

```python
class _BadThenGoodClient:
    """第一次综合返回缺 risks 的坏结构，重试返回合规结构。"""
    def __init__(self):
        self.syn_calls = 0

    def chat_json(self, system, user, attempts=3):
        if "综合研判任务" in user or "schema 修复" in user:
            self.syn_calls += 1
            if self.syn_calls == 1:
                return {"narrative_override": {"risks": "应该是list但给了字符串"}}
            return {
                "panel_insights": "重试后合规：多空分歧集中在 ROE 与估值，整体偏空。",
                "great_divide_override": {
                    "punchline": "ROE 5% 撑不起 25 倍 PE",
                    "bull_say_rounds": ["a1", "a2", "a3"],
                    "bear_say_rounds": ["b1", "b2", "b3"]},
                "narrative_override": {
                    "core_conclusion": "测试标的 35 分回避，安全边际不足。",
                    "risks": ["r1", "r2", "r3"],
                    "buy_zones": {k: {"price": 1.0, "rationale": "xxxxx"}
                                  for k in ("value", "growth", "technical", "youzi")}},
            }
        return {"votes": [{"investor_id": "buffett", "signal": "bearish",
                           "score": 18, "verdict": "回避",
                           "headline": "h" * 25, "reasoning": "r" * 40,
                           "persona_used": "flagship"}],
                "dim_commentary": {"1_financials": "ROE 仅 5% 连续下滑回款承压明显啊啊。"}}


class _RaisingClient:
    def chat_json(self, system, user, attempts=3):
        from lib.llm_panel.client import LLMError
        raise LLMError("network down")


def test_validation_error_triggers_one_retry(tmp_path, monkeypatch):
    cache_mod = _seed_cache(tmp_path, monkeypatch)
    from lib.llm_panel.config import LLMConfig
    from lib.llm_panel.runner import run_llm_review
    fc = _BadThenGoodClient()
    run_llm_review(TICKER, cfg=LLMConfig("k", "b", "m", 0.4, 10, 300), client=fc)
    aa = cache_mod.read_task_output(TICKER, "agent_analysis")
    from lib.agent_analysis_validator import validate
    assert [i for i in validate(aa) if i.severity == "error"] == []
    assert fc.syn_calls == 2  # 恰好一次重试


def test_all_groups_fail_still_writes_but_not_reviewed(tmp_path, monkeypatch):
    cache_mod = _seed_cache(tmp_path, monkeypatch)
    from lib.llm_panel.config import LLMConfig
    from lib.llm_panel.runner import run_llm_review
    ok = run_llm_review(TICKER, cfg=LLMConfig("k", "b", "m", 0.4, 10, 300),
                        client=_RaisingClient())
    assert ok is True
    aa = cache_mod.read_task_output(TICKER, "agent_analysis")
    # 全失败 → 不置 agent_reviewed:true，让 stage2 走骨架降级
    assert aa.get("agent_reviewed") is not True


def test_idempotent_skip_when_already_reviewed(tmp_path, monkeypatch):
    cache_mod = _seed_cache(tmp_path, monkeypatch)
    cache_mod.write_task_output(TICKER, "agent_analysis",
                                {"agent_reviewed": True, "dim_commentary": {}})
    from lib.llm_panel.config import LLMConfig
    from lib.llm_panel.runner import run_llm_review

    class _ShouldNotCall:
        def chat_json(self, *a, **k):
            raise AssertionError("不应调用 LLM")

    ok = run_llm_review(TICKER, cfg=LLMConfig("k", "b", "m", 0.4, 10, 300),
                        client=_ShouldNotCall(), resume=True)
    assert ok is True
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_llm_runner.py -v`
Expected: FAIL — `test_validation_error_triggers_one_retry`（无重试逻辑）和 `test_all_groups_fail_still_writes_but_not_reviewed`（目前 raising client 会让 votes 为空但仍写 agent_reviewed:true）

- [ ] **Step 3: 改实现**

在 `lib/llm_panel/runner.py` 顶部 import 区追加：

```python
from lib.agent_analysis_validator import format_issues, validate
```

把 `run_llm_review` 里"综合调用（墙钟预算内）"那段整体替换为：

```python
    # 综合调用（墙钟预算内）+ 一次 schema 自纠重试
    if all_votes and time.time() - t0 < cfg.max_wall_seconds:
        syn_user = build_synthesis_prompt(all_votes, dim_commentary, dims, panel)
        syn = _synth_with_retry(client, system, syn_user, agent_analysis)
        for k in ("panel_insights", "great_divide_override", "narrative_override"):
            if k in syn:
                agent_analysis[k] = syn[k]
    elif not all_votes:
        print("   ⚠️ 所有分组均失败 · 降级：不置 agent_reviewed（stage2 走骨架）")
        agent_analysis["agent_reviewed"] = False
    else:
        print("   ⏱ 超墙钟预算，跳过综合研判")

    # 最终 schema 校验：仍有 error → 降级（不阻塞报告）
    errs = [i for i in validate(agent_analysis) if i.severity == "error"]
    if errs:
        print(f"   ⚠️ schema 仍有 {len(errs)} 条 error · 降级：不置 agent_reviewed")
        agent_analysis["agent_reviewed"] = False
```

并在文件末尾追加 `_synth_with_retry`：

```python
def _synth_with_retry(client, system, syn_user, partial: dict) -> dict:
    """综合调用 + 一次 schema 自纠重试。返回 syn dict（失败则 {}）。"""
    try:
        syn = client.chat_json(system, syn_user)
    except LLMError as e:
        print(f"   综合研判 ✗（仅写分组结果）: {e}")
        return {}
    trial = dict(partial)
    for k in ("panel_insights", "great_divide_override", "narrative_override"):
        if k in syn:
            trial[k] = syn[k]
    issues = [i for i in validate(trial) if i.severity == "error"]
    if not issues:
        print("   综合研判 ✓")
        return syn
    print(f"   ⚠️ 综合结果有 {len(issues)} 条 schema error · 自纠重试一次")
    fix_user = (syn_user + "\n\n# schema 修复\n上一次输出有结构错误：\n"
                + format_issues(issues)
                + "\n请严格按要求重新输出完整 JSON。")
    try:
        syn2 = client.chat_json(system, fix_user)
        print("   综合研判 ✓（重试后）")
        return syn2
    except LLMError as e:
        print(f"   综合研判重试 ✗: {e}")
        return syn
```

> 注意：`test_all_groups_fail_still_writes_but_not_reviewed` 用 `_RaisingClient`，
> 所有分组抛 `LLMError` → `all_votes` 为空 → 走 `elif not all_votes` 分支
> 置 `agent_reviewed=False`，仍 `write_task_output` 并 `return True`。
> 幂等用例由 Task 4 已有的 `existing.get("agent_reviewed")` 提前 return 覆盖。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_llm_runner.py -v`
Expected: PASS（5 passed — Task4 的 2 个 + 本 Task 的 3 个）

- [ ] **Step 5: 提交**

```bash
git add skills/deep-analysis/scripts/lib/llm_panel/runner.py \
        skills/deep-analysis/scripts/tests/test_llm_runner.py
git commit -m "feat(llm_panel): schema self-correction retry + graceful degradation"
```

---

### Task 6: __init__.py — `maybe_run_llm_review` 边界包装

**Files:**
- Modify: `skills/deep-analysis/scripts/lib/llm_panel/__init__.py`
- Test: `skills/deep-analysis/scripts/tests/test_llm_runner.py`（追加）

- [ ] **Step 1: 追加失败测试**

在 `tests/test_llm_runner.py` 末尾追加：

```python
def test_maybe_run_no_config_returns_false(tmp_path, monkeypatch):
    _seed_cache(tmp_path, monkeypatch)
    for k in ("UZI_LLM_API_KEY", "UZI_LLM_MODEL", "UZI_NO_LLM"):
        monkeypatch.delenv(k, raising=False)
    from lib.llm_panel import maybe_run_llm_review
    assert maybe_run_llm_review(TICKER) is False


def test_maybe_run_never_raises(tmp_path, monkeypatch):
    _seed_cache(tmp_path, monkeypatch)
    monkeypatch.setenv("UZI_LLM_API_KEY", "k")
    monkeypatch.setenv("UZI_LLM_MODEL", "m")
    # runner 内部强制抛非 LLMError 异常，验证边界吞掉
    import lib.llm_panel.runner as rmod

    def _boom(*a, **k):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(rmod, "run_llm_review", _boom)
    from lib.llm_panel import maybe_run_llm_review
    assert maybe_run_llm_review(TICKER) is False  # 不抛，返回 False
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_llm_runner.py -k maybe_run -v`
Expected: FAIL — `ImportError: cannot import name 'maybe_run_llm_review' from 'lib.llm_panel'`

- [ ] **Step 3: 写实现**

把 `skills/deep-analysis/scripts/lib/llm_panel/__init__.py` 整体替换为：

```python
"""lib.llm_panel · CLI 内置 OpenAI 兼容模型 · 自动生成 agent_analysis.json。

唯一对外入口：maybe_run_llm_review(ticker) -> bool
- 未配置 / kill switch → 立即 return False（CLI 行为与今天一致）
- 任何异常都在此吞掉 · 绝不向 stage2 抛出
"""
from __future__ import annotations

import os


def maybe_run_llm_review(ticker: str) -> bool:
    """stage2 入口调用。配置了 UZI_LLM_* 就跑 LLM 评审，否则跳过。

    永不抛异常；永不阻塞报告。返回 True=已写出 / False=跳过。
    """
    try:
        from .config import load_config
        cfg = load_config()
        if cfg is None:
            if os.environ.get("UZI_LLM_API_KEY") and not os.environ.get("UZI_LLM_MODEL"):
                print("ℹ️  已设 UZI_LLM_API_KEY 但缺 UZI_LLM_MODEL · 跳过 LLM 评审")
            else:
                print("ℹ️  未配置 UZI_LLM_* · 跳过 LLM 评审（脚本骨架模式）")
            return False
        print(f"🤖 LLM 评审 · 分组并发 role-play (model={cfg.model})")
        from .runner import run_llm_review
        # resume 语义：UZI_NO_RESUME=1（run.py --no-resume 设置）→ 强制重生成
        resume = os.environ.get("UZI_NO_RESUME") != "1"
        return bool(run_llm_review(ticker, cfg=cfg, resume=resume))
    except Exception as e:  # noqa: BLE001 — 边界必须吞掉一切
        print(f"⚠️  LLM 评审异常（跳过，不阻塞报告）: {type(e).__name__}: {str(e)[:160]}")
        return False
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_llm_runner.py -v`
Expected: PASS（7 passed）

- [ ] **Step 5: 提交**

```bash
git add skills/deep-analysis/scripts/lib/llm_panel/__init__.py \
        skills/deep-analysis/scripts/tests/test_llm_runner.py
git commit -m "feat(llm_panel): maybe_run_llm_review boundary wrapper (never raises)"
```

---

### Task 7: 接入 `stage2()` 单点 hook + 集成回归测试

**Files:**
- Modify: `skills/deep-analysis/scripts/run_real_test.py`（`stage2` 函数体第一行）
- Test: `skills/deep-analysis/scripts/tests/test_llm_integration.py`

- [ ] **Step 1: 写失败测试**

创建 `skills/deep-analysis/scripts/tests/test_llm_integration.py`：

```python
"""stage2 单点 hook 集成回归。"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))


def test_stage2_calls_maybe_run_llm_review(monkeypatch):
    """stage2 必须在读 agent_analysis 之前调用 maybe_run_llm_review。"""
    import run_real_test as rrt
    called = {}

    def _spy(ticker):
        called["ticker"] = ticker
        return False

    monkeypatch.setattr("lib.llm_panel.maybe_run_llm_review", _spy)

    # stage2 缺数据会 raise RuntimeError，但 hook 在 raise 之前已被调用
    try:
        rrt.stage2("NOPE999.SZ")
    except Exception:
        pass
    assert called.get("ticker") == "NOPE999.SZ"


def test_hook_is_first_statement_in_stage2_source():
    """源码层面确认 hook 在 stage2 体内、且在 read agent_analysis 之前。"""
    src = (SCRIPTS / "run_real_test.py").read_text(encoding="utf-8")
    body = src.split("def stage2(", 1)[1]
    i_hook = body.find("maybe_run_llm_review")
    i_read = body.find('read_task_output(ti.full, "agent_analysis")')
    assert i_hook != -1, "stage2 必须调用 maybe_run_llm_review"
    assert i_read != -1
    assert i_hook < i_read, "hook 必须在读取 agent_analysis 之前"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_llm_integration.py -v`
Expected: FAIL — `test_hook_is_first_statement_in_stage2_source`（源码尚无 hook）；
`test_stage2_calls_maybe_run_llm_review` 也 FAIL（spy 未被调用）

- [ ] **Step 3: 改 `run_real_test.py`**

定位 `def stage2(ticker: str) -> str:`（约 632 行）。当前函数体开头是：

```python
    from lib.cache import read_task_output
    ti = parse_ticker(ticker)
```

替换为（在 docstring 之后、`from lib.cache import` 之前插入 hook）：

```python
    # v3.5 · CLI 内置 LLM 评审单点 hook（pipeline + 全部 legacy 路径唯一汇合点）
    # 未配置 UZI_LLM_* / 已有 reviewed 结果 / 任何异常 → 立即返回，stage2 行为不变
    try:
        from lib.llm_panel import maybe_run_llm_review
        maybe_run_llm_review(ticker)
    except Exception as _llm_e:  # 双保险：边界已吞，这里再兜一层
        print(f"⚠️  LLM hook 跳过: {type(_llm_e).__name__}: {str(_llm_e)[:120]}")

    from lib.cache import read_task_output
    ti = parse_ticker(ticker)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_llm_integration.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 全量回归 — 确认无配置时行为零变化**

Run: `pytest tests/ -q -x`
Expected: PASS（全部既有测试 + 新增 llm_panel 测试通过；无 `UZI_LLM_*` 时 stage2 与今天完全一致）

- [ ] **Step 6: 提交**

```bash
git add skills/deep-analysis/scripts/run_real_test.py \
        skills/deep-analysis/scripts/tests/test_llm_integration.py
git commit -m "feat(llm_panel): wire single hook at stage2 entry (pipeline+legacy convergence)"
```

---

### Task 8: 文档 + `.env.example`

**Files:**
- Modify: `.env.example`（若不存在则 Create）
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `skills/deep-analysis/SKILL.md`

- [ ] **Step 1: 更新 `.env.example`**

先查看现有内容：`cat .env.example`（仓库根目录）。在文件末尾追加：

```bash
# ─── CLI 内置 OpenAI 兼容模型（v3.5）─────────────────────────────
# 配了 UZI_LLM_API_KEY + UZI_LLM_MODEL 后，python run.py <ticker> 会在
# stage1 之后自动 role-play 51 评委生成 agent_analysis.json，无需 Claude/Codex。
# 不配则保持脚本骨架模式（行为与今天一致）。
UZI_LLM_API_KEY=
UZI_LLM_MODEL=gpt-5.5
# 可选（有默认值）：
# UZI_LLM_BASE_URL=https://api.openai.com/v1   # DeepSeek: https://api.deepseek.com/v1
# UZI_LLM_TEMPERATURE=0.4
# UZI_LLM_TIMEOUT=90
# UZI_LLM_MAX_WALL_SECONDS=300
# UZI_NO_LLM=1                                  # 临时禁用（即使已配置）
```

- [ ] **Step 2: 更新 `README.md`**

在 README 介绍"快速模式 / 网络受限环境"附近新增一节（紧跟 `MX_APIKEY` 说明之后）：

```markdown
### 纯 CLI 全流程（无需 Claude/Codex · v3.5）

在 `.env` 配置一个 OpenAI 兼容模型即可让 `python run.py <ticker>` 自动完成
51 评委 role-play，输出深度报告，不再需要切到 Claude/Codex：

```bash
UZI_LLM_API_KEY=sk-xxx
UZI_LLM_MODEL=gpt-5.5            # 或 deepseek-chat / qwen-max ...
# UZI_LLM_BASE_URL=https://api.deepseek.com/v1   # 非 OpenAI 端点时设置
```

未配置时行为与之前完全一致（脚本骨架模式）。`UZI_NO_LLM=1` 可临时禁用。
当前范围：基于已采集数据做判断综合，不做联网调研（`qualitative_deep_dive`
留空，CLI 模式仅 warning）。
```

- [ ] **Step 3: 更新 `AGENTS.md`**

在 AGENTS.md 的"深浅两套路径"表格之后，新增说明段：

```markdown
### 路径 C · CLI 内置 LLM（v3.5 · 无 agent 工具）

`.env` 配了 `UZI_LLM_API_KEY` + `UZI_LLM_MODEL` 时，`stage2()` 入口的
`maybe_run_llm_review` 会自动用该模型分组 role-play 51 评委并写
`agent_analysis.json`，然后 stage2 照常 merge。对 agent 透明：
- 你（Claude/Codex）若已介入并写了 `agent_analysis.json`（`agent_reviewed:true`），
  LLM 步骤幂等跳过，不覆盖你的成果。
- 未配置 `UZI_LLM_*` 时此步骤完全不触发，行为与今天一致。
- 范围仅判断综合（A），不做 `qualitative_deep_dive`。
```

- [ ] **Step 4: 更新 `skills/deep-analysis/SKILL.md`**

在 SKILL.md 的"快速模式（跳过 agent 介入）"小节之后新增：

```markdown
### CLI 内置 LLM 模式（v3.5）

配置 `.env` 的 `UZI_LLM_API_KEY` + `UZI_LLM_MODEL` 后，CLI 直跑会在
stage1 与 stage2 之间自动调用该 OpenAI 兼容模型生成 `agent_analysis.json`
（分组 role-play 7 派系 + 1 次综合），实现无 Claude/Codex 的端到端深度分析。
幂等：若已存在 `agent_reviewed:true` 的 `agent_analysis.json` 则跳过。
范围：仅判断综合，不做联网 `qualitative_deep_dive`（CLI 模式仅 warning）。
配置项见仓库根 `.env.example`。
```

- [ ] **Step 5: 提交**

```bash
git add .env.example README.md AGENTS.md skills/deep-analysis/SKILL.md
git commit -m "docs(llm_panel): document UZI_LLM_* CLI-native LLM mode"
```

---

### Task 9: 收尾验证

- [ ] **Step 1: 全量测试**

Run: `cd skills/deep-analysis/scripts && pytest tests/ -q`
Expected: 全绿（既有 332+ 测试 + 新增 ~20 个 llm_panel 测试）

- [ ] **Step 2: 覆盖率检查（新模块 ≥80%）**

Run: `cd skills/deep-analysis/scripts && pytest tests/test_llm_*.py --cov=lib.llm_panel --cov-report=term-missing -q`
Expected: `lib/llm_panel/*` 综合覆盖率 ≥ 80%

- [ ] **Step 3: 冒烟（无配置 = 行为不变）**

Run: `cd skills/deep-analysis/scripts && python -c "import os; os.environ.pop('UZI_LLM_API_KEY', None); from lib.llm_panel import maybe_run_llm_review; print(maybe_run_llm_review('600519.SH'))"`
Expected: 打印 `ℹ️  未配置 UZI_LLM_* · 跳过 LLM 评审（脚本骨架模式）` 然后 `False`

- [ ] **Step 4: 最终提交（如有未提交的收尾改动）**

```bash
git status
git add -A && git commit -m "chore(llm_panel): finalize CLI-native LLM panel feature" || echo "nothing to commit"
```

---

## Self-Review（计划对照 spec）

- **范围 A（仅判断综合，不联网）**：Task 3/4 prompt 不含 web 检索；`qualitative_deep_dive` 不生成 → ✅ §1 范围决策
- **分组多调用（7 派系 + 1 综合）**：Task 4 `_group_investors` + ThreadPool + Task 5 synthesis → ✅ §3 数据流
- **配了就自动跑 / `UZI_NO_LLM` kill switch**：Task 1 `load_config` + Task 6 wrapper → ✅ §1 触发方式
- **单点 hook 在 stage2 入口**：Task 7 改 `run_real_test.py` + 源码断言测试 → ✅ §2 调用点（已与用户确认的 refinement）
- **复用 lib/personas（prefix-stable system）**：Task 3 `build_system` 委托 `build_system_message` + `to_prompt_block` → ✅ §1 非目标"不重写 personas"
- **复用 agent_analysis_validator**：Task 5 `validate` / `format_issues` → ✅ §3 step 8
- **永不阻塞报告（所有失败降级）**：Task 5 降级分支 + Task 6 边界吞异常 + Task 7 双保险 → ✅ §4 错误处理
- **幂等 / resume**：Task 4 `existing.get("agent_reviewed")` + Task 6 `UZI_NO_RESUME` 联动 → ✅ §5
- **零新依赖**：client 用 `requests`（已有），无 `openai` → ✅ §1 非目标
- **测试 mock client、无真实 API、≥80%**：Task 2/4/5/6/7 全 fake client，Task 9 覆盖率门槛 → ✅ §7
- **控制台 banner**：Task 6 `🤖 LLM 评审` + Task 4 `[Group X] ✓` + `✅ agent_analysis.json` → ✅ §6
- **类型一致性**：`LLMConfig`（Task1）→ `LLMClient(cfg)`（Task2）→ `run_llm_review(ticker, cfg=, client=, resume=)`（Task4/5）→ `maybe_run_llm_review(ticker)`（Task6）→ stage2 调用（Task7），签名贯穿一致 ✅
- **占位符扫描**：无 TBD/TODO，每个改码步骤含完整代码 ✅

无遗漏的 spec 需求；无占位符；签名一致。计划可执行。
