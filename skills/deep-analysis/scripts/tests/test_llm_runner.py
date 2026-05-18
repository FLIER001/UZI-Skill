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


def test_second_investor_also_overwritten(tmp_path, monkeypatch):
    cache_mod = _seed_cache(tmp_path, monkeypatch)
    from lib.llm_panel.config import LLMConfig
    from lib.llm_panel.runner import run_llm_review
    run_llm_review(TICKER, cfg=LLMConfig("k", "b", "m", 0.4, 10, 300),
                   client=_FakeClient())
    panel = cache_mod.read_task_output(TICKER, "panel")
    bj = [i for i in panel["investors"] if i["investor_id"] == "bj_cj"][0]
    assert bj["score"] == 18
    assert bj["headline"].startswith("LLM 覆盖")


class _AlwaysBadSynthClient:
    """分组正常，但综合调用两次都返回非法结构（risks 为字符串）。"""
    def chat_json(self, system, user, attempts=3):
        if "综合研判任务" in user:
            return {"narrative_override": {"risks": "still a string, invalid"}}
        return {"votes": [{"investor_id": "buffett", "signal": "bearish",
                           "score": 18, "verdict": "回避",
                           "headline": "h" * 25, "reasoning": "r" * 40,
                           "persona_used": "flagship"}],
                "dim_commentary": {"1_financials": "ROE 仅 5% 连续下滑回款承压明显啊啊。"}}


def test_retry_still_bad_degrades_not_reviewed(tmp_path, monkeypatch):
    cache_mod = _seed_cache(tmp_path, monkeypatch)
    from lib.llm_panel.config import LLMConfig
    from lib.llm_panel.runner import run_llm_review
    ok = run_llm_review(TICKER, cfg=LLMConfig("k", "b", "m", 0.4, 10, 300),
                        client=_AlwaysBadSynthClient())
    assert ok is True
    aa = cache_mod.read_task_output(TICKER, "agent_analysis")
    # 综合两次都非法 → 最终校验仍有 error → 必须降级，绝不能写 agent_reviewed:true
    assert aa.get("agent_reviewed") is not True


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


def test_maybe_run_hint_key_without_model(tmp_path, monkeypatch, capsys):
    _seed_cache(tmp_path, monkeypatch)
    monkeypatch.setenv("UZI_LLM_API_KEY", "sk-xxx")
    monkeypatch.delenv("UZI_LLM_MODEL", raising=False)
    monkeypatch.delenv("UZI_NO_LLM", raising=False)
    from lib.llm_panel import maybe_run_llm_review
    assert maybe_run_llm_review(TICKER) is False
    assert "缺 UZI_LLM_MODEL" in capsys.readouterr().out


def test_maybe_run_kill_switch_message(tmp_path, monkeypatch, capsys):
    _seed_cache(tmp_path, monkeypatch)
    monkeypatch.setenv("UZI_LLM_API_KEY", "sk-xxx")
    monkeypatch.setenv("UZI_LLM_MODEL", "gpt-5.5")
    monkeypatch.setenv("UZI_NO_LLM", "1")
    from lib.llm_panel import maybe_run_llm_review
    assert maybe_run_llm_review(TICKER) is False
    assert "kill switch" in capsys.readouterr().out


def test_maybe_run_configured_success_returns_true(tmp_path, monkeypatch):
    _seed_cache(tmp_path, monkeypatch)
    monkeypatch.setenv("UZI_LLM_API_KEY", "k")
    monkeypatch.setenv("UZI_LLM_MODEL", "gpt-5.5")
    monkeypatch.delenv("UZI_NO_LLM", raising=False)
    import lib.llm_panel.runner as rmod
    monkeypatch.setattr(rmod, "run_llm_review", lambda *a, **k: True)
    from lib.llm_panel import maybe_run_llm_review
    assert maybe_run_llm_review(TICKER) is True
