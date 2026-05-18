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

    # 生效依赖 stage2 内的延迟 import（from lib.llm_panel import ... 在调用时执行）
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
    i_raw_read = body.find('read_task_output(ti.full, "raw_data")')
    assert i_raw_read != -1
    assert i_hook < i_raw_read, "hook 必须在所有 cache 读取之前"


def test_end_to_end_configured_writes_agent_reviewed(tmp_path, monkeypatch):
    """配置 LLM + fake client → maybe_run_llm_review 写出 agent_reviewed:true。"""
    from lib import cache as cache_mod
    monkeypatch.setattr(cache_mod, "CACHE_ROOT", tmp_path / ".cache")
    T = "000001.SZ"
    cache_mod.write_task_output(T, "raw_data", {"ticker": T, "dimensions": {}})
    cache_mod.write_task_output(T, "dimensions", {"fundamental_score": 31})
    cache_mod.write_task_output(T, "panel", {
        "ticker": T, "panel_consensus": 30.0,
        "vote_distribution": {"buy": 1}, "signal_distribution": {"bearish": 1},
        "school_scores": {"A": {"label": "经典价值派", "verdict": "回避", "avg_score": 25}},
        "investors": [{"investor_id": "buffett", "name": "巴菲特", "group": "A",
                       "headline": "skel", "score": 20, "signal": "bearish",
                       "pass": [], "fail": []}],
    })

    class _FC:
        def chat_json(self, system, user, attempts=3):
            if "综合研判任务" in user:
                return {
                    "panel_insights": "看空为主，价值派因 ROE 过低集体回避，分歧集中在估值水平。",
                    "great_divide_override": {
                        "punchline": "ROE 5% 撑不起 25 倍 PE，多空为故事定价",
                        "bull_say_rounds": ["a1", "a2", "a3"],
                        "bear_say_rounds": ["b1", "b2", "b3"]},
                    "narrative_override": {
                        "core_conclusion": "测试标的 35 分回避，安全边际不足明显。",
                        "risks": ["r1", "r2", "r3"],
                        "buy_zones": {k: {"price": 1.0, "rationale": "xxxxx"}
                                      for k in ("value", "growth", "technical", "youzi")}},
                }
            return {"votes": [{"investor_id": "buffett", "signal": "bearish",
                               "score": 18, "verdict": "回避", "headline": "h" * 25,
                               "reasoning": "r" * 40, "persona_used": "flagship"}],
                    "dim_commentary": {"1_financials": "ROE 仅 5% 连续下滑回款承压明显啊。"}}

    monkeypatch.setenv("UZI_LLM_API_KEY", "k")
    monkeypatch.setenv("UZI_LLM_MODEL", "gpt-5.5")
    monkeypatch.delenv("UZI_NO_LLM", raising=False)
    monkeypatch.delenv("UZI_NO_RESUME", raising=False)

    import lib.llm_panel.runner as rmod
    _orig = rmod.run_llm_review
    monkeypatch.setattr(rmod, "run_llm_review",
                        lambda ticker, cfg, client=None, *, resume=True:
                        _orig(ticker, cfg, client=_FC(), resume=resume))

    from lib.llm_panel import maybe_run_llm_review
    assert maybe_run_llm_review(T) is True
    aa = cache_mod.read_task_output(T, "agent_analysis")
    assert aa["agent_reviewed"] is True
    assert aa["_llm_generated"] is True
    from lib.agent_analysis_validator import validate
    assert [i for i in validate(aa) if i.severity == "error"] == []
