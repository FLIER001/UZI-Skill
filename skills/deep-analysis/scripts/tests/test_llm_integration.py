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
