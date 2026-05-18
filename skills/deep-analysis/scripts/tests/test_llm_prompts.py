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


def test_synthesis_prompt_headline_none_safe():
    from lib.llm_panel.prompts import build_synthesis_prompt
    votes = [{"investor_id": "x", "signal": "bearish", "score": 10, "headline": None}]
    s = build_synthesis_prompt(votes, {}, {"fundamental_score": 30}, {})
    assert "x" in s  # no crash


def test_market_snapshot_ignores_non_dict_school_scores():
    import json
    from lib.llm_panel.prompts import build_market_snapshot
    panel = {"school_scores": {"A": "corrupt_value"}}
    obj = json.loads(build_market_snapshot("X", {}, panel))
    assert obj["school_scores"] == {}


def test_group_prompt_empty_investors():
    from lib.llm_panel.prompts import build_group_prompt
    p = build_group_prompt("Z", "empty", [], {})
    assert "0 位" in p  # degrades gracefully, no crash
