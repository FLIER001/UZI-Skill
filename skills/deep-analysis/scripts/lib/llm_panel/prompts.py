"""lib.llm_panel.prompts · 构建 snapshot + 分组 + 综合 prompt。

复用 lib.personas：
- build_system(snapshot) → lib.personas.build_system_message（prefix-stable）
- 组内每个 persona 用 load_persona(id).to_prompt_block()
"""
from __future__ import annotations

import json

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
    tally: dict = {}
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
