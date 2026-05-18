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
