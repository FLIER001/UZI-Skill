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
            if os.environ.get("UZI_NO_LLM") == "1":
                print("ℹ️  UZI_NO_LLM=1 · 跳过 LLM 评审（kill switch）")
            elif os.environ.get("UZI_LLM_API_KEY") and not os.environ.get("UZI_LLM_MODEL"):
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
