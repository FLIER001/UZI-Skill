# Design · CLI 内置 OpenAI 兼容模型 · 自动生成 agent_analysis.json

- 日期：2026-05-18
- 状态：已批准（待写实现计划）
- 分支：`feat/llm-panel-cli`

## 1. 背景与目标

当前 `python run.py <ticker>` 走 v3.0 pipeline：`collect → score → synthesize`。
`score` 段（`score_from_cache`）用规则引擎生成 51 评委的**骨架分**并写
`.cache/{ticker}/panel.json`；随后 `synthesize`（→ `stage2`）读取
`.cache/{ticker}/agent_analysis.json` 合并进报告。

问题：`agent_analysis.json` 原本要靠 Claude / Codex 等 agent 工具
role-play 51 评委后写入。没有 agent 介入时，CLI 直跑只能用脚本骨架
（报告打 "未检测到 agent_analysis.json" 警告，质量显著下降）。

**目标**：在 `.env` 配置了 OpenAI 兼容模型（GPT-5.5 / DeepSeek / Qwen /
Kimi 等）后，CLI 自动在 `score` 与 `synthesize` 之间调用该模型完成
51 评委 role-play + 综合研判，生成合规 `agent_analysis.json`，
实现**不依赖 Claude/Codex，纯 CLI 跑完整流程**。

### 范围决策（已与用户确认）

| 决策 | 选择 | 含义 |
|---|---|---|
| 分析范围 | **仅判断综合（A）** | LLM 只基于 stage1 已采集数据做 51 评委 role-play + `dim_commentary`/`panel_insights`/`great_divide_override`/`narrative_override`。**不**做联网调研。`qualitative_deep_dive` 故意留空（CLI 模式仅 warning，不阻塞报告）。 |
| 调用策略 | **分组多调用** | 按 panel 的 7 个派系（A–G）分别调用 + 1 次综合调用，persona 保真度高。 |
| 触发方式 | **配了就自动跑** | `.env` 配了 LLM key/base_url/model 即自动启用；未配保持现状（脚本骨架模式）。`UZI_NO_LLM=1` 可临时禁用。 |

### 非目标（YAGNI）

- 不做联网调研 / web search / Playwright（范围 A 明确排除）。
- 不生成 `qualitative_deep_dive`（CLI 模式仅 warning）。
- 不引入 `openai` pip 依赖（项目坚持近零依赖，统一用 `requests`）。
- 不改 `stage2` 的 merge/校验逻辑、不改 `agent_analysis_validator` / `assemble_report`
  （这些已能消费任意来源的 `agent_analysis.json`）。**仅在 `stage2()` 入口加一句
  additive 调用**——见 §2 调用点。
- 不做 ETF/LOF/中文名解析相关改动（pipeline preflight 已 fallback legacy 处理）。
- 不重写 `lib/personas.py`：复用其 `load_persona` / `Persona.to_prompt_block` /
  `build_system_message`（已是 prefix-stable / prompt-cache 优化）/
  `FRAMEWORK_INSTRUCTIONS_ZH`（已含严格 JSON 输出契约）。

## 2. 架构

新增包 `skills/deep-analysis/scripts/lib/llm_panel/`：

| 文件 | 职责 | 约束 |
|---|---|---|
| `config.py` | 读 `UZI_LLM_*` 环境变量，返回 `LLMConfig` 或 `None`。`None` ⇒ 整步跳过（保持现状）。 | 纯函数，无网络 |
| `client.py` | 极薄 OpenAI 兼容 `/v1/chat/completions` 客户端（`requests`，重试/超时，JSON mode）。镜像 `lib/mx_api.py` 模式。 | ~120 行，零新依赖 |
| `prompts.py` | 构建**字节级稳定**的 system prompt + 各分组 user prompt（读 `panel.json` / `dimensions.json` / persona YAML）。 | 纯函数 |
| `runner.py` | 编排：7 组并发调用 → 1 次综合调用 → 组装 `agent_analysis.json` → 复用 `agent_analysis_validator` 校验 → 1 次自纠重试 → 写 cache。 | 所有异常在边界吞掉 |
| `__init__.py` | 暴露 `maybe_run_llm_review(ticker) -> bool`。 | 唯一对外入口 |

### 配置环境变量（沿用 `MX_APIKEY` 的 `.env` 约定）

| 变量 | 默认 | 说明 |
|---|---|---|
| `UZI_LLM_API_KEY` | （无） | **必填**，缺失即功能关闭 |
| `UZI_LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI 兼容端点 |
| `UZI_LLM_MODEL` | （必填） | 如 `gpt-5.5` / `deepseek-chat` / `qwen-max` |
| `UZI_LLM_TEMPERATURE` | `0.4` | 采样温度 |
| `UZI_LLM_TIMEOUT` | `90` | 单次调用秒数 |
| `UZI_LLM_MAX_WALL_SECONDS` | `300` | LLM 总墙钟预算，超时放弃走骨架 |
| `UZI_NO_LLM` | （无） | `=1` 即使配置了也跳过（kill switch） |

> 注意：**不复用 `OPENAI_API_KEY`**。`run.py` 的 `detect_environment()`
> 已把 `OPENAI_API_KEY` 存在性当作 "is_codex" 信号，复用会污染该判断。
> 用独立 `UZI_LLM_*` 命名空间。

### 调用点（单点插入 · stage2 入口）

**唯一插入点**：`run_real_test.stage2()` 函数体第一行（在它读取
`agent_analysis.json` 之前）插入 `maybe_run_llm_review(ticker)`。

理由：legacy 路径在 `run.py` 里有 3+ 个 `stage2` 入口（中文名解析后、
stage1 成功后、`run_real_test.main()` 内部），pipeline 路径经
`synthesize_and_render` → 也调 `rrt.stage2()`。**`stage2()` 是 pipeline
与全部 legacy 变体的唯一汇合点**，单点插入一次即全覆盖，杜绝"pipeline
跑了 LLM、中文名 legacy 没跑"的不一致。

这是一句 **additive 调用**（在函数最前，幂等、失败不抛），**不改动**
`stage2` 既有的 merge/校验逻辑，也不改 `agent_analysis_validator` /
`assemble_report`——它们本就设计为消费任意来源（含非 Claude 模型）的
`agent_analysis.json`。`maybe_run_llm_review` 自身保证：未配置 / 已存在
reviewed 结果 / 任何异常 → 立即 return，stage2 行为与今天完全一致。

## 3. 数据流

`maybe_run_llm_review(ticker)` 内部：

```
1. cfg = load_config()
      None → 打印 "ℹ️ 未配置 LLM · 跳过" → return False（骨架模式，不变）
2. 若 .cache/{t}/agent_analysis.json 已有 agent_reviewed:true 且 resume 模式 → 跳过（幂等）
   （--no-resume 强制重生成）
3. 读 .cache/{ticker}/ 下 panel.json, dimensions.json, raw_data.json
4. 按 panel["school_scores"] 把 51 评委分到 A–G 七组
5. 各组并发（ThreadPoolExecutor max_workers=4）：
      system = personas.build_system_message(snapshot_json)  # 复用 · 8 次字节级一致 → prompt cache
      user   = build_group_prompt(g, 组内 investors, dims)
               # 组内每个 persona 用 personas.load_persona(id).to_prompt_block() 拼接
               # + 该 investor 的 panel 骨架分（headline/score/signal/pass/fail）
      resp   = client.chat_json(system, user)          # JSON mode · 返回 PersonaVote 数组
      → 每个 investor: {headline, reasoning, score, signal} + 该组 dim_commentary 片段
6. 综合调用（1 次）：
      输入 = 合并各组输出 + dims summary + DCF/LBO/panel_consensus
      输出 = panel_insights, great_divide_override,
             narrative_override(core_conclusion, risks≥3,
             buy_zones{value,growth,technical,youzi})
7. 组装 agent_analysis.json：
      { agent_reviewed:true, _llm_generated:true, _llm_model:<model>,
        dim_commentary{≥5}, panel_insights,
        great_divide_override, narrative_override }
      （qualitative_deep_dive 故意省略 — 范围 A）
8. validate()（lib.agent_analysis_validator）
      有 error → 把 format_issues() 喂回模型，1 次自纠重试
      仍有 error → 写部分结果但不置 agent_reviewed:true，
                   交给 stage2 既有的优雅降级路径
9. 同步回写 panel.json investors[].{headline,reasoning,score,signal}
   （对齐完整 agent 流程，提升报告里 51 评委卡片质量）
10. 写 agent_analysis.json → return True
```

### 分组依据

7 组已存在于 `panel["school_scores"]`：
A 经典价值 / B 成长 / C 宏观 / D 技术 / E 中式价投 / F A 股游资 / G 量化。

- F 组 19–23 人，但按 `HARD-GATE-PERSONA-ROLEPLAY`，stub persona 以规则
  引擎为准，prompt 保持紧凑。
- 12 个 flagship persona（巴菲特/芒格/林奇/木头姐/索罗斯/达里奥/段永平/
  张坤/赵老哥/章盟主等）注入 `personas/{id}.yaml` 的 `key_metrics`/`voice`，
  保证历史立场一致（巴菲特不会对 PE-882 说买入）。

### Prompt 缓存纪律

system prompt 在全部 8 次调用中字节级一致（只变 user message），复用项目
既有 prefix-stable 约定，支持缓存的 provider（OpenAI/DeepSeek）在第 2–8 次
命中缓存，省成本。

## 4. 错误处理（绝不阻塞报告）

| 失败 | 行为 |
|---|---|
| 未配 `UZI_LLM_API_KEY` | 打印 `ℹ️ 未配置 LLM · 跳过` → return False → stage2 骨架模式（与现状完全一致） |
| `UZI_NO_LLM=1` | 同上，显式 kill switch |
| 网络错误 / 超时 / HTTP 4xx-5xx | 单次调用最多重试 2× + 退避；某组仍失败 → 该组保留规则骨架分，其他组照常 |
| 模型返回非 JSON / 截断 | 一次 "只返回合法 JSON" 重问；仍坏 → 该组回退骨架 |
| 组装结果过 validator（error 级） | 喂回 `format_issues()` 做 1 次自纠重试；仍 error → 写部分 `agent_analysis.json` **但不置** `agent_reviewed:true`，走 stage2 既有降级 |
| LLM 总墙钟 > `UZI_LLM_MAX_WALL_SECONDS` | 放弃 LLM 步骤，走 stage2 骨架模式 |

所有失败路径终点都是"报告照常生成"——LLM 步骤严格只增不减。
所有异常在 `maybe_run_llm_review` 边界 catch，绝不向 pipeline 抛出。

## 5. 幂等 / resume

`.cache/{ticker}/agent_analysis.json` 已存在且 `agent_reviewed:true`、
且**未**传 `--no-resume` → 跳过 LLM 步骤（与 pipeline 既有 resume 语义一致）。
`--no-resume` 强制重生成。

## 6. 控制台 UX

对齐既有 stage banner 风格：

```
🤖 LLM 评审 · 7 组并发 role-play (model=gpt-5.5)
   [A 经典价值] 6 人 ✓   [B 成长] 4 人 ✓   ...
   综合研判 ✓
✅ agent_analysis.json 已生成 · schema 校验通过
```

让纯 CLI 用户清楚看到"发生了真实分析"，而非骨架模式。

## 7. 测试（pytest，落到既有 `scripts/tests/`）

- `test_llm_config.py` — env 解析；缺 key ⇒ `None`；kill switch
- `test_llm_prompts.py` — 分组 prompt 含 flagship persona 的 `key_metrics`；
  system prompt 跨组字节级稳定
- `test_llm_runner.py` — **mock client**（预设 JSON，无网络）：happy path 写出
  合法 `agent_analysis.json`；validator-error 路径恰好触发一次重试；
  API 异常路径返回 False 且不写文件；幂等跳过
- `test_llm_integration.py` — mock client 端到端：`score →
  maybe_run_llm_review → stage2` 产出含 `agent_reviewed:true` 的 HTML；
  且 LLM 关闭时与今天骨架输出一致（回归保护）
- 全部测试用注入的 fake client，**测试套件零真实 API 调用**。
  新模块目标覆盖率 ≥ 80%。

## 8. 影响面

- **新增**：`lib/llm_panel/`（5 文件）+ `scripts/tests/test_llm_*.py`（4 文件）
- **修改**：`run_real_test.py` `stage2()` 入口（**+1 行** additive 调用，
  不动其 merge/校验逻辑）、`.env.example` / README / AGENTS.md / SKILL.md
  （文档说明新 env）
- **不改**：`agent_analysis_validator`、`assemble_report`、`lib/personas.py`、
  `lib/pipeline/*`、22 个 fetcher、score 段、`run.py`
