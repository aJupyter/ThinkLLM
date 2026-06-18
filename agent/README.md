# Agent 核心算法

Agent 推理与规划的核心范式：**ReAct / Chain-of-Thought / 自我反思（Reflexion）/ Tree-of-Thought**，以及工具调用解析。本模块**自包含、纯标准库可运行**，无需真实大模型或联网。

为离线跑通做两处「hack」：
- **工具 hack**：用本地事实字典 + 安全算术求值模拟搜索引擎与计算器；
- **LLM hack**：用手写规则策略（mock LLM）扮演大模型，把 Agent 控制流清晰暴露出来。

> 📌 返回 [项目主 README](../README.md) ｜ 总览见主页 [模块导航](../README.md#modules)

## 文件清单

| 文件 | 你将学到 | 关键实现 |
| --- | --- | --- |
| [`react_agent.ipynb`](./react_agent.ipynb) | ReAct / CoT / 自我反思 / Tree-of-Thought 与工具调用解析 | `search` · `calculator`（工具 hack）· `parse_action`（方括号 / JSON 两种工具调用解析）· `react_agent`（手写 ReAct 循环）· `reflexion_loop`（反思重试）· `tot_solve_24`（思维树搜索 24 点） |

## 各方法一句话核心

| 方法 | 核心 | 适用场景 |
| --- | --- | --- |
| CoT | 显式写出中间推理步骤 | 多步推理 / 算术 / 常识 |
| ReAct | 推理与工具调用交替（Thought → Action → Observation） | 需要外部知识 / 工具的任务 |
| Reflexion | 失败后写反思并重试 | 可验证结果、允许多次尝试 |
| Tree-of-Thought | 在思维树上生成分支 + 评估 + 搜索 | 需要探索 / 回溯的难题（如 24 点、规划） |

接入真实 LLM 时，只需把 `mock_react_llm` / `attempt` 等替换为对模型的调用，把 `TOOLS` 换成真实工具即可。

## 依赖

仅需 Python 标准库（`re` / `json` / `itertools` / `collections`）。

```bash
pip install jupyter   # 仅用于打开 notebook
```
