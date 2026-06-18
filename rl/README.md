# 强化学习对齐 RL

RLHF / 大模型对齐中最常用的两种策略优化损失：**PPO** 与 **GRPO**。本模块**自包含、可独立运行**（仅依赖 `torch` + `numpy`）。

> 📌 返回 [项目主 README](../README.md) ｜ 总览见主页 [模块导航](../README.md#modules)

## 文件清单

| 文件 | 你将学到 | 关键实现 |
| --- | --- | --- |
| [`ppo_grpo_loss.ipynb`](./ppo_grpo_loss.ipynb) | PPO 与 GRPO 损失的数学原理与最小可运行实现，并附玩具模型训练步 | `compute_gae`（GAE 优势）· `ppo_loss`（裁剪目标 + 价值 + 熵）· `grpo_group_advantages`（组内标准化）· `grpo_loss`（无 Critic + k3 KL 惩罚） |

## 关键概念

- **PPO**：用 GAE + 价值网络（Critic）估计优势，配合裁剪式重要性采样目标限制更新幅度。
- **GRPO**：去掉 Critic，用「同一 prompt 采样一组回答」的组内相对奖励作为优势，并显式加 per-token KL 惩罚（DeepSeekMath / DeepSeek-R1 路线）。

接入真实训练循环时，把 notebook 中的玩具模型换成你的 LLM、把合成奖励换成 reward model 打分即可。

## 依赖

```bash
pip install torch numpy jupyter
```
