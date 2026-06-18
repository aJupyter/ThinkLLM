# 混合专家模型 MoE

用最小可读的代码理解 **MoE（Mixture of Experts）**：专家网络、路由（门控）、稀疏激活与负载均衡。

> 📌 返回 [项目主 README](../README.md) ｜ 总览见主页 [模块导航](../README.md#modules)

## 文件清单

| 文件 | 你将学到 | 关键实现 |
| --- | --- | --- |
| [`moe.ipynb`](./moe.ipynb) | 从单个专家到可用于大模型训练的 Sparse MoE | `BasicExpert` · `BasicMOE` · `MOERouter`（Top-K 门控）· `MOEConfig` · `SparseMOE` |

<div align="center"><img src="./images/moe_base.png" alt="MoE 基础结构" width="420"/></div>

## 推荐学习路径

1. **BasicExpert / BasicMOE**：理解「多个专家 + 加权组合」的稠密版本；
2. **MOERouter**：理解 Top-K 门控如何为每个 token 选择专家；
3. **SparseMOE**：理解只激活部分专家的稀疏计算（大模型训练实际使用）。

## 依赖

```bash
pip install torch jupyter
```
