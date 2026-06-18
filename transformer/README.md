# Transformer 核心组件

从零理解一个 Transformer 是如何"算"出来的：注意力、位置编码、归一化、前馈网络逐个拆解。优先 NumPy / 朴素 PyTorch 实现，先讲清原理再做工程优化。

> 📌 返回 [项目主 README](../README.md) ｜ 总览见主页 [模块导航](../README.md#modules)

## 文件清单

| 文件 | 你将学到 | 关键实现 |
| --- | --- | --- |
| [`transformer_basics.ipynb`](./transformer_basics.ipynb) | 纯 NumPy 复现完整 Transformer（编码器 / 解码器），看清每一步矩阵运算 | `softmax` · `positional_encoding` · `scaled_dot_product_attention` · `multi_head_attention` · `feed_forward_network` · `layer_norm` · `encoder_layer` · `decoder_layer` |
| [`attention_mha_gqa_mqa.ipynb`](./attention_mha_gqa_mqa.ipynb) | MHA / MQA / GQA 的概念、原理与显存对比 | 三种注意力对比 + PyTorch 实现 |
| [`attention_mha_gqa_mqa_mla.ipynb`](./attention_mha_gqa_mqa_mla.ipynb) | 用 `nn.Module` 封装多头与分组查询注意力 | `MultiHeadAttention` · `GroupQueryAttention` |
| [`mla_flash_attention.ipynb`](./mla_flash_attention.ipynb) | 线性注意力的计算简化、FlashAttention 的分块思想 | `Linear Attention` · `FlashAttention` |

## 推荐学习路径

1. **`transformer_basics`**：建立直觉，用 NumPy 看清每一步矩阵运算；
2. **`attention_mha_gqa_mqa(_mla)`**：理解 KV Cache 显存优化（MQA / GQA）；
3. **`mla_flash_attention`**：理解推理加速（线性注意力 / FlashAttention）。

## 依赖

```bash
pip install numpy torch jupyter
```
