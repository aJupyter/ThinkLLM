# Tokenization 分词

模型看到的是 token，不是文字。本模块从零实现 **BPE（Byte-Pair Encoding）**，讲清楚它是怎么通过"不断合并最高频的相邻单元对"来学会切词的。

> 📌 返回 [项目主 README](../README.md) ｜ 总览见主页 [模块导航](../README.md#modules)

## 文件清单

| 文件 | 你将学到 | 关键实现 |
| --- | --- | --- |
| [`bpe.ipynb`](./bpe.ipynb) | BPE 的训练流程（统计相邻对 → 合并最高频对 → 迭代）与编 / 解码流程 | `train_tokenizer` · `eval_tokenizer` |
| [`bpe.jsonl`](./bpe.jsonl) | 训练用语料（每行一条 JSON 文本） | — |

## 运行

打开 `bpe.ipynb` 依次执行即可，默认读取同目录下的 `./bpe.jsonl` 训练词表并测试编 / 解码。

```bash
pip install tokenizers jupyter
```

## 关键概念

- **训练**：从字符 / 字节开始，反复合并出现频率最高的相邻对，直到达到目标词表大小。
- **编码**：按学到的合并规则把文本切成子词，再映射为 token id；特殊 token 直接映射。
