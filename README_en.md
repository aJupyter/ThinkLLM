# 🚀 ThinkLLM: Implementation of Large Language Model Algorithms and Components

<div align="center">
    <a href="./README.md">简体中文</a> | English
</div>

<div align="center">
  <img src="images/logo.png" alt="ThinkLLM Logo" width="200"/>
  <br/>
  <p>🚀 Lightweight and Efficient Large Language Model Algorithm Implementations</p>

  [![GitHub stars](https://img.shields.io/github/stars/aJupyter/ThinkLLM?style=flat-square)](https://github.com/aJupyter/ThinkLLM/stargazers)
  [![GitHub forks](https://img.shields.io/github/forks/aJupyter/ThinkLLM?style=flat-square)](https://github.com/aJupyter/ThinkLLM/network)
  [![GitHub issues](https://img.shields.io/github/issues/aJupyter/ThinkLLM?style=flat-square)](https://github.com/aJupyter/ThinkLLM/issues)
  [![GitHub license](https://img.shields.io/github/license/aJupyter/ThinkLLM?style=flat-square)](https://github.com/aJupyter/ThinkLLM/blob/main/LICENSE)
  [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/aJupyter/ThinkLLM/pulls)
</div>

## Project Overview 🌟

**ThinkLLM** is an open-source project focused on the implementation of core algorithms for large language models. With minimal dependencies and concise code, we reimplement key algorithms and components across LLM / multimodal / RAG / MoE / RL / Agent from scratch, helping developers and researchers understand the underlying mechanisms of large models **through runnable code**.

Design principles:

- **Runnable**: every module is a Notebook or script you can open and run directly — not pseudocode.
- **Self-contained**: every Notebook / script **does not depend on any other file in the project**; open it alone and run top-to-bottom.
- **Readable**: NumPy / plain PyTorch first — explain the principle before any engineering optimization.

> If you're interested in full-stack practice with large models, check out the fully open-source [EmoLLM](https://github.com/SmartFlowAI/EmoLLM).
> You can also use [DeepWiki](https://deepwiki.com/aJupyter/ThinkLLM) to help understand this project.

## Updates 🔥

- **[2025.6]** Added [Agent module](./agent): ReAct / CoT / reflection / ToT (hand-written demo + hacked tools); added [RL module](./rl): PPO / GRPO loss from scratch; unified and standardized the directory structure
- **[2025.5]** Added [DeepWiki](https://deepwiki.com/aJupyter/ThinkLLM) support; added [MLA / FlashAttention](./transformer/mla_flash_attention.ipynb) and the [multimodal](./multimodal) series
- **[2025.4]** Added the [RAG algorithm library](./rag) and [BPE tokenizer](./tokenizer/bpe.ipynb)
- **[2025.3]** Added [MHA / GQA / MQA](./transformer/attention_mha_gqa_mqa_mla.ipynb) and [ViT](./multimodal/vit.ipynb)

## Directory Structure 🗂️

```
ThinkLLM/
├── transformer/     # Core Transformer components (attention / positional encoding / norm …)
├── tokenizer/       # Tokenization algorithms (BPE …)
├── multimodal/      # Multimodal (ViT / feature extraction / cross-modal projection)
├── rag/             # Retrieval-Augmented Generation library (vector retrieval / optimization)
├── moe/             # Mixture of Experts (MoE)
├── rl/              # RL alignment (PPO / GRPO …)
├── agent/           # Agent algorithms (ReAct / CoT / reflection / ToT)
├── images/          # Shared image assets
├── README.md
└── README_en.md
```

> Naming convention: directories and files use **lowercase + underscore (snake_case)**, no spaces or special characters.

> 📂 **Per-module docs**: [transformer](./transformer/README.md) · [tokenizer](./tokenizer/README.md) · [multimodal](./multimodal/README.md) · [rag](./rag/README.md) · [moe](./moe/README.md) · [rl](./rl/README.md) · [agent](./agent/README.md)

## Module Navigation 🧭

The table below lists what is **already implemented**. Click a **Module** to open its folder docs, or an **Entry file** to read / run it directly. In `Status`, ✅ means implemented and 🚧 means planned (see the [Roadmap](#roadmap-) at the end).

| Module | Topic | Entry file | Form | Status |
| --- | --- | --- | --- | --- |
| [📂 transformer](./transformer/README.md) | Full Transformer components (NumPy) | [`transformer_basics.ipynb`](./transformer/transformer_basics.ipynb) | Notebook | ✅ |
| [📂 transformer](./transformer/README.md) | MHA / GQA / MQA (concept + theory) | [`attention_mha_gqa_mqa.ipynb`](./transformer/attention_mha_gqa_mqa.ipynb) | Notebook | ✅ |
| [📂 transformer](./transformer/README.md) | Modular MHA / GQA implementation | [`attention_mha_gqa_mqa_mla.ipynb`](./transformer/attention_mha_gqa_mqa_mla.ipynb) | Notebook | ✅ |
| [📂 transformer](./transformer/README.md) | Linear Attention / FlashAttention | [`mla_flash_attention.ipynb`](./transformer/mla_flash_attention.ipynb) | Notebook | ✅ |
| [📂 tokenizer](./tokenizer/README.md) | BPE tokenizer training & evaluation | [`bpe.ipynb`](./tokenizer/bpe.ipynb) | Notebook | ✅ |
| [📂 multimodal](./multimodal/README.md) | ViT (Vision Transformer) | [`vit.ipynb`](./multimodal/vit.ipynb) | Notebook | ✅ |
| [📂 multimodal](./multimodal/README.md) | Image feature extraction & mapping | [`image_feature_extraction.ipynb`](./multimodal/image_feature_extraction.ipynb) | Notebook | ✅ |
| [📂 multimodal](./multimodal/README.md) | Cross-modal projection & fusion | [`projection_layer.ipynb`](./multimodal/projection_layer.ipynb) | Notebook | ✅ |
| [📂 rag](./rag/README.md) | Vector retrieval + retrieval optimization | [`rag/`](./rag) · [docs](./rag/README.md) | Scripts + Notebook | ✅ |
| [📂 moe](./moe/README.md) | Basic MoE & Sparse MoE | [`moe.ipynb`](./moe/moe.ipynb) | Notebook | ✅ |
| [📂 rl](./rl/README.md) | PPO / GRPO loss from scratch | [`ppo_grpo_loss.ipynb`](./rl/ppo_grpo_loss.ipynb) | Notebook | ✅ |
| [📂 agent](./agent/README.md) | ReAct / CoT / reflection / ToT (hand-written demo) | [`react_agent.ipynb`](./agent/react_agent.ipynb) | Notebook | ✅ |

## Setup & Quick Start 💡

```bash
# 1. Clone the repo
git clone https://github.com/aJupyter/ThinkLLM.git
cd ThinkLLM

# 2. Install common dependencies (install per module as needed)
pip install numpy torch matplotlib jupyter
pip install jieba          # Chinese tokenization for rag
pip install tokenizers     # tokenizer (BPE) module

# 3. Open any Notebook
jupyter notebook
```

> Modules are independent; there is no unified `requirements.txt`. Every Notebook / script is **self-contained** and runs on its own.

---

## Implemented Modules in Detail 📦

### 1. Core Transformer Components ([`transformer/`](./transformer/README.md))

> Understand from scratch how a Transformer is actually "computed": attention, positional encoding, normalization, and the feed-forward network, piece by piece.

| File | What you'll learn | Key implementations |
| --- | --- | --- |
| [`transformer_basics.ipynb`](./transformer/transformer_basics.ipynb) | A full Transformer in pure NumPy, understanding every matrix op | `softmax` · `positional_encoding` · `scaled_dot_product_attention` · `multi_head_attention` · `feed_forward_network` · `layer_norm` · `encoder_layer` · `decoder_layer` |
| [`attention_mha_gqa_mqa.ipynb`](./transformer/attention_mha_gqa_mqa.ipynb) | Concepts, theory and memory trade-offs of MHA / MQA / GQA | Comparison + PyTorch implementations |
| [`attention_mha_gqa_mqa_mla.ipynb`](./transformer/attention_mha_gqa_mqa_mla.ipynb) | Multi-head & grouped-query attention as `nn.Module` | `MultiHeadAttention` · `GroupQueryAttention` |
| [`mla_flash_attention.ipynb`](./transformer/mla_flash_attention.ipynb) | Linear attention simplification and FlashAttention tiling | `Linear Attention` · `FlashAttention` |

**Suggested path**: read `transformer_basics` (intuitive NumPy version) first, then `attention_*` (memory optimization), and finally `mla_flash_attention` (inference acceleration).

### 2. Tokenization ([`tokenizer/`](./tokenizer/README.md))

> Models see tokens, not text. This module explains how BPE "learns" to segment text.

| File | What you'll learn | Key implementations |
| --- | --- | --- |
| [`bpe.ipynb`](./tokenizer/bpe.ipynb) | BPE training (count adjacent pairs → merge the most frequent → iterate) and encode/decode | `train_tokenizer` · `eval_tokenizer`, with corpus [`bpe.jsonl`](./tokenizer/bpe.jsonl) |

### 3. Multimodal ([`multimodal/`](./multimodal/README.md))

> Let the model "see" images: from patch encoding to aligning visual features into the language space.

| File | What you'll learn | Key steps |
| --- | --- | --- |
| [`vit.ipynb`](./multimodal/vit.ipynb) | The full Vision Transformer pipeline | Patch Embedding → positional encoding → Transformer Encoder → full ViT → patch visualization |
| [`image_feature_extraction.ipynb`](./multimodal/image_feature_extraction.ipynb) | Image feature extraction and mapping | Feature extractor → mapping module → full pipeline → feature visualization → end-to-end example |
| [`projection_layer.ipynb`](./multimodal/projection_layer.ipynb) | Cross-modal projection & alignment | Projection layer → cross-modal fusion → contrastive learning → end-to-end training |

### 4. Retrieval-Augmented Generation ([`rag/`](./rag/README.md))

> A **one-command-runnable** RAG library covering vector retrieval and retrieval optimization. See [rag/README.md](./rag/README.md).

**Quick demo** (run from the repo root):

```bash
python -m rag.rag_algorithms_demo
```

| Sub-module | File | Key implementations |
| --- | --- | --- |
| Vector retrieval | [`cosine_dot_product_similarity.py`](./rag/vector_retrieval/cosine_dot_product_similarity.py) | `cosine_similarity` · `dot_product_similarity` |
| Vector retrieval | [`approximate_nearest_neighbor.py`](./rag/vector_retrieval/approximate_nearest_neighbor.py) | LSH ANN `LSHIndex` |
| Vector retrieval | [`hnsw_index.py`](./rag/vector_retrieval/hnsw_index.py) | HNSW index `HNSWIndex` |
| Vector retrieval | [`context_compression.py`](./rag/vector_retrieval/context_compression.py) | Context compression `ContextCompressor` · `MapReduceCompressor` |
| Retrieval optimization | [`hybrid_retrieval_sort.py`](./rag/retrieval_optimization/hybrid_retrieval_sort.py) | BM25 + vector hybrid sort `bm25_score` · `hybrid_retrieval_sort` |
| Retrieval optimization | [`query_rewrite_expansion.py`](./rag/retrieval_optimization/query_rewrite_expansion.py) | Query rewrite & expansion `QueryRewriter` |
| Retrieval optimization | [`hyde_algorithm.py`](./rag/retrieval_optimization/hyde_algorithm.py) | Hypothetical Document Embedding `HyDERetriever` |
| Retrieval optimization | [`retrieval_reranking.py`](./rag/retrieval_optimization/retrieval_reranking.py) | Reranking (BM25 / contextual / RRF) `RetrievalReranker` |

### 5. Mixture of Experts ([`moe/`](./moe/README.md))

> Understand MoE with the smallest readable code: experts, routing, sparse activation and load balancing.

| File | What you'll learn | Key implementations |
| --- | --- | --- |
| [`moe.ipynb`](./moe/moe.ipynb) | From a single expert to a Sparse MoE usable in large-model training | `BasicExpert` · `BasicMOE` · `MOERouter` (Top-K gating) · `MOEConfig` · `SparseMOE` |

<div align="center"><img src="moe/images/moe_base.png" alt="MoE basic structure" width="420"/></div>

### 6. RL Alignment ([`rl/`](./rl/README.md))

> The two most common policy-optimization losses for RLHF / LLM alignment — **self-contained and runnable** (only `torch` + `numpy`).

| File | What you'll learn | Key implementations |
| --- | --- | --- |
| [`ppo_grpo_loss.ipynb`](./rl/ppo_grpo_loss.ipynb) | The math and minimal runnable implementations of PPO and GRPO losses, plus toy training steps | `compute_gae` (GAE advantage) · `ppo_loss` (clipped objective + value + entropy) · `grpo_group_advantages` (group normalization) · `grpo_loss` (no Critic + k3 KL penalty) |

**Key idea**: PPO estimates the advantage with GAE + a Critic; GRPO drops the Critic and uses the group-relative reward of "a group of responses to the same prompt" as the advantage, with an explicit per-token KL penalty (the DeepSeekMath / DeepSeek-R1 route).

### 7. Agent Core Algorithms ([`agent/`](./agent/README.md))

> Core reasoning-and-planning paradigms for agents — **self-contained, runs on the standard library only**: a "tool hack (local dict + safe arithmetic) + mock LLM (hand-written rule policy)" makes the control flow run end-to-end with no real LLM or network.

| File | What you'll learn | Key implementations |
| --- | --- | --- |
| [`react_agent.ipynb`](./agent/react_agent.ipynb) | ReAct / CoT / self-reflection / Tree-of-Thought and tool-call parsing | `search` · `calculator` (hacked tools) · `parse_action` (bracket / JSON tool-call parsing) · `react_agent` (hand-written ReAct loop) · `reflexion_loop` (reflect & retry) · `tot_solve_24` (Tree-of-Thought search on the 24 game) |

**Key idea**: CoT writes out intermediate steps; ReAct alternates `Thought → Action → Observation`; Reflexion reflects after a failure and retries; ToT generates branches on a thought tree, evaluates and searches. To use a real LLM, just replace the mock policy and tools with real implementations.

---

## Roadmap 🗺️

The following items are planned but not yet implemented — contributions welcome (🚧). We'd like every item to ultimately map to **a self-contained, runnable script or Notebook**.

<details>
<summary><b>Transformer (advanced)</b></summary>

- Positional encoding: Sinusoidal / Learnable PE / Relative PE / RoPE / ALiBi
- Normalization: RMSNorm, GroupNorm, Pre-LN vs Post-LN stability
- Activation & FFN: GELU / SiLU, SwiGLU, GLU variants
- More tokenization: WordPiece, Byte-level BPE (BBPE)

</details>

<details>
<summary><b>Training & Optimization</b></summary>

- Pre-training objectives: CLM / MLM / PrefixLM / DAE
- Sequence generation: Greedy / Beam Search / Nucleus / Typical Sampling / MCTS
- Optimizers & schedules: AdamW, Lion, warm-up + cosine decay, gradient accumulation & clipping, AMP

</details>

<details>
<summary><b>Efficient Inference & Deployment</b></summary>

- Inference: KV Cache, Continuous Batching, Paged Attention, Speculative Decoding
- Quantization: INT8/INT4/NF4, ZeroQuant, GPTQ, AWQ, QLoRA

</details>

<details>
<summary><b>Long-Sequence Processing</b></summary>

- Long context: Position Interpolation, Sliding Window Attention, Longformer sparse attention, Mamba
- Memory augmentation: external memory retrieval, GateLoop, RWKV, StreamingLLM

</details>

<details>
<summary><b>Agent & RL</b></summary>

- Agent (advanced): multi-agent collaboration, long-term memory, ReWOO, Plan-and-Execute, real tool/function-calling integration
- RL / alignment: reward model training, DPO, preference contrastive learning, alignment-tax measurement

</details>

## Contribution Guide 👏

Contributions of any kind are welcome! The preferred form is **a self-contained, one-command-runnable entry file or Notebook**:

1. Fork the repo and create a branch;
2. Add your implementation under the matching module directory (Notebook or script), keeping the "principle first, then code, runnable" style and **no dependency on other scripts**;
3. Use lowercase + underscore (snake_case) for file/directory names;
4. Add your entry and status to the [Module Navigation](#module-navigation-) table and open a PR describing the idea and how to run it.

## Reference Resources 🪐

This section is continuously improved and aims to share high-value learning resources:

- Papers on algorithm principles
- References for excellent implementations
- Recommended learning paths

## License 😄

This project is licensed under the [Apache License 2.0](./LICENSE).

## Star History ✨

[![Star History Chart](https://api.star-history.com/svg?repos=aJupyter/ThinkLLM&type=Date)](https://star-history.com/#aJupyter/ThinkLLM&Date)

## Contributors

[![ThinkLLM contributors](https://contrib.rocks/image?repo=aJupyter/ThinkLLM&max=50)](https://github.com/aJupyter/ThinkLLM/graphs/contributors)
