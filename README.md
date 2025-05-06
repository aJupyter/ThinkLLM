# 🚀ThinkLLM：大语言模型算法与组件实现

<div align="center">
    简体中文| <a href="README_EN.md" >English</a>
</div>



## 项目简介🌟

ThinkLLM是一个专注于大语言模型核心算法实现的开源项目。本仓库包含了各种LLM、MLLM、RAG、Agent、LRM、RL和MoE的关键算法和组件的Python实现，帮助开发者和研究者通过具体代码深入理解大模型的底层机制。每个算法实现都保持简洁明确，便于学习和二次开发。

如果对大模型全栈实践感兴趣，可以参考完全开源的[EmoLLM](https://github.com/SmartFlowAI/EmoLLM)。

<div align="center">
  <img src="images/logo.png" alt="ThinkLLM Logo" width="200"/>
  <br/>
  <p>🚀 轻量、高效的大语言模型算法实现</p>

  [![GitHub stars](https://img.shields.io/github/stars/aJupyter/ThinkLLM?style=flat-square)](https://github.com/aJupyter/ThinkLLM/stargazers)
  [![GitHub forks](https://img.shields.io/github/forks/aJupyter/ThinkLLM?style=flat-square)](https://github.com/aJupyter/ThinkLLM/network)
  [![GitHub issues](https://img.shields.io/github/issues/aJupyter/ThinkLLM?style=flat-square)](https://github.com/aJupyter/ThinkLLM/issues)
  [![GitHub license](https://img.shields.io/github/license/aJupyter/ThinkLLM?style=flat-square)](https://github.com/aJupyter/ThinkLLM/blob/main/LICENSE)
  [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/aJupyter/ThinkLLM/pulls)
</div>

## 更新🔥

更多、更好的内容在路上！

- [2025.5] 可以使用[deepwiki](https://deepwiki.com/aJupyter/ThinkLLM)辅助理解该项目
- [2025.4] 更新[RAG](./RAG)、[BPE](./LLM/BPE.ipynb)
- [2025.3] 更新[MHA/GQA/MQA](./transformer_component/MHA_GQA_MQA_MLA.ipynb)、[VIT](./multimodal/ViT.ipynb)

## 项目大纲📖

### 1. Transformer核心算法

- **注意力机制算法**
  - 经典自注意力机制（Self-Attention）实现
  - 多头注意力（MHA）的前向与反向传播算法
  - 分组查询注意力（GQA）的高效实现
  - 多查询注意力（MQA）的省内存算法
  - 混合线性注意力（MLA）计算与优化
  - FlashAttention算法实现与性能分析
- **位置编码算法**
  - 正弦余弦位置编码（Sinusoidal）实现
  - 可学习位置编码（Learnable PE）训练算法
  - 相对位置编码（Relative PE）计算方法
  - 旋转位置编码（RoPE）的数学原理与向量变换算法
  - ALiBi位置偏置算法实现
- **归一化层算法**
  - LayerNorm的前向与反向传播实现
  - RMSNorm算法与计算优化
  - GroupNorm在Transformer中的应用
  - 归一化位置（Pre/Post-LN）对训练稳定性的影响
- **激活函数与前馈网络**
  - GELU/SiLU激活函数实现
  - SwiGLU门控变换算法
  - GLU（门控线性单元）及其变体的实现
  - MoE-FFN（专家混合前馈网络）的条件计算

### 2. 模型训练与优化算法

- **预训练算法**
  - 因果语言模型（CLM）预训练目标实现
  - 掩码语言模型（MLM）训练算法
  - 前缀语言模型（PrefixLM）实现
  - 去噪自编码器（DAE）训练策略
- **序列生成算法**
  - 贪婪解码（Greedy Decoding）实现
  - 束搜索（Beam Search）算法
  - 核采样（Nucleus Sampling）和温度采样算法
  - 典型相关性采样（Typical Sampling）实现
  - MCTS（蒙特卡洛树搜索）在语言生成中的应用
- **优化算法**
  - AdamW优化器实现与权重衰减分离
  - Lion优化器算法实现
  - 学习率预热与余弦衰减策略
  - 梯度累积与梯度裁剪实现
  - 混合精度训练（AMP）算法

### 3. 高效推理与部署算法

- **推理优化算法**
  - KV缓存实现与管理策略
  - 连续批处理（Continuous Batching）算法
  - 推理阶段激活值量化方法
  - 页面注意力（Paged Attention）内存管理算法
  - Speculative Decoding推理加速技术
- **量化算法**
  - 权重量化算法（INT8/INT4/NF4）
  - ZeroQuant量化算法实现
  - GPTQ量化过程与优化
  - AWQ（感知激活量化）算法
  - QLoRA量化微调算法

### 4. 长序列处理算法

- **长上下文技术**
  - 位置插值（Position Interpolation）算法
  - Sliding Window Attention实现
  - Longformer式稀疏注意力算法
  - 递归状态空间模型（Mamba）核心算法
  - 高效Recomputation策略算法
- **记忆增强机制**
  - 外部记忆检索算法
  - GateLoop记忆增强循环机制
  - RWKV线性注意力算法实现
  - StreamingLLM无限上下文算法

### 5. 多模态算法

- **视觉编码算法**
  - ViT（Vision Transformer）基础算法实现
  - CLIP视觉编码器前向传播算法
  - 图像特征提取与映射算法
  - 视觉分割与特征融合技术
- **跨模态融合算法**
  - 投影层设计与实现
  - 跨模态注意力计算方法
  - 对齐空间构建算法
  - 视觉-语言表征对齐方法

### 6. 检索增强生成(RAG)算法

- **向量检索算法**
  - 余弦相似度与点积相似度计算
  - 近似最近邻（ANN）快速检索算法
  - HNSW索引构建与查询算法
  - 混合检索排序算法实现
- **检索优化算法**
  - 查询重写与扩展算法
  - HyDE（假设性文档嵌入）算法
  - 上下文压缩与信息保留算法
  - 检索结果重排序与打分机制

### 7. Agent与规划算法

- **推理与规划算法**
  - ReAct框架核心算法实现
  - 思维链（Chain-of-Thought）引导算法
  - 自我反思与修正算法
  - 思维树（Tree-of-Thought）搜索算法
- **工具使用算法**
  - 工具调用解析与参数提取算法
  - 输出格式控制算法
  - 工具结果整合与后续推理算法
  - 循环工具调用与终止条件

### 8. 强化学习(RL)与人类反馈

- **基于策略优化的算法**
  - PPO（近端策略优化）在LLM中的实现
  - 奖励模型训练算法
  - KL惩罚项计算与应用
  - DPO（直接偏好优化）算法实现
- **基于人类反馈的算法**
  - RLHF数据处理与训练算法
  - 偏好对比学习算法
  - 人类偏好建模与排序学习
  - 对齐税（Alignment Tax）测量与优化

### 9. 混合专家模型(MoE)算法

- **路由算法**
  - Top-K门控机制实现
  - 基于Hash的专家分配算法
  - 软路由与硬路由算法对比
  - 负载均衡路由算法
- **专家系统算法**
  - 专家并行训练算法
  - 专家选择与组合算法
  - 条件计算与激活稀疏性
  - 专家参数共享与更新策略

## 使用指南💡

- 找到感兴趣的文件夹阅读学习即可，每个文件夹都具有相应的依赖。

## 贡献指南👏

- 我们欢迎任何的贡献，具体形式最好是一个可一键运行的入口文件或者notebook。

## 参考资源🪐

该部分待完善，旨在分享一些高价值学习资料。
- 算法原理相关论文
- 优秀实现参考
- 推荐学习路径

## 许可证😄
本项目采用[Apache License](./LICENSE)。
