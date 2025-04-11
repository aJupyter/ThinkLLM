# 检索增强生成（RAG）算法库

本目录包含一系列针对检索增强生成（Retrieval-Augmented Generation, RAG）的算法实现，旨在提高大型语言模型生成内容的准确性、相关性和可靠性。

## 目录结构

- `vector_retrival/`: 向量检索相关算法
  - `cosine_dot_product_similarity.py`: 余弦相似度和点积相似度计算
  - `approximate_nearest_neighbor.py`: 近似最近邻（LSH）算法实现
  - `hnsw_index.py`: 层次化可导航小世界（HNSW）索引实现
  - `context_compression.py`: 上下文压缩和信息保留算法

- `retrival_optimization/`: 检索优化相关算法
  - `hybrid_retrieval_sort.py`: 混合检索排序算法
  - `query_rewrite_expansion.py`: 查询重写与扩展算法
  - `hyde_algorithm.py`: 假设性文档嵌入（HyDE）算法
  - `retrieval_reranking.py`: 检索结果重排序与打分机制

- `rag_algorithms_demo.py`: 演示所有RAG算法的主程序

## 主要功能

### 1. 向量检索

- **相似度计算**：实现了余弦相似度和点积相似度两种计算方式，适用于不同的应用场景。
- **近似最近邻算法**：包含LSH和HNSW两种高效的ANN算法，大幅提升大规模向量检索的效率。
- **上下文压缩**：提供多种上下文压缩算法，在保留关键信息的同时减少输入令牌数。

### 2. 检索优化

- **混合检索排序**：结合语义检索和关键词匹配的优势，实现更精准的检索排序。
- **查询重写与扩展**：通过同义词替换和相关术语扩展，提高查询的覆盖范围。
- **HyDE算法**：利用假设性文档嵌入技术，增强语义检索的效果。
- **重排序机制**：支持BM25重排序、上下文相关重排序以及倒数排名融合等多种重排序策略。

## 使用方法

### 快速开始

运行演示脚本以查看所有算法的效果：

```bash
python rag_algorithms_demo.py
```

### 自定义使用

以下是使用本库中各组件的代码示例：

#### 向量检索示例

```python
from RAG.vector_retrival.cosine_dot_product_similarity import cosine_similarity
from RAG.vector_retrival.approximate_nearest_neighbor import LSHIndex

# 计算向量相似度
similarity = cosine_similarity(query_vector, document_vector)

# 使用LSH索引进行近似最近邻搜索
lsh_index = LSHIndex(dim=64, num_tables=10, num_bits=16)
lsh_index.index(vectors)
results = lsh_index.query(query_vector, k=5)
```

#### 检索优化示例

```python
from RAG.retrival_optimization.query_rewrite_expansion import QueryRewriter
from RAG.retrival_optimization.retrieval_reranking import RetrievalReranker

# 查询重写与扩展
rewriter = QueryRewriter()
rewriter.load_synonyms(synonyms_dict)
rewritten_query, expanded_terms = rewriter.full_rewrite_and_expand(query)

# 检索结果重排序
reranker = RetrievalReranker()
reranked_results = reranker.bm25_rerank(query, documents, doc_ids, top_k=10)
```

#### 完整RAG流程

```python
# 请参考rag_algorithms_demo.py中的complete_rag_pipeline函数
from RAG.rag_algorithms_demo import complete_rag_pipeline
from RAG.vector_retrival.context_compression import ContextCompressor

# 初始化组件
embedder = SimpleEmbedder(embedding_dim=64)
documents = [...] # 您的文档集合

# 执行完整RAG流程
results = complete_rag_pipeline(query, documents, embedder)
```

## 性能优化

- 所有向量检索算法均支持大规模数据集，通过近似算法实现高效检索。
- 上下文压缩算法根据不同需求提供多种压缩策略，平衡压缩率和信息保留。
- 混合检索算法通过可调整的权重参数实现灵活的检索策略。
- BM25重排序支持中英文，针对中文文本使用jieba进行分词处理。
- 所有算法都经过优化，支持批量处理和缓存机制，提高效率。

## 注意事项

- 本库主要用于演示和教学目的，生产环境可能需要更多优化。
- 中文分词依赖jieba库，请确保已安装：`pip install jieba`。
- 可视化功能需要matplotlib库，请确保已安装：`pip install matplotlib`。
- 在Windows环境下使用时，请注意文件编码问题，保存文件时使用UTF-8编码。

## 参考文献

- Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering.
- Gao, L., et al. (2023). Precise Zero-Shot Dense Retrieval without Relevance Labels.
- Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
- Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.
- Robertson, S. E., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. 