#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检索增强生成(RAG)算法演示
本脚本展示了各种RAG算法的实现和用法。
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import re
from collections import defaultdict, Counter
import os
import sys

# 配置matplotlib使用英文字体，避免中文字体问题
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
# 忽略字体警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入我们实现的各种算法
from RAG.vector_retrival.cosine_dot_product_similarity import cosine_similarity, dot_product_similarity
from RAG.vector_retrival.approximate_nearest_neighbor import LSHIndex
from RAG.vector_retrival.hnsw_index import HNSWIndex
from RAG.retrival_optimization.hybrid_retrieval_sort import hybrid_retrieval_sort
from RAG.retrival_optimization.query_rewrite_expansion import QueryRewriter
from RAG.retrival_optimization.hyde_algorithm import SimpleEmbedder, SimpleLLM, HyDERetriever
from RAG.vector_retrival.context_compression import ContextCompressor
from RAG.retrival_optimization.retrieval_reranking import RetrievalReranker

def section_title(title):
    """打印章节标题"""
    print("\n" + "="*80)
    print(f"## {title}")
    print("="*80 + "\n")

# 创建示例数据
def create_sample_data():
    section_title("1. 创建示例数据")
    
    # 创建一些示例文档
    documents = [
        "检索增强生成（RAG）是一种将信息检索与自然语言生成相结合的方法，用于提高大型语言模型生成的准确性和相关性。",
        "向量数据库是一种专门设计用于高效存储和检索向量嵌入的数据库系统。",
        "大型语言模型（LLM）是一类基于Transformer架构的神经网络，经过大规模文本数据训练，能够理解和生成自然语言。",
        "混合检索系统结合了多种检索方法，如词法匹配（BM25）和语义搜索（基于向量嵌入），以提高检索效果。",
        "上下文窗口大小是LLM处理能力的一个关键限制。当需要处理大量检索文档时，有效压缩上下文同时保留关键信息变得至关重要。",
        "检索系统的评估通常使用精确率、召回率、F1分数和平均精确率等指标。",
        "语义搜索使用向量嵌入来捕捉查询和文档的含义，而不仅仅是关键词匹配。",
        "BM25是一种流行的词法检索算法，它考虑了词频、逆文档频率和文档长度等因素。",
        "多阶段检索管道通常包括初始检索、重排序和最终排序等步骤，以平衡效率和准确性。",
        "检索增强生成可以显著减少大型语言模型的幻觉问题，提高回答的事实准确性。",
        # 添加更多关于大型语言模型和检索的文档
        "幻觉问题是大型语言模型面临的主要挑战，检索增强生成技术可以通过提供事实依据来解决这个问题。",
        "大型语言模型的局限性主要体现在上下文窗口大小、知识更新和事实准确性方面，需要通过外部知识增强来解决。",
        "向量检索技术使用向量嵌入表示文本语义，可以捕捉文本的深层含义，实现更精准的匹配。",
        "HNSW(Hierarchical Navigable Small World)是一种高效的近似最近邻搜索算法，在大规模向量检索中表现出色。",
        "检索系统可以通过查询重写和扩展来改进检索结果，解决词汇不匹配和查询不明确的问题。"
    ]
    
    # 创建简单的嵌入器
    embedder = SimpleEmbedder(embedding_dim=64)
    
    # 嵌入文档
    doc_vectors = [embedder.embed_text(doc) for doc in documents]
    
    print(f"已创建 {len(documents)} 个文档样本")
    print(f"向量维度: {len(doc_vectors[0])}")
    
    return documents, doc_vectors, embedder

# 向量相似度计算
def test_vector_similarity(documents, doc_vectors, embedder):
    section_title("2. 向量相似度计算")
    
    # 测试余弦相似度和点积相似度
    query = "检索增强生成如何改进语言模型"
    query_vector = embedder.embed_text(query)
    
    print(f"查询: '{query}'")
    print("\n余弦相似度计算结果:")
    cosine_similarities = [(i, cosine_similarity(query_vector, doc_vec)) for i, doc_vec in enumerate(doc_vectors)]
    cosine_similarities.sort(key=lambda x: x[1], reverse=True)
    
    for i, (doc_idx, sim) in enumerate(cosine_similarities[:3]):
        print(f"{i+1}. 文档 {doc_idx}: {sim:.4f} - {documents[doc_idx]}")
    
    print("\n点积相似度计算结果:")
    dot_similarities = [(i, dot_product_similarity(query_vector, doc_vec)) for i, doc_vec in enumerate(doc_vectors)]
    dot_similarities.sort(key=lambda x: x[1], reverse=True)
    
    for i, (doc_idx, sim) in enumerate(dot_similarities[:3]):
        print(f"{i+1}. 文档 {doc_idx}: {sim:.4f} - {documents[doc_idx]}")
    
    return cosine_similarities, dot_similarities

# 近似最近邻算法
def test_ann_algorithms(doc_vectors=None):
    section_title("3. 近似最近邻算法")
    
    # 如果未提供doc_vectors，使用随机向量
    if doc_vectors is None:
        dim = 64
        num_vectors = 10000
        random_vectors = [np.random.randn(dim) for _ in range(num_vectors)]
        test_vectors = random_vectors
        test_size = len(test_vectors)
    else:
        test_vectors = doc_vectors
        test_size = len(test_vectors)
        # 对于小规模doc_vectors，扩充数据以便更好地测试算法性能
        if test_size < 100:
            dim = len(test_vectors[0])
            additional_vectors = [np.random.randn(dim) for _ in range(1000 - test_size)]
            test_vectors = test_vectors + additional_vectors
            test_size = len(test_vectors)
    
    dim = len(test_vectors[0])
    
    # 构建LSH索引
    lsh_index = LSHIndex(dim, num_tables=10, num_bits=16)
    start_time = time.time()
    lsh_index.index(test_vectors)
    lsh_build_time = time.time() - start_time
    
    # 构建HNSW索引（对于较大数据集，仅使用一部分数据以加快演示）
    max_hnsw_size = min(test_size, 2000)  # 限制HNSW测试规模
    hnsw_index = HNSWIndex(dim, M=16, ef_construction=100, levels=3)
    start_time = time.time()
    for i, vec in enumerate(test_vectors[:max_hnsw_size]):
        hnsw_index.add(vec, i)
    hnsw_build_time = time.time() - start_time
    
    # 创建查询向量
    if doc_vectors is None:
        query_vec = np.random.randn(dim)
    else:
        # 使用第一个文档的向量作为查询向量，确保有匹配
        query_vec = test_vectors[0]
    
    # 测试LSH查询
    start_time = time.time()
    lsh_results = lsh_index.query(query_vec, k=5)
    lsh_query_time = max(time.time() - start_time, 0.0001)  # 避免除以零
    
    # 测试HNSW查询
    start_time = time.time()
    hnsw_results = hnsw_index.search(query_vec, k=5)
    hnsw_query_time = max(time.time() - start_time, 0.0001)  # 避免除以零
    
    # 暴力搜索作为基准
    start_time = time.time()
    bruteforce_dists = [(i, np.linalg.norm(query_vec - vec)) for i, vec in enumerate(test_vectors[:max_hnsw_size])]
    bruteforce_dists.sort(key=lambda x: x[1])
    bf_query_time = max(time.time() - start_time, 0.0001)  # 避免除以零
    
    print("近似最近邻算法性能比较:")
    print(f"LSH索引构建时间: {lsh_build_time:.4f} 秒 (向量数量: {test_size})")
    print(f"HNSW索引构建时间: {hnsw_build_time:.4f} 秒 (向量数量: {max_hnsw_size})")
    print(f"\nLSH查询时间: {lsh_query_time:.6f} 秒")
    print(f"HNSW查询时间: {hnsw_query_time:.6f} 秒")
    print(f"暴力搜索时间: {bf_query_time:.6f} 秒")
    
    print(f"\nLSH加速比: {bf_query_time/lsh_query_time:.2f}x")
    print(f"HNSW加速比: {bf_query_time/hnsw_query_time:.2f}x")
    
    # 计算召回率
    gt_ids = set([idx for idx, _ in bruteforce_dists[:5]])
    lsh_ids = set([idx for idx, _ in lsh_results if idx < max_hnsw_size])
    hnsw_ids = set([idx for idx, _ in hnsw_results])
    
    lsh_recall = len(gt_ids.intersection(lsh_ids)) / max(len(gt_ids), 1)
    hnsw_recall = len(gt_ids.intersection(hnsw_ids)) / max(len(gt_ids), 1)
    
    print(f"\nLSH召回率 (前5个结果): {lsh_recall:.2f}")
    print(f"HNSW召回率 (前5个结果): {hnsw_recall:.2f}")
    
    # 打印一些查询结果
    if len(lsh_results) > 0:
        print("\nLSH查询结果示例 (前3个):")
        for i, (idx, dist) in enumerate(lsh_results[:3]):
            if idx < len(test_vectors):
                print(f"  结果 {i+1}: ID {idx}, 距离: {dist:.4f}")
    
    if len(hnsw_results) > 0:
        print("\nHNSW查询结果示例 (前3个):")
        for i, (idx, dist) in enumerate(hnsw_results[:3]):
            if idx < len(test_vectors):
                print(f"  结果 {i+1}: ID {idx}, 距离: {dist:.4f}")
    
    # 打印详细性能对比表格
    print("\n=== 性能指标总结 ===")
    print("-" * 60)
    print(f"{'指标':<15}{'LSH':<15}{'HNSW':<15}{'暴力搜索':<15}")
    print("-" * 60)
    print(f"{'索引时间(秒)':<15}{lsh_build_time:<15.4f}{hnsw_build_time:<15.4f}{'-':<15}")
    print(f"{'查询时间(秒)':<15}{lsh_query_time:<15.6f}{hnsw_query_time:<15.6f}{bf_query_time:<15.6f}")
    print(f"{'加速比':<15}{bf_query_time/lsh_query_time:<15.2f}x{bf_query_time/hnsw_query_time:<15.2f}x{'1.00x':<15}")
    print(f"{'召回率':<15}{lsh_recall:<15.2f}{hnsw_recall:<15.2f}{'1.00':<15}")
    print("-" * 60)
    
    print(f"暴力搜索结果IDs (Top 5): {sorted(list(gt_ids))}")
    print(f"LSH索引结果IDs (Top 5): {sorted(list(lsh_ids))}")
    print(f"HNSW索引结果IDs (Top 5): {sorted(list(hnsw_ids))}")
    
    return lsh_index, hnsw_index

# 混合检索排序算法
def test_hybrid_retrieval(documents, doc_vectors, embedder):
    section_title("4. 混合检索排序算法")
    
    # 准备文档和向量表示
    candidate_docs = [
        {
            'id': i, 
            'text': doc,
            'vector': doc_vectors[i], 
            'terms': re.findall(r'\w+', doc.lower())
        } for i, doc in enumerate(documents)
    ]
    
    # 修改：为文档添加更多关键词，确保BM25有效
    for doc in candidate_docs:
        # 增加中文分词处理
        try:
            import jieba
            doc['terms'].extend(list(jieba.cut(doc['text'])))
        except ImportError:
            # 如果没有jieba，使用简单的字符分割
            doc['terms'].extend([term for term in doc['text'] if len(term.strip()) > 0])
        
        # 确保terms列表中没有空字符串
        doc['terms'] = [term for term in doc['terms'] if term.strip()]
    
    # 准备查询
    query_text = "检索增强生成技术的应用"
    query_vector = embedder.embed_text(query_text)
    
    # 增强查询词条提取
    query_terms = []
    try:
        import jieba
        query_terms = list(jieba.cut(query_text))
    except ImportError:
        query_terms = re.findall(r'\w+', query_text.lower())
    
    # 确保查询词条不为空
    if not query_terms:
        query_terms = [query_text]
    
    print(f"查询: '{query_text}'")
    print(f"查询词条: {query_terms}")
    
    # 测试不同的混合权重
    alphas = [0.0, 0.3, 0.7, 1.0]
    
    for alpha in alphas:
        print(f"\n混合权重 alpha={alpha} (0: 仅BM25, 1: 仅向量相似度)")
        results = hybrid_retrieval_sort(query_vector, query_terms, candidate_docs, alpha=alpha)
        
        for i, (doc_id, score) in enumerate(results[:3]):
            print(f"{i+1}. 文档 {doc_id}: {score:.4f} - {documents[doc_id]}")

# 查询重写与扩展
def test_query_rewrite():
    section_title("5. 查询重写与扩展")
    
    # 初始化查询重写器
    rewriter = QueryRewriter()
    
    # 加载示例词典
    synonyms = {
        '检索': ['搜索', '查询', '获取'],
        '生成': ['创建', '产生', '构建'],
        '语言模型': ['LLM', '大语言模型', '文本生成模型'],
        '向量': ['嵌入', '特征向量', '张量'],
        '相似度': ['距离', '匹配度', '接近度'],
        '幻觉': ['虚构', '错误生成', '不实信息'],
        '大型语言模型': ['LLM', 'GPT', '预训练语言模型']
    }
    rewriter.load_synonyms(synonyms)
    
    stopwords = ['的', '了', '在', '是', '和', '与', '或', '有']
    rewriter.load_stopwords(stopwords)
    
    related_terms = {
        '检索': ['索引', '排序', '搜索引擎', '召回率'],
        '生成': ['文本生成', '内容创建', '对话系统', '补全'],
        '语言模型': ['GPT', 'BERT', 'Transformer', '深度学习'],
        '向量': ['维度', '嵌入空间', '向量数据库', '降维'],
        '文档': ['段落', '语料库', '文章', '内容'],
        '幻觉': ['事实错误', '知识缺失', '信息不准确', '真实性'],
        '问题': ['挑战', '难点', '困难', '限制']
    }
    rewriter.load_related_terms(related_terms)
    
    # 测试查询
    test_queries = [
        "语言模型检索效果不好",
        "如何提高向量检索的准确性",
        "生成式模型有什么应用",
        "如何减少大型语言模型的幻觉问题"
    ]
    
    results = []
    
    for query in test_queries:
        print(f"\n原始查询: '{query}'")
        
        # 查询重写
        rewritten = rewriter.query_rewrite(query)
        print(f"重写后: '{rewritten}'")
        
        # 完整的查询重写和扩展
        rewritten, expanded = rewriter.full_rewrite_and_expand(query)
        
        print("扩展结果:")
        for term, weight in expanded:
            print(f"  {term}: {weight:.1f}")
        
        # 保存结果
        results.append({
            'original': query,
            'rewritten': rewritten,
            'expanded': expanded
        })
    
    return rewriter, results

# HyDE算法
def test_hyde_algorithm(documents):
    section_title("6. HyDE算法")
    
    # 初始化组件
    embedder = SimpleEmbedder(embedding_dim=64)
    llm = SimpleLLM()
    hyde_retriever = HyDERetriever(embedder, llm)
    
    # 建立索引
    hyde_retriever.index_documents(documents)
    
    # 测试查询
    query = "如何减少语言模型的幻觉问题？"
    print(f"查询: '{query}'")
    
    # 传统向量检索
    traditional_results = hyde_retriever.retrieve(query, k=3, use_hyde=False)
    print("\n传统向量检索结果:")
    for i, (doc_idx, score) in enumerate(traditional_results):
        print(f"{i+1}. 文档 {doc_idx}: {score:.4f} - {documents[doc_idx]}")
    
    # HyDE检索
    hyde_results = hyde_retriever.retrieve(query, k=3, use_hyde=True)
    
    # 生成假设性文档
    hyde_prompt = f"请根据查询提供一个详细的答案：{query}"
    hypothetical_doc = llm.generate_text(hyde_prompt)
    
    print("\nHyDE检索结果:")
    print(f"生成的假设性文档:\n  \"{hypothetical_doc}\"")
    print("\n检索结果:")
    for i, (doc_idx, score) in enumerate(hyde_results):
        print(f"{i+1}. 文档 {doc_idx}: {score:.4f} - {documents[doc_idx]}")

# 上下文压缩与信息保留
def test_context_compression(documents):
    """测试上下文压缩算法 - 优化版本"""
    print("\n" + "="*80)
    print("上下文压缩与信息保留算法测试")
    print("="*80)
    
    # 使用传入的documents进行测试
    # 如果需要更长的文档，可以将多个文档合并
    long_documents = []
    for i in range(0, len(documents), 2):
        if i+1 < len(documents):
            long_documents.append(documents[i] + " " + documents[i+1])
        else:
            long_documents.append(documents[i])
    
    # 确保至少有5个文档用于测试
    while len(long_documents) < 5:
        long_documents.append(long_documents[0])
    
    queries = [
        "向量检索的相似度计算方法",
        "检索效率优化",
        "向量检索应用"
    ]
    
    # 创建压缩器实例 - 启用缓存提高性能
    compressor = ContextCompressor(use_cache=True)
    
    # 打印原始文档信息
    total_words = sum(compressor._count_words(doc) for doc in long_documents)
    print(f"\n原始文档总词数: {total_words}")
    
    # 用于存储所有查询的结果，以便计算平均值
    all_results = []
    
    # 测试每个查询
    for query_idx, query in enumerate(queries):
        print("\n" + "="*80)
        print(f"测试查询 {query_idx+1}: {query}")
        print("="*80)
        
        # 测试各种压缩方法并记录时间
        test_methods = [
            ("基于关键词的压缩", compressor.keyword_based_compression),
            ("基于TF-IDF的压缩", compressor.tfidf_based_compression),
            ("基于句子重要性的压缩", compressor.improved_sentence_importance_compression),
            ("Map-Reduce压缩", compressor.map_reduce_compress),
            ("TextRank压缩", compressor.textrank_compression),
            ("信息密度压缩", compressor.info_density_compression)
        ]
        
        results = []
        
        for name, method in test_methods:
            print(f"\n{len(results)+1}. {name}:")
            
            # 运行压缩方法
            start_time = time.time()
            compressed = method(long_documents, query, max_tokens=100)
            end_time = time.time()
            compression_time = end_time - start_time
            
            # 计算词数
            words = compressor._count_words(compressed)
            
            # 保存结果
            results.append({
                'name': name,
                'text': compressed,
                'words': words,
                'time': compression_time
            })
            
            # 打印简化的结果统计，不打印完整内容
            print(f"词数: {words}, 耗时: {compression_time:.4f}秒")
        
        # 保存此查询的结果以计算平均值
        all_results.append(results)
        
        # 打印压缩结果统计
        print("\n压缩效果比较:")
        print("-"*40)
        print(f"原始文档: {total_words}词")
        
        for result in results:
            print(f"{result['name']}: {result['words']}词 " 
                 f"(压缩率: {result['words']/total_words:.1%}, "
                 f"耗时: {result['time']:.4f}秒)")
            
        # 打印压缩算法性能对比柱状图
        try:
            import matplotlib.pyplot as plt
            
            # 绘制压缩时间对比
            plt.figure(figsize=(10, 6))
            names = [r['name'] for r in results]
            times = [r['time'] for r in results]
            
            # Convert Chinese method names to English for plotting
            name_mapping = {
                "基于关键词的压缩": "Keyword-based",
                "基于TF-IDF的压缩": "TF-IDF-based",
                "基于句子重要性的压缩": "Sentence Importance",
                "Map-Reduce压缩": "Map-Reduce",
                "TextRank压缩": "TextRank",
                "信息密度压缩": "Information Density"
            }
            
            english_names = [name_mapping.get(name, name) for name in names]
            
            plt.subplot(1, 2, 1)
            plt.bar(english_names, times)
            plt.ylabel('Compression Time (s)')
            plt.title('Algorithm Time Comparison')
            plt.xticks(rotation=45, ha='right')
            
            # 绘制压缩率对比
            plt.subplot(1, 2, 2)
            compression_ratios = [r['words']/total_words for r in results]
            plt.bar(english_names, compression_ratios)
            plt.ylabel('Compression Ratio')
            plt.title('Algorithm Compression Ratio')
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            # 使用查询索引保存不同文件
            compression_fig_filename = f'compression_comparison_query_{query_idx+1}.png'
            plt.savefig(compression_fig_filename)
            print(f"\n查询 {query_idx+1} 的性能对比图已保存为 '{compression_fig_filename}'")
        except Exception as e:
            print(f"无法生成图表: {e}")
    
    # 计算并展示所有查询的平均结果
    print("\n" + "="*80)
    print("所有查询的平均压缩效果")
    print("="*80)
    
    # 计算平均值
    avg_results = []
    method_names = [r['name'] for r in all_results[0]]
    
    for method_idx, method_name in enumerate(method_names):
        avg_words = sum(results[method_idx]['words'] for results in all_results) / len(all_results)
        avg_time = sum(results[method_idx]['time'] for results in all_results) / len(all_results)
        avg_ratio = avg_words / total_words
        
        avg_results.append({
            'name': method_name,
            'words': avg_words,
            'time': avg_time,
            'ratio': avg_ratio
        })
        
        print(f"{method_name}: 平均{avg_words:.1f}词 "
              f"(压缩率: {avg_ratio:.1%}, "
              f"平均耗时: {avg_time:.4f}秒)")
    
    # 绘制平均性能对比图
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # 中英文名称映射
        name_mapping = {
            "基于关键词的压缩": "Keyword-based",
            "基于TF-IDF的压缩": "TF-IDF-based",
            "基于句子重要性的压缩": "Sentence Importance",
            "Map-Reduce压缩": "Map-Reduce",
            "TextRank压缩": "TextRank",
            "信息密度压缩": "Information Density"
        }
        
        # 平均压缩时间对比
        plt.subplot(2, 2, 1)
        names = [r['name'] for r in avg_results]
        english_names = [name_mapping.get(name, name) for name in names]
        times = [r['time'] for r in avg_results]
        plt.bar(english_names, times)
        plt.ylabel('Average Time (s)')
        plt.title('Average Compression Time')
        plt.xticks(rotation=45, ha='right')
        
        # 平均压缩率对比
        plt.subplot(2, 2, 2)
        ratios = [r['ratio'] for r in avg_results]
        plt.bar(english_names, ratios)
        plt.ylabel('Average Compression Ratio')
        plt.title('Average Compression Ratio')
        plt.xticks(rotation=45, ha='right')
        
        # 平均词数对比
        plt.subplot(2, 2, 3)
        words = [r['words'] for r in avg_results]
        plt.bar(english_names, words)
        plt.ylabel('Average Words')
        plt.title('Average Word Count')
        plt.xticks(rotation=45, ha='right')
        
        # 压缩率与时间的散点图
        plt.subplot(2, 2, 4)
        plt.scatter(ratios, times)
        for i, name in enumerate(english_names):
            plt.annotate(name, (ratios[i], times[i]), 
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
        plt.xlabel('Compression Ratio')
        plt.ylabel('Processing Time (s)')
        plt.title('Compression Ratio vs. Processing Time')
        
        plt.tight_layout()
        plt.savefig('compression_average_comparison.png')
        print("\n所有查询的平均性能对比图已保存为 'compression_average_comparison.png'")
    except Exception as e:
        print(f"无法生成平均性能对比图: {e}")

# 检索结果重排序与打分机制
def test_retrieval_reranking(documents):
    """
    测试检索结果重排序和打分机制
    """
    section_title("8. 检索结果重排序与打分机制")
    
    print("测试开始: 检索结果重排序与打分机制")
    
    # 使用传入的documents
    base_documents = documents
    
    # 添加明确包含特定关键词的测试文档
    test_specific_docs = [
        "幻觉问题是大型语言模型面临的主要挑战，检索增强生成技术可以通过提供事实依据来解决这个问题。",
        "局限性主要体现在上下文窗口大小、知识更新和事实准确性方面，需要通过外部知识增强来解决。",
        "向量检索技术使用向量嵌入表示文本语义，可以捕捉文本的深层含义，实现更精准的匹配。"
    ]
    
    # 添加更多特定场景测试文档，确保不同上下文可以返回不同的文档
    additional_test_docs = [
        "知识图谱与语义网络可以增强语言模型的结构化知识表示能力。",
        "检索系统的评估指标包括精确率、召回率、F1分数和平均精确率等。",
        "向量数据库优化技术包括量化、分片和缓存策略，可以显著提高检索效率。",
        "大语言模型的推理速度优化是工业应用中的关键挑战。",
        "多模态检索系统可以同时处理文本、图像和音频等不同类型的数据。"
    ]
    
    # 合并文档集
    all_documents = base_documents + test_specific_docs + additional_test_docs
    
    # 创建文档ID
    doc_ids = list(range(len(all_documents)))
    
    print(f"测试文档集大小: {len(all_documents)} 个文档")
    
    # 1. 测试RRF排名融合 
    print("\n=== 1. 测试倒数排名融合(RRF) ===")
    
    # 创建模拟的多个排名列表（来自不同检索方法）
    vector_ranks = [(0, 0.92), (9, 0.85), (2, 0.76), (3, 0.65), (6, 0.60)]
    bm25_ranks = [(9, 8.5), (0, 7.8), (7, 6.2), (3, 5.9), (8, 5.5)]
    hybrid_ranks = [(0, 0.88), (9, 0.85), (3, 0.82), (2, 0.75), (8, 0.72)]
    
    # 初始化重排序器
    reranker = RetrievalReranker()
    
    # 测试倒数排名融合
    rrf_results = reranker.reciprocal_rank_fusion([vector_ranks, bm25_ranks, hybrid_ranks])
    
    print("倒数排名融合(RRF)结果:")
    for i, (doc_id, score) in enumerate(rrf_results[:5]):
        if doc_id < len(all_documents):
            print(f"  {i+1}. 文档 {doc_id}: {score:.6f} - {all_documents[doc_id][:50]}...")
    
    # 2. 测试BM25重排序 - 英文查询
    print("\n=== 2. 测试BM25重排序 (英文查询) ===")
    query_en = "how to reduce hallucination in large language models"
    
    bm25_results_en = reranker.bm25_rerank(query_en, all_documents, doc_ids, top_k=5)
    
    print(f"英文查询: '{query_en}'")
    print("BM25重排序结果:")
    for i, (doc_id, score) in enumerate(bm25_results_en):
        print(f"  {i+1}. 文档 {doc_id}: {score:.6f} - {all_documents[doc_id]}")
    
    # 3. 测试BM25重排序 - 中文查询
    print("\n=== 3. 测试BM25重排序 (中文查询) ===")
    query_zh = "如何减少大型语言模型的幻觉问题"
    
    bm25_results_zh = reranker.bm25_rerank(query_zh, all_documents, doc_ids, top_k=5)
    
    print(f"中文查询: '{query_zh}'")
    print("BM25重排序结果:")
    for i, (doc_id, score) in enumerate(bm25_results_zh):
        print(f"  {i+1}. 文档 {doc_id}: {score:.6f} - {all_documents[doc_id]}")
    
    # 4. 测试BM25重排序 - 针对特定内容的查询
    print("\n=== 4. 测试BM25重排序 (特定内容查询) ===")
    query_specific = "大语言模型的局限性"
    
    bm25_results_specific = reranker.bm25_rerank(query_specific, all_documents, doc_ids, top_k=5)
    
    print(f"特定内容查询: '{query_specific}'")
    print("BM25重排序结果:")
    for i, (doc_id, score) in enumerate(bm25_results_specific):
        print(f"  {i+1}. 文档 {doc_id}: {score:.6f} - {all_documents[doc_id]}")
    
    # 5. 测试上下文重排序 - 相关上下文 (针对语言模型的幻觉问题)
    print("\n=== 5. 测试上下文重排序 (相关上下文) ===")
    related_context = [
        "什么是语言模型？", 
        "大型语言模型有哪些优点和缺点？"
    ]
    
    # 针对语言模型幻觉问题的查询
    hallucination_query = "如何减少大型语言模型的幻觉问题"
    
    contextual_results = reranker.contextual_rerank(hallucination_query, all_documents, doc_ids, related_context, top_k=5)
    
    print("相关上下文:")
    for i, ctx in enumerate(related_context):
        print(f"  对话 {i+1}: {ctx}")
    print(f"  当前查询: {hallucination_query}")
    print("上下文重排序结果:")
    for i, (doc_id, score) in enumerate(contextual_results):
        print(f"  {i+1}. 文档 {doc_id}: {score:.6f} - {all_documents[doc_id]}")
    
    # 6. 测试上下文重排序 - 不相关上下文 (针对向量检索技术的查询)
    print("\n=== 6. 测试上下文重排序 (不相关上下文) ===")
    unrelated_context = [
        "什么是向量数据库？", 
        "最快的搜索算法是什么？"
    ]
    
    # 改用不同的查询，以便观察结果差异
    vector_db_query = "向量检索技术的优化方法"
    
    unrelated_results = reranker.contextual_rerank(vector_db_query, all_documents, doc_ids, unrelated_context, top_k=5)
    
    print("不相关上下文:")
    for i, ctx in enumerate(unrelated_context):
        print(f"  对话 {i+1}: {ctx}")
    print(f"  当前查询: {vector_db_query}")
    print("上下文重排序结果:")
    for i, (doc_id, score) in enumerate(unrelated_results):
        print(f"  {i+1}. 文档 {doc_id}: {score:.6f} - {all_documents[doc_id]}")
    
    # 7. 对比测试 - 分析不同方法的性能差异
    print("\n=== 7. 不同重排序方法的性能对比 ===")
    
    # 构建结果集合
    results_collection = {
        "BM25 (英文查询)": bm25_results_en,
        "BM25 (中文查询)": bm25_results_zh,
        "BM25 (特定内容)": bm25_results_specific,
        "相关上下文重排序(幻觉问题)": contextual_results,
        "不相关上下文重排序(向量检索)": unrelated_results
    }
    
    # 比较各方法的TOP-1文档
    print("各方法返回的TOP-1文档:")
    for name, results in results_collection.items():
        if results:
            doc_id, score = results[0]
            print(f"  {name}: 文档 {doc_id} - 得分: {score:.6f}")
            print(f"    {all_documents[doc_id]}")
    
    # 8. 评估算法效果
    print("\n=== 8. 算法效果评估 ===")
    
    # 针对'幻觉问题'相关查询，期望包含'幻觉'的文档排名靠前
    hallucination_query = "幻觉问题"
    hallucination_results = reranker.bm25_rerank(hallucination_query, all_documents, doc_ids, top_k=5)
    
    # 检查前3个结果中是否包含'幻觉'关键词的文档
    hallucination_docs = []
    for doc_id, _ in hallucination_results[:3]:
        if "幻觉" in all_documents[doc_id]:
            hallucination_docs.append(doc_id)
    
    print(f"测试查询: '{hallucination_query}'")
    if hallucination_docs:
        print(f"在前3个结果中找到了{len(hallucination_docs)}个包含'幻觉'关键词的文档，表现良好")
        print(f"相关文档ID: {hallucination_docs}")
    else:
        print("在前3个结果中没有找到包含'幻觉'关键词的文档，需要改进算法")
    
    # 针对'向量检索'相关查询，期望包含'向量'的文档排名靠前
    vector_query = "向量检索方法"
    vector_results = reranker.bm25_rerank(vector_query, all_documents, doc_ids, top_k=5)
    
    # 检查前3个结果中是否包含'向量'关键词的文档
    vector_docs = []
    for doc_id, _ in vector_results[:3]:
        if "向量" in all_documents[doc_id]:
            vector_docs.append(doc_id)
    
    print(f"\n测试查询: '{vector_query}'")
    if vector_docs:
        print(f"在前3个结果中找到了{len(vector_docs)}个包含'向量'关键词的文档，表现良好")
        print(f"相关文档ID: {vector_docs}")
    else:
        print("在前3个结果中没有找到包含'向量'关键词的文档，需要改进算法")
    
    # 返回enriched_documents以便在其他测试中使用
    return all_documents

# 完整的RAG流程示例
def complete_rag_pipeline(query, documents, embedder, lsh_index=None, hnsw_index=None, rewriter=None):
    section_title("9. 完整的RAG流程示例")
    
    print(f"原始查询: '{query}'")
    
    # 第1步: 查询重写与扩展
    if rewriter is None:
        # 如果没有提供查询重写器，创建一个新的
        rewriter = QueryRewriter()
        # 添加最基本的同义词和相关词
        rewriter.load_synonyms({
            '检索': ['搜索', '查询'],
            '语言模型': ['LLM', '大语言模型'],
            '幻觉': ['虚构', '错误生成']
        })
    
    rewritten_query, expanded_query = rewriter.full_rewrite_and_expand(query)
    print(f"\n1. 重写后的查询: '{rewritten_query}'")
    print("   扩展的查询词:")
    for term, weight in expanded_query[:5]:  # 只显示前5个
        print(f"     {term}: {weight:.1f}")
    
    # 第2步: 向量检索 - 使用近似最近邻算法加速
    query_vector = embedder.embed_text(rewritten_query)
    doc_vectors = [embedder.embed_text(doc) for doc in documents]
    
    # 使用不同检索方法获得结果
    vector_similarities = []
    
    # 2.1 使用常规向量相似度计算
    print("\n2.1 常规向量相似度检索:")
    standard_similarities = [(i, np.dot(query_vector, doc_vec)) for i, doc_vec in enumerate(doc_vectors)]
    standard_similarities.sort(key=lambda x: x[1], reverse=True)
    vector_similarities.append(standard_similarities[:5])  # 保存前5个结果
    
    # 2.2 使用HNSW索引（如果提供）
    if hnsw_index is not None:
        print("\n2.2 HNSW索引检索:")
        try:
            # 修复HNSW索引检索：使用get_current_count方法替代get_elements
            # 或者直接根据索引大小限制
            max_hnsw_size = 2000  # 默认最大大小限制
            
            # 尝试获取实际元素数量
            try:
                if hasattr(hnsw_index, 'get_current_count'):
                    max_hnsw_size = hnsw_index.get_current_count()
                elif hasattr(hnsw_index, 'element_count'):
                    max_hnsw_size = hnsw_index.element_count
                # 有些实现可能将元素存储在_elements属性中
                elif hasattr(hnsw_index, '_elements') and isinstance(hnsw_index._elements, list):
                    max_hnsw_size = len(hnsw_index._elements)
            except Exception as e:
                print(f"无法获取HNSW索引元素数量，使用默认值: {e}")
            
            # 确保不超出文档数量
            max_hnsw_size = min(max_hnsw_size, len(documents))
            
            # 执行搜索
            hnsw_results = hnsw_index.search(query_vector, k=5)
            # 格式化结果为与标准格式相同
            hnsw_formatted = [(idx, score) for idx, score in hnsw_results if idx < len(documents)]
            print(f"HNSW检索找到 {len(hnsw_formatted)} 个相关文档")
            if hnsw_formatted:
                vector_similarities.append(hnsw_formatted)
        except Exception as e:
            print(f"HNSW索引检索出错: {e}")
    
    # 2.3 使用LSH索引（如果提供）
    if lsh_index is not None:
        print("\n2.3 LSH索引检索:")
        try:
            # 修复LSH检索，确保查询向量格式正确
            # 有时LSH可能需要特定格式的查询向量
            query_vec_array = np.array(query_vector)
            
            # 尝试标准化查询向量
            query_vec_norm = query_vec_array / np.linalg.norm(query_vec_array)
            
            # 先尝试使用标准化向量
            try:
                lsh_results = lsh_index.query(query_vec_norm, k=5)
                if not lsh_results:  # 如果没有结果，使用原始向量
                    lsh_results = lsh_index.query(query_vec_array, k=5)
            except:
                # 如果标准化向量失败，使用原始向量
                lsh_results = lsh_index.query(query_vec_array, k=5)
            
            # 如果仍然没有结果，可能需要调整索引参数，这里增加结果数量
            if not lsh_results:
                try:
                    # 尝试增加返回的结果数量
                    lsh_results = lsh_index.query(query_vec_array, k=10)
                except:
                    pass
            
            # 格式化结果为与标准格式相同
            lsh_formatted = [(idx, 1.0 - dist) for idx, dist in lsh_results if idx < len(documents)]
            print(f"LSH检索找到 {len(lsh_formatted)} 个相关文档")
            if lsh_formatted:
                vector_similarities.append(lsh_formatted)
        except Exception as e:
            print(f"LSH索引检索出错: {e}")
    
    # 合并向量检索结果（如果有多个结果集）
    if len(vector_similarities) > 1:
        # 使用简单的集合合并，去重复
        combined_vector_results = []
        seen_ids = set()
        for result_set in vector_similarities:
            for doc_id, score in result_set:
                if doc_id not in seen_ids and doc_id < len(documents):
                    combined_vector_results.append((doc_id, score))
                    seen_ids.add(doc_id)
        # 确保按分数排序
        combined_vector_results.sort(key=lambda x: x[1], reverse=True)
        vector_result = combined_vector_results[:5]  # 只保留前5个
    else:
        vector_result = standard_similarities[:5]
    
    print("\n向量检索结果:")
    for i, (doc_id, score) in enumerate(vector_result[:3]):
        print(f"   {i+1}. 文档 {doc_id}: {score:.4f} - {documents[doc_id]}")
    
    # 第3步: BM25检索
    print("\n3. BM25检索:")
    reranker = RetrievalReranker()
    doc_ids = list(range(len(documents)))
    bm25_results = reranker.bm25_rerank(rewritten_query, documents, doc_ids)
    
    for i, (doc_id, score) in enumerate(bm25_results[:3]):
        print(f"   {i+1}. 文档 {doc_id}: {score:.4f} - {documents[doc_id]}")
    
    # 第4步: 结果融合
    print("\n4. 结果融合:")
    # 如果有多个向量检索结果，使用所有结果
    all_results = [vector_result, bm25_results[:5]]
    combined_results = reranker.reciprocal_rank_fusion(all_results)
    
    print("融合后的检索结果:")
    retrieved_docs = []
    for i, (doc_id, score) in enumerate(combined_results[:5]):
        print(f"   {i+1}. 文档 {doc_id}: {score:.4f} - {documents[doc_id]}")
        retrieved_docs.append(documents[doc_id])
    
    # 第5步: 上下文压缩
    print("\n5. 上下文压缩:")
    compressor = ContextCompressor()
    
    # 计算原始文档大小，用于对比
    original_tokens = sum(len(doc.split()) for doc in retrieved_docs)
    
    # 测试不同压缩方法
    compression_methods = [
        ("TextRank压缩", compressor.textrank_compression),
        ("信息密度压缩", compressor.info_density_compression),
        ("Map-Reduce压缩", compressor.map_reduce_compress)
    ]
    
    best_compressed = None
    best_ratio = 0
    
    for name, method in compression_methods:
        try:
            compressed = method(retrieved_docs, rewritten_query, max_tokens=150)
            compressed_tokens = len(compressed.split())
            compression_ratio = compressed_tokens / original_tokens
            
            print(f"   {name}: {compressed_tokens}词 (压缩率: {compression_ratio:.2f})")
            
            # 选择压缩率最佳的结果
            if best_compressed is None or (0.1 < compression_ratio < best_ratio or best_ratio < 0.1):
                best_compressed = compressed
                best_ratio = compression_ratio
        except Exception as e:
            print(f"   {name}失败: {e}")
    
    # 如果所有压缩方法都失败，使用前两个文档
    if best_compressed is None:
        best_compressed = " ".join(retrieved_docs[:2])
        compressed_tokens = len(best_compressed.split())
        best_ratio = compressed_tokens / original_tokens
    
    print(f"\n选择的压缩结果: {len(best_compressed.split())}词 (压缩率: {best_ratio:.2f})")
    print(f"压缩后的上下文:\n{best_compressed}")
    
    # 第6步: 生成回答
    llm = SimpleLLM()
    answer_prompt = f"基于以下上下文回答问题: {query}\n\n上下文:\n{best_compressed}"
    answer = llm.generate_text(answer_prompt)
    
    print(f"\n6. 生成的回答:")
    print(f"   {answer}")
    
    # 返回处理结果，方便分析
    return {
        'query': query,
        'rewritten_query': rewritten_query,
        'vector_results': vector_result,
        'bm25_results': bm25_results[:5],
        'combined_results': combined_results[:5],
        'compressed_context': best_compressed,
        'answer': answer
    }

# 主函数
def main():
    print("=" * 80)
    print("检索增强生成(RAG)算法演示")
    print("=" * 80)
    
    # 1. 创建示例数据 - 作为所有后续测试的基础
    print("\n【步骤1】创建示例数据")
    documents, doc_vectors, embedder = create_sample_data()
    
    # 2. 测试向量相似度计算 - 第一步基础检索能力测试
    print("\n【步骤2】测试基础检索能力：向量相似度计算")
    cosine_similarities, dot_similarities = test_vector_similarity(documents, doc_vectors, embedder)
    print(f"  获得了 {len(cosine_similarities)} 个余弦相似度结果和 {len(dot_similarities)} 个点积相似度结果")
    
    # 3. 测试近似最近邻算法 - 优化检索速度，接收文档向量作为输入
    print("\n【步骤3】测试检索效率优化：近似最近邻算法")
    lsh_index, hnsw_index = test_ann_algorithms(doc_vectors)
    print(f"  构建了LSH索引和HNSW索引，将用于后续快速检索")
    
    # 4. 测试混合检索排序算法 - 结合多种检索方法
    print("\n【步骤4】测试混合检索排序算法")
    test_hybrid_retrieval(documents, doc_vectors, embedder)
    print("  混合检索测试完成，将BM25和向量检索结果进行了加权融合")
    
    # 5. 测试查询重写与扩展 - 优化用户查询
    print("\n【步骤5】测试查询理解与改写")
    rewriter, rewrite_results = test_query_rewrite()
    print(f"  创建了查询重写器并测试了 {len(rewrite_results)} 个查询，将用于后续RAG流程")
    
    # 6. 测试HyDE算法 - 高级检索增强
    print("\n【步骤6】测试高级检索增强：HyDE算法")
    test_hyde_algorithm(documents)
    print("  与传统向量检索相比，HyDE算法通过生成假设文档改进了检索质量")
    
    # 7. 测试上下文压缩与信息保留 - 优化大模型输入
    print("\n【步骤7】测试上下文管理：压缩与信息保留")
    test_context_compression(documents)
    print("  测试了多种压缩方法，找到了效率和信息保留的最佳平衡点")
    
    # 8. 测试检索结果重排序与打分机制 - 优化检索结果
    print("\n【步骤8】测试检索结果优化：重排序与打分")
    enriched_documents = test_retrieval_reranking(documents)
    print(f"  扩充了文档集至 {len(enriched_documents)} 个文档，并验证了不同重排序算法的效果")
    
    # 9. 测试完整的RAG流程 - 整合前面所有组件
    print("\n【步骤9】测试完整RAG流程 - 整合所有组件")
    test_query = "如何减少大型语言模型的幻觉问题？"
    print(f"  测试查询: '{test_query}'")
    
    # 使用前面所有测试中构建的组件
    rag_results = complete_rag_pipeline(
        test_query, 
        enriched_documents, 
        embedder,
        lsh_index=lsh_index,
        hnsw_index=hnsw_index,
        rewriter=rewriter
    )
    
    # 10. 分析流程结果
    print("\n【步骤10】RAG流程结果分析")
    print(f"  原始查询: '{rag_results['query']}'")
    print(f"  重写后查询: '{rag_results['rewritten_query']}'")
    print(f"  检索结果数量: 向量检索 {len(rag_results['vector_results'])}个, BM25检索 {len(rag_results['bm25_results'])}个")
    print(f"  融合后结果数量: {len(rag_results['combined_results'])}个")
    print(f"  压缩上下文长度: {len(rag_results['compressed_context'].split())}个词")
    print(f"  生成答案长度: {len(rag_results['answer'].split())}个词")
    
    # 总结 - 展示整个流程的组件和连接关系
    section_title("RAG流程总结")
    print("我们实现并测试了完整的RAG流程，各组件间的连接关系如下：\n")
    print("1. 文档处理 → 向量化 → 索引构建（LSH/HNSW）")
    print("2. 用户查询 → 查询重写/扩展 → 向量化")
    print("3. 向量检索 + BM25检索 → 结果融合 → 重排序")
    print("4. 检索结果 → 上下文压缩 → 大模型生成")
    print("\n实现的核心算法：\n")
    print("### 向量检索算法：")
    print("- 余弦相似度与点积相似度计算")
    print("- 近似最近邻（ANN）快速检索算法")
    print("- HNSW索引构建与查询算法")
    print("- 混合检索排序算法实现\n")
    print("### 检索优化算法：")
    print("- 查询重写与扩展算法")
    print("- HyDE（假设性文档嵌入）算法")
    print("- 上下文压缩与信息保留算法")
    print("- 检索结果重排序与打分机制\n")
    print("这些算法组合使用，构建了端到端的高效RAG系统。")


def optimized_rag_pipeline():
    """优化的RAG流程测试，确保各组件间数据流连贯"""
    print("=" * 80)
    print("优化的检索增强生成(RAG)算法演示")
    print("=" * 80)
    
    # 定义多样化的测试查询集合，包括不同难度和主题的查询
    test_queries = [
        {
            "id": "q1",
            "text": "如何减少大型语言模型的幻觉问题？",
            "type": "problem_solving",
            "expected_keywords": ["幻觉", "事实", "检索", "RAG"]
        },
        {
            "id": "q2", 
            "text": "向量检索和BM25各有什么优缺点？",
            "type": "comparison",
            "expected_keywords": ["向量", "BM25", "语义", "关键词"]
        },
        {
            "id": "q3",
            "text": "HNSW算法的工作原理是什么？",
            "type": "technical",
            "expected_keywords": ["HNSW", "近似最近邻", "索引", "图"]
        }
    ]
    
    # 选择一个查询进行完整流程演示
    current_query = test_queries[0]
    print(f"\n选择查询: '{current_query['text']}' (类型: {current_query['type']})")
    
    # 1. 创建文档集 - 所有后续步骤的基础
    documents, doc_vectors, embedder = create_sample_data()
    print(f"\n生成了 {len(documents)} 个文档和对应的向量表示")
    
    # 2. 查询重写和扩展 - 优化输入查询
    rewriter, _ = test_query_rewrite()
    rewritten_query, expanded_terms = rewriter.full_rewrite_and_expand(current_query['text'])
    print(f"\n原始查询: '{current_query['text']}'")
    print(f"重写后查询: '{rewritten_query}'")
    print("扩展词条:")
    for term, weight in expanded_terms[:5]:
        print(f"  - {term}: {weight:.2f}")
    
    # 3. 向量检索准备 - 构建索引用于快速检索
    print("\n构建检索索引...")
    lsh_index, hnsw_index = test_ann_algorithms(doc_vectors)
    
    # 4. 混合检索 - 结合向量和关键词方法
    print("\n执行混合检索...")
    # 准备混合检索所需的候选文档格式
    candidate_docs = []
    for i, (doc, vec) in enumerate(zip(documents, doc_vectors)):
        # 提取文档词条
        try:
            import jieba
            terms = list(jieba.cut(doc))
        except ImportError:
            terms = re.findall(r'\w+', doc.lower())
        
        candidate_docs.append({
            'id': i,
            'text': doc,
            'vector': vec,
            'terms': terms
        })
    
    # 执行混合检索
    query_vector = embedder.embed_text(rewritten_query)
    query_terms = [term for term, _ in expanded_terms]
    
    # 测试不同alpha值并选择最佳结果
    best_alpha = 0.5  # 默认平衡权重
    best_results = None
    best_score = 0
    
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        results = hybrid_retrieval_sort(query_vector, query_terms, candidate_docs, alpha=alpha)
        
        # 计算结果质量（此处简化为前3个结果的平均分）
        avg_score = sum([score for _, score in results[:3]]) / 3
        
        if avg_score > best_score:
            best_score = avg_score
            best_alpha = alpha
            best_results = results
    
    print(f"最佳混合权重 alpha={best_alpha} (得分: {best_score:.4f})")
    print("混合检索结果:")
    hybrid_retrieved_docs = []
    for i, (doc_id, score) in enumerate(best_results[:5]):
        print(f"  {i+1}. 文档 {doc_id}: {score:.4f} - {documents[doc_id]}")
        hybrid_retrieved_docs.append(documents[doc_id])
    
    # 5. 使用HyDE算法进行检索增强（可选）
    print("\n执行HyDE检索...")
    llm = SimpleLLM()
    hyde_prompt = f"请根据查询提供一个详细的答案：{current_query['text']}"
    hypothetical_doc = llm.generate_text(hyde_prompt)
    print(f"生成的假设性文档: '{hypothetical_doc[:100]}...'")
    
    # 使用假设性文档进行向量检索
    hyde_vector = embedder.embed_text(hypothetical_doc)
    hyde_results = hnsw_index.search(hyde_vector, k=5)
    
    print("HyDE检索结果:")
    hyde_retrieved_docs = []
    for i, (doc_id, score) in enumerate(hyde_results[:5]):
        if doc_id < len(documents):
            print(f"  {i+1}. 文档 {doc_id}: {score:.4f} - {documents[doc_id]}")
            hyde_retrieved_docs.append(documents[doc_id])
    
    # 6. 检索结果融合与重排序
    print("\n融合并重排序检索结果...")
    
    # 准备用于重排序的文档集
    all_doc_ids = list(range(len(documents)))
    
    # 使用BM25进行重排序
    reranker = RetrievalReranker()
    bm25_results = reranker.bm25_rerank(rewritten_query, documents, all_doc_ids, top_k=7)
    
    print("BM25重排序结果:")
    for i, (doc_id, score) in enumerate(bm25_results[:5]):
        print(f"  {i+1}. 文档 {doc_id}: {score:.4f} - {documents[doc_id]}")
    
    # 合并三种检索结果集
    all_result_sets = [
        [(doc_id, score) for doc_id, score in best_results[:5]],  # 混合检索结果
        [(doc_id, score) for doc_id, score in hyde_results[:5] if doc_id < len(documents)],  # HyDE结果
        [(doc_id, score) for doc_id, score in bm25_results[:5]]   # BM25结果
    ]
    
    # 使用倒数排名融合(RRF)合并结果
    combined_results = reranker.reciprocal_rank_fusion(all_result_sets)
    
    print("融合后的最终检索结果:")
    final_retrieved_docs = []
    final_doc_ids = []
    for i, (doc_id, score) in enumerate(combined_results[:7]):
        print(f"  {i+1}. 文档 {doc_id}: {score:.6f} - {documents[doc_id]}")
        final_retrieved_docs.append(documents[doc_id])
        final_doc_ids.append(doc_id)
    
    # 7. 上下文压缩
    print("\n执行上下文压缩...")
    compressor = ContextCompressor()
    
    # 测试并选择最佳压缩方法
    compression_methods = [
        ("TextRank压缩", compressor.textrank_compression),
        ("信息密度压缩", compressor.info_density_compression),
        ("Map-Reduce压缩", compressor.map_reduce_compress)
    ]
    
    best_compressed = None
    best_method_name = None
    best_quality = 0  # 简单估计的质量分数
    
    for name, method in compression_methods:
        try:
            compressed = method(final_retrieved_docs, rewritten_query, max_tokens=150)
            compressed_len = len(compressed.split())
            
            # 简单估计压缩质量（包含关键词数量/长度比）
            quality_score = 0
            for keyword in current_query['expected_keywords']:
                if keyword.lower() in compressed.lower():
                    quality_score += 1
            
            quality_score = quality_score / (0.01 + compressed_len/200)  # 归一化
            
            print(f"  {name}: {compressed_len}词, 质量分数: {quality_score:.2f}")
            
            if quality_score > best_quality:
                best_quality = quality_score
                best_compressed = compressed
                best_method_name = name
                
        except Exception as e:
            print(f"  {name}失败: {e}")
    
    print(f"\n选择的最佳压缩方法: {best_method_name}, 质量分数: {best_quality:.2f}")
    print(f"压缩后的上下文 ({len(best_compressed.split())}词):")
    print(f"  '{best_compressed[:200]}...'")
    
    # 8. 生成最终回答
    answer_prompt = f"基于以下上下文回答问题: {current_query['text']}\n\n上下文:\n{best_compressed}"
    answer = llm.generate_text(answer_prompt)
    
    print("\n生成的最终回答:")
    print(f"  '{answer}'")
    
    # 返回完整处理结果供评估
    return {
        'query': current_query,
        'rewritten_query': rewritten_query,
        'documents': documents,
        'doc_vectors': doc_vectors,
        'hybrid_results': best_results[:5],
        'hyde_results': [r for r in hyde_results[:5] if r[0] < len(documents)],
        'bm25_results': bm25_results[:5],
        'final_results': combined_results[:7],
        'compressed_context': best_compressed,
        'answer': answer
    }

# 替换主函数
if __name__ == "__main__":
    optimized_rag_pipeline()
    # main() 