"""
上下文压缩算法模块

本模块实现了多种上下文压缩算法，用于RAG系统中对检索到的文档进行智能压缩，
在保留关键信息的同时减少输入token数量，提高大型语言模型处理效率。

主要功能:
1. 关键词压缩 - 提取与查询相关的关键句，忽略无关内容
2. TF-IDF压缩 - 基于词频-逆文档频率判断句子重要性
3. 句子重要性压缩 - 综合考虑多种因素对句子进行评分和排序
4. TextRank压缩 - 使用图算法提取中心句子
5. Map-Reduce压缩 - 分布式处理长文档集合
6. 信息密度压缩 - 根据信息熵和相关性进行压缩

应用场景:
- RAG系统：减少发送到LLM的token数量，降低成本并提高效率
- 文档摘要：生成保留核心信息的摘要
- 长文本处理：帮助处理超出LLM上下文窗口的长文本

实现原理:
基于多种NLP技术和算法，智能判断文本中的关键句子和重要信息，
通过多维度评分、关键信息抽取、冗余去除等方式，在最大限度保留
与查询相关信息的同时，大幅减少文本长度。

优化特性:
- 缓存机制加速重复计算
- 多种评分因子自适应权重
- 中英文双语支持
- 去重与冗余消除
"""

import numpy as np
import re
from collections import Counter, defaultdict
import time
import heapq
from typing import List, Set, Dict, Tuple, Optional
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import sys
import matplotlib
import matplotlib.pyplot as plt
import json
from typing import List, Dict, Tuple, Set, Optional, Union, Any
import jieba
import jieba.analyse
from sklearn.metrics.pairwise import cosine_similarity

# 配置matplotlib使用英文字体，避免中文字体问题
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
# 忽略字体警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

class ContextCompressor:
    """
    上下文压缩与信息保留算法类
    
    用于压缩检索到的文档，同时保留与查询相关的关键信息。
    提供多种压缩算法，适用于不同场景和需求。
    
    核心思想:
        在RAG系统中，发送给LLM的文档内容越多，成本越高且效率越低。
        通过智能压缩技术，可以在保留核心信息的同时大幅减少token数量。
        
    压缩算法原理:
        1. 关键词压缩：提取查询关键词，保留包含这些关键词的重要句子
        2. TF-IDF压缩：使用TF-IDF模型计算词语和句子重要性
        3. 句子重要性压缩：评估句子相关性、信息量、位置等多维因素
        4. TextRank压缩：基于PageRank算法的图方法提取中心句子
        5. Map-Reduce压缩：分布式处理大规模文档集合
        6. 信息密度压缩：综合考虑信息熵和查询相关性进行压缩
        
    评分因素:
        - 查询相关性：与查询关键词的匹配程度
        - 信息密度：句子包含的信息量（基于信息熵）
        - 中心性：句子在文档网络中的中心地位（基于图算法）
        - 位置重要性：文档首尾句通常包含关键信息
        - 长度均衡：避免过长或过短的句子
        
    优化技术:
        - 缓存机制：避免重复计算提高效率
        - 冗余检测：识别并移除内容高度相似的句子
        - 自适应阈值：根据文档特性动态调整参数
        - 中英文双语支持：使用不同的分词和处理策略
    """
    
    def __init__(self, embedder=None, use_cache: bool = True):
        """
        初始化上下文压缩器
        
        参数:
            embedder: 可选的嵌入器，用于语义压缩
            use_cache: 是否使用缓存加速重复计算
        """
        self.embedder = embedder
        self.word_to_idx = {}  # 词到索引的映射
        self.idx_to_word = {}  # 索引到词的映射
        self.stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could'])
        self.use_cache = use_cache
        self._keyword_cache = {}  # 关键词提取缓存
        self._sentence_cache = {}  # 句子分割缓存
        self._similarity_cache = {}  # 相似度计算缓存
        
        # 中文常见无意义字符
        self.invalid_chars = {'的', '了', '和', '与', '或', '及', '等', '中', '对', '从', '到', '把', '被', '让', '使', '由', '于', '在', '是', '有', '将', '能', '会', '可以'}
        
        # 设置压缩目标：尝试将文本压缩到目标比例
        self.target_compression_ratio = 0.5  # 默认目标为原始文本的50%
    
    def keyword_based_compression(self, documents: List[str], query: str, max_tokens: int = 200) -> str:
        """
        基于关键词的压缩算法
        
        原理:
            通过提取查询中的关键词，找出文档中包含这些关键词的重要句子。
            优先保留包含多个关键词的句子，按关键词匹配得分排序，从而实现高效压缩。
        
        算法步骤:
            1. 从查询中提取关键词
            2. 对每个文档计算关键词匹配得分
            3. 按得分对文档排序
            4. 从高分文档中提取包含关键词的句子
            5. 合并这些句子，直到达到token上限
        
        优点:
            - 实现简单，速度快
            - 对长文档集合效果好
            - 直接聚焦于查询相关内容
            
        缺点:
            - 可能会错过不包含关键词但语义相关的句子
            - 对不同语义表达的查询敏感
        
        参数:
            documents (List[str]): 文档列表
            query (str): 查询文本
            max_tokens (int): 最大标记数，默认200
            
        返回:
            str: 压缩后的文本
        """
        start_time = time.time()
        
        # 快速路径检查
        if not documents:
            return ""
            
        # 1. 从查询中提取关键词
        print("\n分析查询:")
        keywords = self._extract_keywords(query)
        
        # 如果没有关键词，直接返回第一个文档的部分内容
        if not keywords:
            return " ".join(documents[0].split()[:max_tokens])
        
        # 2. 对每个文档进行评分 - 并行计算所有文档的得分
        doc_scores = [(doc, self._calculate_keyword_score(doc, keywords)) 
                      for doc in documents]
        
        # 3. 按分数排序
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 4. 选择得分最高的文档片段
        selected_text = ""
        for doc, score in doc_scores:
            if score > 0:  # 只处理包含关键词的文档
                relevant_sentences = self._extract_relevant_sentences(doc, keywords)
                if relevant_sentences:
                    selected_text += " ".join(relevant_sentences) + " "
                    # 检查是否达到了最大标记数量
                    if len(selected_text.split()) >= max_tokens:
                        selected_text = " ".join(selected_text.split()[:max_tokens])
                        break
        
        # 5. 如果结果为空，返回原始文档的一部分
        if not selected_text:
            selected_text = " ".join(documents[0].split()[:max_tokens])
        
        end_time = time.time()
        print(f"\n压缩用时: {end_time - start_time:.4f} 秒")
        
        return selected_text.strip()
    
    def tfidf_based_compression(self, documents: List[str], query: str, max_tokens: int = 200) -> str:
        """
        基于TF-IDF的压缩算法
        
        原理:
            使用TF-IDF(词频-逆文档频率)模型评估词语在文档中的重要性。
            TF-IDF综合考虑:
            - 词频(TF): 一个词在文档中出现的频率越高越重要
            - 逆文档频率(IDF): 一个词在越少的文档中出现越重要
            通过计算文档和查询的TF-IDF向量，并使用余弦相似度度量相关性。
        
        算法步骤:
            1. 构建文档-词项矩阵，生成词汇表
            2. 计算TF-IDF权重矩阵
            3. 构建查询向量
            4. 计算文档与查询的余弦相似度
            5. 按相似度对文档排序
            6. 计算每个句子与查询的相似度
            7. 按相似度选择句子，直到达到token上限
        
        优点:
            - 考虑词语的统计重要性，不仅仅依赖关键词匹配
            - 对语义相关性有一定的捕捉能力
            - 不需要外部模型，计算效率较高
            
        缺点:
            - 基于词袋模型，没有考虑词序和上下文
            - 对多文档集合的计算复杂度较高
        
        参数:
            documents (List[str]): 文档列表
            query (str): 查询文本
            max_tokens (int): 最大标记数，默认200
            
        返回:
            str: 压缩后的文本
        """
        start_time = time.time()
        
        # 快速路径检查
        if not documents:
            return ""
            
        # 1. 构建文档-词项矩阵
        doc_term_matrix, self.word_to_idx, self.idx_to_word = self._build_doc_term_matrix(documents)
        
        # 检查矩阵是否为空
        if doc_term_matrix.size == 0:
            return " ".join(documents[0].split()[:max_tokens])
            
        # 2. 计算TF-IDF权重
        tfidf_matrix = self._calculate_tfidf(doc_term_matrix)
        
        # 3. 构建查询向量
        query_vector = self._build_query_vector(query, doc_term_matrix)
        
        # 4. 计算文档与查询的相似度
        similarities = [(i, self._cosine_similarity(doc_vector, query_vector)) 
                       for i, doc_vector in enumerate(tfidf_matrix)]
        
        # 5. 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 6. 选择最相关的文档片段
        selected_text = ""
        processed_sentences = set()  # 避免重复添加句子
        
        for doc_idx, similarity in similarities:
            if similarity <= 0:  # 跳过不相关文档
                continue
                
            doc = documents[doc_idx]
            sentences = self._split_into_sentences(doc)
            
            # 预计算所有句子向量，避免重复计算
            sentence_vectors = {}
            for sentence in sentences:
                if sentence:
                    sentence_vector = self._build_query_vector(sentence, doc_term_matrix)
                    sentence_similarity = self._cosine_similarity(sentence_vector, query_vector)
                    sentence_vectors[sentence] = sentence_similarity
            
            # 按相似度排序句子
            sorted_sentences = sorted(sentence_vectors.items(), key=lambda x: x[1], reverse=True)
            
            # 添加相似度高的句子
            for sentence, sent_similarity in sorted_sentences:
                if sent_similarity > 0.1 and sentence not in processed_sentences:  # 降低阈值，增加结果多样性
                    processed_sentences.add(sentence)
                    selected_text += sentence + "。 "
                    # 检查是否达到最大标记数
                    if len(selected_text.split()) >= max_tokens:
                        selected_text = " ".join(selected_text.split()[:max_tokens])
                        break
                        
            if len(selected_text.split()) >= max_tokens:
                break
        
        # 7. 如果结果为空，返回原始文档的一部分
        if not selected_text:
            selected_text = " ".join(documents[0].split()[:max_tokens])
        
        end_time = time.time()
        print(f"\n压缩用时: {end_time - start_time:.4f} 秒")
        
        return selected_text.strip()
    
    def improved_sentence_importance_compression(self, documents: List[str], query: str, max_tokens: int = 200) -> str:
        """
        改进的基于句子重要性的压缩算法
        
        原理:
            综合多种因素对句子进行多维度评分，并使用自适应阈值进行筛选。
            评分因素包括:
            - 查询相关性：与查询关键词的匹配程度
            - 位置重要性：文档首尾句通常包含关键信息
            - 信息熵：衡量句子包含的信息量
            最后根据目标压缩率自适应选择句子，并消除冗余。
        
        算法改进:
            1. 多维度评分：综合考虑多种因素，不仅仅是关键词匹配
            2. 自适应阈值：根据文档长度和复杂度动态调整压缩比例
            3. 冗余检测：识别并移除内容高度相似的句子
            4. 结构保持：尝试保持原文档的句子顺序，提高可读性
        
        优点:
            - 综合考虑多种因素，压缩质量高
            - 自适应调整，适用于不同长度和复杂度的文档
            - 保持文档结构，提高可读性
            - 去除冗余信息，提高信息密度
            
        缺点:
            - 计算复杂度较高
            - 参数调优难度大
        
        参数:
            documents (List[str]): 文档列表
            query (str): 查询文本
            max_tokens (int): 最大标记数，默认200
            
        返回:
            str: 压缩后的文本
        """
        start_time = time.time()
        
        # 快速路径检查
        if not documents:
            return ""
        
        # 提取查询关键词
        query_keywords = self._extract_keywords(query)
        
        # 自适应阈值 - 根据文档长度和查询复杂度调整
        total_words = sum(len(doc.split()) for doc in documents)
        target_ratio = min(0.6, max(0.3, max_tokens / total_words))  # 目标压缩比例30%-60%
        
        # 收集所有句子及其得分
        sentence_scores = []
        processed_sentences = set()
        
        for doc in documents:
            sentences = self._split_into_sentences(doc)
            for i, sentence in enumerate(sentences):
                if sentence and sentence not in processed_sentences:
                    processed_sentences.add(sentence)
                    sentence_with_period = sentence + '。'
                    
                    # 计算基础得分 - 与查询的相关性
                    base_score = self._calculate_sentence_importance(sentence_with_period, query_keywords)
                    
                    # 增加位置权重 - 文档开头和结尾的句子更重要
                    position_weight = 1.0
                    if i == 0 or i == len(sentences) - 1:
                        position_weight = 1.2  # 首尾句加权
                    
                    # 计算信息熵 - 信息量大的句子更重要
                    entropy = self._calculate_information_entropy(sentence)
                    
                    # 综合得分
                    final_score = 0.6 * base_score + 0.2 * position_weight + 0.2 * entropy
                    
                    # 只保留得分大于阈值的句子，提高筛选严格性
                    if final_score > 0.1:  # 设置一个基本阈值
                        sentence_scores.append((sentence_with_period, final_score, i, doc))
        
        # 使用强制排序 - 我们真正只想要最重要的句子
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 确定需要保留的句子数量 - 基于目标压缩率
        num_sentences_to_keep = max(1, min(int(len(sentence_scores) * target_ratio), 
                                        int(max_tokens / 10)))  # 假设平均每句10个词
        
        # 选择得分最高的句子
        selected_sentences = sentence_scores[:num_sentences_to_keep]
        
        # 检测并去除冗余句子
        final_sentences = []
        for sentence, score, pos, doc in selected_sentences:
            # 检查是否与已选句子存在高度相似
            is_redundant = False
            for existing, _, _, _ in final_sentences:
                similarity = self._sentence_similarity(sentence, existing)
                if similarity > 0.7:  # 相似度阈值
                    is_redundant = True
                    break
            
            if not is_redundant:
                final_sentences.append((sentence, score, pos, doc))
        
        # 按原始文档顺序排列句子，保持文档结构
        # 首先按文档分组
        doc_groups = defaultdict(list)
        for sent, score, pos, doc in final_sentences:
            doc_groups[doc].append((sent, pos))
        
        # 然后按照每个文档中句子的原始位置排序
        ordered_sentences = []
        for doc in documents:
            if doc in doc_groups:
                # 按位置排序当前文档的句子
                doc_sentences = sorted(doc_groups[doc], key=lambda x: x[1])
                ordered_sentences.extend([s for s, _ in doc_sentences])
        
        # 生成最终文本，确保不超过token限制
        selected_text = ""
        current_tokens = 0
        
        for sentence in ordered_sentences:
            sentence_tokens = len(sentence.split())
            if current_tokens + sentence_tokens <= max_tokens:
                selected_text += sentence + " "
                current_tokens += sentence_tokens
            else:
                break
        
        # 如果结果为空，返回得分最高的一句话
        if not selected_text and sentence_scores:
            selected_text = sentence_scores[0][0]
        
        # 计算最终压缩率
        end_time = time.time()
        compression_time = end_time - start_time
        compressed_tokens = len(selected_text.split())
        compression_ratio = compressed_tokens / total_words if total_words > 0 else 1.0
        
        print(f"\n改进的句子重要性压缩:")
        print(f"压缩用时: {compression_time:.4f} 秒")
        print(f"原始文档总词数: {total_words}")
        print(f"压缩后词数: {compressed_tokens}")
        print(f"压缩率: {compression_ratio:.2f}")
        
        return selected_text.strip()
    
    def _calculate_information_entropy(self, text: str) -> float:
        """计算文本的信息熵，评估信息量"""
        if not text:
            return 0.0
        
        # 计算字符频率
        char_freq = Counter(text)
        length = len(text)
        
        # 计算信息熵
        entropy = 0.0
        for count in char_freq.values():
            probability = count / length
            entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _sentence_similarity(self, sent1: str, sent2: str) -> float:
        """计算两个句子的相似度"""
        keywords1 = self._extract_keywords(sent1)
        keywords2 = self._extract_keywords(sent2)
        
        # 使用Jaccard相似度
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        return intersection / union if union > 0 else 0.0
    
    def sentence_importance_compression(self, documents: List[str], query: str, max_tokens: int = 200) -> str:
        """基于句子重要性的压缩算法 - 优化版本"""
        start_time = time.time()
        
        # 快速路径检查
        if not documents:
            return ""
            
        # 1. 提取查询关键词
        print("\n分析查询:")
        query_keywords = self._extract_keywords(query)
        
        # 如果没有关键词，直接返回第一个文档的部分
        if not query_keywords:
            return " ".join(documents[0].split()[:max_tokens])
        
        # 2. 计算每个句子的重要性得分
        processed_sentences = set()  # 避免句子重复
        
        # 使用堆来保持最高得分的句子，避免全部排序
        top_sentences = []
        
        for doc in documents:
            sentences = self._split_into_sentences(doc)
            for sentence in sentences:
                if sentence and sentence not in processed_sentences:
                    processed_sentences.add(sentence)
                    sentence_with_period = sentence + '。'
                    score = self._calculate_sentence_importance(sentence_with_period, query_keywords)
                    
                    # 只保留得分大于0的句子
                    if score > 0:
                        # 使用最小堆保持前N个最高得分的句子
                        if len(top_sentences) < max_tokens // 5:  # 估计每个句子平均5个词
                            heapq.heappush(top_sentences, (score, sentence_with_period))
                        elif score > top_sentences[0][0]:
                            heapq.heappushpop(top_sentences, (score, sentence_with_period))
        
        # 3. 将堆转换为列表并按得分降序排序
        all_sentences = [(s, sc) for sc, s in top_sentences]
        all_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # 4. 选择最重要的句子
        selected_text = ""
        current_tokens = 0
        
        for sentence, score in all_sentences:
            sentence_tokens = len(sentence.split())
            if current_tokens + sentence_tokens <= max_tokens:
                selected_text += sentence + " "
                current_tokens += sentence_tokens
            else:
                # 如果此句会超出限制，但总体还很少，则添加句子的一部分
                if current_tokens < max_tokens * 0.7:
                    remaining = max_tokens - current_tokens
                    partial = " ".join(sentence.split()[:remaining])
                    selected_text += partial + "... "
                break
        
        # 5. 如果结果为空，返回原始文档的一部分
        if not selected_text:
            selected_text = " ".join(documents[0].split()[:max_tokens])
        
        end_time = time.time()
        print(f"\n压缩用时: {end_time - start_time:.4f} 秒")
        
        return selected_text.strip()
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """提取文本中的关键词 - 优化版本"""
        # 使用缓存加速重复文本的关键词提取
        if self.use_cache and text in self._keyword_cache:
            return self._keyword_cache[text]
        
        keywords = set()
        
        # 快速处理空文本情况
        if not text or text.isspace():
            return keywords
        
        # 预处理文本 - 批量替换标点符号提高效率
        text = text.replace('，', ' ').replace('。', ' ').replace('、', ' ').strip()
        
        # 处理短语 - 使用分割而不是正则表达式提高速度
        phrases = text.split()
        
        # 减少嵌套循环，直接处理有效短语
        for phrase in phrases:
            if len(phrase) >= 2 and not any(char in self.invalid_chars for char in phrase):
                keywords.add(phrase)
            
            # 仅对较长短语进行子串提取
            if len(phrase) > 4:
                self._extract_subphrases(phrase, keywords)
        
        # 缓存结果
        if self.use_cache:
            self._keyword_cache[text] = keywords
            
        # 测试函数运行时完全禁用关键词分析打印，以减少混乱
        return keywords
    
    def _extract_subphrases(self, phrase: str, keywords: set) -> None:
        """从短语中提取有效子短语 - 分离为独立函数提高可维护性"""
        # 使用滑动窗口而不是双重循环，提高效率
        min_length = 2
        max_length = min(5, len(phrase))
        
        for length in range(min_length, max_length + 1):
            for i in range(len(phrase) - length + 1):
                subphrase = phrase[i:i+length]
                if self._is_valid_phrase(subphrase):
                    # 检查是否是更长关键词的子串
                    is_substring = any(subphrase in kw and subphrase != kw for kw in keywords)
                    if not is_substring:
                        keywords.add(subphrase)
    
    def _is_valid_phrase(self, phrase: str) -> bool:
        """检查短语是否有效 - 优化版本"""
        # 快速路径检查以减少计算
        if len(phrase) < 2 or len(phrase) > 8:
            return False

        # 使用集合操作而不是循环，更高效
        if any(word in phrase for word in self.stopwords):
            return False
            
        if not phrase.strip():  # 检查是否为空白字符
            return False
            
        # 检查是否含有无意义字符 - 使用集合交集操作而不是循环
        if set(phrase) & self.invalid_chars:
            return False
            
        return True
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割成句子 - 优化版本"""
        # 使用缓存加速重复文本的句子分割
        if self.use_cache and text in self._sentence_cache:
            return self._sentence_cache[text]
        
        # 批量替换中文标点以提高性能
        text = text.replace('！', '。').replace('？', '。').replace('；', '。')
        
        # 直接使用split而不是正则表达式，提高性能
        sentences = [s.strip() for s in text.split('。') if s.strip()]
        
        # 缓存结果
        if self.use_cache:
            self._sentence_cache[text] = sentences
            
        return sentences
    
    def _cosine_similarity(self, vec1, vec2):
        """计算余弦相似度 - 优化版本"""
        # 创建缓存键
        if self.use_cache:
            # 使用向量的哈希值作为缓存键
            cache_key = (hash(vec1.tobytes()), hash(vec2.tobytes()))
            if cache_key in self._similarity_cache:
                return self._similarity_cache[cache_key]
        
        # numpy的优化计算
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # 快速路径检查零向量
        if norm1 == 0 or norm2 == 0:
            return 0
        
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        
        # 缓存结果
        if self.use_cache:
            self._similarity_cache[cache_key] = similarity
            
        return similarity
    
    def _calculate_keyword_score(self, text: str, keywords: set) -> float:
        """计算文本中包含关键词的得分 - 优化版本"""
        # 快速路径检查
        if not keywords:
            return 0
            
        # 避免重复提取文本关键词    
        text_keywords = self._extract_keywords(text)
        
        # 使用集合操作加速匹配计算
        matches = 0
        for kw in keywords:
            for text_kw in text_keywords:
                if kw in text_kw or text_kw in kw:
                    matches += 1
                    break
        
        return matches / len(keywords)
    
    def _calculate_sentence_importance(self, sentence: str, query_keywords: set) -> float:
        """计算句子的重要性得分 - 优化版本"""
        # 快速路径检查
        if not sentence or not query_keywords:
            return 0.0
            
        sentence_keywords = self._extract_keywords(sentence)
        
        # 如果句子没有关键词，给一个小的基础分数，而不是0
        if not sentence_keywords:
            return 0.001
        
        # 计算关键词匹配得分 - 使用更宽松的匹配标准
        # 1. 精确匹配 - 完全匹配的关键词
        exact_matches = len(query_keywords.intersection(sentence_keywords))
        
        # 2. 模糊匹配 - 部分包含的关键词
        fuzzy_matches = sum(1 for qk in query_keywords 
                           if any(qk in sk or sk in qk for sk in sentence_keywords))
        
        # 3. 字符重叠匹配 - 检查字符级别的重叠
        char_overlap = 0
        for qk in query_keywords:
            for sk in sentence_keywords:
                # 计算两个词的字符重叠率
                common_chars = set(qk).intersection(set(sk))
                if len(common_chars) >= 2:  # 至少有2个字符重叠
                    char_overlap += len(common_chars) / max(len(qk), len(sk))
        
        # 综合匹配得分，使用更灵活的公式
        keyword_score = (0.5 * exact_matches + 0.3 * fuzzy_matches + 0.2 * char_overlap) / len(query_keywords)
        
        # 计算句子长度得分（偏好适中长度的句子）
        length = len(sentence)
        ideal_length = 50  # 假设理想长度为50个字符
        length_score = max(0, 1 - abs(length - ideal_length) / ideal_length)
        
        # 对短句子给予额外加分，便于选择更简洁的句子
        brevity_bonus = 0
        if length < 30:
            brevity_bonus = 0.2 * (1 - length / 30)
        
        # 综合得分（关键词匹配占60%，长度得分占30%，简洁加分占10%）
        final_score = 0.6 * keyword_score + 0.3 * length_score + 0.1 * brevity_bonus
        
        # 确保即使相关性很小，也有一个最小分数
        return max(0.001, final_score)
    
    def _extract_relevant_sentences(self, text: str, keywords: set) -> List[str]:
        """提取包含关键词的相关句子 - 优化版本"""
        # 快速路径检查
        if not keywords:
            return []
            
        sentences = self._split_into_sentences(text)
        relevant = []
        
        # 减少重复计算，预先提取每个句子的关键词
        sentence_keywords = {}
        for sentence in sentences:
            if sentence:
                sentence_keywords[sentence] = self._extract_keywords(sentence)
        
        # 使用更高效的匹配逻辑
        for sentence, sent_keywords in sentence_keywords.items():
            # 使用任意匹配优化循环
            if any(any(kw in sk or sk in kw for sk in sent_keywords) for kw in keywords):
                relevant.append(sentence.strip() + '。')
                
        return relevant
    
    def _build_doc_term_matrix(self, documents):
        """构建文档-词矩阵 - 优化版本"""
        # 快速路径检查
        if not documents:
            return np.array([]), {}, {}
            
        # 收集文档词汇
        all_words = set()
        doc_words_list = []
        
        # 分离循环减少计算
        for doc in documents:
            words = self._extract_keywords(doc)
            all_words.update(words)
            doc_words_list.append(words)
        
        # 创建词到索引的映射
        word_to_idx = {word: i for i, word in enumerate(all_words)}
        idx_to_word = {i: word for i, word in enumerate(all_words)}
        
        # 使用numpy的高效矩阵操作构建文档-词矩阵
        matrix = np.zeros((len(documents), len(all_words)))
        
        # 优化填充矩阵的方法
        for i, words in enumerate(doc_words_list):
            for word in words:
                if word in word_to_idx:
                    matrix[i, word_to_idx[word]] += 1
        
        return matrix, word_to_idx, idx_to_word
    
    def _calculate_tfidf(self, doc_term_matrix):
        """计算TF-IDF权重 - 优化版本"""
        # 快速路径检查
        if doc_term_matrix.size == 0:
            return np.array([])
            
        # 避免除零错误
        row_sums = np.sum(doc_term_matrix, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 防止除以零
        
        # 计算词频(TF) - 优化为单步操作
        tf = doc_term_matrix / row_sums
        
        # 计算逆文档频率(IDF) - 使用log1p避免除零问题
        df = np.sum(doc_term_matrix > 0, axis=0)
        num_docs = len(doc_term_matrix)
        idf = np.log(num_docs / (df + 1)) + 1  # 加1是拉普拉斯平滑，避免除以零或对数为零
        
        # 计算TF-IDF - 使用numpy的广播特性
        tfidf = tf * idf
        
        return tfidf
    
    def _build_query_vector(self, query, doc_term_matrix):
        """构建查询向量 - 优化版本"""
        # 快速路径检查
        if doc_term_matrix.size == 0:
            return np.array([])
            
        query_words = self._extract_keywords(query)
        query_vector = np.zeros(doc_term_matrix.shape[1])
        
        # 使用更高效的词频计数
        word_counts = Counter()
        for word in query_words:
            if word in self.word_to_idx:
                word_counts[self.word_to_idx[word]] += 1
        
        # 一次性更新向量
        for idx, count in word_counts.items():
            query_vector[idx] = count
        
        return query_vector
    
    def _count_words(self, text: str) -> int:
        """统计文本中的词数 - 优化版本"""
        # 空文本快速返回
        if not text:
            return 0
            
        # 使用正则表达式一次性替换所有标点符号
        text = re.sub(r'[，。、！？；：""《》（）]', '', text)
        
        # 分词并计数 - 对中文文本每个字符视为一个词
        words = text.split()
        return sum(len(word) for word in words)

    def map_reduce_compress(self, documents, query, max_tokens=200):
        """
        使用Map-Reduce方法压缩文档 - 优化版本
        
        参数:
            documents: 文档列表
            query: 查询文本
            max_tokens: 最大标记数
            
        返回:
            压缩后的文本
        """
        start_time = time.time()
        
        # 快速路径检查
        if not documents:
            return ""
            
        print("\n执行Map-Reduce压缩:")
        print("-" * 40)
        
        # 提取查询关键词
        query_keywords = self._extract_keywords(query)
        
        # 如果没有关键词，采用简单的截断策略
        if not query_keywords:
            all_text = " ".join(documents)
            return " ".join(all_text.split()[:max_tokens])
        
        # Map阶段：为每个文档并行提取重要句子
        mapped_docs = []
        
        # 使用堆排序优化每个文档的句子选择
        for i, doc in enumerate(documents):
            sentences = self._split_into_sentences(doc)
            if not sentences:
                continue
                
            # 使用最小堆保持得分最高的k个句子
            top_k = 3  # 每个文档最多取3个句子
            top_sentences = []
            
            for sentence in sentences:
                score = self._calculate_sentence_importance(sentence, query_keywords)
                if len(top_sentences) < top_k:
                    heapq.heappush(top_sentences, (score, sentence))
                elif score > top_sentences[0][0]:
                    heapq.heappushpop(top_sentences, (score, sentence))
            
            # 将堆转换为列表并按得分排序
            sorted_sentences = sorted([(s, sc) for sc, s in top_sentences], 
                                     key=lambda x: x[1], reverse=True)
            
            # 仅添加得分大于低阈值的句子，降低筛选条件以保留更多句子
            selected = [s for s, sc in sorted_sentences if sc > 0.001]
            
            # 如果没有选择到任何句子，强制选择得分最高的前2个句子
            if not selected and sorted_sentences:
                selected = [s for s, _ in sorted_sentences[:2]]
            
            if selected:
                mapped_docs.append(" ".join(selected))
        
        # Reduce阶段：按重要性合并文档段落
        if not mapped_docs:
            # 如果以上方法没有找到相关句子，则采用强制压缩策略
            # 从每个文档中选择首句和尾句
            forced_selections = []
            for doc in documents:
                sentences = self._split_into_sentences(doc)
                if sentences:
                    if len(sentences) > 2:
                        # 选择首句和尾句
                        forced_selections.append(sentences[0])
                        forced_selections.append(sentences[-1])
                    else:
                        # 文档较短，直接添加所有句子
                        forced_selections.extend(sentences)
            
            # 最多保留到max_tokens
            all_text = " ".join(forced_selections)
            return " ".join(all_text.split()[:max_tokens])
        
        # 对映射后的文档按与查询的相关性排序
        doc_scores = [(doc, self._calculate_keyword_score(doc, query_keywords)) 
                     for doc in mapped_docs]
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 按重要性选择段落直到达到最大标记数
        combined_text = ""
        current_tokens = 0
        
        for doc, score in doc_scores:
            if score > 0.001:  # 降低阈值，允许更多的文档被选择
                doc_tokens = len(doc.split())
                if current_tokens + doc_tokens <= max_tokens:
                    combined_text += doc + " "
                    current_tokens += doc_tokens
                else:
                    # 部分添加最后一个文档
                    remaining = max_tokens - current_tokens
                    if remaining > 0:
                        partial = " ".join(doc.split()[:remaining])
                        combined_text += partial
                    break
        
        # 如果组合文本为空或太少，加入得分最高的一些文档，不管其score
        if len(combined_text.split()) < max_tokens * 0.5:
            for doc, _ in doc_scores:
                if len(combined_text.split()) >= max_tokens * 0.5:
                    break
                    
                if doc not in combined_text:  # 避免重复添加
                    doc_tokens = len(doc.split())
                    if current_tokens + doc_tokens <= max_tokens:
                        combined_text += doc + " "
                        current_tokens += doc_tokens
                    else:
                        # 部分添加
                        remaining = max_tokens - current_tokens
                        if remaining > 0:
                            partial = " ".join(doc.split()[:remaining])
                            combined_text += partial
                        break
        
        # 如果组合文本仍为空，退回到简单的截断策略
        if not combined_text:
            all_text = " ".join(documents)
            combined_text = " ".join(all_text.split()[:max_tokens])
        
        end_time = time.time()
        compression_time = end_time - start_time
        
        # 计算压缩率
        original_tokens = sum(len(doc.split()) for doc in documents)
        compressed_tokens = len(combined_text.split())
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        
        print(f"压缩用时: {compression_time:.4f} 秒")
        print(f"原始文档总词数: {original_tokens}")
        print(f"压缩后词数: {compressed_tokens}")
        print(f"压缩率: {compression_ratio:.2f}")
        
        return combined_text.strip()

    def sentence_compression(self, text, max_sentences=3):
        """
        句子级别的压缩 - 优化版本
        
        参数:
            text: 文本内容
            max_sentences: 最大句子数
            
        返回:
            压缩后的文本
        """
        start_time = time.time()
        
        # 快速路径检查
        if not text:
            return ""
            
        # 分割句子
        sentences = self._split_into_sentences(text)
        if not sentences:
            return ""
            
        # 使用最小堆优化句子选择
        sentence_scores = []
        
        # 预先计算所有句子的关键词，避免重复提取
        sentence_keywords = {sentence: self._extract_keywords(sentence) for sentence in sentences}
        
        for sentence in sentences:
            # 基于句子长度和关键词密度的简单评分
            words = sentence.split()
            word_count = len(words)
            
            # 较长的句子得分较低
            length_score = 1.0 / (1 + word_count / 20)  
            
            # 关键词密度得分 - 使用预先计算的关键词
            keywords = sentence_keywords[sentence]
            keyword_density = len(keywords) / (word_count + 1)
            
            # 长词密度得分 - 高效计算词长度 >= 3 的比例
            long_words = sum(1 for w in words if len(w) >= 3)
            long_word_score = long_words / (word_count + 1)
            
            # 综合得分
            score = 0.3 * length_score + 0.4 * keyword_density + 0.3 * long_word_score
            
            sentence_scores.append((sentence, score))
        
        # 选择得分最高的句子
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        selected_sentences = [s for s, _ in sentence_scores[:max_sentences]]
        
        # 按原始顺序排列句子
        ordered_sentences = [s for s in sentences if s in selected_sentences]
        
        compressed_text = " ".join(ordered_sentences)
        
        end_time = time.time()
        print(f"\n句子压缩用时: {end_time - start_time:.4f} 秒")
        
        return compressed_text

    def textrank_compression(self, documents: List[str], query: str, max_tokens: int = 200) -> str:
        """
        基于TextRank的压缩算法
        
        使用TextRank算法为句子评分，挑选最重要的句子
        考虑句子间的相互关系，而不仅是与查询的相关性
        
        参数:
            documents: 文档列表
            query: 查询文本
            max_tokens: 最大标记数
            
        返回:
            压缩后的文本
        """
        start_time = time.time()
        
        # 快速路径检查
        if not documents:
            return ""
        
        # 提取所有句子
        all_sentences = []
        doc_indices = []  # 记录每个句子来自哪个文档
        for doc_idx, doc in enumerate(documents):
            sentences = self._split_into_sentences(doc)
            all_sentences.extend(sentences)
            doc_indices.extend([doc_idx] * len(sentences))
        
        if not all_sentences:
            return " ".join(documents[0].split()[:max_tokens])
        
        # 提取查询关键词
        query_keywords = self._extract_keywords(query)
        
        # 创建句子相似度图
        similarity_matrix = self._build_similarity_graph(all_sentences)
        
        # 应用PageRank算法
        scores = self._apply_textrank(similarity_matrix)
        
        # 考虑查询相关性进行加权
        if query_keywords:
            for i, sentence in enumerate(all_sentences):
                query_relevance = self._calculate_sentence_importance(sentence, query_keywords)
                # 增加查询相关性的权重，从30%提高到50%
                scores[i] = 0.5 * scores[i] + 0.5 * query_relevance
        
        # 额外加分：为包含关键词的句子加分
        if query_keywords:
            for i, sentence in enumerate(all_sentences):
                sentence_keywords = self._extract_keywords(sentence)
                keyword_match = 0
                for qk in query_keywords:
                    if qk in sentence_keywords:
                        keyword_match += 1
                if keyword_match > 0:
                    # 包含关键词的句子加分
                    scores[i] += 0.2 * (keyword_match / len(query_keywords))
        
        # 将句子和得分配对并排序
        ranked_sentences = [(i, all_sentences[i], scores[i], doc_indices[i]) 
                          for i in range(len(all_sentences))]
        
        # 按得分排序
        ranked_sentences.sort(key=lambda x: x[2], reverse=True)
        
        # 选择压缩率 - 默认选择30%的句子，但至少要5个句子
        keep_sentence_count = max(5, int(len(all_sentences) * 0.3))
        top_sentences = ranked_sentences[:keep_sentence_count]
        
        # 按照原文档顺序重新排列
        selected_sentences = sorted(top_sentences, key=lambda x: (x[3], x[0]))
        
        # 选择最重要的句子直到达到token限制
        selected_text = ""
        current_tokens = 0
        
        for _, sentence, _, _ in selected_sentences:
            sentence_with_period = sentence + "。"
            sentence_tokens = len(sentence_with_period.split())
            
            if current_tokens + sentence_tokens <= max_tokens:
                selected_text += sentence_with_period + " "
                current_tokens += sentence_tokens
            else:
                # 如果剩余空间不多但仍有一些，添加一部分最后的句子
                if current_tokens < max_tokens * 0.8:
                    remaining = max_tokens - current_tokens
                    words = sentence_with_period.split()
                    partial = " ".join(words[:remaining])
                    selected_text += partial + "... "
                break
        
        # 如果没有选择任何句子，放松筛选标准再试一次
        if not selected_text.strip():
            # 按得分排序选择至少一些句子
            for _, sentence, _, _ in ranked_sentences[:3]:
                sentence_with_period = sentence + "。"
                selected_text += sentence_with_period + " "
                if len(selected_text.split()) >= max_tokens:
                    break
        
        # 仍然没有句子，回退到基本方法
        if not selected_text.strip():
            selected_text = " ".join(documents[0].split()[:max_tokens])
        
        end_time = time.time()
        compression_time = end_time - start_time
        
        # 计算压缩率
        original_tokens = sum(len(doc.split()) for doc in documents)
        compressed_tokens = len(selected_text.split())
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        
        print(f"\nTextRank压缩:")
        print(f"压缩用时: {compression_time:.4f} 秒")
        print(f"原始文档总词数: {original_tokens}")
        print(f"压缩后词数: {compressed_tokens}")
        print(f"压缩率: {compression_ratio:.2f}")
        
        return selected_text.strip()
    
    def info_density_compression(self, documents: List[str], query: str, max_tokens: int = 200) -> str:
        """
        基于信息密度的压缩算法
        
        计算每个句子的信息密度，保留信息最丰富的部分
        同时考虑查询相关性和信息量
        
        参数:
            documents: 文档列表
            query: 查询文本
            max_tokens: 最大标记数
            
        返回:
            压缩后的文本
        """
        start_time = time.time()
        
        # 快速路径检查
        if not documents:
            return ""
        
        # 提取所有句子
        all_sentences = []
        doc_indices = []  # 记录每个句子来自哪个文档
        for doc_idx, doc in enumerate(documents):
            sentences = self._split_into_sentences(doc)
            all_sentences.extend(sentences)
            doc_indices.extend([doc_idx] * len(sentences))
        
        if not all_sentences:
            return " ".join(documents[0].split()[:max_tokens])
        
        # 提取查询关键词
        query_keywords = self._extract_keywords(query)
        
        # 计算每个句子的信息密度
        density_scores = []
        for i, sentence in enumerate(all_sentences):
            # 计算信息密度
            density = self._calculate_information_density(sentence)
            
            # 计算与查询的相关性
            relevance = 0
            if query_keywords:
                relevance = self._calculate_sentence_importance(sentence, query_keywords)
            
            # 为包含查询关键词的句子加分
            keyword_bonus = 0
            if query_keywords:
                sentence_keywords = self._extract_keywords(sentence)
                keyword_matches = sum(1 for kw in query_keywords if kw in sentence_keywords)
                keyword_bonus = 0.3 * (keyword_matches / len(query_keywords)) if keyword_matches > 0 else 0
            
            # 综合评分：60%信息密度 + 40%查询相关性 + 关键词额外加分
            combined_score = 0.6 * density + 0.4 * relevance + keyword_bonus
            density_scores.append((i, sentence, combined_score, doc_indices[i]))
        
        # 按得分排序
        density_scores.sort(key=lambda x: x[2], reverse=True)
        
        # 选择压缩率 - 默认选择30%的句子，但至少要4个句子
        keep_sentence_count = max(4, int(len(all_sentences) * 0.3))
        top_sentences = density_scores[:keep_sentence_count]
        
        # 按照原文档顺序重新排列，保持文档结构
        selected_sentences = sorted(top_sentences, key=lambda x: (x[3], x[0]))
        
        # 选择密度最高的句子直到达到token限制
        selected_text = ""
        current_tokens = 0
        
        for _, sentence, _, _ in selected_sentences:
            sentence_with_period = sentence + "。"
            sentence_tokens = len(sentence_with_period.split())
            
            if current_tokens + sentence_tokens <= max_tokens:
                selected_text += sentence_with_period + " "
                current_tokens += sentence_tokens
            else:
                # 如果空间不足，添加部分句子
                if current_tokens < max_tokens * 0.8:
                    remaining = max_tokens - current_tokens
                    words = sentence_with_period.split()
                    partial = " ".join(words[:remaining])
                    selected_text += partial + "... "
                break
        
        # 如果没有选择任何句子，放松筛选标准
        if not selected_text.strip():
            # 按得分排序选择至少一些句子
            for _, sentence, _, _ in density_scores[:3]:
                sentence_with_period = sentence + "。"
                selected_text += sentence_with_period + " "
                if len(selected_text.split()) >= max_tokens:
                    break
        
        # 仍然没有句子，回退到基本方法
        if not selected_text.strip():
            selected_text = " ".join(documents[0].split()[:max_tokens])
        
        end_time = time.time()
        compression_time = end_time - start_time
        
        # 计算压缩率
        original_tokens = sum(len(doc.split()) for doc in documents)
        compressed_tokens = len(selected_text.split())
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        
        print(f"\n信息密度压缩:")
        print(f"压缩用时: {compression_time:.4f} 秒")
        print(f"原始文档总词数: {original_tokens}")
        print(f"压缩后词数: {compressed_tokens}")
        print(f"压缩率: {compression_ratio:.2f}")
        
        return selected_text.strip()
    
    def _build_similarity_graph(self, sentences: List[str]) -> np.ndarray:
        """
        构建句子相似度图
        
        参数:
            sentences: 句子列表
            
        返回:
            相似度矩阵
        """
        n = len(sentences)
        similarity_matrix = np.zeros((n, n))
        
        # 对每个句子预先提取关键词
        sentence_keywords = {}
        for i, sentence in enumerate(sentences):
            sentence_keywords[i] = self._extract_keywords(sentence)
        
        # 构建相似度矩阵
        for i in range(n):
            for j in range(n):
                if i != j:  # 不考虑自相似
                    # 使用Jaccard相似度
                    similarity = self._calculate_jaccard_similarity(
                        sentence_keywords[i], 
                        sentence_keywords[j]
                    )
                    similarity_matrix[i][j] = similarity
        
        # 正则化
        for i in range(n):
            row_sum = np.sum(similarity_matrix[i])
            if row_sum > 0:
                similarity_matrix[i] /= row_sum
        
        return similarity_matrix
    
    def _apply_textrank(self, similarity_matrix: np.ndarray, d: float = 0.85, max_iter: int = 50, tol: float = 1e-5) -> np.ndarray:
        """
        应用TextRank算法
        
        参数:
            similarity_matrix: 相似度矩阵
            d: 阻尼系数
            max_iter: 最大迭代次数
            tol: 收敛容差
            
        返回:
            句子评分
        """
        n = len(similarity_matrix)
        scores = np.ones(n) / n  # 初始化均匀分布
        
        # Power Method迭代
        for _ in range(max_iter):
            new_scores = (1 - d) / n + d * (similarity_matrix.T @ scores)
            
            # 检查收敛
            if np.linalg.norm(new_scores - scores) < tol:
                break
                
            scores = new_scores
        
        return scores
    
    def _calculate_jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """
        计算Jaccard相似度
        
        参数:
            set1: 第一个集合
            set2: 第二个集合
            
        返回:
            相似度值
        """
        if not set1 or not set2:
            return 0.0
            
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_information_density(self, text: str) -> float:
        """
        计算文本的信息密度
        
        参数:
            text: 输入文本
            
        返回:
            信息密度得分
        """
        if not text:
            return 0.0
        
        # 提取关键词
        keywords = self._extract_keywords(text)
        if not keywords:
            return 0.0
        
        # 计算词汇多样性
        words = text.split()
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        # 计算关键词密度
        keyword_density = len(keywords) / word_count
        
        # 计算长词比例
        long_words = sum(1 for w in words if len(w) >= 3)
        long_word_ratio = long_words / word_count if word_count > 0 else 0
        
        # 计算词汇多样性
        unique_words = len(set(words))
        diversity = unique_words / word_count if word_count > 0 else 0
        
        # 综合得分
        density_score = 0.3 * keyword_density + 0.3 * long_word_ratio + 0.4 * diversity
        
        return density_score

    def improved_map_reduce_compression(self, documents: List[str], query: str, max_tokens: int = 200) -> str:
        """
        改进的Map-Reduce压缩算法
        
        Map阶段：对每个文档提取精华内容
        Reduce阶段：合并并消除冗余
        
        参数:
            documents: 文档列表
            query: 查询文本
            max_tokens: 最大标记数
            
        返回:
            压缩后的文本
        """
        start_time = time.time()
        
        # 快速路径检查
        if not documents:
            return ""
        
        print("\n执行改进的Map-Reduce压缩:")
        print("-" * 40)
        
        # 提取查询关键词
        query_keywords = self._extract_keywords(query)
        
        # 目标压缩率 - 如果文档很长，压缩更多
        total_tokens = sum(len(doc.split()) for doc in documents)
        target_compression = min(0.5, max(0.3, max_tokens / total_tokens))
        
        # Map阶段：为每个文档提取最相关核心句子
        mapped_docs = []
        doc_sentences_map = {}  # 用于追踪每个句子所属的原文档
        
        for doc_idx, doc in enumerate(documents):
            sentences = self._split_into_sentences(doc)
            if not sentences:
                continue
            
            # 为每个句子计算多维评分
            sentence_features = []
            for sentence_idx, sentence in enumerate(sentences):
                # 相关性得分 - 与查询的相关性
                relevance = self._calculate_sentence_importance(sentence, query_keywords)
                
                # 信息密度 - 计算信息量
                info_density = self._calculate_information_density(sentence)
                
                # 位置得分 - 首尾句更重要
                position = 1.0
                if sentence_idx == 0:
                    position = 1.5  # 首句加权
                elif sentence_idx == len(sentences) - 1:
                    position = 1.2  # 尾句加权
                
                # 综合得分
                score = 0.5 * relevance + 0.3 * info_density + 0.2 * position
                
                # 仅保留得分超过阈值的句子
                if score > 0.1:  # 设置一个基本阈值
                    sentence_features.append((sentence, score))
                    # 记录句子所属文档
                    doc_sentences_map[sentence] = doc_idx
            
            # 每个文档只保留真正核心的句子 (自适应比例)
            target_sentences = max(1, min(int(len(sentences) * target_compression), int(max_tokens / (2 * len(documents)))))
            
            if sentence_features:  # 确保有句子可排序
                sentence_features.sort(key=lambda x: x[1], reverse=True)
                core_sentences = []
                
                # 添加前N个句子，同时避免内容过于相似
                for sent, score in sentence_features[:target_sentences * 2]:  # 先选择更多候选
                    # 检查是否与已选句子有高度重叠
                    if not any(self._sentence_similarity(sent, existing) > 0.6 for existing in core_sentences):
                        core_sentences.append(sent)
                        if len(core_sentences) >= target_sentences:
                            break
                
                if core_sentences:
                    # 确保句子按照它们在原文档中的顺序
                    ordered_core = []
                    for sent in sentences:
                        if sent in core_sentences:
                            ordered_core.append(sent)
                    
                    core_text = " ".join([s + "。" for s in ordered_core])
                    mapped_docs.append(core_text)
        
        # Reduce阶段：相似度聚类和合并
        if not mapped_docs:
            return " ".join(documents[0].split()[:max_tokens])
        
        # 对映射后的文档按与查询的相关性排序
        doc_scores = []
        for idx, doc in enumerate(mapped_docs):
            # 计算与查询的相关性
            query_score = self._calculate_keyword_score(doc, query_keywords)
            # 计算信息密度
            info_score = self._calculate_information_density(doc)
            # 综合评分
            combined_score = 0.7 * query_score + 0.3 * info_score
            doc_scores.append((idx, doc, combined_score))
        
        # 按综合得分排序
        doc_scores.sort(key=lambda x: x[2], reverse=True)
        
        # 聚类和去冗余 - 只保留差异化内容
        selected_docs = []
        selected_doc_indices = set()
        
        for _, doc, _ in doc_scores:
            # 检查当前文档是否与已选文档高度相似
            is_redundant = False
            for selected_doc in selected_docs:
                similarity = self._document_similarity(doc, selected_doc)
                if similarity > 0.6:  # 相似度阈值
                    is_redundant = True
                    break
            
            # 只添加非冗余文档
            if not is_redundant:
                selected_docs.append(doc)
        
        # 按重要性选择段落直到达到最大标记数
        combined_text = ""
        current_tokens = 0
        
        for doc in selected_docs:
            doc_tokens = len(doc.split())
            if current_tokens + doc_tokens <= max_tokens:
                combined_text += doc + " "
                current_tokens += doc_tokens
            else:
                # 部分添加最后一个文档
                remaining = max_tokens - current_tokens
                if remaining > 0:
                    partial = " ".join(doc.split()[:remaining])
                    combined_text += partial
                break
        
        # 如果组合文本为空，退回到简单的截断策略
        if not combined_text:
            all_text = " ".join(documents)
            combined_text = " ".join(all_text.split()[:max_tokens])
        
        # 计算最终压缩率
        end_time = time.time()
        compression_time = end_time - start_time
        compressed_tokens = len(combined_text.split())
        compression_ratio = compressed_tokens / total_tokens if total_tokens > 0 else 1.0
        
        print(f"压缩用时: {compression_time:.4f} 秒")
        print(f"原始文档总词数: {total_tokens}")
        print(f"压缩后词数: {compressed_tokens}")
        print(f"压缩率: {compression_ratio:.2f}")
        
        return combined_text.strip()
    
    def _document_similarity(self, doc1: str, doc2: str) -> float:
        """计算两个文档的相似度"""
        # 提取关键词
        keywords1 = self._extract_keywords(doc1)
        keywords2 = self._extract_keywords(doc2)
        
        # 使用Jaccard相似度
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        return intersection / union if union > 0 else 0.0

    def plot_compression_comparison(self, text: str, query: str, methods: List[str] = None):
        """
        绘制不同压缩方法的时间和压缩率对比图表
        """
        if methods is None:
            methods = ["sentence_importance", "map_reduce", "improved_sentence_importance",
                       "textrank", "information_density", "improved_map_reduce"]
            
        # 中英文方法名称映射
        method_name_mapping = {
            "sentence_importance": "Original Sentence Compression",
            "map_reduce": "Original Map-Reduce",
            "improved_sentence_importance": "Improved Sentence Compression",
            "textrank": "TextRank Compression",
            "information_density": "Information Density Compression",
            "improved_map_reduce": "Improved Map-Reduce"
        }
        
        compression_times = []
        compression_ratios = []
        
        for method in methods:
            start_time = time.time()
            compressed_text = getattr(self, method + "_compression")(text, query)
            end_time = time.time()
            
            compression_time = end_time - start_time
            compression_ratio = len(compressed_text) / len(text) if text else 0
            
            compression_times.append(compression_time)
            compression_ratios.append(compression_ratio)
            
            print(f"{method_name_mapping.get(method, method)}:")
            print(f"  - Compression Time: {compression_time:.4f} seconds")
            print(f"  - Compression Ratio: {compression_ratio:.4f}")
            print(f"  - Characters: {len(compressed_text)} / {len(text)}")
            print()
            
        # 创建两个子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 显示英文方法名称
        display_names = [method_name_mapping.get(method, method) for method in methods]
        
        # 时间对比图
        ax1.bar(display_names, compression_times, color='skyblue')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Compression Time Comparison')
        ax1.set_xticklabels(display_names, rotation=45, ha='right')
        
        # 在柱状图上添加数值标签
        for i, v in enumerate(compression_times):
            ax1.text(i, v + 0.01, f"{v:.2f}s", ha='center')
        
        # 压缩率对比图
        ax2.bar(display_names, compression_ratios, color='lightgreen')
        ax2.set_ylabel('Compression Ratio (lower is better)')
        ax2.set_title('Compression Ratio Comparison')
        ax2.set_ylim(0, 1.0)
        ax2.set_xticklabels(display_names, rotation=45, ha='right')
        
        # 在柱状图上添加数值标签
        for i, v in enumerate(compression_ratios):
            ax2.text(i, v + 0.02, f"{v:.2f}", ha='center')
            
        plt.tight_layout()
        plt.savefig('compression_methods_comparison.png')
        print("Compression methods comparison chart saved as 'compression_methods_comparison.png'")
        
        return compression_times, compression_ratios

class MapReduceCompressor:
    """
    专门用于Map-Reduce压缩的类 - 优化版本
    """
    
    def __init__(self, cache_enabled=True):
        """初始化Map-Reduce压缩器"""
        self.cache_enabled = cache_enabled
        
    def compress(self, documents, query, max_tokens=1000):
        """
        使用Map-Reduce方法压缩多个文档
        
        参数:
            documents: 文档列表
            query: 查询文本
            max_tokens: 压缩后的最大标记数
            
        返回:
            压缩后的文本
        """
        compressor = ContextCompressor(use_cache=self.cache_enabled)
        return compressor.map_reduce_compress(documents, query, max_tokens)

def test_context_compression():
    """测试上下文压缩算法 - 优化版本"""
    print("\n" + "="*80)
    print("上下文压缩与信息保留算法测试")
    print("="*80)
    
    # 创建测试数据 - 使用更长的文档来更好地测试压缩效果
    documents = [
        "向量检索是一种高效的相似度搜索方法，它通过将文本转换为向量表示来实现。这种方法能够捕捉语义关系，比传统的基于关键词匹配的检索方法更加精确。在向量检索中，文档和查询都被转换为高维空间中的向量，然后通过计算向量之间的距离或相似度来确定它们的相关性。",
        "在向量检索中，常用的相似度计算方法包括余弦相似度和点积相似度。余弦相似度测量两个向量之间的夹角，而点积则同时考虑了方向和大小。这些方法各有优缺点，选择哪种方法通常取决于具体应用场景和向量的特性。此外，欧几里得距离也经常被用于某些特定的向量检索任务。",
        "近似最近邻搜索算法如HNSW和LSH可以显著提高大规模向量检索的效率。HNSW(Hierarchical Navigable Small World)利用多层图结构实现高效导航，而LSH(Locality-Sensitive Hashing)则通过哈希函数将相似向量映射到相同的桶中。这些算法在保持一定准确率的同时，大大降低了计算复杂度，使得在数百万甚至数十亿向量的数据集上进行实时检索成为可能。",
        "检索增强生成技术结合了向量检索和语言模型，能够提供更准确的回答。这种方法首先使用向量检索从知识库中找到与查询相关的内容，然后将这些内容作为上下文提供给语言模型，引导模型生成基于事实的回答。这种技术有效减少了大型语言模型的幻觉问题，并使模型能够访问最新信息。",
        "向量检索的性能受到维度灾难的影响，高维空间中的距离计算变得困难。随着向量维度的增加，计算效率下降，且向量间的距离变得越来越相似，难以区分。为了解决这个问题，研究人员开发了多种降维技术和索引结构。此外，向量量化和稀疏向量表示等方法也被广泛用于优化向量存储和检索性能。向量数据库如FAISS、Milvus等专门设计用于高效管理和检索大规模向量数据。"
    ]
    
    queries = [
        "向量检索的相似度计算方法",
        "检索效率优化",
        "向量检索应用"
    ]
    
    # 创建压缩器实例 - 启用缓存提高性能
    compressor = ContextCompressor(use_cache=True)
    
    # 打印原始文档信息
    total_words = sum(compressor._count_words(doc) for doc in documents)
    print(f"\n原始文档总词数: {total_words}")
    
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
            compressed = method(documents, query, max_tokens=100)
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

if __name__ == "__main__":
    test_context_compression() 