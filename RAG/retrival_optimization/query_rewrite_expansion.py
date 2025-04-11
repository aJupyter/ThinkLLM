"""
查询重写与扩展算法模块

本模块实现了查询重写与扩展技术，用于改善检索效果。通过对原始查询进行修改和扩展，
可以提高检索系统的召回率和精确度，解决用户查询与文档表达之间的词汇鸿沟问题。

主要功能:
1. 查询重写 - 修正拼写错误、标准化格式、同义词替换等
2. 查询扩展 - 添加同义词和相关术语，增加查询覆盖面
3. 多语言支持 - 同时支持中英文查询处理

应用场景:
- 搜索引擎优化：解决用户查询不精确、表达不规范的问题
- 信息检索系统：弥合查询与文档表达之间的词汇差异
- RAG系统：提高检索组件的召回率，为生成提供更全面的上下文

实现方法采用了基于规则和统计的方法，支持同义词替换、相关术语扩展和拼写纠正等功能。
"""

import numpy as np
import re
from collections import Counter, defaultdict
import time
import random
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class QueryRewriter:
    """
    查询重写与扩展器，用于改善检索效果
    
    原理:
        查询重写和扩展技术通过修改原始查询来弥合用户表达与文档内容之间的词汇鸿沟：
        1. 查询重写：修正语法错误、拼写错误，规范化表达，替换同义词等
        2. 查询扩展：添加语义相关的词语，扩大查询的覆盖范围
        
    优点:
        - 提高检索系统的召回率，尤其是对含义相同但表达不同的情况
        - 增强检索的容错能力，对拼写错误和非标准表达更鲁棒
        - 可以引入领域知识，提高特定场景下的检索效果
        
    局限性:
        - 过度扩展可能引入噪声，导致精确度下降
        - 词典维护成本高，需要针对不同领域构建专用词典
        - 需要平衡召回率和精确度的权衡
    """
    
    def __init__(self):
        """
        初始化查询重写器，创建同义词、停用词和关联词的数据结构
        """
        # 同义词词典
        self.synonyms = {}
        # 停用词列表
        self.stopwords = set()
        # 关联词词典
        self.related_terms = defaultdict(list)
        
    def load_synonyms(self, synonym_dict):
        """
        加载同义词词典
        
        参数:
            synonym_dict (dict): 同义词词典，格式为 {词: [同义词列表]}
        """
        self.synonyms = synonym_dict
        
    def load_stopwords(self, stopwords_list):
        """
        加载停用词列表
        
        参数:
            stopwords_list (list): 停用词列表，包含应被过滤的常见词
        """
        self.stopwords = set(stopwords_list)
        
    def load_related_terms(self, related_terms_dict):
        """
        加载关联词词典
        
        参数:
            related_terms_dict (dict): 关联词词典，格式为 {词: [关联词列表]}
        """
        self.related_terms = related_terms_dict
        
    def _tokenize(self, text):
        """
        简单分词，支持中英文
        
        处理流程:
            1. 将文本转为小写
            2. 对中文文本，识别常见词组并按词切分
            3. 对英文文本，去除标点符号并按空格切分
            4. 过滤停用词
        
        参数:
            text (str): 输入文本
            
        返回:
            list: 分词后的词条列表
        """
        # 简单分词，实际应用中应使用更复杂的分词器
        text = text.lower()
        
        # 对于中文，按字符切分（实际应使用专业中文分词库如jieba）
        if any(u'\u4e00' <= char <= u'\u9fff' for char in text):
            # 中文文本，尝试按词切分
            # 简单规则：把常见词组作为整体识别
            common_terms = [
                '检索增强生成', '语言模型', '向量嵌入', '检索系统', '生成模型', 
                '大型语言模型', '效果不好', '检索效果', '准确性', '向量检索', '应用'
            ]
            
            # 先尝试找到常见词组
            tokens = []
            remaining_text = text
            
            for term in common_terms:
                if term in remaining_text:
                    remaining_text = remaining_text.replace(term, " PLACEHOLDER ")
                    tokens.append(term)
            
            # 对剩余文本切分
            for word in remaining_text.split():
                if word != "PLACEHOLDER":
                    tokens.append(word)
        else:
            # 英文文本，去除标点符号并按空格切分
            text = re.sub(r'[^\w\s]', '', text)
            tokens = text.split()
        
        # 去除停用词
        tokens = [token for token in tokens if token not in self.stopwords]
        return tokens
        
    def query_expansion(self, query, method='all', max_terms=3, weights=None):
        """
        查询扩展：通过添加同义词和相关词丰富原始查询
        
        扩展策略:
            1. 同义词扩展：查找原始查询中每个词的同义词
            2. 关联词扩展：查找原始查询中每个词的相关词
            3. 混合扩展：同时使用上述两种扩展方式
            
        权重机制:
            - 原始词条权重最高
            - 同义词次之
            - 关联词权重最低
        
        参数:
            query (str): 原始查询
            method (str): 扩展方法，可选值为 'synonym'(同义词扩展), 'related'(关联词扩展), 'all'(所有方法)
            max_terms (int): 每个原始词最多扩展的词数，默认3
            weights (dict): 不同扩展方法的权重，如 {'original': 1.0, 'synonym': 0.8, 'related': 0.5}
            
        返回:
            list: 扩展后的查询词条及其权重列表，格式为 [(词, 权重)]
        """
        if weights is None:
            weights = {
                'original': 1.0, 
                'synonym': 0.8, 
                'related': 0.5
            }
            
        # 分词
        tokens = self._tokenize(query)
        print(f"DEBUG: 分词结果: {tokens}")  # 调试信息
        
        # 初始化扩展结果
        expanded_query = [(token, weights['original']) for token in tokens]
        expanded_terms = set(tokens)
        
        # 根据指定方法扩展
        for token in tokens:
            # 同义词扩展
            if method in ['synonym', 'all'] and token in self.synonyms:
                synonyms = self.synonyms[token][:max_terms]
                print(f"DEBUG: 词'{token}'的同义词: {synonyms}")  # 调试信息
                for syn in synonyms:
                    if syn not in expanded_terms:
                        expanded_query.append((syn, weights['synonym']))
                        expanded_terms.add(syn)
            
            # 关联词扩展
            if method in ['related', 'all'] and token in self.related_terms:
                related = self.related_terms[token][:max_terms]
                print(f"DEBUG: 词'{token}'的相关词: {related}")  # 调试信息
                for rel in related:
                    if rel not in expanded_terms:
                        expanded_query.append((rel, weights['related']))
                        expanded_terms.add(rel)
                        
        return expanded_query
    
    def query_rewrite(self, query):
        """
        查询重写：包括拼写纠正、标准化、同义词替换等
        
        重写步骤:
            1. 将查询转为小写
            2. 去除多余空格，标准化格式
            3. 进行拼写纠正（使用预定义的常见错误映射）
            4. 概率性同义词替换（增加多样性）
        
        参数:
            query (str): 原始查询
            
        返回:
            str: 重写后的查询
        """
        print(f"DEBUG: 原始查询: '{query}'")  # 调试信息
        
        # 1. 转小写
        query = query.lower()
        
        # 2. 去除多余空格
        query = ' '.join(query.split())
        
        # 3. 拼写纠正（简化版）
        # 英文拼写纠正
        common_misspellings = {
            'retreival': 'retrieval',
            'alogrithm': 'algorithm',
            'serach': 'search',
            'vecter': 'vector',
            'similiarity': 'similarity'
        }
        
        # 4. 同义词替换（概率性替换，增加多样性）
        tokens = self._tokenize(query)
        rewritten_tokens = []
        
        for token in tokens:
            # 检查拼写错误
            if token in common_misspellings:
                token = common_misspellings[token]
            
            # 概率性同义词替换（30%概率）
            if token in self.synonyms and random.random() < 0.3:
                synonym = random.choice(self.synonyms[token])
                print(f"DEBUG: 替换词'{token}'为同义词'{synonym}'")  # 调试信息
                rewritten_tokens.append(synonym)
            else:
                rewritten_tokens.append(token)
                
        rewritten_query = ' '.join(rewritten_tokens)
        print(f"DEBUG: 重写后查询: '{rewritten_query}'")  # 调试信息
        return rewritten_query
        
    def full_rewrite_and_expand(self, query):
        """
        执行完整的查询重写和扩展流程
        
        处理流程:
            1. 首先进行查询重写（拼写纠正、标准化等）
            2. 然后对重写后的查询进行扩展（添加同义词、相关词）
        
        参数:
            query (str): 原始查询
            
        返回:
            tuple: (重写后的查询字符串, 扩展后的查询词条与权重列表)
        """
        # 首先进行查询重写
        rewritten_query = self.query_rewrite(query)
        
        # 然后进行查询扩展
        expanded_query = self.query_expansion(rewritten_query)
        
        return rewritten_query, expanded_query

def test_query_rewrite_expansion():
    """
    测试查询重写和扩展算法，展示在不同查询上的效果
    
    测试内容:
        1. 中英文查询的处理
        2. 拼写错误纠正
        3. 同义词替换和扩展
        4. 相关词扩展
        
    测试数据包括:
        - 英文查询：包含拼写错误、专业术语等
        - 中文查询：包含领域术语、复合词等
    """
    # 初始化查询重写器
    rewriter = QueryRewriter()
    
    # 加载英文示例同义词
    en_synonyms = {
        'retrieval': ['search', 'fetch', 'lookup'],
        'algorithm': ['method', 'procedure', 'technique'],
        'vector': ['embedding', 'array', 'tensor'],
        'search': ['lookup', 'find', 'query'],
        'similarity': ['closeness', 'resemblance', 'likeness']
    }
    
    # 加载中文示例同义词
    cn_synonyms = {
        '检索': ['搜索', '查询', '获取'],
        '生成': ['创建', '产生', '构建'],
        '语言模型': ['LLM', '大语言模型', '文本生成模型'],
        '向量': ['嵌入', '特征向量', '张量'],
        '相似度': ['距离', '匹配度', '接近度'],
        '效果': ['性能', '表现', '准确率'],
        '提高': ['增强', '优化', '改进'],
        '应用': ['用途', '场景', '使用案例']
    }
    
    # 合并词典
    all_synonyms = {**en_synonyms, **cn_synonyms}
    rewriter.load_synonyms(all_synonyms)
    
    # 加载示例停用词
    stopwords = ['a', 'an', 'the', 'in', 'on', 'at', 'for', 'with', 'by', 
                '的', '了', '在', '是', '和', '与', '或', '有']
    rewriter.load_stopwords(stopwords)
    
    # 加载示例关联词
    en_related_terms = {
        'retrieval': ['index', 'query', 'rank', 'relevance'],
        'algorithm': ['complexity', 'efficiency', 'implementation'],
        'vector': ['dimension', 'distance', 'space', 'scalar'],
        'search': ['engine', 'result', 'ranking', 'index'],
        'similarity': ['distance', 'metric', 'measure', 'proximity']
    }
    
    # 加载中文示例关联词
    cn_related_terms = {
        '检索': ['索引', '排序', '搜索引擎', '召回率'],
        '生成': ['文本生成', '内容创建', '对话系统', '补全'],
        '语言模型': ['GPT', 'BERT', 'Transformer', '深度学习'],
        '向量': ['维度', '嵌入空间', '向量数据库', '降维'],
        '文档': ['段落', '语料库', '文章', '内容'],
        '效果': ['准确率', '召回率', 'F1分数', '评估指标'],
        '提高': ['方法', '技术', '策略', '优化方案'],
        '应用': ['实例', '产品', '解决方案', '落地场景']
    }
    
    # 合并词典
    all_related_terms = {**en_related_terms, **cn_related_terms}
    rewriter.load_related_terms(all_related_terms)
    
    # 英文测试查询
    en_test_queries = [
        "vecter similiarity alogrithm",  # 包含拼写错误
        "retrieval algorithm for search engine",
        "vector similarity in high dimensions",
        "efficient serach methods"
    ]
    
    # 中文测试查询
    cn_test_queries = [
        "检索效果不好",  # 直接匹配词典中的词
        "如何提高向量检索的准确性",  # 包含'向量'、'提高'等词典中的词
        "语言模型有什么应用",  # 使用'语言模型'词组
        "检索增强生成技术"  # 测试复合词
    ]
    
    print("测试开始: 查询重写与扩展算法")
    
    print("\n==== 测试英文查询 ====")
    for i, query in enumerate(en_test_queries):
        print(f"\n查询 {i+1}: \"{query}\"")
        
        # 测试查询重写
        rewritten = rewriter.query_rewrite(query)
        print(f"重写后: \"{rewritten}\"")
        
        # 完整的查询重写和扩展
        rewritten, expanded_all = rewriter.full_rewrite_and_expand(query)
        print("查询扩展结果:")
        for term, weight in expanded_all:
            print(f"  {term}: {weight:.1f}")
    
    print("\n==== 测试中文查询 ====")
    for i, query in enumerate(cn_test_queries):
        print(f"\n查询 {i+1}: \"{query}\"")
        
        # 测试查询重写
        rewritten = rewriter.query_rewrite(query)
        print(f"重写后: \"{rewritten}\"")
        
        # 完整的查询重写和扩展
        rewritten, expanded_all = rewriter.full_rewrite_and_expand(query)
        print("查询扩展结果:")
        for term, weight in expanded_all:
            print(f"  {term}: {weight:.1f}")
        
if __name__ == "__main__":
    test_query_rewrite_expansion() 