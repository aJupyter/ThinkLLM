"""
混合检索排序算法模块

本模块实现了结合语义检索和关键词匹配的混合检索排序算法。
主要包含以下功能:
1. BM25词法相似度计算 - 基于词频和逆文档频率的经典信息检索算法
2. 向量语义相似度计算 - 支持余弦相似度、点积和欧氏距离三种方式
3. 混合检索排序 - 通过可调整权重参数平衡语义检索和词法匹配的重要性

混合检索的核心思想是结合两种检索范式的优点:
- 向量检索：擅长捕捉语义相关性，即使词汇不完全匹配
- 词法检索：擅长精确匹配关键词，对特定术语查询效果好

该算法通过alpha参数控制两种检索方法的权重，实现灵活的混合策略。
"""

import numpy as np
import time
from collections import defaultdict

def bm25_score(query_terms, document_terms, corpus_stats, k1=1.5, b=0.75):
    """
    计算查询和文档之间的BM25相似度分数
    
    原理:
        BM25 (Best Matching 25) 是一种常用的信息检索排序算法，它基于词频 (TF) 和逆文档频率 (IDF)。
        相比传统的TF-IDF，BM25引入了两个可调参数k1和b：
        - k1: 控制词频饱和度。较高的k1意味着词频对分数的影响会更快达到饱和。
        - b:  控制文档长度归一化的程度。b=1表示完全归一化，b=0表示不归一化。
        公式通常包含三部分：
        1. IDF项：衡量词的稀有度。一个词在越少的文档中出现，IDF值越高。
        2. TF项：衡量词在当前文档中的重要性，经过k1参数饱和处理。
        3. 文档长度归一化项：使用b参数调整文档长度对分数的影响，惩罚过长的文档。
    
    参数:
        query_terms (List[str]): 查询中的词条列表。
        document_terms (List[str]): 文档中的词条列表。
        corpus_stats (dict): 包含整个语料库统计信息的字典:
            - 'total_docs' (int): 语料库中的文档总数 (N)。
            - 'avg_doc_len' (float): 语料库中文档的平均长度。
            - 'doc_freq' (defaultdict[str, int]): 包含每个词条的文档频率 (df) 的字典。
        k1 (float): BM25的词频饱和参数，默认1.5。
        b (float): BM25的文档长度归一化参数，默认0.75。
        
    返回:
        float: 计算得到的BM25分数。分数越高表示文档与查询越相关。
    """
    # 计算当前文档的词频
    doc_freq = defaultdict(int)
    for term in document_terms:
        doc_freq[term] += 1
        
    # 获取当前文档的长度
    doc_len = len(document_terms)
    
    # 初始化BM25分数
    score = 0.0
    
    # 遍历查询中的每个词条
    for term in query_terms:
        # 确保该词条在语料库统计中存在，并且也在当前文档中出现
        if term in corpus_stats['doc_freq'] and term in doc_freq:
            # 获取该词条的文档频率 (df) 和语料库总文档数 (N)
            df = corpus_stats['doc_freq'][term]
            N = corpus_stats['total_docs']
            
            # 计算逆文档频率 (IDF)
            # 加1是为了处理df=N的情况，并避免IDF为负
            # +0.5是为了平滑处理，避免df=0或df=N时出现问题
            idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
            
            # 获取词条在当前文档中的词频 (tf)
            tf = doc_freq[term]
            
            # 计算归一化的词频部分
            # 分子: tf * (k1 + 1) - 词频越高，分数越高，但受k1饱和控制
            # 分母: tf + k1 * (1 - b + b * doc_len / avg_doc_len)
            #       - 包含词频自身
            #       - 包含文档长度归一化因子:
            #         - doc_len / avg_doc_len: 当前文档长度相对于平均长度的比例
            #         - b * ...: 由参数b控制归一化强度
            #         - 1 - b: 保留一部分不受长度影响的分数
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / corpus_stats['avg_doc_len'])
            norm_tf = numerator / denominator
            
            # 将当前词条的贡献加到总分中 (IDF * 归一化TF)
            score += idf * norm_tf
            
    return score

def vector_similarity(query_vec, doc_vec, method='cosine'):
    """
    计算两个向量之间的相似度
    
    支持多种相似度计算方法: 余弦相似度, 点积, 欧氏距离 (转换为相似度)
    
    参数:
        query_vec (np.ndarray): 查询向量。
        doc_vec (np.ndarray): 文档向量。
        method (str): 要使用的相似度计算方法。可选值为 'cosine', 'dot', 'euclidean'。默认为 'cosine'。
        
    返回:
        float: 计算得到的相似度分数。
               - 余弦相似度: 范围 [-1, 1]，1表示完全相似。
               - 点积: 范围 (-inf, +inf)，受向量模长影响。
               - 欧氏距离相似度: 范围 (0, 1]，1表示完全相同 (距离为0)。
               
    异常:
        ValueError: 如果提供了不支持的 `method`。
    """
    if method == 'cosine':
        # 计算点积
        dot_product = np.dot(query_vec, doc_vec)
        # 计算向量的L2范数 (模长)
        norm_q = np.linalg.norm(query_vec)
        norm_d = np.linalg.norm(doc_vec)
        # 检查是否存在零向量，避免除以零
        if norm_q == 0 or norm_d == 0:
            return 0.0
        # 计算余弦相似度: 点积 / (模长乘积)
        return dot_product / (norm_q * norm_d)
    
    elif method == 'dot':
        # 直接计算点积
        return np.dot(query_vec, doc_vec)
    
    elif method == 'euclidean':
        # 计算欧氏距离
        dist = np.linalg.norm(query_vec - doc_vec)
        # 将距离转换为相似度: 1 / (1 + 距离)
        # 距离越小，相似度越接近1
        return 1.0 / (1.0 + dist)
    
    else:
        # 如果方法无效，抛出错误
        raise ValueError(f"不支持的相似度方法: {method}")

def hybrid_retrieval_sort(query_vec, query_terms, candidate_docs, alpha=0.5):
    """
    实现混合检索排序算法，结合向量语义相似度和BM25词法相似度
    
    原理:
        混合检索旨在结合两种不同检索范式的优点：
        1. 向量检索 (语义相似度): 擅长捕捉语义相关性，即使词语不完全匹配。
        2. 词法检索 (如BM25): 擅长匹配精确关键词，对于特定术语查询效果好。
        通过线性加权组合两种相似度分数，得到一个综合的混合分数，用于最终排序。
        混合权重 alpha 控制两者的相对重要性。alpha=1 时只考虑向量相似度，alpha=0 时只考虑BM25。
        
    步骤:
        1. 计算语料库的统计信息 (文档总数、平均长度、词的文档频率)。
        2. 对每个候选文档:
           a. 计算其与查询向量的向量相似度 (例如余弦相似度)。
           b. 计算其与查询词条的BM25分数。
           c. (可选但推荐) 将两种分数归一化到相似范围 (例如 [0, 1])。
           d. 使用 alpha 参数计算混合分数: alpha * norm_vec_score + (1 - alpha) * norm_bm25_score。
        3. 按混合分数对所有候选文档进行降序排序。
        
    参数:
        query_vec (np.ndarray): 查询的向量表示。
        query_terms (List[str]): 查询的词条表示 (用于BM25)。
        candidate_docs (List[dict]): 候选文档列表。每个文档是字典，至少包含:
            - 'id': 文档的唯一标识符。
            - 'vector' (np.ndarray): 文档的向量表示。
            - 'terms' (List[str]): 文档的词条表示。
        alpha (float): 混合权重参数，范围 [0, 1]。控制向量相似度 (alpha) 和BM25分数 (1-alpha) 的相对重要性。默认为0.5。
        
    返回:
        List[Tuple[any, float]]: 按最终混合相似度降序排序的文档列表，格式为 [(文档ID, 混合分数)]。
    """
    # 存储最终结果的列表
    results = []
    
    # 1. 计算语料库统计信息
    num_docs = len(candidate_docs)
    # 计算所有文档的平均词条数 (平均长度)
    avg_doc_len = np.mean([len(doc['terms']) for doc in candidate_docs]) if candidate_docs else 0
    # 计算每个词条在多少个文档中出现 (文档频率)
    doc_freq_map = defaultdict(int)
    for doc in candidate_docs:
        unique_terms_in_doc = set(doc['terms'])
        for term in unique_terms_in_doc:
            doc_freq_map[term] += 1
            
    corpus_stats = {
        'total_docs': num_docs,
        'avg_doc_len': avg_doc_len,
        'doc_freq': doc_freq_map
    }
    
    # 2. 遍历每个候选文档，计算分数
    for doc in candidate_docs:
        # a. 计算向量相似度分数 (使用余弦相似度)
        vec_score = vector_similarity(query_vec, doc['vector'], method='cosine')
        
        # b. 计算BM25分数
        bm25_score_val = bm25_score(query_terms, doc['terms'], corpus_stats)
        
        # c. 归一化分数 (简化处理)
        # 余弦相似度自然在[-1, 1]范围内，对于非负向量通常在[0, 1]
        # BM25分数范围不固定，这里假设一个最大值进行归一化，实际应用中可能需要更复杂的归一化策略
        max_vec_score = 1.0
        max_bm25_score = 10.0  # 假设BM25的最大值，需要根据实际数据调整
        
        # 确保分数不为负（对于相似度）且避免除以零
        norm_vec_score = max(0, vec_score) / max_vec_score
        norm_bm25_score = max(0, bm25_score_val) / (max_bm25_score + 1e-6) # 加小量避免除零
        
        # d. 计算混合分数
        hybrid_score = alpha * norm_vec_score + (1 - alpha) * norm_bm25_score
        
        # 将结果添加到列表
        results.append((doc['id'], hybrid_score))
    
    # 3. 按混合分数降序排序
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results

def test_hybrid_retrieval_sort():
    """
    测试混合检索排序算法的功能和效果
    
    使用一个小的电影数据集进行演示。
    展示不同alpha值下，排序结果如何从偏向词法匹配过渡到偏向语义匹配。
    """
    # 模拟电影文档数据
    docs = [
        {
            'id': 1,
            'title': '星球大战：新希望',
            'vector': np.array([0.8, 0.3, 0.5, 0.2]),
            'terms': ['星球', '大战', '科幻', '太空', '绝地武士', '原力'] # 强关键词匹配
        },
        {
            'id': 2,
            'title': '星球大战：帝国反击战',
            'vector': np.array([0.85, 0.35, 0.45, 0.25]), # 与查询向量更接近
            'terms': ['星球', '大战', '帝国', '反击战', '科幻', '太空', '绝地武士', '原力']
        },
        {
            'id': 3,
            'title': '黑客帝国',
            'vector': np.array([0.4, 0.8, 0.3, 0.1]),
            'terms': ['黑客', '帝国', '科幻', '虚拟现实', '矩阵'] # 部分关键词匹配
        },
        {
            'id': 4,
            'title': '盗梦空间',
            'vector': np.array([0.3, 0.75, 0.4, 0.2]),
            'terms': ['盗梦', '空间', '科幻', '梦境', '潜意识'] # 只有"科幻"匹配
        },
        {
            'id': 5,
            'title': '银翼杀手',
            'vector': np.array([0.6, 0.6, 0.4, 0.3]), # 向量相似度中等
            'terms': ['银翼', '杀手', '科幻', '未来', '人工智能', '复制人']
        }
    ]
    
    # 定义测试查询
    query = {
        'text': '科幻太空原力电影', # 包含多个关键词
        'terms': ['科幻', '太空', '原力', '电影'], # 用于BM25
        'vector': np.array([0.75, 0.4, 0.5, 0.3]) # 用于向量相似度
    }
    
    print("测试开始: 混合检索排序算法")
    print(f"查询: {query['text']}")
    
    # 测试不同的混合权重 alpha
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    for alpha in alphas:
        print(f"混合权重 alpha={alpha}")
        if alpha == 0.0:
            print("  (仅使用BM25词法匹配 - 预期《星球大战：新希望》排名靠前，因关键词完全匹配)")
        elif alpha == 1.0:
            print("  (仅使用向量相似度 - 预期《星球大战：帝国反击战》排名靠前，因向量更接近)")
        else:
             print(f"  (结合BM25和向量相似度, 权重 {1-alpha:.1f} / {alpha:.1f})")
        
        # 调用混合检索排序函数
        results = hybrid_retrieval_sort(
            query['vector'], query['terms'], docs, alpha=alpha
        )
        
        # 打印排序结果
        for i, (doc_id, score) in enumerate(results):
            # 找到对应文档的标题
            doc = next(d for d in docs if d['id'] == doc_id)
            print(f"  {i+1}. {doc['title']} (ID: {doc_id}, 分数: {score:.4f})")
    
if __name__ == "__main__":
    test_hybrid_retrieval_sort() 