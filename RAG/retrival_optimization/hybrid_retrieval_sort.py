import numpy as np
import time
from collections import defaultdict

def bm25_score(query_terms, document_terms, corpus_stats, k1=1.5, b=0.75):
    """
    计算BM25相似度分数
    
    参数:
        query_terms: 查询词条列表
        document_terms: 文档词条列表
        corpus_stats: 语料库统计，包含avg_doc_len和doc_freq
        k1: 词频饱和参数，通常在1.2-2.0之间
        b: 文档长度归一化参数，通常为0.75
        
    返回:
        BM25分数
    """
    # 文档词频
    doc_freq = defaultdict(int)
    for term in document_terms:
        doc_freq[term] += 1
        
    # 文档长度
    doc_len = len(document_terms)
    
    # 计算BM25分数
    score = 0.0
    for term in query_terms:
        if term in corpus_stats['doc_freq'] and term in doc_freq:
            # 逆文档频率 (IDF)
            df = corpus_stats['doc_freq'][term]
            N = corpus_stats['total_docs']
            idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
            
            # 词频，带饱和度和文档长度归一化
            tf = doc_freq[term]
            norm_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / corpus_stats['avg_doc_len']))
            
            score += idf * norm_tf
            
    return score

def vector_similarity(query_vec, doc_vec, method='cosine'):
    """
    计算向量相似度
    
    参数:
        query_vec: 查询向量
        doc_vec: 文档向量
        method: 相似度方法 ('cosine', 'dot', 'euclidean')
        
    返回:
        相似度分数
    """
    if method == 'cosine':
        # 余弦相似度
        dot_product = np.dot(query_vec, doc_vec)
        norm_q = np.linalg.norm(query_vec)
        norm_d = np.linalg.norm(doc_vec)
        if norm_q == 0 or norm_d == 0:
            return 0
        return dot_product / (norm_q * norm_d)
    
    elif method == 'dot':
        # 点积相似度
        return np.dot(query_vec, doc_vec)
    
    elif method == 'euclidean':
        # 欧几里得距离（转换为相似度）
        dist = np.linalg.norm(query_vec - doc_vec)
        return 1.0 / (1.0 + dist)  # 将距离转换为相似度
    
    else:
        raise ValueError(f"不支持的相似度方法: {method}")

def hybrid_retrieval_sort(query_vec, query_terms, candidate_docs, alpha=0.5):
    """
    混合检索排序算法，结合向量相似度和词法相似度
    
    参数:
        query_vec: 查询的向量表示
        query_terms: 查询的词条表示
        candidate_docs: 候选文档列表，每个文档是一个包含id、vector、terms和corpus_stats的字典
        alpha: 混合权重参数，控制向量相似度和BM25分数的相对重要性
        
    返回:
        按最终相似度排序的文档列表 [(id, score)]
    """
    results = []
    
    # 计算语料库统计
    corpus_stats = {
        'total_docs': len(candidate_docs),
        'avg_doc_len': np.mean([len(doc['terms']) for doc in candidate_docs]),
        'doc_freq': defaultdict(int)
    }
    
    # 计算文档频率
    for doc in candidate_docs:
        unique_terms = set(doc['terms'])
        for term in unique_terms:
            corpus_stats['doc_freq'][term] += 1
    
    for doc in candidate_docs:
        # 计算向量相似度分数
        vec_score = vector_similarity(query_vec, doc['vector'], method='cosine')
        
        # 计算BM25分数
        bm25_score_val = bm25_score(query_terms, doc['terms'], corpus_stats)
        
        # 归一化分数
        max_vec_score = 1.0  # 余弦相似度的最大值为1
        max_bm25_score = 10.0  # 假设BM25的最大值，实际中可能需要调整
        
        norm_vec_score = vec_score / max_vec_score
        norm_bm25_score = bm25_score_val / max_bm25_score
        
        # 混合分数
        hybrid_score = alpha * norm_vec_score + (1 - alpha) * norm_bm25_score
        
        results.append((doc['id'], hybrid_score))
    
    # 按分数降序排序
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def test_hybrid_retrieval_sort():
    """
    测试混合检索排序算法
    """
    # 模拟数据
    # 假设我们有一些电影文档，每个都有向量表示和词条表示
    docs = [
        {
            'id': 1,
            'title': '星球大战：新希望',
            'vector': np.array([0.8, 0.3, 0.5, 0.2]),
            'terms': ['星球', '大战', '科幻', '太空', '绝地武士', '原力']
        },
        {
            'id': 2,
            'title': '星球大战：帝国反击战',
            'vector': np.array([0.85, 0.35, 0.45, 0.25]),
            'terms': ['星球', '大战', '帝国', '反击战', '科幻', '太空', '绝地武士', '原力']
        },
        {
            'id': 3,
            'title': '黑客帝国',
            'vector': np.array([0.4, 0.8, 0.3, 0.1]),
            'terms': ['黑客', '帝国', '科幻', '虚拟现实', '矩阵']
        },
        {
            'id': 4,
            'title': '盗梦空间',
            'vector': np.array([0.3, 0.75, 0.4, 0.2]),
            'terms': ['盗梦', '空间', '科幻', '梦境', '潜意识']
        },
        {
            'id': 5,
            'title': '银翼杀手',
            'vector': np.array([0.6, 0.6, 0.4, 0.3]),
            'terms': ['银翼', '杀手', '科幻', '未来', '人工智能', '复制人']
        }
    ]
    
    # 测试查询
    query = {
        'text': '科幻太空原力电影',
        'terms': ['科幻', '太空', '原力', '电影'],
        'vector': np.array([0.75, 0.4, 0.5, 0.3])
    }
    
    print("测试开始: 混合检索排序算法")
    print(f"查询: {query['text']}")
    
    # 测试不同的混合权重
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    for alpha in alphas:
        print(f"\n混合权重 alpha={alpha}")
        print(f"  (alpha=0: 仅使用BM25词法匹配, alpha=1: 仅使用向量相似度)")
        
        results = hybrid_retrieval_sort(
            query['vector'], query['terms'], docs, alpha=alpha
        )
        
        # 打印结果
        for i, (doc_id, score) in enumerate(results):
            doc = next(d for d in docs if d['id'] == doc_id)
            print(f"  {i+1}. {doc['title']} (ID: {doc_id}, 分数: {score:.4f})")
    
if __name__ == "__main__":
    test_hybrid_retrieval_sort() 