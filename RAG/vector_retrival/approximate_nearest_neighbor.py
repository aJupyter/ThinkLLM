import numpy as np
from collections import defaultdict
import time
import random

class LSHIndex:
    """
    使用局部敏感哈希(LSH)实现的近似最近邻算法
    这是一种简单的ANN实现，专注于随机投影LSH
    """
    
    def __init__(self, dim, num_tables=5, num_bits=10):
        """
        初始化LSH索引
        
        参数:
            dim: 输入向量的维度
            num_tables: 哈希表的数量
            num_bits: 每个哈希值的比特数
        """
        self.dim = dim
        self.num_tables = num_tables
        self.num_bits = num_bits
        self.tables = [defaultdict(list) for _ in range(num_tables)]
        
        # 为每个表创建随机投影向量
        self.random_vectors = [
            np.random.randn(num_bits, dim) for _ in range(num_tables)
        ]
        
    def _hash_vector(self, vector, table_idx):
        """
        使用随机投影将向量哈希为二进制字符串
        
        参数:
            vector: 要哈希的向量
            table_idx: 使用哪个哈希表
            
        返回:
            哈希值作为字符串
        """
        projections = np.dot(self.random_vectors[table_idx], vector)
        return ''.join(['1' if p > 0 else '0' for p in projections])
    
    def index(self, vectors, ids=None):
        """
        将向量集合索引到哈希表中
        
        参数:
            vectors: 要索引的向量列表
            ids: 向量的ID（如果未提供，则使用索引）
        """
        if ids is None:
            ids = list(range(len(vectors)))
            
        for i, vector in enumerate(vectors):
            for table_idx in range(self.num_tables):
                hash_val = self._hash_vector(vector, table_idx)
                self.tables[table_idx][hash_val].append((ids[i], vector))
    
    def query(self, query_vector, k=5):
        """
        查询最近的k个邻居
        
        参数:
            query_vector: 查询向量
            k: 要返回的近邻数量
            
        返回:
            (id, 距离) 的列表，按距离排序
        """
        # 使用列表存储候选项而不是集合，因为元组(id, vector)不可哈希
        candidates = []
        seen_ids = set()  # 跟踪已添加的ID，防止重复
        
        # 从所有表中收集候选项
        for table_idx in range(self.num_tables):
            hash_val = self._hash_vector(query_vector, table_idx)
            # 获取与查询向量在同一个桶中的所有向量
            matches = self.tables[table_idx].get(hash_val, [])
            
            # 添加主哈希值的匹配
            for id_vec in matches:
                if id_vec[0] not in seen_ids:
                    candidates.append(id_vec)
                    seen_ids.add(id_vec[0])
            
            # 提高召回率: 搜索汉明距离为1的哈希桶
            if len(candidates) < k * 2:  # 如果候选数量不足，扩大搜索范围
                for bit_idx in range(self.num_bits):
                    flipped_hash = list(hash_val)
                    flipped_hash[bit_idx] = '1' if hash_val[bit_idx] == '0' else '0'
                    flipped_hash = ''.join(flipped_hash)
                    
                    flipped_matches = self.tables[table_idx].get(flipped_hash, [])
                    for id_vec in flipped_matches:
                        if id_vec[0] not in seen_ids:
                            candidates.append(id_vec)
                            seen_ids.add(id_vec[0])

            # 提高召回率: 如果仍然候选不足，搜索汉明距离为2的哈希桶(限制在特定条件下)
            if len(candidates) < k and self.num_bits <= 16:
                for i in range(self.num_bits):
                    for j in range(i+1, self.num_bits):
                        double_flipped_hash = list(hash_val)
                        double_flipped_hash[i] = '1' if hash_val[i] == '0' else '0'
                        double_flipped_hash[j] = '1' if hash_val[j] == '0' else '0'
                        double_flipped_hash = ''.join(double_flipped_hash)
                        
                        double_matches = self.tables[table_idx].get(double_flipped_hash, [])
                        for id_vec in double_matches:
                            if id_vec[0] not in seen_ids:
                                candidates.append(id_vec)
                                seen_ids.add(id_vec[0])
                        
                        if len(candidates) >= k * 3:  # 一旦有足够候选，停止搜索
                            break
                    if len(candidates) >= k * 3:
                        break
        
        # 如果仍然没有找到足够候选项，添加随机样本
        if len(candidates) < k:
            # 从所有表中随机选择
            for table_idx in range(self.num_tables):
                for hash_bucket in list(self.tables[table_idx].values()):
                    for id_vec in hash_bucket:
                        if id_vec[0] not in seen_ids:
                            candidates.append(id_vec)
                            seen_ids.add(id_vec[0])
                            if len(candidates) >= k * 5:  # 确保有足够多的候选
                                break
                    if len(candidates) >= k * 5:
                        break
                if len(candidates) >= k * 5:
                    break
        
        # 如果仍然没有找到候选项，返回空列表
        if not candidates:
            return []
        
        # 向量归一化以提高召回率(使用余弦相似度而不是欧几里得距离)
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_unit = query_vector / query_norm
            
            # 使用余弦相似度排序
            candidates_with_distances = []
            for idx, vec in candidates:
                vec_norm = np.linalg.norm(vec)
                if vec_norm > 0:
                    similarity = np.dot(query_unit, vec / vec_norm)
                    # 转换为距离(值越小越相似)
                    distance = 1 - similarity
                    candidates_with_distances.append((idx, distance))
                else:
                    candidates_with_distances.append((idx, 2.0))  # 默认最大距离
        else:
            # 如果查询向量是零向量，使用欧几里得距离
            candidates_with_distances = [
                (idx, np.linalg.norm(query_vector - vec)) 
                for idx, vec in candidates
            ]
        
        # 按距离排序并返回前k个
        candidates_with_distances.sort(key=lambda x: x[1])
        return candidates_with_distances[:k]

def test_ann_algorithm():
    """
    测试ANN算法的性能
    """
    # 创建一些随机向量
    dim = 64
    num_vectors = 10000
    vectors = [np.random.randn(dim) for _ in range(num_vectors)]
    
    # 创建测试查询 - 使用其中一个向量作为查询，确保能找到至少一个精确匹配
    query_idx = random.randint(0, num_vectors-1)
    query = vectors[query_idx].copy()
    
    print("测试开始: 近似最近邻算法 (LSH)")
    print(f"使用向量 #{query_idx} 作为查询向量，确保至少有一个完全匹配")
    
    # 构建LSH索引
    start_time = time.time()
    lsh = LSHIndex(dim, num_tables=15, num_bits=12)  # 增加表数量，减少位数以提高召回率
    lsh.index(vectors)
    index_time = time.time() - start_time
    print(f"LSH索引构建时间: {index_time:.4f} 秒 (向量数量: {num_vectors})")
    
    # 测试LSH查询
    start_time = time.time()
    lsh_results = lsh.query(query, k=5)
    lsh_query_time = max(time.time() - start_time, 0.0001)  # 避免除以零
    
    # 暴力搜索作为基准
    start_time = time.time()
    bruteforce_dists = [(i, np.linalg.norm(query - vec)) for i, vec in enumerate(vectors)]
    bruteforce_dists.sort(key=lambda x: x[1])
    bf_query_time = max(time.time() - start_time, 0.0001)  # 避免除以零
    
    print("\n近似最近邻算法性能比较:")
    print(f"LSH索引构建时间: {index_time:.4f} 秒 (向量数量: {num_vectors})")
    print(f"\nLSH查询时间: {lsh_query_time:.6f} 秒")
    print(f"暴力搜索时间: {bf_query_time:.6f} 秒")
    
    print(f"\nLSH加速比: {bf_query_time/lsh_query_time:.2f}x")
    
    # 计算召回率
    gt_ids = set([idx for idx, _ in bruteforce_dists[:5]])
    lsh_ids = set([idx for idx, _ in lsh_results])
    recall = len(gt_ids.intersection(lsh_ids)) / max(len(gt_ids), 1)
    print(f"LSH召回率 (前5个结果): {recall:.2f}")
    
    # 是否找到查询向量自身
    query_found = query_idx in lsh_ids
    print(f"查询向量 #{query_idx} 是否被找到: {'是' if query_found else '否'}")
    
    # 详细打印结果对比信息
    print("\n=== 搜索结果详细对比 ===")
    print("\nLSH查询结果 (ID, 距离):")
    for i, (idx, dist) in enumerate(lsh_results):
        print(f"  #{i+1}: ID {idx}, 距离: {dist:.4f}")
    
    print("\n暴力搜索结果 (前5个) (ID, 距离):")
    for i, (idx, dist) in enumerate(bruteforce_dists[:5]):
        print(f"  #{i+1}: ID {idx}, 距离: {dist:.4f}")
    
    # 打印性能指标表格
    print("\n=== 性能指标总结 ===")
    print("-" * 50)
    print(f"{'指标':<15}{'LSH':<15}{'暴力搜索':<15}{'对比':<15}")
    print("-" * 50)
    print(f"{'索引时间(秒)':<15}{index_time:<15.4f}{'-':<15}{'-':<15}")
    print(f"{'查询时间(秒)':<15}{lsh_query_time:<15.6f}{bf_query_time:<15.6f}{bf_query_time/lsh_query_time:<15.2f}x")
    print(f"{'召回率':<15}{recall:<15.2f}{'1.00':<15}{recall/1.0:<15.2f}")
    print("-" * 50)
    
    # 打印召回率详细信息
    print("\n=== 召回率详细信息 ===")
    print(f"暴力搜索找到的Top 5 IDs: {sorted(list(gt_ids))}")
    print(f"LSH索引找到的Top 5 IDs: {sorted(list(lsh_ids))}")
    print(f"重叠的IDs: {sorted(list(gt_ids.intersection(lsh_ids)))}")
    print(f"召回率: {len(gt_ids.intersection(lsh_ids))}/{len(gt_ids)} = {recall:.2f}")
    
    return lsh
    
if __name__ == "__main__":
    test_ann_algorithm() 