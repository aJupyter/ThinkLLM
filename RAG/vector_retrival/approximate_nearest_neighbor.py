import numpy as np
from collections import defaultdict
import time
import random

class LSHIndex:
    """
    使用局部敏感哈希(LSH)实现的近似最近邻(ANN)搜索索引
    
    原理:
        LSH的基本思想是设计一组哈希函数，使得相似的向量有更高的概率哈希到同一个"桶"中，
        而不相似的向量有更高的概率哈希到不同的桶中。
        查询时，只需要比较查询向量所在桶内的向量，从而大大减少比较次数。
        此实现使用随机投影(Random Projection)作为LSH族：
        1. 生成多组随机向量(投影平面)。
        2. 将输入向量投影到这些随机向量上。
        3. 根据投影结果（通常是正负）生成哈希值（二进制编码）。
        4. 使用多个哈希表（每个表使用不同的随机投影集）来提高召回率。
        
    优点:
        - 构建索引速度相对较快。
        - 适用于高维数据。
        
    缺点:
        - 为了获得高召回率，通常需要较多的哈希表和/或哈希位数，可能导致内存占用较大。
        - 召回率可能不如其他一些ANN算法（如HNSW）稳定。
        - 对参数（哈希表数量、哈希位数）敏感。
    """
    
    def __init__(self, dim, num_tables=5, num_bits=10):
        """
        初始化LSH索引
        
        参数:
            dim (int): 输入向量的维度
            num_tables (int): 哈希表的数量。增加表数量可以提高召回率，但会增加内存和查询时间。
            num_bits (int): 每个哈希值的比特数（即随机投影向量的数量）。
                           增加比特数可以提高精度（减少误报），但可能降低召回率（相似向量落入不同桶的概率增加）。
        """
        self.dim = dim
        self.num_tables = num_tables
        self.num_bits = num_bits
        # 创建 num_tables 个哈希表，每个表是一个字典，键是哈希值，值是 (id, vector) 列表
        self.tables = [defaultdict(list) for _ in range(num_tables)]
        
        # 为每个哈希表生成一组随机投影向量 (num_bits 个 dim 维向量)
        # 这些向量定义了哈希函数
        self.random_vectors = [
            np.random.randn(num_bits, dim) for _ in range(num_tables)
        ]
        
    def _hash_vector(self, vector, table_idx):
        """
        计算给定向量在指定哈希表中的哈希值
        
        参数:
            vector (np.ndarray): 要哈希的向量
            table_idx (int): 使用哪个哈希表的索引 (0 到 num_tables-1)
            
        返回:
            str: 表示哈希值的二进制字符串 (长度为 num_bits)
        """
        # 计算向量与该表随机投影向量的点积
        # projections 的形状是 (num_bits,)
        projections = np.dot(self.random_vectors[table_idx], vector)
        # 根据点积的正负生成二进制哈希值：>0 为 '1', <=0 为 '0'
        return ''.join(['1' if p > 0 else '0' for p in projections])
    
    def index(self, vectors, ids=None):
        """
        将一组向量添加到LSH索引中
        
        参数:
            vectors (List[np.ndarray]): 要索引的向量列表
            ids (List, optional): 每个向量对应的ID列表。如果为None，则使用向量的索引作为ID。
        """
        if ids is None:
            ids = list(range(len(vectors)))
            
        # 遍历每个向量
        for i, vector in enumerate(vectors):
            # 将向量添加到所有 num_tables 个哈希表中
            for table_idx in range(self.num_tables):
                # 计算该向量在当前哈希表中的哈希值
                hash_val = self._hash_vector(vector, table_idx)
                # 将 (向量ID, 向量本身) 添加到对应哈希桶的列表中
                self.tables[table_idx][hash_val].append((ids[i], vector))
    
    def query(self, query_vector, k=5):
        """
        查询与给定查询向量最接近的k个邻居
        
        参数:
            query_vector (np.ndarray): 查询向量
            k (int): 要返回的最近邻数量
            
        返回:
            List[Tuple[int, float]]: 包含 (ID, 距离) 的列表，按距离升序排序
        """
        # 候选邻居列表，存储 (id, vector)
        candidates = []
        # 用于记录已经添加的候选向量ID，避免重复计算距离
        seen_ids = set()  
        
        # 1. 从所有哈希表中收集候选向量
        for table_idx in range(self.num_tables):
            # 计算查询向量在当前表的哈希值
            hash_val = self._hash_vector(query_vector, table_idx)
            
            # 获取与查询向量哈希值完全相同的桶中的所有向量
            exact_matches = self.tables[table_idx].get(hash_val, [])
            for id_vec in exact_matches:
                if id_vec[0] not in seen_ids:
                    candidates.append(id_vec)
                    seen_ids.add(id_vec[0])
            
            # 2. (可选) 搜索邻近桶以提高召回率 - 汉明距离为1的桶
            # 仅在候选数量不足时执行，以平衡效率和召回率
            if len(candidates) < k * 2: 
                for bit_idx in range(self.num_bits):
                    # 翻转哈希值的某一位，生成邻近桶的哈希值
                    flipped_hash = list(hash_val)
                    flipped_hash[bit_idx] = '1' if hash_val[bit_idx] == '0' else '0'
                    flipped_hash = ''.join(flipped_hash)
                    
                    # 获取邻近桶中的向量
                    flipped_matches = self.tables[table_idx].get(flipped_hash, [])
                    for id_vec in flipped_matches:
                        if id_vec[0] not in seen_ids:
                            candidates.append(id_vec)
                            seen_ids.add(id_vec[0])

            # 3. (可选, 更进一步) 搜索汉明距离为2的桶
            # 仅在候选极少且哈希位数不高时执行（避免组合爆炸）
            if len(candidates) < k and self.num_bits <= 16:
                for i in range(self.num_bits):
                    for j in range(i+1, self.num_bits):
                        # 翻转哈希值的两位
                        double_flipped_hash = list(hash_val)
                        double_flipped_hash[i] = '1' if hash_val[i] == '0' else '0'
                        double_flipped_hash[j] = '1' if hash_val[j] == '0' else '0'
                        double_flipped_hash = ''.join(double_flipped_hash)
                        
                        double_matches = self.tables[table_idx].get(double_flipped_hash, [])
                        for id_vec in double_matches:
                            if id_vec[0] not in seen_ids:
                                candidates.append(id_vec)
                                seen_ids.add(id_vec[0])
                        
                        # 如果候选数量已足够，停止内层循环
                        if len(candidates) >= k * 3:  
                            break
                    # 如果候选数量已足够，停止外层循环
                    if len(candidates) >= k * 3:
                        break
        
        # 4. 如果候选向量仍然太少，进行更广泛的采样 (代价较高)
        if len(candidates) < k:
            # 遍历所有哈希表的所有桶，随机添加一些向量作为候选
            all_vectors = []
            for table in self.tables:
                for bucket in table.values():
                    all_vectors.extend(bucket)
            
            needed = k * 5 - len(candidates) # 需要补充的候选数量
            if needed > 0 and all_vectors:
                sample_indices = np.random.choice(len(all_vectors), size=min(needed, len(all_vectors)), replace=False)
                for idx in sample_indices:
                    id_vec = all_vectors[idx]
                    if id_vec[0] not in seen_ids:
                        candidates.append(id_vec)
                        seen_ids.add(id_vec[0])
        
        # 如果最终没有找到任何候选向量，返回空列表
        if not candidates:
            return []
        
        # 5. 计算所有候选向量与查询向量的精确距离
        # 使用余弦相似度作为距离度量（转换为距离：1 - similarity）通常效果更好
        query_norm = np.linalg.norm(query_vector)
        candidates_with_distances = []
        
        if query_norm > 0:
            query_unit = query_vector / query_norm # 归一化查询向量
            for idx, vec in candidates:
                vec_norm = np.linalg.norm(vec)
                if vec_norm > 0:
                    similarity = np.dot(query_unit, vec / vec_norm) # 计算余弦相似度
                    # 将相似度转换为距离 (0-2范围, 0表示最相似)
                    distance = 1.0 - similarity
                    candidates_with_distances.append((idx, distance))
                else:
                    # 零向量与任何非零向量的距离视为最大
                    candidates_with_distances.append((idx, 2.0)) 
        else:
            # 如果查询向量是零向量，则使用欧氏距离
            for idx, vec in candidates:
                distance = np.linalg.norm(query_vector - vec)
                candidates_with_distances.append((idx, distance))
        
        # 6. 按计算出的精确距离排序
        candidates_with_distances.sort(key=lambda x: x[1])
        
        # 7. 返回前k个最近邻
        return candidates_with_distances[:k]

def test_ann_algorithm():
    """
    测试LSH近似最近邻算法的性能和准确性
    
    步骤:
    1. 生成随机高维向量数据集。
    2. 选择一个向量作为查询向量（确保真实最近邻是它自己）。
    3. 构建LSH索引。
    4. 使用LSH索引查询最近邻。
    5. 使用暴力搜索计算精确最近邻作为基准。
    6. 比较LSH和暴力搜索的结果（时间、召回率）。
    7. 打印详细的性能指标和结果对比。
    """
    # 参数设置
    dim = 64         # 向量维度
    num_vectors = 10000 # 数据集大小
    k_neighbors = 5   # 查询的最近邻数量
    lsh_tables = 15   # LSH哈希表数量
    lsh_bits = 12     # LSH哈希位数
    
    # 1. 生成随机向量数据集
    np.random.seed(42) # 固定随机种子以保证结果可复现
    vectors = [np.random.randn(dim) for _ in range(num_vectors)]
    
    # 2. 选择查询向量 (随机选择一个已存在的向量)
    query_idx = random.randint(0, num_vectors - 1)
    query_vector = vectors[query_idx].copy() # 复制以防意外修改
    
    print("="*60)
    print("开始测试 LSH 近似最近邻算法")
    print(f"数据集: {num_vectors} 个 {dim} 维向量")
    print(f"查询向量: 使用数据集中的第 {query_idx} 个向量")
    print(f"查询近邻数 (k): {k_neighbors}")
    print(f"LSH参数: num_tables={lsh_tables}, num_bits={lsh_bits}")
    print("="*60)
    
    # 3. 构建LSH索引
    start_time = time.time()
    lsh_index = LSHIndex(dim, num_tables=lsh_tables, num_bits=lsh_bits)
    lsh_index.index(vectors)
    index_time = time.time() - start_time
    print(f"[构建索引]")
    print(f"LSH索引构建耗时: {index_time:.4f} 秒")
    
    # 4. 使用LSH查询
    start_time = time.time()
    lsh_results = lsh_index.query(query_vector, k=k_neighbors)
    lsh_query_time = time.time() - start_time
    # 避免查询时间为0导致除零错误
    lsh_query_time = max(lsh_query_time, 1e-6) 
    
    print(f"[LSH查询]")
    print(f"LSH查询耗时: {lsh_query_time:.6f} 秒")
    
    # 5. 暴力搜索 (计算精确最近邻)
    start_time = time.time()
    # 计算所有向量与查询向量的欧氏距离
    bruteforce_distances = [(i, np.linalg.norm(query_vector - vec)) for i, vec in enumerate(vectors)]
    # 按距离排序
    bruteforce_distances.sort(key=lambda x: x[1])
    bf_query_time = time.time() - start_time
    # 避免查询时间为0
    bf_query_time = max(bf_query_time, 1e-6) 
    
    print(f"[暴力搜索基准]")
    print(f"暴力搜索耗时: {bf_query_time:.6f} 秒")
    
    # 6. 比较结果
    print("" + "="*60)
    print("性能与准确性评估")
    print("="*60)
    
    # 计算加速比
    speedup_ratio = bf_query_time / lsh_query_time
    print(f"加速比 (暴力搜索时间 / LSH查询时间): {speedup_ratio:.2f}x")
    
    # 计算召回率 (Recall@k)
    # 获取暴力搜索找到的前k个精确最近邻的ID集合
    ground_truth_ids = set([idx for idx, _ in bruteforce_distances[:k_neighbors]])
    # 获取LSH找到的前k个近似最近邻的ID集合
    lsh_found_ids = set([idx for idx, _ in lsh_results])
    # 计算交集大小
    intersection_count = len(ground_truth_ids.intersection(lsh_found_ids))
    # 计算召回率
    recall = intersection_count / len(ground_truth_ids) if ground_truth_ids else 1.0
    print(f"召回率 (Recall@{k_neighbors}): {recall:.2f} ({intersection_count}/{len(ground_truth_ids)})")
    
    # 检查查询向量本身是否被找到（它应该是距离最近的）
    query_found_by_lsh = query_idx in lsh_found_ids
    print(f"查询向量本身(ID: {query_idx})是否被LSH找到: {'是' if query_found_by_lsh else '否'}")
    
    # 7. 打印详细结果
    print("--- LSH查询结果 (Top 5) ---")
    print("(ID, 距离)")
    for i, (idx, dist) in enumerate(lsh_results):
        print(f"  #{i+1}: ({idx}, {dist:.6f})")
        
    print("--- 暴力搜索精确结果 (Top 5) ---")
    print("(ID, 距离)")
    for i, (idx, dist) in enumerate(bruteforce_distances[:k_neighbors]):
        print(f"  #{i+1}: ({idx}, {dist:.6f})")
        
    print("--- 结果对比 ---")
    print(f"精确最近邻 IDs: {sorted(list(ground_truth_ids))}")
    print(f"LSH找到的 IDs: {sorted(list(lsh_found_ids))}")
    print(f"共同找到的 IDs: {sorted(list(ground_truth_ids.intersection(lsh_found_ids)))}")
    print("="*60)
    
    return lsh_index # 返回构建好的索引，可能在其他地方使用

if __name__ == "__main__":
    # 运行测试函数
    test_ann_algorithm() 