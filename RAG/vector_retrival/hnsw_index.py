import numpy as np
import heapq
import time
from collections import defaultdict, deque
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class HNSWIndex:
    """
    分层可导航小世界（HNSW）索引的简化实现
    HNSW是一种图形化近似最近邻(ANN)算法，表现优异
    """
    
    def __init__(self, dim, M=16, ef_construction=200, levels=4):
        """
        初始化HNSW索引
        
        参数:
            dim: 向量的维度
            M: 每个节点最大连接数
            ef_construction: 构建时搜索列表的大小（影响构建质量）
            levels: 层次结构中的层数
        """
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.max_level = levels - 1
        
        # 数据存储
        self.vectors = []
        self.ids = []
        
        # 图结构：level -> node_id -> [neighbors]
        self.graph = [defaultdict(list) for _ in range(levels)]
        
        # 入口点
        self.entry_point = None
        
    def _distance(self, a, b):
        """计算两个向量之间的欧氏距离"""
        return np.linalg.norm(a - b)
    
    def _select_neighbors(self, q, candidates, M):
        """
        从候选邻居中选择最佳的M个
        简化的贪婪算法（真实HNSW使用更复杂的启发式）
        
        参数:
            q: 查询点
            candidates: 候选邻居列表 (id, distance)
            M: 要选择的邻居数量
            
        返回:
            选择的邻居列表 [id]
        """
        # 按距离排序
        candidates.sort(key=lambda x: x[1])
        # 选择最接近的M个
        return [c[0] for c in candidates[:M]]
        
    def _search_layer(self, q, entry_point, ef, level):
        """
        在单层中搜索最近邻
        
        参数:
            q: 查询向量
            entry_point: 搜索的入口点ID
            ef: 结果集大小
            level: 要搜索的层级
            
        返回:
            最近邻列表 [(id, distance)]
        """
        # 访问过的节点
        visited = set([entry_point])
        
        # 使用最大堆作为结果集(实际距离的负值，这样最小的距离会被pop)
        results = []
        distance = self._distance(q, self.vectors[entry_point])
        heapq.heappush(results, (-distance, entry_point))
        
        # 使用最小堆作为候选集
        candidates = [(distance, entry_point)]
        
        # 当候选集不为空时
        while candidates:
            # 获取最接近的候选者
            current_dist, current_id = heapq.heappop(candidates)
            
            # 如果当前距离超过了结果堆中最远邻居的距离，则停止
            if current_dist > -results[0][0]:
                break
                
            # 检查当前节点的邻居
            for neighbor_id in self.graph[level][current_id]:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    
                    # 计算邻居到查询的距离
                    neighbor_dist = self._distance(q, self.vectors[neighbor_id])
                    
                    # 如果结果集未满或新距离优于当前最差距离
                    if len(results) < ef or neighbor_dist < -results[0][0]:
                        # 添加到候选集
                        heapq.heappush(candidates, (neighbor_dist, neighbor_id))
                        
                        # 添加到结果集
                        heapq.heappush(results, (-neighbor_dist, neighbor_id))
                        
                        # 如果结果集超过ef，移除最远的元素
                        if len(results) > ef:
                            heapq.heappop(results)
        
        # 转换结果为(id, distance)列表并按距离排序
        return [(node_id, -neg_dist) for neg_dist, node_id in results]
    
    def add(self, vector, id=None):
        """
        将向量添加到索引中
        
        参数:
            vector: 要添加的向量
            id: 向量的可选ID(如果未提供，则使用索引)
        """
        if id is None:
            id = len(self.vectors)
            
        # 存储向量和ID
        self.vectors.append(vector)
        self.ids.append(id)
        node_id = len(self.vectors) - 1
        
        # 如果这是第一个点，设为入口点并返回
        if self.entry_point is None:
            self.entry_point = node_id
            return
            
        # 为新节点分配随机层级
        # 简化：使用固定概率1/2决定节点是否上升到下一层
        node_level = 0
        while node_level < self.max_level and np.random.random() < 0.5:
            node_level += 1
            
        # 开始从最高层插入
        current_id = self.entry_point
        
        # 从最高层开始向下搜索
        for level in range(self.max_level, node_level, -1):
            # 在当前层查找最近邻
            nearest = self._search_layer(vector, current_id, 1, level)[0]
            current_id = nearest[0]  # 更新入口点为找到的最近邻
            
        # 插入节点到它自己的层级以及所有更低层级
        for level in range(min(node_level, self.max_level), -1, -1):
            # 在当前层查找ef_construction个候选邻居
            candidates = self._search_layer(vector, current_id, self.ef_construction, level)
            
            # 选择最佳的M个邻居
            neighbors = self._select_neighbors(vector, candidates, self.M)
            
            # 将边添加到图中
            self.graph[level][node_id] = neighbors
            
            # 双向连接：为每个选定的邻居添加回边
            for neighbor_id in neighbors:
                self.graph[level][neighbor_id].append(node_id)
                
                # 如果邻居的连接数超过M，修剪一些连接
                if len(self.graph[level][neighbor_id]) > self.M:
                    neighbor_vector = self.vectors[neighbor_id]
                    
                    # 计算所有邻居到当前邻居的距离
                    distances = [(n_id, self._distance(neighbor_vector, self.vectors[n_id])) 
                                 for n_id in self.graph[level][neighbor_id]]
                    
                    # 选择最佳的M个邻居
                    best_neighbors = self._select_neighbors(neighbor_vector, distances, self.M)
                    self.graph[level][neighbor_id] = best_neighbors
                    
        # 如果新节点的层级高于当前入口点的层级，更新入口点
        if node_level > 0 and (self.entry_point is None or node_level > self.max_level):
            self.entry_point = node_id
                
    def search(self, query, k=1):
        """
        搜索k个最近邻
        
        参数:
            query: 查询向量
            k: 要返回的近邻数量
            
        返回:
            (id, 距离)的列表，按距离排序
        """
        if self.entry_point is None:
            return []
            
        current_id = self.entry_point
        
        # 从最高层开始向下搜索
        for level in range(self.max_level, 0, -1):
            closest = self._search_layer(query, current_id, 1, level)[0]
            current_id = closest[0]
            
        # 在最底层进行完整搜索
        return self._search_layer(query, current_id, k, 0)
        
def test_hnsw_algorithm():
    """
    测试HNSW索引的性能
    """
    # 创建一些随机向量
    dim = 100
    num_vectors = 1000
    vectors = [np.random.randn(dim) for _ in range(num_vectors)]
    
    # 创建测试查询
    query = np.random.randn(dim)
    
    print("测试开始: HNSW索引")
    
    # 构建HNSW索引
    start_time = time.time()
    hnsw = HNSWIndex(dim, M=16, ef_construction=100, levels=3)
    for i, vec in enumerate(vectors):
        hnsw.add(vec, i)
    index_time = time.time() - start_time
    print(f"索引构建时间: {index_time:.4f} 秒")
    
    # 使用HNSW查询
    start_time = time.time()
    hnsw_results = hnsw.search(query, k=5)
    hnsw_query_time = time.time() - start_time
    # 确保查询时间不为零，避免除零错误
    hnsw_query_time = max(hnsw_query_time, 0.0001)
    
    print(f"HNSW查询时间: {hnsw_query_time:.4f} 秒")
    print("HNSW查询结果 (ID, 距离):")
    for idx, dist in hnsw_results:
        print(f"  ID: {idx}, 距离: {dist:.4f}")
    
    # 为比较，进行暴力搜索
    start_time = time.time()
    brute_force_dists = [(i, np.linalg.norm(query - vec)) for i, vec in enumerate(vectors)]
    brute_force_dists.sort(key=lambda x: x[1])
    bf_query_time = time.time() - start_time
    # 确保查询时间不为零，避免除零错误
    bf_query_time = max(bf_query_time, 0.0001)
    
    print(f"暴力搜索时间: {bf_query_time:.4f} 秒")
    print("暴力搜索结果 (前5个) (ID, 距离):")
    for idx, dist in brute_force_dists[:5]:
        print(f"  ID: {idx}, 距离: {dist:.4f}")
    
    # 计算召回率
    gt_ids = set([idx for idx, _ in brute_force_dists[:5]])
    hnsw_ids = set([idx for idx, _ in hnsw_results])
    recall = len(gt_ids.intersection(hnsw_ids)) / len(gt_ids)
    print(f"召回率 (前5个结果): {recall:.2f}")
    print(f"HNSW速度相对于暴力搜索的提升: {bf_query_time/hnsw_query_time:.2f}x")
    
if __name__ == "__main__":
    test_hnsw_algorithm() 