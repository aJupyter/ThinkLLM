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
    分层可导航小世界（Hierarchical Navigable Small World, HNSW）索引的简化实现
    
    原理:
        HNSW是一种基于图的近似最近邻(ANN)搜索算法，它通过构建多层图结构来加速搜索。
        1. **多层结构**: 图分为多个层级，最底层包含所有数据点，层级越高，节点越稀疏，边越长。
        2. **插入**: 新节点随机选择一个层级插入，然后在该层及以下所有层级中找到最近邻，并建立连接。
           连接建立时会使用启发式规则（如选择多样化的邻居）来优化图的导航性能。
        3. **搜索**: 从最高层的入口点开始，贪婪地向查询点移动到更近的邻居。当在某一层无法找到更近的邻居时，
           进入下一层继续搜索，直到到达最底层。在最底层进行更精细的搜索以找到最终结果。
           
    优点:
        - 查询速度非常快，尤其是在高维空间和大规模数据集上。
        - 召回率通常很高，且性能相对稳定。
        - 支持动态插入新数据点。
        
    缺点:
        - 构建索引的时间相对较长，比LSH等方法慢。
        - 内存占用较高，需要存储图结构。
        - 实现相对复杂，参数调优（M, ef_construction）对性能影响较大。
    """
    
    def __init__(self, dim, M=16, ef_construction=200, levels=4):
        """
        初始化HNSW索引
        
        参数:
            dim (int): 输入向量的维度。
            M (int): 图中每个节点的最大连接数（出度）。控制图的密度，影响内存和搜索速度。
            ef_construction (int): 构建索引时搜索候选列表的大小。值越大，索引质量越高（召回率可能更高），但构建时间越长。
            levels (int): 图的层数。影响搜索速度和内存占用。
        """
        self.dim = dim
        self.M = M # 每个节点的最大连接数
        self.ef_construction = ef_construction # 构建时的搜索深度
        self.max_level = levels - 1 # 最高层索引 (0-based)
        
        # 数据存储
        self.vectors = [] # 存储实际向量
        self.ids = []     # 存储向量对应的原始ID
        
        # 图结构: 使用列表存储每一层，每层是一个字典 {node_index: [neighbor_indices]}
        self.graph = [defaultdict(list) for _ in range(levels)]
        
        # 最高层的入口点，初始为None
        self.entry_point = None
        self.element_count = 0 # 记录已添加的元素数量
        
    def _distance(self, vec1, vec2):
        """计算两个向量之间的欧氏距离的平方 (避免开方，提高效率)"""
        # 使用欧氏距离的平方，避免开方运算，比较大小时等价
        diff = vec1 - vec2
        return np.dot(diff, diff)
    
    def _select_neighbors_heuristic(self, query_vec, candidates, M, level):
        """
        使用启发式规则从候选者中选择邻居 (HNSW论文中的方法)
        目标是选择与查询点近且互相之间距离较远的邻居，增加多样性。

        参数:
            query_vec (np.ndarray): 查询向量。
            candidates (List[Tuple[int, float]]): 候选邻居列表，格式为 (node_index, distance_to_query)。
            M (int): 需要选择的邻居数量。
            level (int): 当前操作的图层级。

        返回:
            List[int]: 选择出的邻居节点的索引列表。
        """
        # 按到查询点的距离升序排序候选者
        candidates.sort(key=lambda x: x[1])
        
        selected_neighbors = []
        selected_indices = set()

        # 迭代候选者，尝试选择M个最佳邻居
        for cand_idx, cand_dist in candidates:
            if len(selected_neighbors) >= M:
                break # 已选够M个

            # 检查当前候选者是否比已选中的所有邻居都更接近查询点
            # 这是HNSW论文中的一个优化，确保选中的邻居质量
            is_closer_than_all_selected = True
            for selected_idx in selected_neighbors:
                dist_cand_to_selected = self._distance(self.vectors[cand_idx], self.vectors[selected_idx])
                if dist_cand_to_selected < cand_dist:
                    is_closer_than_all_selected = False
                    break
            
            # 如果满足条件，则选择该候选者
            if is_closer_than_all_selected:
                selected_neighbors.append(cand_idx)
                selected_indices.add(cand_idx)
        
        # 如果启发式选择不足M个，用最近的补齐 (简化处理)
        if len(selected_neighbors) < M:
            remaining_candidates = [c[0] for c in candidates if c[0] not in selected_indices]
            needed = M - len(selected_neighbors)
            selected_neighbors.extend(remaining_candidates[:needed])
            
        return selected_neighbors

    def _search_layer(self, query_vec, entry_point_idx, ef, level):
        """
        在指定层级上执行搜索，查找最近的ef个邻居。

        参数:
            query_vec (np.ndarray): 查询向量。
            entry_point_idx (int): 此层搜索的起始节点索引。
            ef (int): 动态候选列表的大小 (搜索深度)。ef越大，召回率越高，但速度越慢。
            level (int): 当前搜索的图层级。

        返回:
            List[Tuple[int, float]]: 找到的最近邻列表，格式为 (node_index, distance_to_query)，按距离升序排序。
        """
        # 记录访问过的节点，避免重复计算
        visited = set([entry_point_idx])
        
        # 计算入口点到查询点的距离
        entry_dist = self._distance(query_vec, self.vectors[entry_point_idx])
        
        # 候选列表 (最小堆)，存储 (distance, node_index)
        candidates = [(entry_dist, entry_point_idx)]
        # 结果列表 (最大堆)，存储 (-distance, node_index)，方便获取最远邻居
        results = [(-entry_dist, entry_point_idx)]
        
        # 主搜索循环
        while candidates:
            # 从候选列表中取出距离最近的节点
            cand_dist, cand_idx = heapq.heappop(candidates)
            
            # 获取结果列表中最远的距离 (负数)
            farthest_result_neg_dist = results[0][0]
            
            # 优化: 如果当前候选节点的距离比结果中最远的还要远，则停止搜索
            if cand_dist > -farthest_result_neg_dist:
                break
            
            # 遍历当前节点的邻居
            neighbors = self.graph[level].get(cand_idx, [])
            for neighbor_idx in neighbors:
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    
                    # 计算邻居到查询点的距离
                    neighbor_dist = self._distance(query_vec, self.vectors[neighbor_idx])
                    farthest_result_neg_dist = results[0][0]
                    
                    # 如果结果列表未满 (少于ef个) 或 当前邻居比结果中最远的更近
                    if len(results) < ef or neighbor_dist < -farthest_result_neg_dist:
                        # 将邻居加入候选列表和结果列表
                        heapq.heappush(candidates, (neighbor_dist, neighbor_idx))
                        heapq.heappush(results, (-neighbor_dist, neighbor_idx))
                        
                        # 如果结果列表超出ef个，移除最远的那个
                        if len(results) > ef:
                            heapq.heappop(results)
                            
        # 将结果列表转换为 (node_index, distance) 并按距离升序排序
        final_results = [(idx, -neg_dist) for neg_dist, idx in results]
        final_results.sort(key=lambda x: x[1])
        return final_results

    def add(self, vector, id=None):
        """
        将向量添加到HNSW索引中。
        
        参数:
            vector (np.ndarray): 要添加的向量。
            id (optional): 向量的原始ID。如果为None，则使用内部索引。
        """
        if id is None:
            internal_id = self.element_count
        else:
            internal_id = id # 注意：如果使用外部ID，需要确保其唯一性且从0开始连续，或修改存储方式
            
        node_idx = self.element_count # 新节点的内部索引
        self.vectors.append(vector)
        self.ids.append(internal_id) # 存储原始ID或内部索引
        self.element_count += 1
        
        # 1. 确定新节点要插入的最高层级 (随机化)
        # 使用 HNSW 论文中的概率模型: P(level) = normalize_factor * exp(-level / mL)
        # 这里简化为指数衰减概率 (更易实现)
        mL = 1 / np.log(self.M) # 标准化因子，论文建议
        node_max_level = 0
        while np.random.rand() < np.exp(-node_max_level / mL) and node_max_level < self.max_level:
             node_max_level += 1
        
        # 如果没有入口点 (第一个节点)，将其设为入口点
        if self.entry_point is None:
            self.entry_point = node_idx
            # 将此节点添加到所有层级（虽然它没有邻居）
            for level in range(node_max_level, -1, -1):
                 self.graph[level][node_idx] = []
            return
            
        # 2. 从最高层开始，逐层向下查找插入点的最近邻
        current_nearest_idx = self.entry_point
        for level in range(self.max_level, node_max_level, -1):
            # 在当前层找到距离新节点最近的一个邻居，作为下一层的入口点
            search_results = self._search_layer(vector, current_nearest_idx, ef=1, level=level)
            if search_results: # 确保搜索有结果
                 current_nearest_idx = search_results[0][0]
            # 如果某层没有找到邻居（可能图未完全连接），保持上一个入口点
            
        # 3. 在新节点所属的层级 (node_max_level) 及以下所有层级进行插入
        for level in range(min(node_max_level, self.max_level), -1, -1):
            # 在当前层找到 ef_construction 个候选连接点
            candidates = self._search_layer(vector, current_nearest_idx, ef=self.ef_construction, level=level)
            
            # 使用启发式规则选择 M 个最佳邻居进行连接
            neighbors_to_connect = self._select_neighbors_heuristic(vector, candidates, self.M, level)
            
            # 将新节点与选定的邻居互相连接
            self.graph[level][node_idx] = neighbors_to_connect
            for neighbor_idx in neighbors_to_connect:
                # 将新节点添加到邻居的连接列表中
                self.graph[level][neighbor_idx].append(node_idx)
                
                # 4. 连接裁剪: 如果邻居的连接数超过 M_max (通常是M或2M)，进行裁剪
                # HNSW论文建议M_max = M for level > 0, M_max = 2*M for level 0
                M_max = self.M if level > 0 else 2 * self.M
                if len(self.graph[level][neighbor_idx]) > M_max:
                    neighbor_vec = self.vectors[neighbor_idx]
                    # 计算该邻居的所有连接点到它的距离
                    neighbor_connections = self.graph[level][neighbor_idx]
                    distances = [(conn_idx, self._distance(neighbor_vec, self.vectors[conn_idx])) 
                                 for conn_idx in neighbor_connections]
                    # 使用启发式选择保留 M_max 个最佳连接
                    best_connections = self._select_neighbors_heuristic(neighbor_vec, distances, M_max, level)
                    self.graph[level][neighbor_idx] = best_connections
            
            # 更新下一层的入口点为本层找到的最近邻
            if candidates: # 确保有候选者
                 current_nearest_idx = candidates[0][0]
        
        # 更新全局入口点（如果新节点的层级更高）
        # 注意：此简化实现未处理入口点层级更新，实际HNSW需要
        # if node_max_level > self.get_level(self.entry_point): # 假设有get_level函数
        #    self.entry_point = node_idx
        pass # 简化：入口点固定或只在第一个节点时设置

    def search(self, query_vec, k=5, ef_search=None):
        """
        在HNSW索引中搜索与查询向量最接近的k个邻居。

        参数:
            query_vec (np.ndarray): 查询向量。
            k (int): 要返回的最近邻数量。
            ef_search (int, optional): 搜索时的动态候选列表大小。如果为None，则使用ef_construction。
                                      通常 ef_search >= k。

        返回:
            List[Tuple[int, float]]: 包含 (原始ID, 距离) 的列表，按距离升序排序。
        """
        if self.entry_point is None:
            return [] # 索引为空
            
        if ef_search is None:
            ef_search = max(k, self.ef_construction // 2) # 默认搜索深度
        elif ef_search < k:
             print(f"Warning: ef_search ({ef_search}) is less than k ({k}). Setting ef_search=k.")
             ef_search = k

        # 1. 从最高层开始，逐层向下找到最底层搜索的入口点
        current_nearest_idx = self.entry_point
        for level in range(self.max_level, 0, -1):
            search_results = self._search_layer(query_vec, current_nearest_idx, ef=1, level=level)
            if search_results: # 确保搜索有结果
                 current_nearest_idx = search_results[0][0]
            # 如果某层没有找到邻居，保持上一个入口点
            
        # 2. 在最底层 (level 0) 使用更大的ef进行精确搜索
        bottom_level_results = self._search_layer(query_vec, current_nearest_idx, ef=ef_search, level=0)
        
        # 3. 截取前k个结果，并返回 (原始ID, 真实距离)
        final_results = []
        for node_idx, dist_sq in bottom_level_results[:k]:
             original_id = self.ids[node_idx] # 获取原始ID
             real_distance = np.sqrt(dist_sq) # 计算真实欧氏距离
             final_results.append((original_id, real_distance))
             
        return final_results

def test_hnsw_algorithm():
    """
    测试HNSW索引的性能和准确性。
    
    步骤:
    1. 设置参数 (维度、向量数量、HNSW参数)。
    2. 生成随机向量数据集。
    3. 构建HNSW索引并计时。
    4. 选择一个查询向量。
    5. 使用HNSW索引查询k个最近邻并计时。
    6. 使用暴力搜索计算精确最近邻作为基准并计时。
    7. 比较HNSW和暴力搜索的结果（时间、召回率、加速比）。
    8. 打印详细的性能指标和结果对比。
    """
    # 1. 参数设置
    dim = 100         # 向量维度
    num_vectors = 2000  # 数据集大小 (增加数量以更好测试)
    k_neighbors = 10    # 查询的最近邻数量
    hnsw_M = 16       # HNSW参数: 每个节点最大连接数
    hnsw_ef_construction = 150 # HNSW参数: 构建时搜索深度
    hnsw_levels = 4     # HNSW参数: 图的层数
    hnsw_ef_search = 100   # HNSW参数: 查询时搜索深度
    
    # 2. 生成随机向量数据集
    np.random.seed(42) # 固定随机种子
    vectors = [np.random.randn(dim) for _ in range(num_vectors)]
    vector_ids = list(range(num_vectors)) # 使用从0开始的连续ID
    
    print("="*60)
    print("开始测试 HNSW 近似最近邻算法")
    print(f"数据集: {num_vectors} 个 {dim} 维向量")
    print(f"查询近邻数 (k): {k_neighbors}")
    print(f"HNSW参数: M={hnsw_M}, ef_construction={hnsw_ef_construction}, levels={hnsw_levels}, ef_search={hnsw_ef_search}")
    print("="*60)
    
    # 3. 构建HNSW索引
    start_time = time.time()
    hnsw_index = HNSWIndex(dim, M=hnsw_M, ef_construction=hnsw_ef_construction, levels=hnsw_levels)
    # 使用原始ID添加向量
    for i, vec in enumerate(vectors):
        hnsw_index.add(vec, id=vector_ids[i]) 
    index_time = time.time() - start_time
    print(f"[构建索引]")
    print(f"HNSW索引构建耗时: {index_time:.4f} 秒")
    
    # 4. 选择查询向量 (随机选择一个)
    query_idx = np.random.randint(0, num_vectors)
    query_vector = vectors[query_idx]
    print(f"[查询准备]")
    print(f"使用数据集中的第 {query_idx} 个向量作为查询向量。")
    
    # 5. 使用HNSW查询
    start_time = time.time()
    hnsw_results = hnsw_index.search(query_vector, k=k_neighbors, ef_search=hnsw_ef_search)
    hnsw_query_time = time.time() - start_time
    hnsw_query_time = max(hnsw_query_time, 1e-6) # 避免除零
    
    print(f"[HNSW查询]")
    print(f"HNSW查询耗时: {hnsw_query_time:.6f} 秒")
    
    # 6. 暴力搜索 (计算精确最近邻)
    start_time = time.time()
    # 计算所有向量与查询向量的真实欧氏距离
    bruteforce_distances = [(vector_ids[i], np.linalg.norm(query_vector - vec)) 
                            for i, vec in enumerate(vectors)]
    # 按距离排序
    bruteforce_distances.sort(key=lambda x: x[1])
    bf_query_time = time.time() - start_time
    bf_query_time = max(bf_query_time, 1e-6) # 避免除零
    
    print(f"[暴力搜索基准]")
    print(f"暴力搜索耗时: {bf_query_time:.6f} 秒")
    
    # 7. 比较结果
    print("="*60)
    print("性能与准确性评估")
    print("="*60)
    
    # 计算加速比
    speedup_ratio = bf_query_time / hnsw_query_time
    print(f"加速比 (暴力搜索时间 / HNSW查询时间): {speedup_ratio:.2f}x")
    
    # 计算召回率 (Recall@k)
    ground_truth_ids = set([id for id, _ in bruteforce_distances[:k_neighbors]])
    hnsw_found_ids = set([id for id, _ in hnsw_results])
    intersection_count = len(ground_truth_ids.intersection(hnsw_found_ids))
    recall = intersection_count / len(ground_truth_ids) if ground_truth_ids else 1.0
    print(f"召回率 (Recall@{k_neighbors}): {recall:.2f} ({intersection_count}/{len(ground_truth_ids)})")
    
    # 8. 打印详细结果
    print("--- HNSW查询结果 (Top 5) ---")
    print("(ID, 距离)")
    for i, (id, dist) in enumerate(hnsw_results[:5]): # 只显示前5个以便查看
        print(f"  #{i+1}: ({id}, {dist:.6f})")
        
    print("--- 暴力搜索精确结果 (Top 5) ---")
    print("(ID, 距离)")
    for i, (id, dist) in enumerate(bruteforce_distances[:5]): # 只显示前5个
        print(f"  #{i+1}: ({id}, {dist:.6f})")
        
    print("--- 结果对比 ---")
    print(f"精确最近邻 IDs (Top {k_neighbors}): {sorted(list(ground_truth_ids))}")
    print(f"HNSW找到的 IDs (Top {k_neighbors}): {sorted(list(hnsw_found_ids))}")
    print(f"共同找到的 IDs: {sorted(list(ground_truth_ids.intersection(hnsw_found_ids)))}")
    print("="*60)

if __name__ == "__main__":
    test_hnsw_algorithm() 