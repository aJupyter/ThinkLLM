import numpy as np
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度
    
    参数:
        vec1: 第一个向量
        vec2: 第二个向量
        
    返回:
        余弦相似度值，范围在[-1, 1]之间
    """
    # 计算点积
    dot_product = np.dot(vec1, vec2)
    # 计算向量模长
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    # 避免除以零
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    # 计算余弦相似度
    return dot_product / (norm_vec1 * norm_vec2)

def dot_product_similarity(vec1, vec2):
    """
    计算两个向量之间的点积相似度
    
    参数:
        vec1: 第一个向量
        vec2: 第二个向量
        
    返回:
        点积值
    """
    return np.dot(vec1, vec2)

def test_similarity_functions():
    """
    测试余弦相似度和点积相似度函数
    """
    # 定义两个测试向量
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([4, 5, 6])
    
    # 计算余弦相似度
    cos_sim = cosine_similarity(vec1, vec2)
    print(f"余弦相似度: {cos_sim:.4f}") # 0.9746
    
    # 计算点积相似度
    dot_sim = dot_product_similarity(vec1, vec2)
    print(f"点积相似度: {dot_sim:.4f}") # 32.0000
    
    # 测试正交向量（余弦相似度应为0）
    vec3 = np.array([1, 0, 0])
    vec4 = np.array([0, 1, 0])
    cos_sim_orth = cosine_similarity(vec3, vec4)
    print(f"正交向量的余弦相似度: {cos_sim_orth:.4f}") # 0.0000
    
    # 测试相同方向的向量（余弦相似度应为1）
    vec5 = np.array([1, 2, 3])
    vec6 = np.array([2, 4, 6])  # 与vec5方向相同，但大小不同
    cos_sim_same = cosine_similarity(vec5, vec6)
    print(f"相同方向向量的余弦相似度: {cos_sim_same:.4f}") # 1.0000

if __name__ == "__main__":
    test_similarity_functions() 
    # # 
    # 余弦相似度: 0.9746
    # 点积相似度: 32.0000
    # 正交向量的余弦相似度: 0.0000
    # 相同方向向量的余弦相似度: 1.0000