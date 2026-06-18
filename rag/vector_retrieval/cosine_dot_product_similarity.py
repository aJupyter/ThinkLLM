import numpy as np
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度
    
    原理:
        余弦相似度通过计算两个向量夹角的余弦值来衡量它们的方向相似性。
        公式: cos(theta) = (vec1 . vec2) / (||vec1|| * ||vec2||)
        值域: [-1, 1]，1表示方向完全相同，-1表示方向完全相反，0表示正交。
        优点: 对向量的绝对大小不敏感，只关心方向。
        
    参数:
        vec1 (np.ndarray): 第一个向量
        vec2 (np.ndarray): 第二个向量
        
    返回:
        float: 余弦相似度值，范围在[-1, 1]之间
    """
    # 计算点积
    dot_product = np.dot(vec1, vec2)
    # 计算向量模长 (L2范数)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    # 避免除以零 (如果任一向量模长为0，则相似度为0)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    # 计算余弦相似度
    similarity = dot_product / (norm_vec1 * norm_vec2)
    # 确保结果在[-1, 1]范围内 (由于浮点数精度问题可能略微超出)
    return np.clip(similarity, -1.0, 1.0)

def dot_product_similarity(vec1, vec2):
    """
    计算两个向量之间的点积相似度
    
    原理:
        点积衡量两个向量在方向上的对齐程度以及它们的大小。
        公式: vec1 . vec2 = sum(vec1[i] * vec2[i])
        值域: (-inf, +inf)
        优点: 计算简单快速。
        缺点: 受向量大小影响很大，长向量的点积通常更大。
        
    参数:
        vec1 (np.ndarray): 第一个向量
        vec2 (np.ndarray): 第二个向量
        
    返回:
        float: 点积值
    """
    return np.dot(vec1, vec2)

def test_similarity_functions():
    """
    测试余弦相似度和点积相似度函数
    包含一些典型场景的测试用例
    """
    print("开始测试向量相似度函数...")
    # 定义两个测试向量
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([4, 5, 6])
    print(f"向量1: {vec1}")
    print(f"向量2: {vec2}")
    
    # 计算余弦相似度
    cos_sim = cosine_similarity(vec1, vec2)
    print(f"余弦相似度: {cos_sim:.4f} (预期接近 0.9746)") # 结果应接近1，表示方向相似
    
    # 计算点积相似度
    dot_sim = dot_product_similarity(vec1, vec2)
    print(f"点积相似度: {dot_sim:.4f} (预期为 32.0000)") # 1*4 + 2*5 + 3*6 = 32
    
    print("\n测试特殊情况...")
    # 测试正交向量（余弦相似度应为0）
    vec3 = np.array([1, 0, 0])
    vec4 = np.array([0, 1, 0])
    print(f"正交向量1: {vec3}")
    print(f"正交向量2: {vec4}")
    cos_sim_orth = cosine_similarity(vec3, vec4)
    dot_sim_orth = dot_product_similarity(vec3, vec4)
    print(f"正交向量的余弦相似度: {cos_sim_orth:.4f} (预期为 0.0000)")
    print(f"正交向量的点积相似度: {dot_sim_orth:.4f} (预期为 0.0000)")
    
    # 测试相同方向的向量（余弦相似度应为1）
    vec5 = np.array([1, 2, 3])
    vec6 = np.array([2, 4, 6])  # 与vec5方向相同，但大小不同
    print(f"\n同向向量1: {vec5}")
    print(f"同向向量2: {vec6}")
    cos_sim_same = cosine_similarity(vec5, vec6)
    dot_sim_same = dot_product_similarity(vec5, vec6)
    print(f"相同方向向量的余弦相似度: {cos_sim_same:.4f} (预期为 1.0000)")
    print(f"相同方向向量的点积相似度: {dot_sim_same:.4f} (预期为 28.0000)") # 1*2 + 2*4 + 3*6 = 28
    
    # 测试反向向量（余弦相似度应为-1）
    vec7 = np.array([-1, -2, -3])
    print(f"\n反向向量: {vec7}")
    cos_sim_opp = cosine_similarity(vec5, vec7)
    dot_sim_opp = dot_product_similarity(vec5, vec7)
    print(f"反向向量的余弦相似度: {cos_sim_opp:.4f} (预期为 -1.0000)")
    print(f"反向向量的点积相似度: {dot_sim_opp:.4f} (预期为 -14.0000)") # 1*(-1) + 2*(-2) + 3*(-3) = -14

    # 测试零向量
    vec_zero = np.array([0, 0, 0])
    print(f"\n零向量: {vec_zero}")
    cos_sim_zero = cosine_similarity(vec5, vec_zero)
    dot_sim_zero = dot_product_similarity(vec5, vec_zero)
    print(f"与零向量的余弦相似度: {cos_sim_zero:.4f} (预期为 0.0000)")
    print(f"与零向量的点积相似度: {dot_sim_zero:.4f} (预期为 0.0000)")
    print("测试完成。")

if __name__ == "__main__":
    test_similarity_functions() 
    # 预期输出示例:
    # 余弦相似度: 0.9746
    # 点积相似度: 32.0000
    # 正交向量的余弦相似度: 0.0000
    # 相同方向向量的余弦相似度: 1.0000