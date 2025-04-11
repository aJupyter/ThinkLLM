import numpy as np
import time
from collections import defaultdict
import re

class SimpleEmbedder:
    """
    简化的文本嵌入器，用于演示目的
    在实际应用中，应使用预训练的语言模型如BERT、Sentence-BERT或OpenAI的Embeddings API
    """
    
    def __init__(self, embedding_dim=64, vocabulary_size=10000):
        """
        初始化嵌入器
        
        参数:
            embedding_dim: 嵌入向量的维度
            vocabulary_size: 词汇表大小
        """
        # 初始化词嵌入矩阵（随机值）
        np.random.seed(42)  # 为了可重复性
        self.word_embeddings = np.random.randn(vocabulary_size, embedding_dim) / np.sqrt(embedding_dim)
        self.dim = embedding_dim
        
        # 简单的词典，将词映射到索引
        self.word_to_index = {}
        self.current_index = 0
        
    def get_word_embedding(self, word):
        """
        获取单词的嵌入
        
        参数:
            word: 输入单词
            
        返回:
            单词的嵌入向量
        """
        # 如果单词不在词典中，添加它
        if word not in self.word_to_index:
            if self.current_index < len(self.word_embeddings):
                self.word_to_index[word] = self.current_index
                self.current_index += 1
            else:
                # 如果词汇表已满，使用哈希
                self.word_to_index[word] = hash(word) % len(self.word_embeddings)
                
        return self.word_embeddings[self.word_to_index[word]]
        
    def embed_text(self, text):
        """
        获取文本的嵌入（简单平均池化）
        
        参数:
            text: 输入文本
            
        返回:
            文本的嵌入向量
        """
        # 简单的分词
        words = re.findall(r'\w+', text.lower())
        
        if not words:
            return np.zeros(self.dim)
            
        # 获取所有单词的嵌入
        embeddings = [self.get_word_embedding(word) for word in words]
        
        # 平均池化
        embedding = np.mean(embeddings, axis=0)
        
        # 标准化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding

class SimpleLLM:
    """
    简化的大语言模型，用于演示目的
    在实际应用中，应使用真实的LLM，如GPT系列、LLaMA等
    """
    
    def generate_text(self, prompt, max_tokens=100):
        """
        根据提示生成文本
        
        参数:
            prompt: 输入提示
            max_tokens: 生成的最大词元数
            
        返回:
            生成的文本
        """
        # 这只是一个模拟，返回与提示相关的简单回应
        topics = {
            "搜索引擎": "搜索引擎是一种软件系统，旨在执行Web搜索。它系统地搜索万维网上的信息，根据用户的查询提供特定的结果。搜索引擎通常使用网络爬虫收集信息，使用索引来存储数据，并使用查询处理器来处理用户查询并返回相关结果。",
            "机器学习": "机器学习是人工智能的一个分支，使计算机系统能够自动从数据中学习和改进经验，而无需显式编程。机器学习算法建立数学模型，基于样本数据（称为训练数据）进行预测或决策，无需按照静态的程序指令。",
            "自然语言处理": "自然语言处理（NLP）是计算机科学、人工智能和语言学的交叉学科，专注于计算机与人类语言之间的交互。NLP旨在使计算机能够理解、解释和生成人类语言。",
            "向量检索": "向量检索或向量相似度搜索是一种搜索方法，它使用向量（或嵌入）表示查询和文档，并找到最相似的文档。常用的相似度度量包括余弦相似度、点积和欧几里得距离。",
            "数据库": "数据库是有组织的数据集合，通常以电子形式存储在计算机系统中。数据库由数据库管理系统（DBMS）控制，DBMS与应用程序和数据库本身一起构成数据库系统。"
        }
        
        # 检查提示中是否包含特定主题
        response = ""
        for topic, content in topics.items():
            if topic.lower() in prompt.lower():
                response += content + " "
                
        if not response:
            response = "这是一个关于" + prompt + "的假设性响应。该主题涉及到各种概念和应用。在实际情境中，这里会包含有关该主题的详细、相关和准确的信息。"
                
        return response[:max_tokens * 5]  # 简单估计一个词约为5个字符

class HyDERetriever:
    """
    假设性文档嵌入(HyDE)检索器实现
    HyDE使用LLM生成假设性文档，然后对这些文档进行嵌入，而不是直接嵌入查询
    """
    
    def __init__(self, embedder, llm):
        """
        初始化HyDE检索器
        
        参数:
            embedder: 文本嵌入器
            llm: 大语言模型
        """
        self.embedder = embedder
        self.llm = llm
        self.doc_embeddings = []
        self.documents = []
        
    def index_documents(self, documents):
        """
        为文档创建索引
        
        参数:
            documents: 文档列表
        """
        self.documents = documents
        self.doc_embeddings = [self.embedder.embed_text(doc) for doc in documents]
        
    def retrieve(self, query, k=5, use_hyde=True):
        """
        检索与查询最相关的文档
        
        参数:
            query: 查询文本
            k: 要检索的文档数量
            use_hyde: 是否使用HyDE（如果为False，则使用传统向量检索）
            
        返回:
            相关文档的索引和得分 [(idx, score)]
        """
        if use_hyde:
            # 步骤1：使用LLM生成假设性文档
            hyde_prompt = f"请根据查询提供一个详细的答案：{query}"
            hypothetical_doc = self.llm.generate_text(hyde_prompt)
            
            # 步骤2：嵌入假设性文档，而不是原始查询
            query_embedding = self.embedder.embed_text(hypothetical_doc)
        else:
            # 传统向量检索：直接嵌入查询
            query_embedding = self.embedder.embed_text(query)
            
        # 步骤3：计算假设性文档嵌入与语料库中文档嵌入的相似度
        similarities = []
        for i, doc_embedding in enumerate(self.doc_embeddings):
            # 使用余弦相似度
            similarity = np.dot(query_embedding, doc_embedding)
            similarities.append((i, similarity))
            
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]

def test_hyde_algorithm():
    """
    测试HyDE算法
    """
    print("测试开始: 假设性文档嵌入(HyDE)算法")
    
    # 初始化嵌入器和LLM
    embedder = SimpleEmbedder(embedding_dim=64)
    llm = SimpleLLM()
    
    # 创建HyDE检索器
    hyde_retriever = HyDERetriever(embedder, llm)
    
    # 创建一些示例文档
    documents = [
        "搜索引擎优化(SEO)是提高网站在搜索引擎自然搜索结果中可见性的过程。",
        "机器学习是人工智能的一个子领域，专注于开发能够从数据中学习的算法。",
        "向量数据库专门设计用于存储和检索向量嵌入，支持向量相似度搜索。",
        "自然语言处理使计算机能够理解、分析和生成人类语言。",
        "向量嵌入是将词语、短语或文档映射到连续向量空间的技术。",
        "余弦相似度是衡量两个非零向量之间夹角余弦值的指标，常用于比较文档相似性。",
        "检索增强生成(RAG)结合了检索系统和文本生成模型，通过检索相关信息来增强生成结果。",
        "大语言模型(LLM)是能够理解和生成类似人类文本的深度学习模型。",
        "数据库索引是一种数据结构，通过提高数据检索操作的速度来改善数据库表的性能。",
        "信息检索是获取来自大型非结构化数据集合的相关信息的科学。"
    ]
    
    # 为文档创建索引
    hyde_retriever.index_documents(documents)
    
    # 测试查询
    test_queries = [
        "搜索引擎如何工作？",
        "向量相似度的应用有哪些？",
        "大型语言模型与信息检索的关系",
        "如何优化数据库查询性能？"
    ]
    
    for i, query in enumerate(test_queries):
        print(f"\n查询 {i+1}: \"{query}\"")
        
        # 使用传统向量检索
        start_time = time.time()
        traditional_results = hyde_retriever.retrieve(query, k=3, use_hyde=False)
        traditional_time = time.time() - start_time
        
        print("传统向量检索结果:")
        for idx, score in traditional_results:
            print(f"  文档 {idx}: 分数 {score:.4f}")
            print(f"    \"{documents[idx]}\"")
        print(f"  检索时间: {traditional_time:.4f} 秒")
        
        # 使用HyDE检索
        start_time = time.time()
        hyde_results = hyde_retriever.retrieve(query, k=3, use_hyde=True)
        hyde_time = time.time() - start_time
        
        # 获取生成的假设性文档
        hyde_prompt = f"请根据查询提供一个详细的答案：{query}"
        hypothetical_doc = llm.generate_text(hyde_prompt)
        
        print("\nHyDE检索结果:")
        print(f"  生成的假设性文档:\n    \"{hypothetical_doc}\"")
        
        for idx, score in hyde_results:
            print(f"  文档 {idx}: 分数 {score:.4f}")
            print(f"    \"{documents[idx]}\"")
        print(f"  检索时间: {hyde_time:.4f} 秒")
        
        # 比较两种方法检索到的文档
        traditional_ids = set([idx for idx, _ in traditional_results])
        hyde_ids = set([idx for idx, _ in hyde_results])
        
        common = traditional_ids.intersection(hyde_ids)
        print(f"\n两种方法共同检索到的文档数: {len(common)}")
        if len(common) < len(traditional_results):
            print("HyDE检索到了不同的文档，可能提供了更多相关信息。")
        
if __name__ == "__main__":
    test_hyde_algorithm() 