"""
检索结果重排序算法模块

本模块实现了多种检索结果重排序算法，用于提高检索质量。
通过对初步检索结果进行再次排序，可以显著提升最终检索结果的相关性和多样性。

主要功能:
1. 倒数排名融合(RRF) - 合并多种检索方法的结果
2. BM25重排序 - 基于经典BM25算法的词法匹配重排序
3. 上下文相关重排序 - 考虑对话历史/会话上下文的重排序
4. 多样性重排序 - 使用最大边际相关性(MMR)算法平衡相关性与多样性

应用场景:
- RAG系统：提升检索组件的质量，为生成模型提供更相关的上下文
- 搜索引擎：改善搜索结果的相关性和多样性
- 对话系统：利用上下文历史提高结果的连贯性

支持中英文双语处理，集成了多种评分因素如词汇匹配、语义匹配、
上下文相关性、文档长度和结果多样性等。
"""

import numpy as np
import time
from collections import defaultdict
import re
import heapq
import jieba  # 添加jieba中文分词

class RetrievalReranker:
    """
    检索结果重排序器
    
    用于实现各种重排序技术，对初步检索的结果进行再次排序，
    综合考虑多种因素提高检索质量。
    
    原理:
        重排序是信息检索系统的关键组件，通常在初步检索后应用。
        初始检索阶段通常优先考虑效率，获取候选文档集合；
        重排序阶段则更注重精度，使用更复杂的模型和特征进行精细排序。
        
    支持的重排序方法:
        1. 倒数排名融合(RRF)：合并多个排序列表的结果
        2. 交叉编码器重排序：模拟Cross-Encoder对query-doc对进行直接打分
        3. BM25重排序：使用经典的BM25算法进行词法匹配重排序
        4. 上下文重排序：考虑对话历史/会话上下文进行重排序
        5. 多样性重排序：平衡相关性和结果多样性
        
    特点:
        - 支持中英文双语处理
        - 提供详细的评分解释
        - 可配置多种重排序参数
        - 支持与嵌入模型集成进行语义排序
    """
    
    def __init__(self, embedder=None):
        """
        初始化重排序器
        
        参数:
            embedder: 可选的嵌入器，用于语义相似度计算
        """
        self.embedder = embedder
        # 中文停用词
        self.stopwords = {'的', '了', '和', '与', '或', '及', '等', '是', '在', '对', '从', '到', '把', '被', '让', '使', '由', 
                       '对于', '如何', '什么', '怎么', '怎样', '这个', '那个', '这种', '那种', '一个', '一种', '有些', '有的',
                       'a', 'an', 'the', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                       'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'may', 'might', 'must', 
                       'can', 'could', 'for', 'to', 'with', 'in', 'on', 'at', 'by', 'of', 'about', 'as', 'if', 'then', 'but', 'so'}
        
    def reciprocal_rank_fusion(self, rank_lists, k=60):
        """
        倒数排名融合(RRF)算法，将多个排名列表合并为单一排名
        
        原理:
            RRF是一种简单而强大的排名融合算法，通过对每个排名列表中文档的排名取倒数，
            然后求和得到最终排名。参数k是一个平滑因子，防止极端排名（如第1位）对结果影响过大。
            
        公式:
            RRF(d) = ∑ 1/(k + r(d))
            其中r(d)是文档d在各排名列表中的排名位置，k是平滑常数。
            
        优点:
            - 实现简单，计算高效
            - 不需要训练，无需调整大量参数
            - 对不同类型和尺度的排名结果具有鲁棒性
            - 能有效融合不同检索方法的优点
        
        参数:
            rank_lists (List[List[Tuple]]): 多个排名列表，每个为[(doc_id, score)]格式
            k (int): RRF常数，用于平滑排名的影响，默认60
            
        返回:
            List[Tuple[any, float]]: 融合后的排名列表 [(doc_id, score)]，按分数降序排序
        """
        # 初始化RRF分数字典
        rrf_scores = defaultdict(float)
        
        # 为每个排名列表计算RRF分数
        for rank_list in rank_lists:
            for rank, (doc_id, _) in enumerate(rank_list):
                # RRF公式: 1 / (k + rank)
                rrf_scores[doc_id] += 1.0 / (k + rank)
                
        # 转换为列表并排序
        result = [(doc_id, score) for doc_id, score in rrf_scores.items()]
        result.sort(key=lambda x: x[1], reverse=True)
        
        return result
        
    def cross_encoder_rerank(self, query, documents, doc_ids, top_k=None):
        """
        使用Cross-Encoder模型思想进行重排序
        
        原理:
            Cross-Encoder模型直接对查询-文档对(query-document pair)进行打分，
            通常比Bi-Encoder(如向量检索)更精确但速度更慢。
            这里提供一个简化实现，综合考虑词汇重叠度和语义相似度(如果提供嵌入器)。
            
        不同于Bi-Encoder的方式:
            - Bi-Encoder: 查询和文档分别编码为向量，然后计算相似度
            - Cross-Encoder: 将查询和文档作为一个整体输入到模型中直接得到分数
            
        在实际应用中，应使用预训练的Cross-Encoder模型如BERT等进行精确打分。
        本实现为简化版，模拟Cross-Encoder的基本思想。
        
        参数:
            query (str): 查询文本
            documents (List[str]): 文档文本列表
            doc_ids (List): 文档ID列表
            top_k (int, optional): 返回的最大文档数，默认为None(返回所有)
            
        返回:
            List[Tuple[any, float]]: 重排序的文档ID和分数列表 [(doc_id, score)]，按分数降序排序
        """
        # 注意：这是简化版本，真实应用中应使用预训练模型
        scores = []
        
        # 简单的相关性打分（模拟）
        for i, doc in enumerate(documents):
            # 1. 计算查询词在文档中的出现次数
            query_terms = set(self._tokenize(query.lower()))
            doc_terms = self._tokenize(doc.lower())
            
            # 避免空白文档导致的除零错误
            if not doc_terms:
                term_overlap_ratio = 0.0
            else:
                term_overlap = sum(1 for term in doc_terms if term in query_terms)
                term_overlap_ratio = term_overlap / len(doc_terms)
            
            # 2. 计算余弦相似度（如果提供了嵌入器）
            semantic_score = 0
            if self.embedder:
                query_embedding = self.embedder.embed_text(query)
                doc_embedding = self.embedder.embed_text(doc)
                semantic_score = np.dot(query_embedding, doc_embedding)
                
            # 3. 计算最终分数
            # 添加基础分数0.1，确保即使没有词条重叠也能返回有意义的分数
            final_score = 0.1 + term_overlap_ratio * 0.4 + semantic_score * 0.5
            
            scores.append((doc_ids[i], final_score))
            
        # 按分数降序排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 限制结果数量
        if top_k:
            scores = scores[:top_k]
            
        return scores
        
    def _extract_keywords(self, text):
        """
        使用jieba提取文本中的关键词
        
        功能:
            根据文本语言类型选择适当的分词方法，提取文本中的有意义关键词。
            - 中文文本：使用jieba分词
            - 英文文本：使用空格分词
            过滤掉停用词和单字词(对中文)，保留更有信息量的词语。
        
        参数:
            text (str): 输入文本
            
        返回:
            List[str]: 提取出的关键词列表
        """
        # 检测是否含有中文字符
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
        
        if has_chinese:
            # 中文文本使用jieba分词
            words = [word for word in jieba.cut(text) if word not in self.stopwords and len(word) > 1]
        else:
            # 英文文本使用常规分词
            tokens = self._tokenize(text.lower())
            words = [word for word in tokens if word not in self.stopwords and len(word) > 1]
        
        # 如果没有提取到关键词，返回原始分词结果
        if not words:
            return self._tokenize(text.lower())
        
        return words

    def bm25_rerank(self, query, documents, doc_ids, top_k=None, k1=1.5, b=0.75):
        """
        使用BM25算法重排序文档
        
        原理:
            BM25是经典的信息检索排序算法，是对TF-IDF的改进版本。
            主要考虑三个因素:
            1. 词频(TF)：词在文档中出现的频率，但有饱和控制
            2. 逆文档频率(IDF)：衡量词的稀有度
            3. 文档长度归一化：偏好与平均长度相近的文档
            
        公式:
            score(q,d) = ∑(IDF(qi) * ((tf(qi,d) * (k1+1)) / (tf(qi,d) + k1 * (1-b+b*|d|/avgdl))))
            
        增强功能:
            1. 对查询词赋予不同权重：
               - 位置权重：前面出现的词更重要
               - 长度权重：长词可能更具鉴别性
            2. 匹配比例奖励：匹配查询词比例越高分数越高
            3. 文档长度因子：适度奖励更长的文档
            4. 位置惩罚：确保排序稳定性
            
        支持中英文双语处理：
            - 使用jieba对中文进行分词
            - 使用关键词提取而非简单分词，提高相关性
        
        参数:
            query (str): 查询文本
            documents (List[str]): 文档文本列表
            doc_ids (List): 文档ID列表
            top_k (int, optional): 返回的最大文档数
            k1 (float): BM25参数，控制词频缩放，默认1.5
            b (float): BM25参数，控制文档长度归一化，默认0.75
            
        返回:
            List[Tuple[any, float]]: 重排序的文档ID和分数列表 [(doc_id, score)]，按分数降序排序
        """
        print("\nBM25重排序调试信息:")
        print(f"查询: '{query}'")
        
        # 使用jieba提取查询关键词
        query_terms = self._extract_keywords(query)
        print(f"提取的查询关键词: {query_terms}")
        
        # 预处理所有文档，提取关键词
        doc_keywords = []
        for doc in documents:
            kw = self._extract_keywords(doc)
            doc_keywords.append(kw)
            
        # 计算文档的平均长度
        doc_lengths = [len(kw) for kw in doc_keywords]
        avg_doc_length = sum(doc_lengths) / len(documents) if documents else 0
        print(f"文档平均长度: {avg_doc_length:.2f} 关键词")
        
        # 计算文档频率（包含词的文档数）
        df = defaultdict(int)
        for doc_terms in doc_keywords:
            doc_terms_set = set(doc_terms)
            for term in query_terms:
                if term in doc_terms_set:
                    df[term] += 1
        
        print(f"文档频率统计:")
        for term in query_terms:
            print(f"  '{term}': 出现在 {df[term]} 个文档中")
        
        # 给查询词赋予不同的权重 - 越重要的词权重越高
        term_weights = {}
        for i, term in enumerate(query_terms):
            # 位置权重 - 一般前面出现的词更重要
            position_weight = 1.0 - 0.1 * min(i, 5)  # 最多降低0.5
            # 长度权重 - 长词可能更有鉴别性
            length_weight = min(1.0, len(term) / 5)
            # 最终权重
            term_weights[term] = position_weight * (0.5 + 0.5 * length_weight)
        
        print(f"查询词权重:")
        for term, weight in term_weights.items():
            print(f"  '{term}': {weight:.2f}")
                    
        # 计算每个文档的BM25分数
        scores = []
        N = len(documents)  # 文档总数
        
        for i, (doc, doc_terms) in enumerate(zip(documents, doc_keywords)):
            doc_len = len(doc_terms)
            
            # 计算文档中每个词的频率
            term_freq = Counter()
            for term in doc_terms:
                term_freq[term] += 1
            
            # 计算文档的BM25分数
            score = 0
            matched_terms = 0
            term_contributions = []  # 记录每个词的贡献
            
            for term in query_terms:
                if term in term_freq and term in df:
                    matched_terms += 1
                    # 计算IDF
                    idf = np.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
                    
                    # 计算TF部分
                    tf = term_freq[term]
                    tf_score = ((k1 + 1) * tf) / (k1 * (1 - b + b * doc_len / avg_doc_length) + tf)
                    
                    # 使用词权重调整分数
                    term_weight = term_weights.get(term, 1.0)
                    term_contribution = idf * tf_score * term_weight
                    score += term_contribution
                    
                    # 记录每个词的贡献
                    term_contributions.append((term, term_contribution))
            
            # 基础分数和匹配奖励
            base_score = 0.1
            
            # 匹配比例奖励 - 匹配的查询词越多，分数越高
            if query_terms:
                match_ratio = matched_terms / len(query_terms)
                match_bonus = match_ratio * 0.6  # 提高匹配奖励的权重
            else:
                match_bonus = 0
            
            # 使用文档长度作为轻微加权因子
            length_factor = min(0.05, doc_len / (avg_doc_length * 10)) if avg_doc_length > 0 else 0
            
            # 最终分数 = 基础分数 + BM25分数 + 匹配奖励
            # 显著放大匹配分数的贡献，减少位置惩罚的影响
            final_score = base_score + score * 2.0 + match_bonus + length_factor
            
            # 对于排名后的文档添加小幅度的惩罚，确保排序稳定且有区分度
            position_penalty = i * 0.01
            final_score -= position_penalty
                
            scores.append((doc_ids[i], final_score))
            
            # 打印前3个文档的详细得分
            if i < 3:
                print(f"\n文档 {i}: 长度 {doc_len} 关键词")
                print(f"  匹配词: {matched_terms}/{len(query_terms)}")
                print(f"  词汇贡献:")
                for term, contribution in term_contributions:
                    print(f"    '{term}': {contribution:.4f}")
                print(f"  匹配奖励: {match_bonus:.4f}")
                print(f"  长度因素: {length_factor:.4f}")
                print(f"  位置惩罚: {position_penalty:.4f}")
                print(f"  最终分数: {final_score:.4f}")
                print(f"  文档内容: {doc[:100]}...")
        
        # 按分数降序排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 限制结果数量
        if top_k:
            scores = scores[:top_k]
            
        return scores
    
    def contextual_rerank(self, query, documents, doc_ids, context=None, top_k=None):
        """
        考虑上下文的重排序，基于当前对话/会话历史
        
        原理:
            在对话系统和交互式搜索中，当前查询往往不是独立的，而是与之前的会话历史相关。
            上下文重排序考虑当前查询和历史上下文，使结果更加连贯和相关。
            
        核心思想:
            1. 从当前查询提取关键词
            2. 从会话历史(上下文)提取关键词，并赋予权重(越近的上下文权重越高)
            3. 计算文档与查询和上下文的匹配程度
            4. 组合多种因素进行综合评分:
               - 查询匹配得分(更高权重)
               - 上下文匹配得分
               - 文档长度奖励
               - 关键词覆盖率奖励
               - 位置稳定性因子
               
        应用场景:
            - 多轮对话系统
            - 交互式信息检索
            - 连续查询的搜索会话
        
        参数:
            query (str): 当前查询文本
            documents (List[str]): 文档文本列表
            doc_ids (List): 文档ID列表
            context (List[str], optional): 对话上下文/历史查询列表
            top_k (int, optional): 返回的最大文档数
            
        返回:
            List[Tuple[any, float]]: 重排序的文档ID和分数列表 [(doc_id, score)]，按分数降序排序
        """
        print("\n上下文重排序调试信息:")
        
        if context is None or not context:
            # 无上下文时，直接使用普通重排序
            print("无上下文信息，使用普通重排序")
            return self.cross_encoder_rerank(query, documents, doc_ids, top_k)
        
        # 从查询和上下文提取关键词
        query_keywords = set(self._extract_keywords(query))
        print(f"查询关键词: {query_keywords}")
        
        # 从上下文中提取关键词，给不同位置的上下文不同的权重
        context_keywords = {}
        for i, ctx in enumerate(context):
            # 越近的上下文权重越高
            recency_weight = 0.5 + 0.5 * (i + 1) / len(context)  # 0.5-1.0范围
            ctx_keywords = self._extract_keywords(ctx)
            print(f"上下文 {i+1} 关键词 (权重 {recency_weight:.2f}): {ctx_keywords}")
            for keyword in ctx_keywords:
                if keyword in context_keywords:
                    context_keywords[keyword] = max(context_keywords[keyword], recency_weight)
                else:
                    context_keywords[keyword] = recency_weight
        
        print(f"合并上下文关键词权重:")
        for keyword, weight in context_keywords.items():
            print(f"  '{keyword}': {weight:.2f}")
        
        # 预处理文档关键词
        doc_keywords_list = [set(self._extract_keywords(doc)) for doc in documents]
        
        # 计算文档与查询+上下文的相关性分数
        scores = []
        
        for i, (doc, doc_keywords) in enumerate(zip(documents, doc_keywords_list)):
            # 1. 基础分数
            base_score = 0.1
            
            # 2. 直接匹配查询的加分 - 提高权重
            query_match_score = 0
            query_overlap = 0
            if query_keywords:
                query_overlap = len(query_keywords.intersection(doc_keywords))
                query_match_score = 0.7 * (query_overlap / len(query_keywords))  # 提高查询匹配的权重
            
            # 3. 匹配上下文的加分
            context_match_score = 0
            matched_context_keywords = []
            if context_keywords:
                context_overlap = 0
                for keyword in doc_keywords:
                    if keyword in context_keywords:
                        context_overlap += context_keywords[keyword]  # 加上权重值
                        matched_context_keywords.append(keyword)
                context_match_score = 0.3 * (context_overlap / (sum(context_keywords.values()) if context_keywords else 1))
            
            # 4. 文档长度权重 - 适度奖励更长的文档，因为它们可能包含更多信息
            doc_len = len(doc_keywords)
            avg_doc_len = sum(len(kw) for kw in doc_keywords_list) / len(documents) if documents else 1
            len_factor = min(0.05, 0.3 * (doc_len / avg_doc_len))
            
            # 5. 关键词覆盖率奖励 - 匹配了多少不同的关键词
            all_keywords = query_keywords.union(context_keywords.keys())
            coverage_bonus = 0
            if all_keywords:
                keyword_coverage = len(doc_keywords.intersection(all_keywords)) / len(all_keywords)
                coverage_bonus = 0.2 * keyword_coverage
            
            # 6. 位置惩罚 - 轻微降低后面文档的分数，确保排序稳定
            position_penalty = 0.005 * i
            
            # 计算最终分数
            final_score = base_score + query_match_score + context_match_score + len_factor + coverage_bonus - position_penalty
            
            scores.append((doc_ids[i], final_score))
            
            # 打印前3个文档的详细得分
            if i < 3:
                print(f"\n文档 {i} 评分详情:")
                print(f"  查询匹配分数: {query_match_score:.4f} ({query_overlap}/{len(query_keywords)} 查询词)")
                print(f"  上下文匹配分数: {context_match_score:.4f} (匹配关键词: {matched_context_keywords})")
                print(f"  长度因素: {len_factor:.4f} (长度: {doc_len})")
                print(f"  覆盖率奖励: {coverage_bonus:.4f}")
                print(f"  位置惩罚: {position_penalty:.4f}")
                print(f"  最终分数: {final_score:.4f}")
                print(f"  文档内容: {doc[:100]}...")
        
        # 按分数降序排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 限制结果数量
        if top_k:
            scores = scores[:top_k]
            
        return scores
    
    def diversity_rerank(self, query, documents, doc_ids, top_k=None, lambda_param=0.5):
        """
        多样性重排序，同时考虑相关性和结果多样性
        
        原理:
            使用最大边际相关性(Maximum Marginal Relevance, MMR)算法平衡相关性与多样性。
            MMR算法在选择文档时，不仅考虑与查询的相关性，还考虑与已选择文档的差异性，
            避免返回过于相似的结果集。
            
        MMR算法:
            迭代选择文档，每次选择能够最大化以下公式的文档:
            MMR = λ * sim(q,d) - (1-λ) * max(sim(d,d_j))
            其中:
            - sim(q,d): 文档d与查询q的相似度
            - max(sim(d,d_j)): 文档d与已选择文档集合中任一文档d_j的最大相似度
            - λ: 权衡相关性与多样性的参数，λ越大越注重相关性，λ越小越注重多样性
            
        应用场景:
            - 搜索结果去冗余
            - 推荐系统提供多样化选择
            - 文档摘要生成
        
        参数:
            query (str): 查询文本
            documents (List[str]): 文档文本列表
            doc_ids (List): 文档ID列表
            top_k (int, optional): 返回的最大文档数
            lambda_param (float): 多样性权重参数(0-1之间)，默认0.5
                                  越大则越注重相关性，越小则越注重多样性
            
        返回:
            List[Tuple[any, float]]: 重排序的文档ID和分数列表 [(doc_id, score)]，按分数降序排序
            
        说明:
            该方法需要embedder进行向量嵌入，如果未提供embedder将抛出异常。
        """
        if not self.embedder:
            raise ValueError("多样性重排序需要提供embedder")
            
        if top_k is None or top_k > len(documents):
            top_k = len(documents)
            
        # 计算查询的嵌入
        query_embedding = self.embedder.embed_text(query)
        
        # 计算所有文档的嵌入和相关性分数
        doc_embeddings = []
        sim_scores = []
        
        for doc in documents:
            doc_embedding = self.embedder.embed_text(doc)
            doc_embeddings.append(doc_embedding)
            
            # 计算与查询的相似度
            sim = np.dot(query_embedding, doc_embedding)
            sim_scores.append(sim)
            
        # MMR算法实现
        selected = []
        selected_ids = []
        
        # 将文档索引、ID和得分放入候选池
        candidates = list(zip(range(len(documents)), doc_ids, sim_scores))
        
        for _ in range(top_k):
            # 计算MMR得分并找到最大值
            mmr_scores = []
            
            for i, doc_id, sim_score in candidates:
                if i in selected:
                    continue
                    
                # 如果尚未选择任何文档，则仅使用相关性
                if not selected:
                    mmr_scores.append((i, doc_id, sim_score))
                    continue
                    
                # 计算与已选文档的最大相似度
                max_sim_to_selected = max(
                    np.dot(doc_embeddings[i], doc_embeddings[j])
                    for j in selected
                )
                
                # MMR公式: lambda * sim(q,d) - (1-lambda) * max(sim(d,d_j))
                mmr_score = lambda_param * sim_score - (1 - lambda_param) * max_sim_to_selected
                mmr_scores.append((i, doc_id, mmr_score))
                
            if not mmr_scores:
                break
                
            # 选择具有最高MMR得分的文档
            mmr_scores.sort(key=lambda x: x[2], reverse=True)
            best_idx, best_id, best_score = mmr_scores[0]
            
            selected.append(best_idx)
            selected_ids.append((best_id, best_score))
            
        return selected_ids
    
    def _tokenize(self, text):
        """
        简单分词，将文本拆分为标记列表
        
        处理流程:
            1. 将文本转为小写
            2. 移除标点符号
            3. 按空格拆分为标记列表
        
        参数:
            text (str): 输入文本
            
        返回:
            List[str]: 分词后的标记列表
        """
        # 转为小写并移除标点
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        
        # 分词
        return text.split()

class Counter(dict):
    """
    简单的计数器实现，用于计算词频
    
    继承自字典类型，对不存在的键返回0而不是抛出异常，
    便于直接进行词频统计。
    """
    def __missing__(self, key):
        """
        处理访问不存在的键时的行为，返回0而不是抛出异常
        
        参数:
            key: 字典键
            
        返回:
            int: 对于不存在的键返回0
        """
        return 0

def test_retrieval_reranking():
    """
    测试检索结果重排序和打分机制
    
    本测试函数全面评估重排序算法在不同场景下的性能:
    
    1. 倒数排名融合(RRF)测试：
       - 模拟向量检索、BM25和混合检索的结果
       - 使用RRF算法融合多个结果列表
       
    2. BM25重排序测试：
       - 英文查询测试："how to reduce hallucination in large language models"
       - 中文查询测试："如何减少大型语言模型的幻觉问题"
       - 特定内容查询测试："大语言模型的局限性"
       
    3. 上下文重排序测试：
       - 相关上下文测试：提供与当前查询相关的历史查询
       - 不相关上下文测试：提供与当前查询无关的历史查询
       
    4. 效果评估：
       - 对比不同方法的TOP-1结果
       - 关键词匹配评估：检查包含特定关键词的文档是否被正确排在前面
       
    测试数据集包含多个中英文文档，涵盖RAG、语言模型和向量检索等主题，
    用于模拟真实应用场景。
    """
    print("测试开始: 检索结果重排序与打分机制")
    
    # 创建一些示例文档，增加一些中文文档
    documents = [
        "检索增强生成（RAG）是一种将信息检索与自然语言生成相结合的方法，用于提高大型语言模型生成的准确性和相关性。",
        "向量数据库是一种专门设计用于高效存储和检索向量嵌入的数据库系统。",
        "大型语言模型（LLM）是一类基于Transformer架构的神经网络，经过大规模文本数据训练，能够理解和生成自然语言。",
        "混合检索系统结合了多种检索方法，如词法匹配（BM25）和语义搜索（基于向量嵌入），以提高检索效果。",
        "上下文窗口大小是LLM处理能力的一个关键限制。当需要处理大量检索文档时，有效压缩上下文同时保留关键信息变得至关重要。",
        "检索系统的评估通常使用精确率、召回率、F1分数和平均精确率等指标。",
        "语义搜索使用向量嵌入来捕捉查询和文档的含义，而不仅仅是关键词匹配。",
        "BM25是一种流行的词法检索算法，它考虑了词频、逆文档频率和文档长度等因素。",
        "多阶段检索管道通常包括初始检索、重排序和最终排序等步骤，以平衡效率和准确性。",
        "检索增强生成可以显著减少大型语言模型的幻觉问题，提高回答的事实准确性。"
    ]
    
    # 添加明确包含特定关键词的测试文档
    test_specific_docs = [
        "幻觉问题是大型语言模型面临的主要挑战，检索增强生成技术可以通过提供事实依据来解决这个问题。",
        "局限性主要体现在上下文窗口大小、知识更新和事实准确性方面，需要通过外部知识增强来解决。",
        "向量检索技术使用向量嵌入表示文本语义，可以捕捉文本的深层含义，实现更精准的匹配。"
    ]
    
    # 合并文档集
    all_documents = documents + test_specific_docs
    
    # 创建文档ID
    doc_ids = list(range(len(all_documents)))
    
    print(f"测试文档集大小: {len(all_documents)} 个文档")
    
    # 1. 测试RRF排名融合 
    print("\n=== 1. 测试倒数排名融合(RRF) ===")
    
    # 创建模拟的多个排名列表（来自不同检索方法）
    vector_ranks = [(0, 0.92), (9, 0.85), (2, 0.76), (3, 0.65), (6, 0.60)]
    bm25_ranks = [(9, 8.5), (0, 7.8), (7, 6.2), (3, 5.9), (8, 5.5)]
    hybrid_ranks = [(0, 0.88), (9, 0.85), (3, 0.82), (2, 0.75), (8, 0.72)]
    
    # 初始化重排序器
    reranker = RetrievalReranker()
    
    # 测试倒数排名融合
    rrf_results = reranker.reciprocal_rank_fusion([vector_ranks, bm25_ranks, hybrid_ranks])
    
    print("倒数排名融合(RRF)结果:")
    for i, (doc_id, score) in enumerate(rrf_results[:5]):
        print(f"  {i+1}. 文档 {doc_id}: {score:.6f} - {all_documents[doc_id][:50]}...")
    
    # 2. 测试BM25重排序 - 英文查询
    print("\n=== 2. 测试BM25重排序 (英文查询) ===")
    query_en = "how to reduce hallucination in large language models"
    
    bm25_results_en = reranker.bm25_rerank(query_en, all_documents, doc_ids, top_k=5)
    
    print(f"英文查询: '{query_en}'")
    print("BM25重排序结果:")
    for i, (doc_id, score) in enumerate(bm25_results_en):
        print(f"  {i+1}. 文档 {doc_id}: {score:.6f} - {all_documents[doc_id]}")
    
    # 3. 测试BM25重排序 - 中文查询
    print("\n=== 3. 测试BM25重排序 (中文查询) ===")
    query_zh = "如何减少大型语言模型的幻觉问题"
    
    bm25_results_zh = reranker.bm25_rerank(query_zh, all_documents, doc_ids, top_k=5)
    
    print(f"中文查询: '{query_zh}'")
    print("BM25重排序结果:")
    for i, (doc_id, score) in enumerate(bm25_results_zh):
        print(f"  {i+1}. 文档 {doc_id}: {score:.6f} - {all_documents[doc_id]}")
    
    # 4. 测试BM25重排序 - 针对特定内容的查询
    print("\n=== 4. 测试BM25重排序 (特定内容查询) ===")
    query_specific = "大语言模型的局限性"
    
    bm25_results_specific = reranker.bm25_rerank(query_specific, all_documents, doc_ids, top_k=5)
    
    print(f"特定内容查询: '{query_specific}'")
    print("BM25重排序结果:")
    for i, (doc_id, score) in enumerate(bm25_results_specific):
        print(f"  {i+1}. 文档 {doc_id}: {score:.6f} - {all_documents[doc_id]}")
    
    # 5. 测试上下文重排序 - 相关上下文
    print("\n=== 5. 测试上下文重排序 (相关上下文) ===")
    related_context = [
        "什么是语言模型？", 
        "大型语言模型有哪些优点和缺点？"
    ]
    
    contextual_results = reranker.contextual_rerank(query_zh, all_documents, doc_ids, related_context, top_k=5)
    
    print("相关上下文:")
    for i, ctx in enumerate(related_context):
        print(f"  对话 {i+1}: {ctx}")
    print(f"  当前查询: {query_zh}")
    print("上下文重排序结果:")
    for i, (doc_id, score) in enumerate(contextual_results):
        print(f"  {i+1}. 文档 {doc_id}: {score:.6f} - {all_documents[doc_id]}")
    
    # 6. 测试上下文重排序 - 不相关上下文
    print("\n=== 6. 测试上下文重排序 (不相关上下文) ===")
    unrelated_context = [
        "什么是向量数据库？", 
        "最快的搜索算法是什么？"
    ]
    
    unrelated_results = reranker.contextual_rerank(query_zh, all_documents, doc_ids, unrelated_context, top_k=5)
    
    print("不相关上下文:")
    for i, ctx in enumerate(unrelated_context):
        print(f"  对话 {i+1}: {ctx}")
    print(f"  当前查询: {query_zh}")
    print("上下文重排序结果:")
    for i, (doc_id, score) in enumerate(unrelated_results):
        print(f"  {i+1}. 文档 {doc_id}: {score:.6f} - {all_documents[doc_id]}")
    
    # 7. 对比测试 - 分析不同方法的性能差异
    print("\n=== 7. 不同重排序方法的性能对比 ===")
    
    # 构建结果集合
    results_collection = {
        "BM25 (英文查询)": bm25_results_en,
        "BM25 (中文查询)": bm25_results_zh,
        "BM25 (特定内容)": bm25_results_specific,
        "相关上下文重排序": contextual_results,
        "不相关上下文重排序": unrelated_results
    }
    
    # 比较各方法的TOP-1文档
    print("各方法返回的TOP-1文档:")
    for name, results in results_collection.items():
        if results:
            doc_id, score = results[0]
            print(f"  {name}: 文档 {doc_id} - 得分: {score:.6f}")
            print(f"    {all_documents[doc_id]}")
    
    # 8. 评估算法效果
    print("\n=== 8. 算法效果评估 ===")
    
    # 针对'幻觉问题'相关查询，期望包含'幻觉'的文档排名靠前
    hallucination_query = "幻觉问题"
    hallucination_results = reranker.bm25_rerank(hallucination_query, all_documents, doc_ids, top_k=5)
    
    # 检查前3个结果中是否包含'幻觉'关键词的文档
    hallucination_docs = []
    for doc_id, _ in hallucination_results[:3]:
        if "幻觉" in all_documents[doc_id]:
            hallucination_docs.append(doc_id)
    
    print(f"测试查询: '{hallucination_query}'")
    if hallucination_docs:
        print(f"在前3个结果中找到了{len(hallucination_docs)}个包含'幻觉'关键词的文档，表现良好")
        print(f"相关文档ID: {hallucination_docs}")
    else:
        print("在前3个结果中没有找到包含'幻觉'关键词的文档，需要改进算法")
    
    # 针对'向量检索'相关查询，期望包含'向量'的文档排名靠前
    vector_query = "向量检索方法"
    vector_results = reranker.bm25_rerank(vector_query, all_documents, doc_ids, top_k=5)
    
    # 检查前3个结果中是否包含'向量'关键词的文档
    vector_docs = []
    for doc_id, _ in vector_results[:3]:
        if "向量" in all_documents[doc_id]:
            vector_docs.append(doc_id)
    
    print(f"\n测试查询: '{vector_query}'")
    if vector_docs:
        print(f"在前3个结果中找到了{len(vector_docs)}个包含'向量'关键词的文档，表现良好")
        print(f"相关文档ID: {vector_docs}")
    else:
        print("在前3个结果中没有找到包含'向量'关键词的文档，需要改进算法")
    
if __name__ == "__main__":
    test_retrieval_reranking() 