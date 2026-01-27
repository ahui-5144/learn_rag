import os
import re

import fitz
import numpy as np
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv('ZHIPU_API_KEY'),
    base_url=os.getenv('ZHIPU_URL'),
)

def extract_text_from_pdf(pdf_path):
    """
    从PDF中提取文本

    Args:
        pdf_path:pdf文件路径

    Returns:
        str: pdf中的文本
    """
    all_text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text = page.get_text("text")
            all_text += text

    return all_text

def chunk_text(text, n, overlap):
    """将文本分割成重叠的块"""
    chunks = []

    for i in range(0, len(text), n - overlap):
        chunks.append(text[i:i + n])

    return chunks




class SimpleVectorStore:
    """简单的向量存储实现"""
    def __init__(self):
        self.vectors = []  # 存储向量（embeddings）
        self.texts = []    # 存储原始文档内容
        self.metadata = [] # 存储元数据

    def add_item(self, text, embedding, metadata=None):
        """向向量存储添加项目"""
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})

    @staticmethod
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def similarity_search(self, query_embeddings, top_k = 5):
        """
        相似度搜索
        Args:
            query_embeddings: 查询语句向量化
            top_k: 返回的最相关的条数

        Returns:
            list[dist]: 相似度最高的结果
        """
        if not self.vectors:
            return []

        similarities = []
        for i,vector in enumerate(self.vectors):
            similarity = self.cosine_similarity(np.array(query_embeddings), vector)
            similarities.append((i, similarity))

        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 返回top_k结果
        results = []
        for i in range(min(top_k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": score
            })

        return results


def create_embeddings(text, model="BAAI/bge-base-en-v1.5"):
    """
    为给定文本创建嵌入向量

    Args:
        text:
        model: BAAI/bge-base-en-v1.5

    Returns:

    """
    embedding_model = HuggingFaceEmbedding(model_name=model)
    if isinstance(text, list):
        response = embedding_model.get_text_embedding_batch(text)
    else:
        response = embedding_model.get_text_embedding(text)

    return response

def rerank_with_llm(query, results, top_n=3, model="GLM-4.5"):
    """
    使用LLM相关性评分对搜索结果进行重排序

    Args:
        query(str): 用户查询
        results(list[Dict]):初始搜索结果
        top_n(int): 重排序后返回的结果数
        model(str):用于评分的模型 meta-llama/Llama-3.2-3B-Instruct

    Returns:
        List[Dict]: 重排序后的结果
    """
    if len(results) <= top_n:
        return results

    scored_results = []

    for i, result in enumerate(results):
        # 为每个结果生成相关性评分
        scoring_prompt = f"""
        请评估以下文档与用户查询的相关性。

        用户查询: {query}

        文档内容: {result['text'][:500]}...

        请给出0-10的相关性评分（10分最相关）。
        只返回数字分数，不要其他内容。
        """

        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role":"system","content":"你是一个专业的文档相关性评估专家。"},
                    {"role": "user", "content": scoring_prompt}
                ]
            )

            # 提取评分
            score_text = response.choices[0].message.content.strip()
            score = float(re.search(r'\d+\.?\d*', score_text).group())

            # 添加到结果中
            result['llm_score'] = score
            scored_results.append(result)
        except Exception as e:
            print(f"评分结果 {i} 时出错: {e}")
            result['llm_score'] = 0
            scored_results.append(result)

    scored_results.sort(key=lambda x: x['llm_score'], reverse=True)

    return scored_results[:top_n]

def rerank_with_keywords(query, results, top_n=3):
    """
    基于关键词匹配对搜索结果进行重排序

    Args:
        query (str): 用户查询
        results (List[Dict]): 初始搜索结果
        top_n (int): 重排序后返回的结果数

    Returns:
        List[Dict]: 重排序后的结果
    """
    # 提取查询中的关键词
    query_keywords = set(re.findall(r'\b\w+\b', query.lower()))

    scored_results = []

    for i, result in enumerate(results):
        text_lower = result["text"].lower()
        text_keywords = re.findall(r'\b\w+\b', text_lower)

        # 计算关键词匹配度
        common_keywords = query_keywords.intersection(text_keywords)
        keyword_score = len(common_keywords) / len(query_keywords) if query_keywords else 0

        # 计算关键词在文档中的频率
        keyword_frequency = sum(text_lower.count(keyword) for keyword in common_keywords)

        # 计算文档长度权重（较短文档可能更相关）
        length_penalty = 1 / (1 + len(result['text']) / 1000)

        # 综合评分：原始相似度 + 关键词匹配 + 频率 + 长度权重
        combined_score = (
                result['similarity'] * 0.4 +
                keyword_score * 0.3 +
                min(keyword_frequency / 10, 0.2) +  # 限制频率权重
                length_penalty * 0.1
        )

        result['keyword_score'] = combined_score
        scored_results.append(result)

    # 按综合评分降序排序
    scored_results.sort(key=lambda x: x['keyword_score'], reverse=True)

    return scored_results[:top_n]

def generate_response(query, context, model="GLM-4.5"):
    """基于上下文生成回答 meta-llama/Llama-3.2-3B-Instruct"""
    system_prompt = "你是一个AI助手,严格基于给定的上下文回答问题.如果无法从提供的上下文中得出答案，请回答：'我没有足够的信息来回答这个问题。'"

    user_prompt = f"""
    上下文：
    {context}
    
    问题： {query}
    
    请基于以上上下文回答问题
    """

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages = [
            {"role":"system","content":system_prompt},
            {"role": "user", "content":user_prompt}
        ]
    )

    return response.choices[0].message.content

def rag_with_reranking(query, vector_store, reranking_method="llm", top_n=3,model="GLM-4.5"):
    """
    使用重排序的RAG系统

    Args:
        query (str): 用户查询
        vector_store: 向量存储
        reranking_method (str): 重排序方法 ("llm", "keywords", "none")
        top_n (int): 最终返回的结果数
        model (str): 使用的模型 meta-llama/Llama-3.2-3B-Instruct

    Returns:
        dict: 包含重排序结果和生成回答的字典
    """
    # 1.初始检索（获取更多候选结果）
    initial_k = max(top_n * 2, 10)
    query_embedding = create_embeddings(query)
    initial_results = vector_store.similarity_search(query_embedding, initial_k)

    print(f"初始检索到{len(initial_results)} 个结果")

    # 2.应用重排序
    if reranking_method == "llm":
        # print("应用LLM重排序...")
        # reranked_results = rerank_with_llm(query, initial_results, top_n, model)
        print("应用LLM批量重排序。。。")
        reranked_results = enhanced_llm_rerank(query, initial_results, top_n)
    elif reranking_method == "keywords":
        print("应用关键词重排序...")
        reranked_results = rerank_with_keywords(query, initial_results, top_n)
    else:
        print("不使用重排序...")
        reranked_results = initial_results[:top_n]

    # 3.准备上下文
    context = "\n\n".join([
        f"段落{i + 1}: {result['text']}"
        for i, result in enumerate(reranked_results)
    ])

    # 4. 生成回答
    response = generate_response(query, context, model)

    return {
        "query": query,
        "reranking_method": reranking_method,
        "initial_results_count": len(initial_results),
        "reranked_results": reranked_results,
        "context": context,
        "response": response
    }


def enhanced_llm_rerank(query, results, top_n=3, model="GLM-4.5", max_retries=3):
    """
    增强的LLM重排序方法

    特点：
    1. 批量评分：一次LLM调用对所有文档打分，提高效率
    2. 重试机制：API失败时自动重试
    3. 结果添加llm_score字段并按评分排序

    Args:
        query (str): 用户查询
        results (List[Dict]): 初始搜索结果
        top_n (int): 重排序后返回的结果数
        model (str): 用于评分的模型
        max_retries (int): 最大重试次数

    Returns:
        List[Dict]: 重排序后的结果
    """
    if len(results) <= top_n:
        return results

    # 构建批量评分的Prompt
    scoring_prompt = f"""
    作为文档相关性专家，请评估以下文档与查询的相关性。

    评估标准：
    1. 内容匹配度：文档是否包含查询所需信息
    2. 答案完整性：文档是否能完整回答查询
    3. 信息质量：文档信息的准确性和可靠性

    查询: {query}

    请为每个文档给出0-10的评分，格式：文档编号:分数|简短理由

    """

    # 批量添加所有文档到Prompt
    for i, result in enumerate(results):
        scoring_prompt += f"\n文档{i + 1}: {result['text'][:400]}...\n"

    scoring_prompt += "\n请按以上格式，为每个文档给出评分（只返回评分结果，不要其他内容）："

    # 带重试机制的LLM调用
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": "你是一个专业的文档相关性评估专家。"},
                    {"role": "user", "content": scoring_prompt}
                ]
            )

            score_text = response.choices[0].message.content.strip()
            print(f"LLM评分结果: {score_text}")

            # 解析批量评分结果
            llm_scores = {}
            for line in score_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                # 匹配 "文档1:8.5|理由" 或 "文档1:8.5" 或 "1:8.5" 等格式
                match = re.search(r'文档?(\d+)[：:]\s*(\d+\.?\d*)', line)
                if match:
                    doc_num = int(match.group(1))
                    score = float(match.group(2))
                    llm_scores[doc_num] = score

            print(f"解析到的评分: {llm_scores}")

            # 添加到结果中
            scored_results = []
            for i, result in enumerate(results):
                doc_num = i + 1
                llm_score = llm_scores.get(doc_num, 0.0)  # 未解析到评分为0

                # 添加到结果中
                result['llm_score'] = llm_score
                scored_results.append(result)

            # 按LLM评分降序排序
            scored_results.sort(key=lambda x: x['llm_score'], reverse=True)

            return scored_results[:top_n]

        except Exception as e:
            print(f"LLM重排序失败（第{attempt + 1}次尝试）: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(1)  # 等待1秒后重试
            else:
                print("所有重试均失败，返回原始相似度排序结果")
                return results[:top_n]

# def advanced_keyword_rerank(query, results):
#     """高级关键词重排序"""
#     # 提取不同类型的关键词
#     important_words = extract_important_words(query)  # 重要词汇
#     entities = extract_entities(query)  # 命名实体
#     concepts = extract_concepts(query)  # 概念词汇
#
#     for result in results:
#         text = result['text']
#
#         # 计算不同类型匹配的权重
#         important_match = calculate_word_match(important_words, text) * 0.4
#         entity_match = calculate_entity_match(entities, text) * 0.3
#         concept_match = calculate_concept_match(concepts, text) * 0.3
#
#         # 考虑位置权重（关键词在文档开头更重要）
#         position_weight = calculate_position_weight(important_words, text)
#
#         combined_score = (important_match + entity_match + concept_match) * position_weight
#         result['advanced_keyword_score'] = combined_score
#
#     return sorted(results, key=lambda x: x['advanced_keyword_score'], reverse=True)

def normalize_scores(scores):
    """默认使用min-max归一化到[0,1]"""
    if not scores:
        return []
    min_val = min(scores)
    max_val = max(scores)
    if max_val - min_val < 1e-8:  # 几乎相等
        return [1.0] * len(scores)  # 倾向给满分，或业务决定给0
    return [(s - min_val) / (max_val - min_val) for s in scores]

def get_llm_scores(query, results):
    # 1. 调用前记录原始索引
    for i, result in enumerate(results):
        result['_original_index'] = i

    # 2. 调用 enhanced_llm_rerank
    reranked = enhanced_llm_rerank(query, results, top_n=len(results))

    # 3. 按原始索引恢复顺序并提取分数
    reranked.sort(key=lambda x: x['_original_index'])

    # 4. 清理临时字段并返回
    scores = [r['llm_score'] for r in reranked]
    for r in reranked:
        r.pop('_original_index', None)

    return scores


def get_keyword_scores(query, results):
    """
    复用 rerank_with_keywords 的评分逻辑
    返回分数列表，保持原始顺序
    """
    # 提取查询中的关键词
    query_keywords = set(re.findall(r'\b\w+\b', query.lower()))
    scores = []

    for result in results:
        text_lower = result["text"].lower()
        text_keywords = re.findall(r'\b\w+\b', text_lower)

        # 计算关键词匹配度
        common_keywords = query_keywords.intersection(text_keywords)
        keyword_score = len(common_keywords) / len(query_keywords) if query_keywords else 0

        # 计算关键词在文档中的频率
        keyword_frequency = sum(text_lower.count(keyword) for keyword in common_keywords)

        # 计算文档长度权重（较短文档可能更相关）
        length_penalty = 1 / (1 + len(result['text']) / 1000)

        # 综合评分：原始相似度 + 关键词匹配 + 频率 + 长度权重
        combined_score = (
            result['similarity'] * 0.4 +
            keyword_score * 0.3 +
            min(keyword_frequency / 10, 0.2) +
            length_penalty * 0.1
        )

        scores.append(combined_score)

    return scores


def hybrid_rerank(query, results, weights={'llm': 0.5, 'keywords': 0.3, 'similarity': 0.2}):
    """混合重排序方法"""
    #1. 获取各种评分
    llm_scores = get_llm_scores(query, results)
    keyword_scores = get_keyword_scores(query, results)
    similarity_scores = [r['similarity'] for r in results]

    # 归一化评分
    llm_scores = normalize_scores(llm_scores)
    keyword_scores = normalize_scores(keyword_scores)
    similarity_scores = normalize_scores(similarity_scores)

    # 计算加权综合评分
    final_scores = []
    for i in range(len(results)):
        weighted_score = (
                llm_scores[i] * weights['llm'] +
                keyword_scores[i] * weights['keywords'] +
                similarity_scores[i] * weights['similarity']
        )
        final_scores.append(weighted_score)
        results[i]['hybrid_score'] = weighted_score

    return sorted(results, key=lambda x: x['hybrid_score'], reverse=True)

if __name__ == "__main__":
    pdf_path = "../basic_rag/data/AI_Information.pdf"
    print("处理文档。。。")

    # 提取文本
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text, 1000, 200)
    print(f"创建了{len(chunks)} 个分块")

    # 创建嵌入和向量存储
    embeddings = create_embeddings(chunks)
    vector_store = SimpleVectorStore()

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector_store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={"index": i, "source": pdf_path}
        )

    print(f"向量存储已建立，包含 {len(chunks)} 个块")

    query = "traditional programming versus machine learning"
    print(f"\n测试查询: {query}")

    # 比较不同重排序方法
    methods = ["llm", "keywords", "none"]
    results = {}

    for method in methods:
        print(f"\n{'=' * 20} {method.upper()} 方法 {'=' * 20}")

        result = rag_with_reranking(
            query=query,
            vector_store=vector_store,
            reranking_method=method,
            top_n=3
        )

        results[method] = result

        print(f"重排序后的结果:")
        for i, res in enumerate(result['reranked_results'], 1):
            print(f"\n结果 {i}:")
            if method == "llm":
                print(f"LLM评分: {res.get('llm_score', 'N/A')}")
            elif method == "keywords":
                print(f"关键词评分: {res.get('keyword_score', 'N/A'):.4f}")
            print(f"原始相似度: {res['similarity']:.4f}")
            print(f"内容: {res['text'][:150]}...")

        print(f"\n生成的回答:")
        print(result['response'])

    # 显示对比总结
    print(f"\n{'=' * 60}")
    print("重排序方法对比总结:")
    print("=" * 60)

    for method, result in results.items():
        print(f"\n【{method.upper()}方法】")
        print(f"回答: {result['response'][:200]}...")