import os

import fitz
import numpy as np
from dotenv import load_dotenv, __main__
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from openai import OpenAI
from sympy import false

load_dotenv()

api_key = os.getenv("ZHIPU_API_KEY")
zhipu_url = os.getenv("ZHIPU_URL")

# client = OpenAILike(
#         model="glm-4.7",  # 或 "glm-4-plus"、"glm-4.7" 等（视您的账号支持而定）
#         api_key=api_key,
#         api_base=zhipu_url,
#         is_chat_model=True
#     )
client = OpenAI(
    base_url="https://open.bigmodel.cn/api/paas/v4/",  # 智谱官方 OpenAI 兼容地址
    api_key=os.getenv("ZHIPU_API_KEY")  # 必须显式传入，不能省略
)

def extra_text_from_pdf(pdf_path):
    """ 从PDF文件中提取文本 """
    mypdf = fitz.open(pdf_path)
    all_text = ""
    for page_num in range(mypdf.page_count):
        my_page = mypdf[page_num]
        text = my_page.get_text("text")
        all_text += text

    return all_text

def analyze_document_characteristics(text):
    """
    分析文档特征以确定最优分块大小

    Args:
        text (str): 文档文本

    Returns:
        dict: 文档特征分析结果
    """
    # 基础统计
    total_length = len(text)
    sentences = text.split(".")
    paragraphs = text.split("\n\n")

    # 计算特征
    avg_sentence_length = np.mean([len(s.strip()) for s in sentences if s.strip()])
    avg_paragraph_length = np.mean([len(p.strip()) for p in paragraphs if p.strip()])

    # 信息密度分析
    unique_words = len(set(text.lower().split()))
    total_words = len(text.split())
    vocabulary_richness = unique_words / total_words if total_words > 0 else 0

    # 结构复杂度
    line_breaks = text.count('\n')
    structural_complexity = line_breaks / total_length if total_length > 0 else 0

    characteristics = {
        'total_length': total_length,
        'avg_sentence_length': avg_sentence_length,
        'avg_paragraph_length': avg_paragraph_length,
        'vocabulary_richness': vocabulary_richness,
        'structural_complexity': structural_complexity,
        'sentence_count': len([s for s in sentences if s.strip()]),
        'paragraph_count': len([p for p in paragraphs if p.strip()])
    }

    return characteristics

def analyze_query_characteristics(query):
    """
    分析查询特征

    Args:
        query (str): 用户查询

    Returns:
        dict: 查询特征分析结果
    """
    query_length = len(query)
    word_count = len(query.split())
    question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
    # has_question_word = any(word.lower() in query.lower()  for word in question_words)
    has_question_word = False
    for word in question_words:
        if word in query.lower():
            has_question_word = True

    # 查询复杂度评估
    complexity_indicators = ['and', 'or', 'compare', 'difference', 'relationship', 'impact']
    # complexity_score = sum(1 for indicator in complexity_indicators if indicator in query.lower())
    complexity_score = 0
    for indicator in complexity_indicators:
        if indicator in query.lower():
            complexity_score += 1

    return {
        'query_length': query_length,
        'word_count': word_count,
        'has_question_word': has_question_word,
        'complexity_indicators': complexity_indicators,
        'complexity_score': complexity_score,
        'is_specific': word_count < 5,
        'is_complex': complexity_score >= 2
    }

def recommend_chunk_size(doc_characteristics, query_characteristics):
    """
    基于文档和查询特征推荐最优分块大小

    Args:
        doc_characteristics (dict): 文档特征
        query_characteristics (dict): 查询特征

    Returns:
        tuple: (推荐的分块大小, 重叠大小, 推荐理由)
    """
    base_chunk_size = 1000

    # 根据文档特征调整
    if doc_characteristics['avg_paragraph_length'] > 500:
        # 段落较长的文档，使用较大的分块
        doc_adjustment = 1.3
        reason = "文档段落较长，"
    elif doc_characteristics['avg_paragraph_length'] < 200:
        # 段落较短的文档，使用较小的分块
        doc_adjustment = 0.7
        reason = "文档段落较短，"
    else:
        doc_adjustment = 1.0
        reason = "文档结构适中，"

    # 根据词汇丰富度调整
    if doc_characteristics['vocabulary_richness'] > 0.7:
        vocab_adjustment = 1.2  # 词汇丰富，需要更大的上下文
        reason += "词汇丰富，"
    elif doc_characteristics['vocabulary_richness'] < 0.4:
        vocab_adjustment = 0.8  # 词汇单一，可以使用较小分块
        reason += "词汇相对单一，"
    else:
        vocab_adjustment = 1.0
        reason += "词汇密度适中，"

    # 根据查询特征调整
    if query_characteristics['is_complex']:
        query_adjustment = 1.4  # 复杂查询需要更多上下文
        reason += "查询复杂需要更多上下文，"
    elif query_characteristics['is_specific']:
        query_adjustment = 0.8  # 具体查询可以使用较小分块
        reason += "查询具体可使用较小分块，"
    else:
        query_adjustment = 1.0
        reason += "查询复杂度适中，"

    # 计算最终分块大小
    final_chunk_size = int(base_chunk_size * doc_adjustment * vocab_adjustment * query_adjustment)

    # 确保分块大小在合理范围内
    final_chunk_size = max(400, min(2000, final_chunk_size))

    # 计算重叠大小（通常为分块大小的20%）
    overlap_size = int(final_chunk_size * 0.2)

    reason += f"推荐分块大小为{final_chunk_size}字符"

    return final_chunk_size, overlap_size, reason

def create_chunks_with_size(text, chunk_size, overlap_size):
    """
    使用指定大小创建文本分块

    Args:
        text (str): 要分块的文本
        chunk_size (int): 分块大小
        overlap_size (int): 重叠大小

    Returns:
        List[str]: 文本分块列表
    """
    chunks = []

    step = chunk_size - overlap_size

    for i in range(0, len(text), step):
        chunk = text[i:i + step]
        if chunk.strip():
            chunks.append(chunk)

    return chunks

def create_embeddings(text, model="BAAI/bge-base-en-v1.5"):
    """ 为给定文本创建向量 """
    embedding_model = HuggingFaceEmbedding(model_name=model)
    if isinstance(text, list):
        response = embedding_model.get_text_embedding_batch(text)
    else:
        response = embedding_model.get_text_embedding(text)

    return response

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def search_with_chunks(query, chunks, embeddings, top_k=5):
    """
    使用给定的分块和嵌入进行搜索

    Args:
        query (str): 查询
        chunks (List[str]): 文本分块
        embeddings (List): 嵌入向量
        top_k (int): 返回的结果数量

    Returns:
        List[Dict]: 搜索结果
    """
    similarities = []
    query_embeddings = create_embeddings(query)
    for i,chunk_embedding  in enumerate(embeddings):
        similarity = cosine_similarity(
            np.array(query_embeddings),
            np.array(chunk_embedding)
        )
        similarities.append((i, similarity, chunks[i]))

    # 按相似度降序排序
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    results = []
    for i in range(min(top_k, len(similarities))):
        idx, score, chunk = similarities[i]
        results.append({
            "index": idx,
            "score": score,
            "chunk": chunk,
        })

    return results

def evaluate_chunk_size_performance(query, text, chunk_size, overlap_size):
    """
    评估特定分块大小的性能

    Args:
        query (str): 查询
        text (str): 文档文本
        chunk_size (int): 分块大小
        overlap_size (int): 重叠大小

    Returns:
        dict: 性能评估结果
    """
    # 创建分块
    chunks = create_chunks_with_size(text, chunk_size, overlap_size)

    # 创建嵌入
    embeddings = create_embeddings(chunks)

    # 执行搜索
    search_results = search_with_chunks(query, chunks, embeddings, top_k=3)
    # 计算性能指标
    avg_similarity = np.mean([result['score'] for result in search_results])
    chunk_count = len(chunks)
    avg_chunk_length = np.mean([len(chunk) for chunk in chunks])

    # 计算上下文覆盖率（top结果的总长度）
    total_context_length = sum(len(result['chunk']) for result in search_results)

    return {
        'chunk_size': chunk_size,
        'overlap_size': overlap_size,
        'chunk_count': chunk_count,
        'avg_chunk_length': avg_chunk_length,
        'avg_similarity': avg_similarity,
        'total_context_length': total_context_length,
        'search_results': search_results
    }

def compare_chunk_sizes(query, text, chunk_sizes=None):
    """
    比较不同分块大小的性能

    Args:
        query (str): 查询
        text (str): 文档文本
        chunk_sizes (List[int], optional): 要比较的分块大小列表

    Returns:
        List[Dict]: 各种分块大小的性能比较结果
    """
    if chunk_sizes is None:
        chunk_sizes = [400, 600, 800, 1000, 1200, 1500]

    results = []

    print(f"比较不同分块大小的性能...")

    for chunk_size in chunk_sizes:
        overlap_size = int(chunk_size * 0.2)  # 20%重叠

        performance = evaluate_chunk_size_performance(
            query, text, chunk_size, overlap_size
        )

        results.append(performance)

        print(f"分块大小 {chunk_size}: 平均相似度 {performance['avg_similarity']:.4f}, "
              f"分块数量 {performance['chunk_count']}")

    # 按平均相似度排序
    results.sort(key=lambda x: x['avg_similarity'], reverse=True)

    return results

def generate_response(query, context, model="glm-4.7"):
    """ 基于上下文生成回答 """
    system_prompt = "你是一个AI助手，严格基于给定的上下文回答问题。如果无法从提供的上下文中得出答案，请回答：'我没有足够的信息来回答这个问题。'"

    user_prompt = f"""
    上下文：
    {context}
    
    问题：{query}
    
    请基于以上上下文回答问题。
    """

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role":"system", "content":system_prompt},
            {"role":"user", "content":user_prompt},
        ]
    )

    return response.choices[0].message.content

def adaptive_chunking_rag(pad_path, query):
    """
    使用自适应分块的完整RAG流程

    Args:
        pdf_path (str): PDF文档路径
        query (str): 用户查询

    Returns:
        dict: 完整的处理结果
    """
    print("开始自适应分块RAG流程...")

    # 1. 提取文档文本
    print("1. 提取文档文本...")
    text = extra_text_from_pdf(pad_path)
    print(f"文档总长度{len(text)}字符")

    # 2. 分析文档特征
    print("2. 分析文档特征...")
    doc_characteristics = analyze_document_characteristics(text)
    print(f"文档特征: 平均段落长度={doc_characteristics['avg_paragraph_length']:.1f}, "
          f"词汇丰富度={doc_characteristics['vocabulary_richness']:.3f}")

    # 3. 分析查询特征
    print("3. 分析查询特征...")
    query_characteristics = analyze_query_characteristics(query)
    print(f"查询特征: 长度={query_characteristics['query_length']}, "
          f"复杂度={query_characteristics['complexity_score']}")

    #4. 推荐最优分块大小
    print("4. 推荐最优分块大小...")
    recommended_chunk_size, recommended_overlap, reason = recommend_chunk_size(
        doc_characteristics, query_characteristics
    )
    print(f"推荐策略: {reason}")

    # 5. 比较不同分块大小的性能
    print("5. 比较不同分块大小的性能...")
    comparison_results = compare_chunk_sizes(
        query, text,
        chunk_sizes=[400, 600, 800, recommended_chunk_size, 1200, 1500]
    )

    # 6. 使用最佳分块大小进行RAG
    print("6. 使用最佳分块大小进行RAG...")
    best_performance = comparison_results[0]
    best_chunk_size = best_performance['chunk_size']

    print(f"选择最佳分块大小: {best_chunk_size}")

    # 7. 生成最终回答
    context = "\n\n".join([
        f"段落{i + 1}: {result['chunk']}"
        for i, result in enumerate(best_performance['search_results'])
    ])

    response = generate_response(query, context)

    return {
        'query': query,
        'doc_characteristics': doc_characteristics,
        'query_characteristics': query_characteristics,
        'recommended_chunk_size': recommended_chunk_size,
        'recommended_reason': reason,
        'comparison_results': comparison_results,
        'best_chunk_size': best_chunk_size,
        'best_performance': best_performance,
        'context': context,
        'response': response
    }

if __name__ == "__main__":
    # 自适应分块RAG完整演示
    pdf_path = "../data/AI_Information.pdf"
    query = "深度学习的主要应用领域有哪些？"

    print(f"查询: {query}")
    print("=" * 60)

    # 执行自适应分块RAG
    result = adaptive_chunking_rag(pdf_path, query)

    # 显示文档分析结果
    print(f"\n📊 文档特征分析:")
    doc_chars = result['doc_characteristics']
    print(f"- 总长度: {doc_chars['total_length']} 字符")
    print(f"- 平均句子长度: {doc_chars['avg_sentence_length']:.1f} 字符")
    print(f"- 平均段落长度: {doc_chars['avg_paragraph_length']:.1f} 字符")
    print(f"- 词汇丰富度: {doc_chars['vocabulary_richness']:.3f}")

    # 显示查询分析结果
    print(f"\n🎯 查询特征分析:")
    query_chars = result['query_characteristics']
    print(f"- 查询长度: {query_chars['query_length']} 字符")
    print(f"- 词数: {query_chars['word_count']}")
    print(f"- 复杂度评分: {query_chars['complexity_score']}")
    print(f"- 是否具体查询: {query_chars['is_specific']}")

    # 显示推荐结果
    print(f"\n💡 推荐策略:")
    print(f"- 推荐分块大小: {result['recommended_chunk_size']} 字符")
    print(f"- 推荐理由: {result['recommended_reason']}")

    # 显示性能比较
    print(f"\n📈 分块大小性能比较:")
    print("分块大小 | 平均相似度 | 分块数量 | 平均分块长度")
    print("-" * 50)
    for perf in result['comparison_results'][:5]:
        print(f"{perf['chunk_size']:^8} | {perf['avg_similarity']:^10.4f} | "
              f"{perf['chunk_count']:^8} | {perf['avg_chunk_length']:^12.1f}")

    print(f"\n🏆 最佳分块大小: {result['best_chunk_size']} 字符")

    # 显示搜索结果
    print(f"\n🔍 搜索结果预览:")
    for i, search_result in enumerate(result['best_performance']['search_results'], 1):
        print(f"\n结果{i} (相似度: {search_result['score']:.4f}):")
        print(f"{search_result['chunk'][:200]}...")

    # 显示最终回答
    print(f"\n🤖 生成的回答:")
    print(result['response'])