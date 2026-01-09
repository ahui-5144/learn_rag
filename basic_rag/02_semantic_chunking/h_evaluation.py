"""
语义分块质量评估模块
用于评估 RAG 系统中分块策略的效果
"""
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# ============== 嵌入相关函数 ==============

def create_embeddings(texts, model_name="BAAI/bge-base-en-v1.5"):
    """
    为文本列表创建嵌入向量

    Args:
        texts: 可以是字符串列表或句子列表的列表（chunks格式）
        model_name: 嵌入模型名称

    Returns:
        numpy数组: 嵌入向量
    """
    # 处理 chunks 格式：将句子列表拼接成字符串
    processed_texts = []
    for text in texts:
        if isinstance(text, list):
            # 如果是句子列表，拼接成字符串
            joined = " ".join(text)
            if joined.strip():  # 过滤空字符串
                processed_texts.append(joined)
        else:
            # 如果已经是字符串，直接使用
            if text.strip():  # 过滤空字符串
                processed_texts.append(text)

    embed_model = HuggingFaceEmbedding(model_name=model_name)
    embeddings = embed_model.get_text_embedding_batch(processed_texts)
    return np.array(embeddings)


# ============== 相似度计算函数 ==============

def cosine_similarity(vec1, vec2):
    """
    计算余弦相似度

    Args:
        vec1: 向量1
        vec2: 向量2

    Returns:
        float: 余弦相似度值
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# ============== 语义搜索函数 ==============

def semantic_search(query, chunks, chunk_embeddings, k=1):
    """
    基于语义相似度搜索最相关的文档块

    Args:
        query: 查询语句
        chunks: 分块列表
        chunk_embeddings: 分块向量化数据
        k: 取的条数

    Returns:
        list: 最相关的k个分块
    """
    # 创建查询的嵌入
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    query_embedding = np.array(embed_model.get_text_embedding(query))

    # 计算查询与每个分块的相似度
    similarity_scores = []
    for i, chunk_embedding in enumerate(chunk_embeddings):
        similarity = cosine_similarity(query_embedding, chunk_embedding)
        similarity_scores.append((i, similarity))

    # 按相似度排序，取top-k
    sorted_similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_indices = [index for index, _ in sorted_similarity_scores[:k]]

    return [chunks[index] for index in top_indices]


# ============== 评估函数 ==============

def evaluate_chunking_quality(chunks, queries, ground_truth):
    """
    评估分块质量

    Args:
        chunks: 分块列表（句子列表的列表）
        queries: 测试查询列表
        ground_truth: 每个查询期望包含在检索结果中的关键词

    Returns:
        dict: 包含评估指标的字典
    """
    # 创建分块嵌入
    print("正在创建分块嵌入...")
    chunk_embeddings = create_embeddings(chunks)

    # 检索测试
    correct_retrievals = 0
    total_queries = len(queries)

    print(f"正在评估 {total_queries} 个查询...")
    for query, truth in zip(queries, ground_truth):
        retrieved = semantic_search(query, chunks, chunk_embeddings, k=1)

        # 将检索结果转换为文本
        retrieved_chunk = retrieved[0]
        if isinstance(retrieved_chunk, list):
            retrieved_text = " ".join(retrieved_chunk)
        else:
            retrieved_text = str(retrieved_chunk)

        # 检查ground truth是否在检索结果中
        if truth.lower() in retrieved_text.lower():
            correct_retrievals += 1
            print(f"  ✓ 查询: '{query}' - 检索正确")
        else:
            print(f"  ✗ 查询: '{query}' - 检索失败")

    accuracy = correct_retrievals / total_queries

    # 分块统计
    chunk_sizes = [len(chunk) if isinstance(chunk, str) else sum(len(s) for s in chunk)
                   for chunk in chunks]
    avg_size = np.mean(chunk_sizes)
    size_variance = np.var(chunk_sizes)

    return {
        'accuracy': accuracy,
        'avg_chunk_size': avg_size,
        'size_variance': size_variance,
        'num_chunks': len(chunks),
        'correct_retrievals': correct_retrievals,
        'total_queries': total_queries
    }


def print_evaluation_report(results):
    """
    打印评估报告

    Args:
        results: evaluate_chunking_quality返回的结果字典
    """
    print("\n" + "=" * 50)
    print("分块质量评估报告")
    print("=" * 50)
    print(f"检索准确率:     {results['accuracy']:.2%}")
    print(f"正确检索数:     {results['correct_retrievals']} / {results['total_queries']}")
    print(f"分块总数:       {results['num_chunks']}")
    print(f"平均分块大小:   {results['avg_chunk_size']:.0f} 字符")
    print(f"分块大小方差:   {results['size_variance']:.0f}")
    print("=" * 50)


# ============== 主程序 ==============

if __name__ == "__main__":
    from semantic_chunking_complete import SemanticChunker

    # 1. 创建语义分块器并处理文档
    print("正在处理文档...")
    chunker = SemanticChunker(similarity_threshold=0.5)
    chunks = chunker.process_document("../data/AI_Information.pdf")

    # 2. 定义测试查询和期望结果
    queries = [
        "What is Artificial Intelligence?",
        "How do neural networks learn?",
        "What are the types of machine learning?"
    ]

    ground_truth = [
        "Artificial Intelligence",
        "neural network",
        "machine learning"
    ]

    # 3. 运行评估
    results = evaluate_chunking_quality(chunks, queries, ground_truth)

    # 4. 打印报告
    print_evaluation_report(results)
