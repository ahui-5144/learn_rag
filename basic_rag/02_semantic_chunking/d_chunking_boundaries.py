from b_sentence_embeddings import create_sentence_embeddings
from c_semantic_similarity import calculate_similarities

def find_chunk_boundaries(similarities, threshold=0.5) -> list[float]:
    """
    基于相似度阈值确定分块边界
    Args:
        similarities: 相邻句子相似度列表
        threshold: 相似度阈值，低于此值则分块

    Returns:
        boundaries: 分块边界位置列表
    """
    boundaries = [0] # 第一个边界总是0

    for i,sim in enumerate(similarities):
        if sim < threshold:
            boundaries.append(i + 1) # 在相似度低的地方设置边界

    boundaries.append(len(similarities) + 1) # 最后一个边界
    return boundaries


# 创建句子嵌入
sentences = ["AI is a branch of computer science.",
             "It aims to create intelligent machines.",
             "Machine learning is a subset of AI."]

embeddings = create_sentence_embeddings(sentences)

# 计算相邻句子相似度
similarities = calculate_similarities(embeddings)

# 确定分块边界
boundaries = find_chunk_boundaries(similarities, threshold=0.6)
print("分块边界:", boundaries)