import numpy as np

from b_sentence_embeddings import create_sentence_embeddings

def cosine_similarity(vec1, vec2):
    """ 计算两个向量的余弦相似度 """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def calculate_similarities(embeddings):
    """计算相邻句子间的相似度"""
    similarities = []
    for i in range(len(embeddings) - 1):
        similarty = cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(similarty)
    return similarities

# 创建句子嵌入
sentences = ["AI is a branch of computer science.",
             "It aims to create intelligent machines.",
             "Machine learning is a subset of AI."]

embeddings = create_sentence_embeddings(sentences)

# 计算相邻句子相似度
similarities = calculate_similarities(embeddings)
print("相邻句子相似度:", similarities)
""" console
相邻句子相似度: [np.float64(0.7584673835092608), np.float64(0.7797275238723944)]
"""