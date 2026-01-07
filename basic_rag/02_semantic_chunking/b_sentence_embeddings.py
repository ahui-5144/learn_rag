import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def create_sentence_embeddings(sentences, model="BAAI/bge-base-en-v1.5"):
    """
    为每个句子创建向量
    Args:
        sentences: 句子
        model: 模型

    Returns:

    """
    embedding_model = HuggingFaceEmbedding(model_name=model)
    embeddings = embedding_model.get_text_embedding_batch(sentences) #(3, 768) 但无shape属性
    return np.array(embeddings) # 转换后的数组，可高效运算

# 创建句子嵌入
sentences = ["AI is a branch of computer science.",
             "It aims to create intelligent machines.",
             "Machine learning is a subset of AI."]

embeddings = create_sentence_embeddings(sentences)
print(f"嵌入矩阵形状: {embeddings.shape}")
""" console
嵌入矩阵形状: (3, 768)   3个向量，每个向量有 768个坐标
"""