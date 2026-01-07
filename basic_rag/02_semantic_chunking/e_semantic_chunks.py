import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def create_semantic_chunks(sentences, boundaries):
    """
    根据边界创建语义分块
    Args:
        sentences: 句子列表
        boundaries: 边界

    Returns:

    """
    chunks = []
    for i in range (len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1]
        chunk = " ".join(sentences[start:end])

        chunks.append(chunk)

    return chunks

# 创建句子嵌入
text = "AI is a branch of computer science。It aims to create intelligent machines。Machine learning is a subset of AI。"

def sentence_splitting(text) -> list[str]:
    result = []
    for s in text.split("。"):
        if s.strip() != "":
            result.append(s.strip() + ".")

    return result

def create_sentence_embeddings(sentences, model="BAAI/bge-base-en-v1.5"):
    embedding_model = HuggingFaceEmbedding(model_name=model)
    sentence_embeddings = embedding_model.get_text_embedding_batch(sentences)
    return np.array(sentence_embeddings)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def calculate_similarities(embeddings):
    similarities = []
    for i in range(len(embeddings) - 1):
        similarity = cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(similarity)
    return similarities

def find_chunk_boundaries(similarities, threshold=0.5):
    boundaries = [0]

    for k,sim in enumerate(similarities):
        if sim < threshold:
            boundaries.append(k + 1)

    boundaries.append(len(similarities) + 1)
    return boundaries

# 将文本分割为句子列表
sentences = sentence_splitting(text)

# 为每个句子创建向量
embeddings = create_sentence_embeddings(sentences)

# 计算相邻句子相似度
similarities = calculate_similarities(embeddings)

# 确定分块边界
boundaries = find_chunk_boundaries(similarities, threshold=0.6)

semantic_chunks = create_semantic_chunks(sentences, boundaries)
print("语义分块结果：")
for i, chunk in enumerate(semantic_chunks):
    print(f"块{i+1}:{chunk}")



if __name__ == "__main__":
    text = "AI is a branch of computer science。It aims to create intelligent machines。Machine learning is a subset of AI."
    sentences = sentence_splitting(text)
    print(sentences)