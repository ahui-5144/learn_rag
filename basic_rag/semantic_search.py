import numpy as np

from chunk_text import chunk_text
from extract_text_from_pdf import extract_text_from_pdf
from create_embeddings import create_embeddings


def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def semantic_search(query, text_chunks, embeddings, k=5):
    """
    基于语义相似度搜索最相关的文档快
    """
    # 为查询创建嵌入向量
    query_embeddings = create_embeddings(query)
    similarity_scores = []

    # 计算查询与每个文档块的相似度
    for i, chunk_embedding in enumerate(embeddings):
        similarity = cosine_similarity(
            np.array(query_embeddings),
            np.array(chunk_embedding)
        )
        similarity_scores.append((i, similarity)) # similarity_scores 预期是一个元组列表（list of tuples），每个元组的结构为 (index, similarity)

    # 按相似度排序并返回top-k结果
    # key=lambda x: x[1] 告诉 sort() 方法：排序时，只关注每个元素 x 的第二个值（即 x[1]，相似度分数）。
    # reverse=True 指定降序排序，即相似度最高的分数排在最前面
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    """
    similarity_scores[:k] 列表切片，取前 k 个元素（因为之前已按相似度降序排序，前 k 个就是最相似的 k 个）
    for index, _ in similarity_scores[:k] 
        遍历这前 k 个元素。每个元素是一个元组 (index, similarity)：
        index：文档块在原始列表中的位置（整数）。
        _：占位符，表示我们不关心第二个值（相似度分数），因此用下划线 _ 忽略它（Python 惯例）
        
    [index for ... ]
        列表推导式：对于遍历到的每个元组，只取出第一个值 index，并收集成一个新列表。    
    """
    top_indices = [index for index, _ in similarity_scores[:k]]
    return [text_chunks[index] for index in top_indices]

# 示例搜索
query = "What is artificial intelligence?"

pdf_path = "data/AI_Information.pdf"
extract_text = extract_text_from_pdf(pdf_path)
text_chunks = chunk_text(extract_text, 1000, 200)
# 创建所有文本块的嵌入
chunks_embeddings = create_embeddings(text_chunks)

top_chunks = semantic_search(query, text_chunks, chunks_embeddings,  k=3)

print(f"查询: {query}")
for i, chunk in enumerate(top_chunks):
    print(f"结果 {i+1}:\n{chunk}\n" + "="*50)

