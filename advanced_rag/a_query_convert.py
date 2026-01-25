import os

import fitz
import numpy as np
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from openai import OpenAI


load_dotenv()

"""
查询重写 (Query Rewriting)
步退提示 (Step-back Prompting)
子查询分解 (Sub-query Decomposition)
"""
client = OpenAI(
    base_url="https://open.bigmodel.cn/api/paas/v4/",  # 智谱官方 OpenAI 兼容地址
    api_key=os.getenv("ZHIPU_API_KEY")  # 必须显式传入，不能省略
)

def rewrite_query(original_query, model="glm-4.7"):
    """
    重写查询使其更具体和详细
    Args:
        original_query(str): 原始查询
        model(str):  用于查询重写的模型  使用的模型 (推荐是meta-llama/Llama-3.2-3B-Instruct)

    Returns:
        str:重写后的查询
    """
    system_prompt = "你是一个专门改进搜索查询的AI助手。你的任务是将用户查询重写得更具体、详细，更有可能检索到相关信息。"

    user_prompt = f"""
        将以下查询重写得更具体和详细。包含相关术语和概念，这些可能有助于检索准确信息。

        原始查询: {original_query}

        重写查询:
        """

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,  # 低温度确保输出确定性
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# original_query = "AI的影响"
# rewrite_query = rewrite_query(original_query)
# print(rewrite_query)

def generate_step_back_query(original_query, model="glm-4.7"):
    """

    Args:
        original_query(str): 原始用户查询
        model: 用于步退查询生成的模型 (推荐是meta-llama/Llama-3.2-3B-Instruct)

    Returns:
        str:步退查询
    """
    system_prompt = "你是一个专门研究搜索策略的AI助手。你的任务是为特定查询生成更广泛、更通用的版本，以检索有用的背景信息。"

    user_prompt = f"""
    为以下查询生成一个更广泛、更通用的版本，这可以帮助检索有用的背景信息。

    原始查询: {original_query}

    步退查询:
    """

    response = client.chat.completions.create(
        model=model,
        temperature=0.1,  # 稍高温度增加一些变化
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content.strip()


# original_query = "如何训练BERT模型？"
# step_back_query = generate_step_back_query(original_query)
# print(step_back_query)

def decompose_query(original_query, num_subqueries = 4, model="glm-4.7"):
    """
    将复杂查询分解为更简单的子查询

    Args:
        original_query(str): 原始复杂查询
        num_subqueries(int): 要生成的子查询数量
        model(str): 用于查询分解的模型(推荐是meta-llama/Llama-3.2-3B-Instruct)

    Returns:
        List[str]:更简单的子查询列表
    """
    system_prompt = "你是一个专门分解复杂问题的AI助手。你的任务是将复杂查询分解为更简单的子问题，这些子问题的答案结合起来可以解决原始查询。"

    user_prompt = f"""
    将以下复杂查询分解为{num_subqueries}个更简单的子查询。每个子查询应关注原始问题的不同方面。
    
    原始查询: {original_query}
    
    生成{num_subqueries}个子查询，每行一个，格式如下：
    1. [第一个子查询]
    2. [第二个子查询]
    以此类推...
    """

    response = client.chat.completions.create(
        model=model,
        temperature=0.2,  # 稍高温度增加一些变化
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    content = response.choices[0].message.content.strip()

    lines = content.split("\n")
    sub_queries = []

    for line in lines:
        if line.strip() and any(line.strip().startswith(f"{i}.") for i in range(1, 10)):
            query = line.strip()
            query = query[query.find(".") + 1:].strip()
            sub_queries.append(query)

    return sub_queries

# sub_queries = decompose_query("AI对就业的影响")
# for query in sub_queries:
#     print(query)

def extract_text_from_pdf(pdf_path: str):
    """
    从pdf提取文件
    Args:
        pdf_path: pdf文件路径

    Returns:
        str:所有文本
    """
    with fitz.open(pdf_path) as my_pdf:
        all_text = ''
        for page_num in range(my_pdf.page_count):
            page = my_pdf[page_num]
            text = page.get_text("text")
            all_text += text

    return all_text

def chunk_text(text, chunk_size = 1000, overlap = 200):
    """
    将文本分割成重叠的块
    Args:
        text: 文本
        chunk_size: 分块大小
        overlap: 上下分块之前的重叠大小

    Returns:

    """
    if not text:
        return []

    chunks = []

    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    return chunks

def create_embedding(text, model="BAAI/bge-base-en-v1.5"):
    """
    为给定文本创建嵌入向量
    Args:
        text: 文本
        model: 模型

    Returns:
        文本向量
    """
    embedding_model = HuggingFaceEmbedding(model_name=model)

    if isinstance(text, list):
        response = embedding_model.get_text_embedding_batch(text)
    else:
        response = embedding_model.get_text_embedding(text)
    return response

class SimpleVectorStore:
    """
    简单的向量存储实现
    """
    def __init__(self):
        self.vectors = []    # 存储向量（embeddings）
        self.documents = []  # 存储原始文档内容
        self.metadata = []   # 存储元数据

    def add_document(self, documents, vectors=None, metadata=None):
        """
        向向量存储添加文档
        """
        if vectors is None:
            vectors = [None] * len(documents) # Python 特性：[None] * n 创建包含 n 个 None 的列表

        if metadata is None:
            metadata = [{} for _ in range(len(documents))] # Python 列表推导式：为每个文档创建一个空字典 {}

        for doc, vec, meta in zip(documents, vectors, metadata):
            self.documents.append(doc)
            self.vectors.append(vec)
            self.metadata.append(meta)


    def search(self, query_vector, top_k = 5):
        """
        搜索最相似的文档
        Args:
            query_vector: 查询向量
            top_k: 获取相似度排名前几的结果

        Returns:
            List[dict]:相似度前几的结果
        """
        if not self.vectors or not self.documents:
            return []

        query_array = np.array(query_vector)

        # 计算相似度
        similarities = []
        for i,vector in enumerate(self.vectors):
            if vector is not None:
                similarity = np.dot(vector, query_array) / (np.linalg.norm(vector) * np.linalg.norm(query_array))
                similarities.append((i, similarity))

        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i , score in similarities[:top_k]:
            result = {
                "document": self.documents[i],
                "socre": score,
                "metadata": self.metadata[i]
            }
            results.append(result)

        return results


def process_document(
        pdf_path: str,
        chunk_size=1000,
        overlap=200
) -> list[SimpleVectorStore]:
    """
    处理文档用于RAG
    Args:
        pdf_path: pdf文档路径
        chunk_size: 分块大小
        overlap: 分块重叠大小

    Returns:
         list[SimpleVectorStore]: 文档分块向量
    """
    print("正在从PDF提取文本...")
    text = extract_text_from_pdf(pdf_path)

    print("准备文档分块")
    chunks = chunk_text(text, chunk_size, overlap)
    print(f"创建了{len(chunks)}个文档分块")

    print("为分块创建向量")
    chunk_embeddings = create_embedding(chunks)

    store = SimpleVectorStore()

