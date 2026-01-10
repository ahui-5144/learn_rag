""" 上下文增强RAG """
import os

import fitz
import json
import numpy as np
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from openai import OpenAI

load_dotenv()

def extract_text_from_pdf(pdf_path):
    """
    从pdf 中提取文本内容
    Args:
        pdf_path (str): PDF文件路径

    Returns:
        str: 提取的文本内容
    """
    my_pdf = fitz.open(pdf_path)
    all_text = ""
    for page_num in range(my_pdf.page_count):
        page = my_pdf[page_num]
        text = page.get_text("text")
        all_text += text

    return all_text

def chunk_text(text:str, n:int, overlap:int):
    """
    将文本切割成重叠的文本块
    Args:
        text (str): 文本
        n (int): 每个文本块的字符数
        overlap (int): 重叠的字符数

    Returns:
        List[str]: 文本块列表

    """
    chunks = []

    step = n - overlap
    for i in range(0, len(text), step):
        chunks.append(text[i:i + n])  # 修复：每个块长度应为 n

    return chunks


client = OpenAI(
    base_url="https://open.bigmodel.cn/api/paas/v4/",  # 智谱官方 OpenAI 兼容地址
    api_key=os.getenv("ZHIPU_API_KEY")  # 必须显式传入，不能省略
)

def create_embeddings(text,model="BAAI/bge-base-en-v1.5"):
    """
    为给定的文本创建
    Args:
        text: 文本或者文本列表
        model: 嵌入模型名称

    Returns:
        List[float]: 嵌入向量
    """
    embedding_model = HuggingFaceEmbedding(model_name = model)

    if isinstance(text, list):
        response = embedding_model.get_text_embedding_batch(text)
    else:
        response = embedding_model.get_text_embedding(text)

    return response


def cosine_similarity(vec1, vec2):
    """
     计算两个向量的余弦相似度
    Args:
        vec1 (np.ndarray): 第一个向量
        vec2 (np.ndarray): 第二个向量

    Returns:
        float: 余弦相似度值
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def context_enriched_search(query, text_chunks, embeddings, k=1, context_size=1):
    """
    执行上下文增强检索

    Args:
        query (str): 查询问题
        text_chunks (List[str]): 文本块列表
        embeddings (List): 嵌入向量列表
        k (int): 检索的相关块数量
        context_size (int): 上下文邻居块数量

            context_size=0: 等同于传统 RAG，只返回最相关块
            context_size=1: 返回相关块及前后各 1 个邻居（推荐）
            context_size=2: 返回相关块及前后各 2 个邻居
            过大的 context_size: 可能引入噪声信息

            context_size选择
                academic_papers = 1-2    # 学术论文，逻辑严密
                technical_docs = 1       # 技术文档，条理清晰
                narrative_text = 2-3     # 叙述性文本，连贯性强

            块大小与上下文大小的平衡
                chunk_size = 1000    # 基础块大小
                context_size = 1     # 总上下文 ≈ 3000字符
                确保总上下文不超过模型限制
            推荐做法
                合理设置上下文大小: 从 1 开始测试，根据效果调整
                考虑文档结构: 对于结构化文档，保持逻辑完整性
                监控性能: 更多上下文意味着更高的计算成本
                评估质量: 定期评估上下文是否提升了回答质量
    Returns:
        List[str]: 包含上下文的相关文本块
    """
    # 将查询转换为嵌入向量
    query_embedding = create_embeddings(query)
    similarity_scores = []

    # 计算查询与每个文本块的相似度
    for i, chunk_embedding in enumerate(embeddings):
        similarity = cosine_similarity(
            np.array(query_embedding),
            np.array(chunk_embedding)
        )
        similarity_scores.append((i, similarity))

    # 按相似度降序排序
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # 获取最相关块的索引
    top_index = similarity_scores[0][0]
    print(f"最相关块索引：{top_index}")

    # 确定上下文反胃，确保不超出边界
    start = max(0, top_index - context_size)
    end = min(len(text_chunks), top_index + context_size)

    # 返回相关块及其邻近上下文
    result = []
    for i in range(start, end):
        chunk = text_chunks[i]
        result.append(chunk)

    return result

def generate_response(system_prompt, user_message, model="glm-4.7"):
    """
    生成AI回答

    Args:
        system_prompt(str): 系统提示词
        user_message(str): 用户消息
        model(str): 使用的模型

    Returns:
        str: AI生成回答
    """
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_message}
        ]
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    pdf_path = "../data/AI_Information.pdf"
    extracted_text = extract_text_from_pdf(pdf_path)

    # 2. 文本分块
    text_chunks = chunk_text(extracted_text, 1000, 200)
    print(f"创建了 {len(text_chunks)} 个文本块")

    # 3. 创建嵌入向量
    embeddings = create_embeddings(text_chunks)

    # 4. 加载测试查询
    with open('../data/val.json', encoding='utf-8') as f:
        data = json.load(f)

    # query = data[0]['question']
    query = "traditional programming versus machine learning"
    print(f"查询: {query}")

    # 5. 执行上下文增强检索
    # context_size=2 表示包含前后各2个邻居块，共5个块
    top_chunks = context_enriched_search(
        query,
        text_chunks,
        embeddings,
        k=1,
        context_size=2
    )

    print(f"检索到 {len(top_chunks)} 个上下文块")

    # 6. 显示检索结果
    for i, chunk in enumerate(top_chunks):
        print(f"上下文 {i + 1}:\n{chunk}\n" + "=" * 50)

    # 7. 生成最终回答
    system_prompt = "你是一个AI助手，严格基于给定的上下文回答问题。如果无法从提供的上下文中得出答案，请回答：'我没有足够的信息来回答这个问题。'"

    # 组合上下文
    context = "\n\n".join([f"上下文{i + 1}: {chunk}" for i, chunk in enumerate(top_chunks)])
    user_message = f"上下文:\n{context}\n\n问题: {query}"

    response = generate_response(system_prompt, user_message)
    print(f"AI回答: {response}")