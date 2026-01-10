""" 上下文头部分块 """
import json
import os
from typing import Any

import fitz
import numpy as np
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# 原始分块
chunk = "Transformers use self-attention mechanisms..."

# 增强后的分块
enhanced_chunk = """
[Document: Attention Is All You Need]
[Section: 3.2 Attention Mechanisms]
[Keywords: transformer, self-attention, neural networks]
[Context: Deep learning architectures]

Transformers use self-attention mechanisms...
"""

def extract_text_from_pdf(pdf_path):
    """
       从PDF文件中提取文本内容

       Args:
           pdf_path (str): PDF文件路径

       Returns:
           str: 提取的文本内容
       """
    mypdf = fitz.open(pdf_path)
    all_text = ""

    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]
        text = page.get_text("text")
        all_text += text

    return all_text

# 初始化OpenAI客户端
client = OpenAI(
    base_url="https://open.bigmodel.cn/api/paas/v4/",  # 智谱官方 OpenAI 兼容地址
    api_key=os.getenv("ZHIPU_API_KEY")  # 必须显式传入，不能省略
)


def generate_chunk_header(chunk, model="glm-4.7"):
    """
    使用LLM为给定的文本块省生成标题/头部

    Args:
        chunk(str): 要生成头部的文本块
        model(str): 使用的模型名称

    Returns:
        str:生成的头部/标题
    """
    # 定义系统提示语
    system_prompt = "为给定的文本生成简洁且信息丰富的标题"

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":chunk},
        ]
    )

    return response.choices[0].message.content.strip()

def chunk_text_with_headers(text, n , overlap):
    """
    将文本分块并为每个块生成头部

    Args:
        text(str): 要分块的完整文本
        n(int): 快大小(字符数)
        overlap(int): 重叠字符数

    Returns:
        List[dict]: 包含 'header' 和 'text' 键的字典列表
    """
    chunks = []
    # 先计算所有分块，确定总数
    chunk_indices = [(i, i + n) for i in range(0, len(text), n - overlap)]

    for start, end in tqdm(chunk_indices, desc="生成头部"):
        chunk = text[start:end]
        header = generate_chunk_header(chunk)
        chunks.append({"header": header, "text": chunk})

    return chunks


def create_embeddings(text, model="BAAI/bge-base-en-v1.5"):
    """
    为给定文本创建嵌入向量
    Args:
        text(str):输入文本
        model(str): 嵌入模型名称

    Returns:
        List[float]:嵌入向量
    """
    embedding_modle = HuggingFaceEmbedding(model_name=model)
    if isinstance(text, list):
        response = embedding_modle.get_text_embedding_batch(text)
    else:
        response = embedding_modle.get_text_embedding(text)

    return response

def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度

    Args:
        vec1(np.array):第一个向量
        vec2(np.array):第二个向量

    Returns:
        float:余弦相似度值
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def semantic_search(query, chunks, top_k=5):
    """
    基于查询搜索最相关的文本块

    Args:
        query (str): 用户查询
        chunks (List[dict]): 包含头部和嵌入的文本块列表
        top_k (int): 返回的结果数

    Returns:
        List[dict]: 最相关的k的文本块
    """
    # 为查询创建嵌入
    query_embedding = create_embeddings(query)

    similarities = []

    for i, chunk in enumerate(chunks):
        # 计算查询与文本内容的相似度
        similarity_text = cosine_similarity(
            np.array(query_embedding),
            np.array(chunk["embedding"])
        )
        # 计算查询与头部的相似度
        similarity_header = cosine_similarity(
            np.array(query_embedding),
            np.array(chunk["header_embedding"])
        )
        # 计算平均相似度
        avg_similarity = (similarity_text + similarity_header) / 2
        similarities.append((chunk, avg_similarity))
    # 按相似度降序排序
    similarities.sort(key=lambda x: x[1], reverse=True)

    # 返回前k个最相关的块
    return [ x[0] for x in similarities[:top_k]]

def generate_response(system_prompt, user_message, model="glm-4.7"):
    """
    生成AI回答

    Args:
        system_prompt (str): 系统提示词
        user_message (str): 用户消息
        model (str): 使用的模型

    Returns:
        str: AI生成的回答
    """
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_message},
        ]
    )

    return response.choices[0].message.content.strip()


""" 优化 """
def avg_similarity_update(sim_text, sim_header):
    # 方法1: 简单平均（推荐）
    avg_similarity = (sim_text + sim_header) / 2

    # 方法2: 加权平均
    weight_text = 0.7
    weight_header = 0.3
    weighted_similarity = weight_text * sim_text + weight_header * sim_header

    # 方法3: 最大值
    max_similarity = max(sim_text, sim_header)

    # 方法4: 动态权重（根据查询类型调整）
    if is_specific_query(query):
        # 具体查询更依赖内容
        avg_similarity = 0.8 * sim_text + 0.2 * sim_header
    else:
        # 概念性查询更依赖头部
        avg_similarity = 0.4 * sim_text + 0.6 * sim_header

def is_specific_query(query):
    pass

""" 
效果对比 传统RAG 和 头部增强RAG 

检索准确度提升: 10-15%
回答相关性提升: 15-20%
计算成本增加: 50-70% (需要生成头部和额外嵌入)
"""

def prompt_update():
    # 基础版提示词
    system_prompt = "为给定的文本生成简洁且信息丰富的标题。"

    # 优化版提示词
    system_prompt = """
    你是一个专业的文档整理专家。为给定的文本段落生成一个准确、简洁的标题。
    要求：
    1. 标题应准确反映文本的主要内容
    2. 使用3-8个词
    3. 包含关键概念和术语
    4. 避免过于宽泛或过于具体
    """

    # 领域特定提示词
    system_prompt = """
    为这段关于人工智能的文本生成标题。
    重点关注：技术概念、算法名称、应用领域。
    格式：[主题] - [具体内容]
    """
    pass

def generate_headers_batch(chunks, batch_size=5):
    """
    批量生成头部以提高效率
    """
    headers = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        # 创建批量提示
        batch_prompt = "为以下文本段落分别生成标题：\n\n"
        for j, chunk in enumerate(batch):
            batch_prompt += f"段落{j+1}:\n{chunk}\n\n"

        batch_prompt += "请按顺序返回标题，每行一个："

        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            temperature=0,
            messages=[
                {"role": "system", "content": "你是文档标题生成专家。"},
                {"role": "user", "content": batch_prompt}
            ]
        )

        # 解析批量结果
        batch_headers = response.choices[0].message.content.strip().split('\n')
        headers.extend(batch_headers[:len(batch)])

    return headers

def adaptive_similarity(query, chunk, query_type="general"):
    query_embedding = create_embeddings(query)
    """
    根据查询类型动态调整头部和内容的权重
    """
    sim_text = cosine_similarity(query_embedding, chunk["embedding"])
    sim_header = cosine_similarity(query_embedding, chunk["header_embedding"])

    if query_type == "conceptual":
        # 概念性查询更依赖头部
        return 0.3 * sim_text + 0.7 * sim_header
    elif query_type == "specific":
        # 具体查询更依赖内容
        return 0.8 * sim_text + 0.2 * sim_header
    else:
        # 通用查询平均权重
        return 0.5 * sim_text + 0.5 * sim_header


if __name__ == "__main__":
    # 1. 文档处理
    pdf_path = "../data/AI_Information.pdf"
    extracted_text = extract_text_from_pdf(pdf_path)

    # 2.分块并生成头部
    print("正在分块并生成头部。。。")
    text_chunks = chunk_text_with_headers(extracted_text, 1000, 200)

    # 3.显示示例
    print("示例块")
    print("头部:", text_chunks[0]['header'])
    print("内容:", text_chunks[0]['text'][:200] + "...")

    # 4.为每个块生成嵌入
    embeddings = []

    print("正在生成嵌入。。。")
    for chunk in tqdm(text_chunks, desc="处理块"):
        # 为文本内容创建嵌入
        text_embedding = create_embeddings(chunk["text"])
        # 为头部创建嵌入
        header_embedding = create_embeddings(chunk["header"])

        # 保存所有信息
        embeddings.append({
            "header": chunk["header"],
            "text": chunk["text"],
            "embedding": text_embedding,
            "header_embedding": header_embedding
        })
    """
     - text_chunks - 要遍历的数据
    - desc="处理块" - 进度条前显示的描述文字

  常用参数
  ┌───────┬──────────────────────────────────────┐
  │ 参数  │                 作用                 │
  ├───────┼──────────────────────────────────────┤
  │ desc  │ 进度条前缀描述                       │
  ├───────┼──────────────────────────────────────┤
  │ total │ 总迭代次数（有时无法自动推断时需要） │
  ├───────┼──────────────────────────────────────┤
  │ unit  │ 单位名称，默认 "it"                  │
  ├───────┼──────────────────────────────────────┤
  │ leave │ 完成后是否保留进度条                 │
  ├───────┼──────────────────────────────────────┤
  │ ncols │ 进度条宽度                           │
  └───────┴──────────────────────────────────────┘
  你的代码会显示

  处理块: 45%|████████████▌         | 45/100 [00:03<00:04, 12.34it/s]
    """

    # 5. 加载测试查询
    # with open('data/val.json') as f:
    #     data = json.load(f)
    #
    # query = data[0]['question']
    query = "traditional programming versus machine learning"
    print(f"查询: {query}")

    # 6. 执行语义搜索
    top_chunks = semantic_search(query, embeddings, top_k=2)

    # 7. 显示结果
    print("搜索结果：")
    for i, chunk in enumerate(top_chunks):
        print(f"\n结果 {i + 1}")
        print(f"头部：{chunk['header']}")
        print(f"内容：{chunk['text'][:300]}")
        print("-" * 50)

    # 8. 生成最终回答
    system_prompt = "你是一个AI助手，严格基于给定的上下文回答问题。如果无法从提供的上下文中得出答案，请回答：'我没有足够的信息来回答这个问题。'"

    # 组合上下文
    context = "\n\n".join([
        f"头部：{chunk['header']} \n内容：{chunk['text']}"
        for chunk in top_chunks
    ])

    user_message = f"上下文：\n{context}\n\n问题：{query}"
    response = generate_response(system_prompt, user_message)
    print(f"\nAI回答：{response}")