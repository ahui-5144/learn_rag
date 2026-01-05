import os

from dotenv import load_dotenv
from llama_index.llms.openai_like import OpenAILike
from openai import OpenAI

from chunk_text import chunk_text
from create_embeddings import create_embeddings
from extract_text_from_pdf import extract_text_from_pdf
from semantic_search import semantic_search

load_dotenv()

client = OpenAI(
    base_url="https://open.bigmodel.cn/api/paas/v4/",  # 智谱官方 OpenAI 兼容地址
    api_key=os.getenv("ZHIPU_API_KEY")  # 必须显式传入，不能省略
)

# 正确初始化：显式传入智谱 API key，不依赖 OPENAI_API_KEY 环境变量

def generate_response(system_prompt: str, user_message: str, model: str = "glm-4.7") -> str:
    """
    基于检索到的上下文回答问题
    Args:
        system_prompt: 系统提示词
        user_message: 用户消息 = 上下文 + 用户问题
        model:

    Returns:
    """

    response = client.chat.completions.create(
        model=model,  # "glm-4.7"（推荐）、"glm-4-plus" 或 "glm-4"
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return response.choices[0].message.content

# 系统提示词（保持不变）
system_prompt = """你是一个AI助手，严格基于给定的上下文回答问题。
如果答案无法从提供的上下文中直接得出，请回答："我没有足够的信息来回答这个问题。" """


# 示例搜索
query = "What is artificial intelligence?"

pdf_path = "data/AI_Information.pdf"
extract_text = extract_text_from_pdf(pdf_path)
text_chunks = chunk_text(extract_text, 1000, 200)
# 创建所有文本块的嵌入
chunks_embeddings = create_embeddings(text_chunks)

top_chunks = semantic_search(query, text_chunks, chunks_embeddings,  k=3)

# 构建用户提示（假设 top_chunks 和 query 已定义）
context = "\n".join([f"上下文 {i+1}:\n{chunk}" for i, chunk in enumerate(top_chunks)])
user_prompt = f"{context}\n\n问题: {query}"

# 生成回答
ai_response = generate_response(system_prompt, user_prompt)
print(f"AI回答: {ai_response}")
"""console
AI回答: 根据提供的上下文，人工智能（AI）能够分析海量数据、解决复杂问题、做出决策并执行创造性任务。
它由三个主要组件驱动：算法、数据和算力，这些解释了机器如何能够表 现出智能行为。
此外，AI还涉及机器学习（或深度学习）以及训练技术（如监督、无监督或强化学习）。
"""