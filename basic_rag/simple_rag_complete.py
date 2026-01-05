import os

import fitz
import numpy as np
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from openai import OpenAI


load_dotenv()

class SimpleRAG:
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.embedding_model = HuggingFaceEmbedding(model_name = model_name)
        self.client = OpenAI(
            base_url="https://open.bigmodel.cn/api/paas/v4/",  # 智谱官方 OpenAI 兼容地址
            api_key=os.getenv("ZHIPU_API_KEY")  # 必须显式传入，不能省略
        )
        self.text_chunks = []
        self.embeddings = []

    def load_document(self, pdf_path):
        """ 加载并处理PDF文档 """
        # 提取文档
        text = self.extract_text_from_pdf(pdf_path)

        # 分块
        self.text_chunks = self.chunk_text(text, 1000, 200)

        # 创建嵌入
        self.embeddings = self.create_embeddings(self.text_chunks)

        print(f"已加载文档，创建了 {len(self.text_chunks)} 个文本块")



    def extract_text_from_pdf(self, pdf_path):
        """
        从 PDF 文件中提取文本内容
        Args:
            pdf_path: 文件路径

        Returns:

        """
        mypdf = fitz.open(pdf_path)
        all_text = ""
        for page_num in range(mypdf.page_count):
            page = mypdf[page_num]
            all_text += page.get_text("text")

        return all_text

    def chunk_text(self, text, n, overlap):
        """
        将文本分割成固定大小的块，支持重叠
        Args:
            text:  文档
            n:  块大小
            overlap: 重叠大小

        Returns:

        """
        chunks = []

        for i in range(0, len(text), n - overlap):
            chunks.append(text[i:i + n])
        return chunks

    def create_embeddings(self, chunks):
        """
        为文本创建向量嵌入
        Args:
            chunks:

        Returns:

        """
        return self.embedding_model.get_text_embedding_batch(chunks)

    def cosine_similarity(self, vec1, vec2):
        """
        计算两个向量的余弦相似度
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def search(self, query, k=3):
        """
        搜索相关文档
        Args:
            query: 查询预计
            k: 返回最相关的条数

        Returns:

        """
        query_embedding = self.embedding_model.get_text_embedding(query)
        similarities = []

        for index, chunk_embedding in enumerate(self.embeddings):
            similarity = self.cosine_similarity(
                np.array(query_embedding),
                np.array(chunk_embedding)
            )
            similarities.append([index, similarity])

        similarities.sort(key=lambda x: x[1], reverse=True)

        # top_indices = [i for i,_ in similarities[:k]]
        top_k_pairs = similarities[:k]
        top_indices = []
        # 步骤3：遍历前 k 个元组，只取出第一个元素（索引），忽略第二个元素（相似度分数）
        for i, _ in top_k_pairs:  # _ 是惯用写法，表示“这个值我不需要”
            top_indices.append(i)

        return [self.text_chunks[i] for i in top_indices]

    def answer(self, query):
        """回答问题"""
        # 检索相关文档
        relevant_chunks = self.search(query)

        # 构建提示
        context = "\n".join([f"上下文 {i + 1}: \n{chunk}"
                             for i, chunk in enumerate(relevant_chunks)])

        system_prompt = """你是一个AI助手，严格基于给定的上下文回答问题。
如果答案无法从提供的上下文中直接得出，请回答："我没有足够的信息来回答这个问题。" """

        user_prompt = f"{context}\n\n问题：{query}"

        # 生成回答
        response = self.client.chat.completions.create(
            model="glm-4.7",
            temperature=0,
            messages=[
                {"role":"system", "content": system_prompt},
                {"role":"user", "content": user_prompt},
            ]
        )

        return response.choices[0].message.content

# 使用示例
if __name__ == "__main__":
    # 创建实例
    rag = SimpleRAG()

    # 加载文档
    rag.load_document("data/AI_Information.pdf")

    # 提问
    question = "What is artificial intelligence?"

    answer = rag.answer(question)

    print(f"提问:{question}")
    print(f"回答:{answer}")

"""console
提问:What is artificial intelligence?
回答:根据提供的上下文，人工智能（AI）被描述为能够分析海量数据、解决复杂问题、做出决策并执行创造性任务的技术。它依赖于高级算法、数据和计算能力这三个组件，使机器能够表现出智能行为。
"""
