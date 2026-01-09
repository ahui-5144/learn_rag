import textwrap

import fitz
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class SemanticChunker:
    def __init__(self, model_name="BAAI/bge-base-en-v1.5", similarity_threshold=0.5):
        self.embedding_model = HuggingFaceEmbedding(model_name=model_name)
        self.similarity_threshold = similarity_threshold

    def extract_text_from_pdf(self, pdf_path):
        mypdf = fitz.open(pdf_path)
        all_text = ""
        for page_num in range(mypdf.page_count):
            mypage = mypdf[page_num]
            all_text += mypage.get_text("text") + " "
        return all_text.strip()

    def split_into_sentence(self, text):
        """分割句子"""
        # result = []
        # sentence = text.split(".")
        # for s in sentence:
        #     if s.strip():
        #         result.append(s + ".")
        # return result
        sentences = text.split(".")
        sentences = [s.strip() + "." for s in sentences if s.strip()]
        return sentences

    def cosine_similarity(self, vec1, vec2):
        """ 计算余弦相似度 """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def calculate_similarities(self, embeddings):
        """计算相邻语句的相似度"""
        similarities = []
        for i in range(len(embeddings) - 1):
            similarity = self.cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(similarity)
        return similarities

    def find_boundaries(self, similarities):
        boundaries = [0]
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                boundaries.append(i + 1)
        boundaries.append(len(similarities) + 1)
        return boundaries

    def chunk_text(self, text):
        # 1. 分割句子
        sentence = self.split_into_sentence(text)
        # 2. 创建句子嵌入（向量化）
        embeddings = self.embedding_model.get_text_embedding_batch(sentence)
        embeddings = np.array(embeddings)

        # 3. 计算相邻句子的相似度
        similarities = self.calculate_similarities(embeddings)

        # 4. 确定边界
        boundaries = self.find_boundaries(similarities)
        # 5. 创建分块
        chunks = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            chunk = sentence[start:end]
            chunks.append(chunk)
        return chunks

    def process_document(self, pdf_path):
        """ 处理整个文档 """
        # 提取文本
        text = self.extract_text_from_pdf(pdf_path)

        # 创建语义块
        chunks = self.chunk_text(text)

        print(f"创建了{len(chunks)}个语义分块")

        return chunks

if __name__ == "__main__":
    # 创建语义分块器
    chunker = SemanticChunker()
    # 处理文档
    chunks = chunker.process_document("../data/AI_Information.pdf")
    # 显示前3个分块
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n==语义分块{i + 1}==")
        print(chunk)
        print("-" * 50)