import textwrap

import fitz


class SemanticChunker:
    def __init__(self, model_name="BAAI/bge-base-en-v1.5", threshold=0.5):
        self.model_name = model_name
        self.threshold = threshold

    def extract_text_from_pdf(self, pdf_path):
        mypdf = fitz.open(pdf_path)
        all_text = ""
        for page_num in range(mypdf.page_count):
            mypage = mypdf[page_num]
            all_text += mypage.get_text("text") + " "
        return all_text.strip()

    def split_into_sentence(self, text):
        result = []
        text.split(".")

    def chunk_text(self, text):
        # 1. 分割句子
        sentence = self.split_into_sentence(text)
        # 2. 创建句子嵌入（向量化）

        # 3. 计算相邻句子的相似度

        # 4. 确定边界

        # 5. 创建分块

    def process_document(self, pdf_path):
        """ 处理整个文档 """
        # 提取文本
        text = self.extract_text_from_pdf(pdf_path)

        # 创建语义块
        chunks = self.chunk_text(text)
