import fitz
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def calculate_dynamic_threshold(similarities, percentile=25):
    """
    基于相似度分布计算动态阈值

    Args:
        similarities: 相似度列表
        percentile: 百分位数（较低的百分位对应更严格的分块）

    Returns:

    """
    threshold = np.percentile(similarities, percentile)
    return threshold

def extract_text_from_pdf(pdf_path):
    mypdf = fitz.open(pdf_path)
    all_text = ""
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]
        all_text += page.get_text("text") + " "
    return all_text.strip()

def split_into_sentence(text):
    sentences = text.split(".")
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def create_sentence_embeddings(sentences, model="BAAI/bge-base-en-v1.5"):
    embedding_model = HuggingFaceEmbedding(model_name=model)
    embeddings = embedding_model.get_text_embedding_batch(sentences)
    return np.array(embeddings)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def calculate_similarities(embeddings):
    similarities = []
    for i in range(len(embeddings) - 1):
        similarity = cosine_similarity(embeddings[i], embeddings[i+1])
        similarities.append(similarity)
    return similarities

text = extract_text_from_pdf("../data/AI_Information.pdf")
sentences = split_into_sentence(text)
embeddings = create_sentence_embeddings(sentences)
similarities = calculate_similarities(embeddings)

dynamic_threshold = calculate_dynamic_threshold(similarities, percentile=30)
print(f"动态阈值:{dynamic_threshold}")