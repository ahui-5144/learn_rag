# 必需的安装（在终端或 notebook 中执行一次）
# pip install llama-index llama-index-llms-openai llama-index-embeddings-openai \
#     llama-index-retrievers-bm25 llama-index-postprocessor-cohere-rerank \
#     llama-index-postprocessor-sentence-transformer-rerank stemmer

import os
from typing import List

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import CohereRerank  # 闭源，效果通常最佳
# 或者使用开源 reranker（无需 API key）
# from llama_index.core.postprocessor import SentenceTransformerRerank

# 可选：设置全局 LLM 和 embedding 模型（这里用 OpenAI，也可换本地模型）
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# -----------------------------
# 1. 加载文档并切分节点（nodes）
# -----------------------------
# 替换为你的数据目录，例如 "./data/" 下放 PDF / TXT / MD 文件
documents = SimpleDirectoryReader(input_dir="./data").load_data()

# 推荐 chunk_size 512–1024，根据文档特性调整
splitter = SentenceSplitter(chunk_size=768, chunk_overlap=128)
nodes = splitter.get_nodes_from_documents(documents)

# -----------------------------
# 2. 构建向量索引（dense retrieval）
# -----------------------------
storage_context = StorageContext.from_defaults()
index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)

# Vector retriever（稠密检索）
vector_retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=40,          # 粗召回多一些，给 reranker 留空间
)

# -----------------------------
# 3. 构建 BM25 retriever（sparse retrieval）
# -----------------------------
bm25_retriever = BM25Retriever.from_defaults(
    docstore=index.docstore,      # 复用 vector index 的 docstore，避免重复存储
    similarity_top_k=40,
    # 可选：中文场景建议使用 jieba 分词或自定义 tokenizer
    # 这里默认使用简单英文 stemmer；生产中可自定义 language="zh" 或传入 tokenizer
)

# -----------------------------
# 4. 融合 retriever（hybrid retrieval with RRF）
# -----------------------------
fusion_retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=20,           # 融合后最终召回数（reranker 前）
    num_queries=1,                 # 若设 >1 会自动生成多查询扩展，可进一步提升
    use_async=True,                # 异步并行召回，提升速度
    verbose=True,                  # 调试时开启
)

# -----------------------------
# 5. 添加 reranker（精排）
# -----------------------------
# 选项1：Cohere Rerank（推荐，工业效果顶尖，需要 Cohere API key）
cohere_rerank = CohereRerank(
    api_key=os.getenv("COHERE_API_KEY"),  # 替换为你的 key
    top_n=8,                               # 最终输出给 LLM 的 top 文档数
    model="rerank-english-v3.0",           # 或 "rerank-multilingual-v3.0" 用于中英混合
)

# 选项2：开源替代（无需 API key，速度快）
# reranker = SentenceTransformerRerank(
#     model="BAAI/bge-reranker-v2-m3",   # 或 v3 / large 版本
#     top_n=8,
#     device="cuda" if torch.cuda.is_available() else "cpu",
# )

# -----------------------------
# 6. 构建最终 Query Engine
# -----------------------------
query_engine = RetrieverQueryEngine(
    retriever=fusion_retriever,
    node_postprocessors=[cohere_rerank],   # 替换为 reranker 变量即可
)

# -----------------------------
# 7. 测试查询
# -----------------------------
query = "traditional programming versus machine learning"
response = query_engine.query(query)

print("Response:", response.response)
print("\nSource nodes:")
for node in response.source_nodes:
    print(f"Score: {node.score:.4f} | {node.node.get_text()[:200]}...\n")