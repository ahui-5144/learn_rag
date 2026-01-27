# 必需的安装（在终端或 notebook 中执行一次）
# pip install llama-index llama-index-llms-openai llama-index-embeddings-openai \
#     llama-index-retrievers-bm25 \
#     llama-index-postprocessor-sentence-transformer-rerank \
#     sentence-transformers jieba python-dotenv

import os

# Windows 上禁用 torch dynamo（避免 resource module 错误）  报错模型不存在，后面自己下载模型试试
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

from dotenv import load_dotenv

# 加载环境变量（.env 文件中配置 ZHIPU_API_KEY 和 ZHIPU_URL）
load_dotenv()

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
from llama_index.core.postprocessor import SentenceTransformerRerank

# 使用智谱 API（兼容 OpenAI 格式）
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.llm = OpenAI(
    model="gpt-4o-mini",            # 使用 OpenAI 兼容模型名，智谱 API 会映射
    temperature=0.0,
    api_key=os.getenv("ZHIPU_API_KEY"),
    api_base=os.getenv("ZHIPU_URL")  # https://open.bigmodel.cn/api/paas/v4
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-en-v1.5",
    device="cpu",  # Windows 上显式指定使用 CPU
    embed_batch_size=10
)

# -----------------------------
# 1. 加载文档并切分节点（nodes）
# -----------------------------
# 替换为你的数据目录，例如 "./data/" 下放 PDF / TXT / MD 文件
#  input_dir  指定目录（加载目录下所有文件） input_dir="./data"
# 或者使用 input_files 指定单个文件
documents = SimpleDirectoryReader(input_files=["../basic_rag/data/AI_Information.pdf"]).load_data()

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
# 可选：中文分词器（取消注释以启用）
# import jieba
# def chinese_tokenizer(text):
#     return list(jieba.cut(text))

bm25_retriever = BM25Retriever.from_defaults(
    docstore=index.docstore,      # 复用 vector index 的 docstore
    similarity_top_k=40,
    # tokenize=chinese_tokenizer,  # 中文文档建议启用
)

# -----------------------------
# 4. 融合 retriever（hybrid retrieval with RRF）
# -----------------------------
fusion_retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=20,           # 融合后最终召回数
    num_queries=1,                 # 若设 >1 会自动生成多查询扩展
    use_async=True,                # 异步并行召回
    verbose=True,
)

# -----------------------------
# 5. 添加 reranker（精排）- 开源方案
# -----------------------------
reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-v2-m3",   # 支持中英文混合
    top_n=8,                            # 最终输出给 LLM 的文档数
    device="cpu",                       # 如果有 GPU 改为 "cuda"
)

# -----------------------------
# 6. 构建最终 Query Engine
# -----------------------------
query_engine = RetrieverQueryEngine(
    retriever=fusion_retriever,
    node_postprocessors=[reranker],
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
