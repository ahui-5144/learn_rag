import os

from dotenv import load_dotenv
from llama_index.core.base.llms.types import ChatMessage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.openai_like import OpenAILike  # 专用兼容类

load_dotenv()

# 测试嵌入模型
print("Testing embedding model....")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
test_embedding = embed_model.get_text_embedding("test")
print(f"✅ Embedding model working! Vector dimension: {len(test_embedding)}")

# # 测试LLM (如果配置了API密钥)
# if os.getenv("GOOGLE_API_KEY"):
#     print("Testing Google Gemini...")
#     llm = GoogleGenAI(model="gemini-2.5-flash")
#     response = llm.complete("Hello, how are you?")
#     print(f"✅ Google Gemini working! Response: {response}")
# else:
#     print("⚠️  Google API key not found, skipping LLM test")
#
# print("🎉 Environment setup complete!")

api_key=os.getenv("ZHIPU_API_KEY")
zhipu_url=os.getenv("ZHIPU_URL")

if api_key:
    print("Testing ZHIPU GLM ...")
    llm = OpenAILike(
        model="glm-4.7",  # 或 "glm-4-plus"、"glm-4.7" 等（视您的账号支持而定）
        api_key=api_key,
        api_base=zhipu_url,
        is_chat_model=True
    )
    # response = llm.complete("Hello, how are you?")
    # 推荐使用 chat 方法（确保调用 /chat/completions 端点，避免 completions 404）
    messages = [ChatMessage(role="user", content="Hello, how are you?")]
    response = llm.chat(messages)
    print(f"✅ ZhiPu working! Response: {response}")
else:
    print("⚠️  ZHIPU key not found, skipping LLM test")

print("🎉 Environment setup complete!")