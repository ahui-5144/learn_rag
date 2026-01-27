# rag_eval_experiment.py
# 使用 Ragas 0.4.3 的 evaluate() 接口，避免 @experiment() 的 backend 问题
import asyncio
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import DiscreteMetric
from langchain_openai import ChatOpenAI
from b_rerank import (
    extract_text_from_pdf,
    chunk_text,
    create_embeddings,
    SimpleVectorStore,
    rag_with_reranking
)

# Ragas 评估用 LLM
evaluator_llm = ChatOpenAI(
    model="glm-4",  # 智谱模型名
    openai_api_key=os.getenv('ZHIPU_API_KEY'),
    openai_api_base=os.getenv('ZHIPU_URL'),
    temperature=0.0,
    max_tokens=512,
)


# 构建向量存储（全局）
def build_vector_store(pdf_path="../basic_rag/data/AI_Information.pdf"):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text, n=800, overlap=150)
    embeddings = create_embeddings(chunks)
    vector_store = SimpleVectorStore()
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vector_store.add_item(chunk, emb, {"index": i, "source": pdf_path})
    return vector_store


vector_store = build_vector_store()

# 自定义指标
correctness_metric = DiscreteMetric(
    name="correctness",
    prompt="""
你是一個嚴格的評估專家。

請判斷以下生成的回答是否完整包含了「评分指引笔记」中列出的所有關鍵點。
- 如果**全部關鍵點**都被涵蓋 → 只回傳 'pass'
- 如果有任何關鍵點遺漏 → 只回傳 'fail'

絕對不要輸出任何其他文字、解釋、理由、標點、空格或換行。

回答內容: {response}
评分指引笔记: {grading_notes}
    """.strip(),
    allowed_values=["pass", "fail"]
)


# 加载数据集（使用 HF Dataset，Ragas 0.4.3 完全兼容）
def load_test_dataset(csv_path="dataset/rag_test_dataset.csv"):
    df = pd.read_csv(csv_path)

    samples = []
    for _, row in df.iterrows():
        samples.append({
            "user_input": row["question"],
            "reference": row["grading_notes"],
        })

    dataset = Dataset.from_list(samples)
    print(f"从 {csv_path} 加载了 {len(samples)} 条测试数据")
    return dataset


dataset = load_test_dataset()


# 新增一个异步评估单个样本的函数
async def evaluate_one_sample(question, answer, grading_notes, contexts, method):
    score_result = await correctness_metric.ascore(
        question=question,
        response=answer,
        grading_notes=grading_notes,   # 注意关键字必须和 prompt 里的占位符一致
        contexts=contexts,
        llm=evaluator_llm
    )
    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "score": score_result.value,   # "pass" 或 "fail"
        "reason": score_result.reason,
        "reranking_method": method,
    }


# 异步评估函数（批量处理所有 rerank 方法）
async def run_evaluation_async(methods=["none", "keywords", "llm"]):
    all_results = []

    for method in methods:
        print(f"\n运行评估：rerank 方法 = {method}")

        tasks = []

        for row in dataset:
            question = row["user_input"]
            grading_notes = row["reference"]

            rag_output = rag_with_reranking(
                query=question,
                vector_store=vector_store,
                reranking_method=method,
                top_n=5,
                model="glm-4"
            )

            answer = rag_output["response"]
            contexts = [res['text'] for res in rag_output["reranked_results"]]

            # 收集异步任务
            tasks.append(
                evaluate_one_sample(question, answer, grading_notes, contexts, method)
            )

        # 并发执行当前 method 的所有样本评估
        method_results = await asyncio.gather(*tasks)

        pass_rate = sum(1 for r in method_results if r["score"] == "pass") / len(method_results)
        print(f"{method} 方法完成，平均 pass 率：{pass_rate:.2%}")

        all_results.extend(method_results)

    # 保存所有结果
    pd.DataFrame(all_results).to_csv("./experiments/rag_evaluation_results.csv", index=False)
    print("\n所有评估结果已保存至 ./experiments/rag_evaluation_results.csv")


# 评估函数（批量处理所有 rerank 方法）
def run_evaluation(methods=["none", "keywords", "llm"]):
    asyncio.run(run_evaluation_async(methods))


if __name__ == "__main__":
    run_evaluation()