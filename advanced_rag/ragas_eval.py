# RAGAS 评估脚本
# 用于评估 b_rerank.py 中的不同重排序方法

# 安装依赖
# pip install ragas datasets pandas

import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

# 加载环境变量
load_dotenv()

# 导入 RAG 系统组件
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from b_rerank import (
    extract_text_from_pdf,
    chunk_text,
    create_embeddings,
    SimpleVectorStore,
    rag_with_reranking,
)

# ==================== 配置区 ====================

# PDF 文档路径
PDF_PATH = "../basic_rag/data/AI_Information.pdf"

# 评估数据集文件
EVAL_DATASET_FILE = "eval_dataset.json"

# 要评估的方法列表
METHODS_TO_EVALUATE = ["none", "llm", "keywords"]

# RAGAS 使用的 LLM（用于评估，默认用 OpenAI，也可配置其他）
# RAGAS 会自动从环境变量读取 OPENAI_API_KEY
# 如果使用智谱，需要在下面单独配置


# ==================== 评估数据集 ====================

# 示例评估数据集（如果使用文件方式，可以删除这部分）
EVAL_DATASET = [
    {
        "question": "What is the difference between traditional programming and machine learning?",
        "ground_truth": "Traditional programming relies on explicit instructions written by programmers to perform tasks, while machine learning learns patterns from data to make decisions without being explicitly programmed for specific rules. In traditional programming, rules are coded by humans; in machine learning, rules are learned from data."
    },
    {
        "question": "What are the main components of an expert system?",
        "ground_truth": "Expert systems consist of two main components: a knowledge base, which stores facts and rules about a specific domain, and an inference engine, which applies logical rules to the knowledge base to derive new conclusions and make decisions."
    },
    {
        "question": "How does machine learning differ from deep learning?",
        "ground_truth": "Deep learning is a specialized subset of machine learning that uses multi-layered neural networks to learn from data. While traditional machine learning often requires manual feature engineering, deep learning can automatically learn features from raw data."
    },
    {
        "question": "What is the role of training data in machine learning?",
        "ground_truth": "Training data in machine learning serves as the example set from which the algorithm learns patterns, relationships, and rules. The quality and quantity of training data directly affect the model's performance and ability to generalize to new, unseen data."
    },
    {
        "question": "什么是机器学习？",
        "ground_truth": "机器学习是人工智能的一个分支，它使用算法从数据中学习模式，并利用这些模式做出预测或决策，而无需为特定规则进行显式编程。"
    }
]


# ==================== 工具函数 ====================

def load_eval_dataset(file_path=None):
    """
    加载评估数据集

    Args:
        file_path: JSON 文件路径，如果为 None 则使用内置示例数据

    Returns:
        list: 评估数据列表
    """
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print(f"使用内置评估数据集（{len(EVAL_DATASET)} 条）")
        return EVAL_DATASET


def init_rag_system(pdf_path):
    """
    初始化 RAG 系统

    Args:
        pdf_path: PDF 文档路径

    Returns:
        SimpleVectorStore: 初始化好的向量存储
    """
    print(f"\n正在加载文档: {pdf_path}")

    # 提取文本
    text = extract_text_from_pdf(pdf_path)

    # 分块
    chunks = chunk_text(text, 1000, 200)
    print(f"创建了 {len(chunks)} 个分块")

    # 创建向量存储
    embeddings = create_embeddings(chunks)
    vector_store = SimpleVectorStore()

    for chunk, embedding in zip(chunks, embeddings):
        vector_store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={"source": pdf_path}
        )

    print(f"向量存储已建立，包含 {len(chunks)} 个块")

    return vector_store


def evaluate_single_method(method_name, eval_dataset, vector_store, model="GLM-4.5"):
    """
    评估单个重排序方法

    Args:
        method_name: 重排序方法名称 ("none", "llm", "keywords")
        eval_dataset: 评估数据集
        vector_store: 向量存储
        model: LLM 模型名称

    Returns:
        dict: 包含问题和结果的数据
    """
    print(f"\n{'='*60}")
    print(f"正在评估方法: {method_name}")
    print('='*60)

    results = {
        "question": [],
        "ground_truth": [],
        "answer": [],
        "contexts": [],
    }

    for i, item in enumerate(eval_dataset, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\n[{i}/{len(eval_dataset)}] 问题: {question[:50]}...")

        try:
            # 调用 RAG 系统
            response = rag_with_reranking(
                query=question,
                vector_store=vector_store,
                reranking_method=method_name,
                top_n=3,
                model=model
            )

            # 提取上下文文本
            contexts = [r["text"] for r in response["reranked_results"]]

            # 保存结果
            results["question"].append(question)
            results["ground_truth"].append(ground_truth)
            results["answer"].append(response["response"])
            results["contexts"].append(contexts)

            print(f"  ✓ 答案生成完成 (检索到 {len(contexts)} 个上下文)")

        except Exception as e:
            print(f"  ✗ 处理失败: {e}")
            # 失败时添加空结果
            results["question"].append(question)
            results["ground_truth"].append(ground_truth)
            results["answer"].append(f"Error: {str(e)}")
            results["contexts"].append([])

    return results


def run_ragas_evaluation(results_dict):
    """
    使用 RAGAS 运行评估

    Args:
        results_dict: 包含 question, ground_truth, answer, contexts 的字典

    Returns:
        Dataset: RAGAS 评估结果
    """
    print("\n正在运行 RAGAS 评估...")

    # 转换为 RAGAS Dataset 格式
    dataset = Dataset.from_dict(results_dict)

    # 运行评估
    score = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
    )

    return score


def save_results(all_scores, all_results, output_file="ragas_evaluation_results.json"):
    """
    保存评估结果到 JSON 文件

    Args:
        all_scores: 各方法的 RAGAS 评分
        all_results: 各方法的详细结果
        output_file: 输出文件路径
    """
    output_data = {
        "summary": {},
        "details": {}
    }

    # 保存评分摘要
    for method, score in all_scores.items():
        score_df = score.to_pandas()
        output_data["summary"][method] = {
            "faithfulness": float(score_df["faithfulness"].mean()),
            "answer_relevancy": float(score_df["answer_relevancy"].mean()),
            "context_precision": float(score_df["context_precision"].mean()),
            "context_recall": float(score_df["context_recall"].mean()),
        }

    # 保存详细结果
    for method, results in all_results.items():
        output_data["details"][method] = results

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_file}")


def print_comparison_table(all_scores):
    """
    打印方法对比表格

    Args:
        all_scores: 各方法的 RAGAS 评分字典
    """
    print(f"\n{'='*80}")
    print(" " * 25 + "方法对比总结")
    print('='*80)

    comparison_data = []

    for method, score in all_scores.items():
        score_df = score.to_pandas()

        row = {
            "Method": method.upper(),
            "Faithfulness": f"{score_df['faithfulness'].mean():.4f}",
            "Answer Relevancy": f"{score_df['answer_relevancy'].mean():.4f}",
            "Context Precision": f"{score_df['context_precision'].mean():.4f}",
            "Context Recall": f"{score_df['context_recall'].mean():.4f}",
        }
        comparison_data.append(row)

    # 打印表格
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))

    print('\n' + '='*80)
    print("指标说明:")
    print("  • Faithfulness (忠实度): 答案是否忠实于检索到的上下文")
    print("  • Answer Relevancy (答案相关性): 答案与问题的相关程度")
    print("  • Context Precision (上下文精确度): 检索到的上下文相关性")
    print("  • Context Recall (上下文召回率): 检索是否覆盖标准答案所需信息")
    print('='*80)


def print_best_method(all_scores):
    """
    打印最佳方法和推荐

    Args:
        all_scores: 各方法的 RAGAS 评分字典
    """
    # 计算每个方法的平均分
    avg_scores = {}
    for method, score in all_scores.items():
        score_df = score.to_pandas()
        avg_scores[method] = score_df[["faithfulness", "answer_relevancy",
                                       "context_precision", "context_recall"]].mean().mean()

    # 找出最佳方法
    best_method = max(avg_scores, key=avg_scores.get)

    print(f"\n{'='*80}")
    print(f"🏆 最佳方法: {best_method.upper()}")
    print(f"   综合得分: {avg_scores[best_method]:.4f}")
    print('='*80)

    # 打印推荐建议
    print("\n💡 使用建议:")

    if best_method == "llm":
        print("   • LLM 重排序效果最好，适合高价值场景")
        print("   • 缺点是需要额外的 API 调用，成本较高")
        print("   • 对于简单查询，可以考虑使用 keywords 方法以降低成本")

    elif best_method == "keywords":
        print("   • 关键词重排序性价比高，无需额外 API 调用")
        print("   • 适合关键词明确、术语规范的场景")
        print("   • 对于语义复杂的问题，可能需要结合 LLM 重排序")

    elif best_method == "none":
        print("   • 原始向量检索效果已经很好")
        print("   • 重排序可能在此场景下提升有限")
        print("   • 建议检查评估数据集是否过于简单")


# ==================== 主函数 ====================

def main():
    """主函数：执行完整的评估流程"""

    print("\n" + "="*80)
    print(" " * 25 + "RAGAS 评估脚本")
    print("="*80)

    # 1. 加载评估数据集
    eval_dataset = load_eval_dataset(EVAL_DATASET_FILE)
    print(f"\n加载了 {len(eval_dataset)} 条评估数据")

    # 2. 初始化 RAG 系统
    vector_store = init_rag_system(PDF_PATH)

    # 3. 评估各个方法
    all_scores = {}
    all_results = {}

    for method in METHODS_TO_EVALUATE:
        # 评估单个方法
        results = evaluate_single_method(method, eval_dataset, vector_store)
        all_results[method] = results

        # 运行 RAGAS 评估
        score = run_ragas_evaluation(results)
        all_scores[method] = score

        # 打印单个方法的评分
        score_df = score.to_pandas()
        print(f"\n{method.upper()} 方法评分:")
        print(score_df.to_string(index=False))

    # 4. 打印对比表格
    print_comparison_table(all_scores)

    # 5. 打印最佳方法推荐
    print_best_method(all_scores)

    # 6. 保存结果
    save_results(all_scores, all_results)

    print("\n✓ 评估完成!")


if __name__ == "__main__":
    main()
