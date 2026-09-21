"""
Runs the full RAG pipeline against app/evaluation/eval_set.json
and scores it with RAGAS.

Run with: python -m app.evaluation.run_eval
"""
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from app.retrieval.hybrid import get_hybrid_retriever
from app.retrieval.reranker import rerank
from app.generation.answer_chain import generate_answer, get_llm
from app.ingestion.embed_and_store import get_embeddings

EVAL_SET_PATH = "app/evaluation/eval_set.json"


def load_eval_set():
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline_for_question(retriever, question: str):
    """
    Runs your actual production pipeline: hybrid retrieve -> rerank -> generate.
    Returns the answer text and the list of context strings used.
    """
    candidates = retriever.invoke(question)
    chunks = rerank(question, candidates)
    answer = generate_answer(question, chunks)
    contexts = [doc.page_content for doc in chunks]
    return answer, contexts


def build_eval_dataset():
    eval_items = load_eval_set()
    retriever = get_hybrid_retriever()

    questions, answers, contexts_list, ground_truths = [], [], [], []

    for item in eval_items:
        question = item["question"]
        print(f"Running: {question}")
        answer, contexts = run_pipeline_for_question(retriever, question)

        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(item["ground_truth"])

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })


def run_evaluation():
    dataset = build_eval_dataset()

    
    judge_llm = LangchainLLMWrapper(get_llm())
    judge_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    print("\n=== RAGAS Evaluation Results ===")
    print(scores)
    return scores


if __name__ == "__main__":
    scores = run_evaluation()
    df = scores.to_pandas()

    print("\n=== Available columns ===")
    print(df.columns.tolist())

    print("\n=== Per-question breakdown ===")
    print(df.to_string())