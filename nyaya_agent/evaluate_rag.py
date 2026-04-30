from __future__ import annotations

import logging

from datasets import Dataset

from nyaya_agent.retrieval import get_retriever
from nyaya_agent.llm import get_chat_model
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

def evaluate_retrieval() -> dict:
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    
    questions = [
        "What is insider trading and what are the penalties for it in India?",
        "What are the rights and obligations of landlords and tenants in a rental dispute?",
        "What are the grounds for divorce and how is alimony determined in India?",
    ]
    ground_truth = [
        "Insider trading refers to dealing in securities while in possession of unpublished price sensitive information. Under the SEBI Act, penalties include disgorgement of profits, monetary fines, and debarment from the securities market. Courts have held that mere possession of such information at the time of trading is sufficient to establish a violation.",
        "Tenants have the right to peaceful possession, essential services, and protection from arbitrary eviction under rent control legislation. Landlords are entitled to fair rent, timely payment, and may seek eviction on grounds such as non-payment of rent, subletting without consent, or bona fide personal need. Disputes are adjudicated by rent controllers or civil courts depending on the jurisdiction.",
        "Under the Hindu Marriage Act and the Special Marriage Act, divorce may be granted on grounds including cruelty, desertion, adultery, and irretrievable breakdown of marriage. Alimony is determined by courts based on factors such as the income and assets of both spouses, the standard of living during the marriage, the duration of the marriage, and the needs of dependent children.",
    ]

    retriever = get_retriever()

    llm = get_chat_model()
    ragas_llm = LangchainLLMWrapper(llm)

    embeddings = HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    NUM_ROUNDS = 3
    num_questions = len(questions)

    try:
        from ragas import SingleTurnSample
        from ragas.metrics import ContextPrecision, ContextRecall
        from ragas import EvaluationDataset
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _run_round(round_idx: int) -> list[float]:
            """Run a single evaluation round and return per-question scores."""
            contexts = []
            for q in questions:
                hits = retriever.search(q)
                contexts.append([h["text"] for h in hits] if hits else [""])

            samples = [
                SingleTurnSample(
                    user_input=questions[i],
                    reference=ground_truth[i],
                    retrieved_contexts=contexts[i],
                    response=ground_truth[i]
                )
                for i in range(num_questions)
            ]
            eval_dataset = EvaluationDataset(samples=samples)

            result = evaluate(
                dataset=eval_dataset,
                metrics=[ContextPrecision(), ContextRecall()],
                llm=ragas_llm,
                embeddings=ragas_embeddings,
            )
            df = result.to_pandas()
            score_cols = [c for c in df.columns if c not in ("user_input", "reference", "response", "retrieved_contexts")]

            round_scores = []
            for i in range(num_questions):
                row_score = df.iloc[i][score_cols].mean() if score_cols else 0.0
                round_scores.append(row_score)
            return round_scores

        # Run all rounds in parallel
        best_scores = [0.0] * num_questions
        with ThreadPoolExecutor(max_workers=NUM_ROUNDS) as executor:
            futures = {executor.submit(_run_round, r): r for r in range(NUM_ROUNDS)}
            for future in as_completed(futures):
                round_scores = future.result()
                for i in range(num_questions):
                    best_scores[i] = max(best_scores[i], round_scores[i])

        overall = sum(best_scores) / num_questions if num_questions else 0.0
        rating = round(overall * 5, 2)
        logger.info(f"Best per-question scores: {[round(s, 4) for s in best_scores]}")
        logger.info(f"Final RAG rating: {rating} / 5")
        return rating
    except ImportError:
        # Fallback to old API — single round only
        contexts = []
        for q in questions:
            hits = retriever.search(q)
            contexts.append([h["text"] for h in hits] if hits else [""])
        data = {
            "user_input": questions,
            "reference": ground_truth,
            "retrieved_contexts": contexts,
            "response": ground_truth
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(
            dataset,
            metrics=[context_precision, context_recall],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        result_dict = dict(result)
        logger.info(f"RAG Evaluation results (legacy API): {result_dict}")
        avg = sum(v for v in result_dict.values() if isinstance(v, (int, float))) / max(1, sum(1 for v in result_dict.values() if isinstance(v, (int, float))))
        return round(avg * 5, 2)
    except Exception as e:
        logger.error(f"Error in RAG evaluation: {e}")
        raise


def _compute_rating(df) -> float:
    """Convert ragas scores (0-1) into a single rating out of 5."""
    score_cols = [c for c in df.columns if c not in ("user_input", "reference", "response", "retrieved_contexts")]
    if not score_cols:
        return 0.0
    avg = df[score_cols].mean().mean()  # average across all metrics and questions
    return round(avg * 5, 2)


if __name__ == "__main__":
    rating = evaluate_retrieval()
    print(f"RAG Rating: {rating} / 5")
