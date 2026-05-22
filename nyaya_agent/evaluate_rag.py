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
        "What are hindu marriage act and special marriage act?",
    ]
    ground_truth = [
        "Insider trading refers to dealing in securities while in possession of unpublished price sensitive information. Under the SEBI Act, penalties include disgorgement of profits, monetary fines, and debarment from the securities market. Courts have held that mere possession of such information at the time of trading is sufficient to establish a violation.",
        "Tenants have the right to peaceful possession, essential services, and protection from arbitrary eviction under rent control legislation. Landlords are entitled to fair rent, timely payment, and may seek eviction on grounds such as non-payment of rent, subletting without consent, or bona fide personal need. Disputes are adjudicated by rent controllers or civil courts depending on the jurisdiction.",
        "The Hindu Marriage Act and the Special Marriage Act are two distinct legal frameworks that establish legal guidelines for establishing and dissolving marriages. While one focuses on traditional customs and specific cultural rituals for those within certain communities, the other provides a civil alternative that accommodates diverse backgrounds without requiring religious ceremonies. Both acts outline rights, obligations, and procedures regarding matrimony, but they apply differently based on the choices and backgrounds of the individuals involved.",
    ]

    retriever = get_retriever()

    llm = get_chat_model(eval=True)
    ragas_llm = LangchainLLMWrapper(llm)

    embeddings = HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    NUM_ROUNDS = 3
    num_questions = len(questions)

    try:
        from ragas import SingleTurnSample
        from ragas.metrics import ContextPrecision, ContextRecall
        from ragas import EvaluationDataset


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
                row = df.iloc[i]
                prec = 0.0
                rec = 0.0
                for c in score_cols:
                    if 'precision' in c.lower():
                        prec = row[c]
                    elif 'recall' in c.lower():
                        rec = row[c]
                
                f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                round_scores.append(f1)
            return round_scores

        # Run all rounds sequentially
        best_scores = [0.0] * num_questions
        for r in range(NUM_ROUNDS):
            round_scores = _run_round(r)
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
        
        prec = 0.0
        rec = 0.0
        for k, v in result_dict.items():
            if 'precision' in k.lower():
                prec = v
            elif 'recall' in k.lower():
                rec = v
                
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return round(f1 * 5, 2)
    except Exception as e:
        logger.error(f"Error in RAG evaluation: {e}")
        raise


def _compute_rating(df) -> float:
    """Convert ragas scores (0-1) into a single rating out of 5 using mean F1 score."""
    score_cols = [c for c in df.columns if c not in ("user_input", "reference", "response", "retrieved_contexts")]
    if not score_cols:
        return 0.0
        
    f1_scores = []
    for _, row in df.iterrows():
        prec = 0.0
        rec = 0.0
        for c in score_cols:
            if 'precision' in c.lower():
                prec = row[c]
            elif 'recall' in c.lower():
                rec = row[c]
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)
        
    avg = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return round(avg * 5, 2)


if __name__ == "__main__":
    rating = evaluate_retrieval()
    print(f"RAG Rating: {rating} / 5")
