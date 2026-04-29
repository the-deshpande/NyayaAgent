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
        "What penalties does SEBI impose for insider trading and how has the Supreme Court interpreted them?",
        "When can a court grant specific performance as a remedy for breach of contract?",
        "What are the fundamental rights guaranteed under the Constitution of India and their limitations?"
    ]
    ground_truth = [
        "SEBI can impose penalties up to twenty-five crore rupees or three times the profits made from insider trading under the SEBI Act. The Supreme Court has held that penalties under SEBI regulations are civil in nature and the mens rea of the insider is not a prerequisite; possession of UPSI while trading is sufficient to attract liability.",
        "Under the Specific Relief Act, 1963, a court may grant specific performance when monetary compensation is inadequate, the contract is certain and enforceable, and the plaintiff has performed or is ready to perform their obligations. The Supreme Court has ruled that specific performance is no longer a discretionary remedy and should be granted unless the contract falls within the statutory exceptions.",
        "Part III of the Constitution of India guarantees fundamental rights including the right to equality, freedom of speech and expression, protection against discrimination, and the right to life and personal liberty under Articles 14 to 32. These rights are not absolute and can be subject to reasonable restrictions imposed by the State in the interests of sovereignty, public order, morality, and security of India."
    ]

    retriever = get_retriever()
    contexts = []
    
    for q in questions:
        hits = retriever.search(q)
        if hits:
            contexts.append([h["text"] for h in hits])
        else:
            contexts.append([""])
        
    data = {
        "user_input": questions,
        "reference": ground_truth,
        "retrieved_contexts": contexts,
        "response": ground_truth # adding response for completeness, some metrics might need it
    }
    
    dataset = Dataset.from_dict(data)
    
    llm = get_chat_model()
    ragas_llm = LangchainLLMWrapper(llm)
    
    embeddings = HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # Note: latest Ragas uses 'user_input', 'reference', 'response', 'retrieved_contexts' 
    # instead of 'question', 'ground_truth', 'answer', 'contexts'.
    
    try:
        from ragas import SingleTurnSample
        from ragas.metrics import ContextPrecision, ContextRecall
        from ragas import EvaluationDataset
        
        # New API for Ragas
        samples = []
        for i in range(len(questions)):
            samples.append(SingleTurnSample(
                user_input=questions[i],
                reference=ground_truth[i],
                retrieved_contexts=contexts[i],
                response=ground_truth[i]
            ))
        eval_dataset = EvaluationDataset(samples=samples)
        
        result = evaluate(
            dataset=eval_dataset,
            metrics=[ContextPrecision(), ContextRecall()],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        df = result.to_pandas()
        records = df.to_dict(orient="records")
        logger.info("RAG Evaluation detailed results:")
        logger.info(records)
        
        return _compute_rating(df)
    except ImportError:
        # Fallback to old API
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
