#!/usr/bin/env python3
"""
Evaluate fine-tuned RAG model on Natural Questions test set.

This script:
1. Loads your fine-tuned model
2. Runs inference on test questions
3. Computes metrics (Exact Match, F1, etc.)
4. Generates sample predictions for inspection
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
import logging
from typing import List, Dict
import re
from collections import Counter

from transformers import BartTokenizer, BartForConditionalGeneration, DPRQuestionEncoder

from rag import RAGSequenceForGeneration, RAGConfig
from rag.retrieval import DPRRetriever, FAISSIndex

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison (from SQuAD evaluation)."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)

    common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common_tokens.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1


def evaluate_rag(
    model_dir: str,
    test_data_file: str,
    index_dir: str,
    output_file: str = None,
    num_samples: int = None,
    batch_size: int = 1,
    num_retrieved_docs: int = 5,
    max_length: int = 50,
    num_beams: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Evaluate fine-tuned RAG model.
    """
    logger.info("="*70)
    logger.info("RAG Model Evaluation")
    logger.info("="*70)
    logger.info(f"Model: {model_dir}")
    logger.info(f"Test data: {test_data_file}")
    logger.info(f"Index: {index_dir}")
    logger.info(f"Device: {device}")

    # Load test data
    logger.info("\n[1/5] Loading test data...")
    test_examples = []
    with open(test_data_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_examples.append(json.loads(line))

    if num_samples:
        test_examples = test_examples[:num_samples]

    logger.info(f"Loaded {len(test_examples)} test examples")

    # Load FAISS index
    logger.info("\n[2/5] Loading FAISS index...")
    index = FAISSIndex.load(f"{index_dir}/index.faiss")

    passages = []
    with open(f"{index_dir}/passages.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            passages.append(json.loads(line))
    logger.info(f"Loaded {len(passages)} passages")

    # Load question encoder
    logger.info("\n[3/5] Loading question encoder...")
    question_encoder_path = f"{model_dir}/question_encoder"

    # Create retriever
    retriever = DPRRetriever(
        question_encoder=question_encoder_path,
        ctx_encoder="facebook/dpr-ctx_encoder-single-nq-base",  # Not fine-tuned
        index=index,
        passages=passages,
        device=device,
        freeze_doc_encoder=True
    )

    # Load fine-tuned generator
    logger.info("\n[4/5] Loading fine-tuned generator...")
    generator = BartForConditionalGeneration.from_pretrained(f"{model_dir}/best_model")
    tokenizer = BartTokenizer.from_pretrained(f"{model_dir}/best_model")
    generator = generator.to(device)
    generator.eval()

    # Load training info if available
    try:
        with open(f"{model_dir}/training_info.json", 'r') as f:
            training_info = json.load(f)
            logger.info(f"Model trained for {training_info.get('global_step', 'unknown')} steps")
            logger.info(f"Best validation loss: {training_info.get('best_val_loss', 'unknown'):.4f}")
    except:
        logger.warning("Training info not found")

    # Evaluate
    logger.info("\n[5/5] Running evaluation...")
    logger.info("-"*70)

    predictions = []
    exact_matches = []
    f1_scores = []

    with torch.no_grad():
        for example in tqdm(test_examples, desc="Evaluating"):
            question = example['question']
            answers = example['answers'] if isinstance(example['answers'], list) else [example['answers']]
            ground_truth = answers[0]  # Use first answer as reference

            try:
                # Retrieve documents
                retrieved = retriever.retrieve([question], k=num_retrieved_docs)
                doc_texts = [doc.text for doc in retrieved[0]]

                # Prepare input
                context = f"{question} {' '.join(doc_texts[:3])}"  # Use top 3 docs

                # Tokenize
                inputs = tokenizer(
                    context,
                    max_length=512,
                    truncation=True,
                    return_tensors="pt"
                ).to(device)

                # Generate
                outputs = generator.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                )

                # Decode prediction
                prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Compute metrics
                em = exact_match_score(prediction, ground_truth)
                f1 = f1_score(prediction, ground_truth)

                exact_matches.append(em)
                f1_scores.append(f1)

                predictions.append({
                    'id': example.get('id', ''),
                    'question': question,
                    'prediction': prediction,
                    'ground_truth': ground_truth,
                    'exact_match': em,
                    'f1': f1,
                    'retrieved_docs': [doc.text[:100] + '...' for doc in retrieved[0][:3]]
                })

            except Exception as e:
                logger.error(f"Error processing example: {e}")
                continue

    # Compute final metrics
    avg_em = sum(exact_matches) / len(exact_matches) * 100 if exact_matches else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) * 100 if f1_scores else 0

    logger.info("\n" + "="*70)
    logger.info("EVALUATION RESULTS")
    logger.info("="*70)
    logger.info(f"Number of examples: {len(predictions)}")
    logger.info(f"Exact Match (EM): {avg_em:.2f}%")
    logger.info(f"F1 Score: {avg_f1:.2f}%")
    logger.info("="*70)

    # Save results
    if output_file:
        results = {
            'metrics': {
                'exact_match': avg_em,
                'f1_score': avg_f1,
                'num_examples': len(predictions)
            },
            'config': {
                'model_dir': model_dir,
                'test_data': test_data_file,
                'num_retrieved_docs': num_retrieved_docs,
                'max_length': max_length,
                'num_beams': num_beams
            },
            'predictions': predictions
        }

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"\nResults saved to: {output_file}")

    # Print sample predictions
    logger.info("\n" + "="*70)
    logger.info("SAMPLE PREDICTIONS")
    logger.info("="*70)

    for i, pred in enumerate(predictions[:5], 1):
        logger.info(f"\nExample {i}:")
        logger.info(f"Question: {pred['question']}")
        logger.info(f"Prediction: {pred['prediction']}")
        logger.info(f"Ground Truth: {pred['ground_truth']}")
        logger.info(f"EM: {pred['exact_match']:.0f} | F1: {pred['f1']:.2f}")
        logger.info(f"Top Retrieved Doc: {pred['retrieved_docs'][0]}")

    return {
        'exact_match': avg_em,
        'f1_score': avg_f1,
        'predictions': predictions
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned RAG model on Natural Questions"
    )
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Directory containing fine-tuned model")
    parser.add_argument("--test_data", type=str, required=True,
                       help="Path to test data JSONL file")
    parser.add_argument("--index_dir", type=str, required=True,
                       help="Directory containing FAISS index")
    parser.add_argument("--output_file", type=str, default="./results/evaluation_results.json",
                       help="Where to save evaluation results")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to evaluate (None = all)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_retrieved_docs", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    evaluate_rag(
        model_dir=args.model_dir,
        test_data_file=args.test_data,
        index_dir=args.index_dir,
        output_file=args.output_file,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_retrieved_docs=args.num_retrieved_docs,
        max_length=args.max_length,
        num_beams=args.num_beams,
        device=args.device
    )


if __name__ == "__main__":
    main()
