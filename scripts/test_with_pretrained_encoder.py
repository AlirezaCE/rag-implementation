#!/usr/bin/env python3
"""
Quick test using pre-trained question encoder (workaround when tokenizer not saved).
"""

import argparse
import json
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

from rag.retrieval import DPRRetriever, FAISSIndex


def quick_test(model_dir: str, index_dir: str):
    """Quick test using pre-trained DPR encoder."""

    print("="*80)
    print("RAG MODEL QUICK TEST (with pre-trained encoder)")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Test questions
    test_questions = [
        "What is the capital of France?",
        "Who invented the telephone?",
        "When did World War II end?",
    ]

    print("\n[1/3] Loading FAISS index...")
    index = FAISSIndex.load(f"{index_dir}/index.faiss")
    passages = []
    with open(f"{index_dir}/passages.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            passages.append(json.loads(line))
    print(f"✓ Loaded {len(passages)} passages")

    print("\n[2/3] Loading model...")
    # Use pre-trained question encoder instead of fine-tuned one
    retriever = DPRRetriever(
        question_encoder="facebook/dpr-question_encoder-single-nq-base",  # Pre-trained
        ctx_encoder="facebook/dpr-ctx_encoder-single-nq-base",
        index=index,
        passages=passages,
        device=device,
        freeze_doc_encoder=True
    )

    # Load fine-tuned generator
    generator = BartForConditionalGeneration.from_pretrained(f"{model_dir}/best_model")
    tokenizer = BartTokenizer.from_pretrained(f"{model_dir}/best_model")
    generator = generator.to(device)
    generator.eval()

    print("✓ Model loaded")

    print("\n[3/3] Testing generation...")
    print("="*80)

    success_count = 0

    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")

        try:
            # Retrieve
            retrieved = retriever.retrieve([question], k=5)
            doc_texts = [doc.text for doc in retrieved[0]]

            print(f"   Top doc: {doc_texts[0][:80]}...")

            # Generate
            context = f"{question} {' '.join(doc_texts[:3])}"
            inputs = tokenizer(
                context,
                max_length=512,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = generator.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=50,
                    num_beams=4,
                    early_stopping=True,
                )

            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"   Answer: {answer}")
            print(f"   ✓ Success")
            success_count += 1

        except Exception as e:
            print(f"   ✗ Error: {e}")

    print("\n" + "="*80)
    print(f"SUMMARY: {success_count}/{len(test_questions)} questions answered")
    print("\nNote: Using pre-trained question encoder since fine-tuned tokenizer wasn't saved")
    print("="*80)

    return success_count == len(test_questions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--index_dir", type=str, required=True)

    args = parser.parse_args()

    quick_test(args.model_dir, args.index_dir)


if __name__ == "__main__":
    main()
