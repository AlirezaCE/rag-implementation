#!/usr/bin/env python3
"""
Quick test script - run this immediately after training to verify your model works.

Usage:
    python scripts/quick_test.py --model_dir ./models/rag_finetuned --index_dir ./data/index_200k
"""

import argparse
import json
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from rag.retrieval import DPRRetriever, FAISSIndex


def quick_test(model_dir: str, index_dir: str):
    """Quick sanity check of trained model."""

    print("="*80)
    print("RAG MODEL QUICK TEST")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Test questions
    test_questions = [
        "What is the capital of France?",
        "Who invented the telephone?",
        "When did World War II end?",
        "What is the largest planet in our solar system?",
        "Who wrote Romeo and Juliet?"
    ]

    print("\n[1/4] Checking model files...")
    try:
        import os
        assert os.path.exists(f"{model_dir}/best_model"), "Generator not found!"
        assert os.path.exists(f"{model_dir}/question_encoder"), "Question encoder not found!"
        print("✓ Model files found")
    except AssertionError as e:
        print(f"✗ Error: {e}")
        return False

    print("\n[2/4] Loading FAISS index...")
    try:
        index = FAISSIndex.load(f"{index_dir}/index.faiss")
        passages = []
        with open(f"{index_dir}/passages.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                passages.append(json.loads(line))
        print(f"✓ Loaded {len(passages)} passages")
    except Exception as e:
        print(f"✗ Error loading index: {e}")
        return False

    print("\n[3/4] Loading model...")
    try:
        # Load retriever
        retriever = DPRRetriever(
            question_encoder=f"{model_dir}/question_encoder",
            ctx_encoder="facebook/dpr-ctx_encoder-single-nq-base",
            index=index,
            passages=passages,
            device=device,
            freeze_doc_encoder=True
        )

        # Load generator
        generator = BartForConditionalGeneration.from_pretrained(
            f"{model_dir}/best_model"
        )
        tokenizer = BartTokenizer.from_pretrained(f"{model_dir}/best_model")
        generator = generator.to(device)
        generator.eval()

        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n[4/4] Testing generation...")
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
    print(f"SUMMARY: {success_count}/{len(test_questions)} questions answered successfully")

    if success_count == len(test_questions):
        print("✓ ALL TESTS PASSED! Your model is working correctly.")
        print("\nNext steps:")
        print("  1. Run full evaluation: python scripts/evaluate_rag.py ...")
        print("  2. Try interactive mode: python scripts/interactive_test.py ...")
        print("  3. See TESTING_GUIDE.md for more options")
    elif success_count > 0:
        print("⚠ PARTIAL SUCCESS - Some tests failed")
        print("  Check the errors above and verify your model/index")
    else:
        print("✗ ALL TESTS FAILED")
        print("  Your model may not be working correctly")
        print("  Check the error messages above")

    print("="*80)

    return success_count == len(test_questions)


def main():
    parser = argparse.ArgumentParser(description="Quick test of trained RAG model")
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Directory containing trained model")
    parser.add_argument("--index_dir", type=str, required=True,
                       help="Directory containing FAISS index")

    args = parser.parse_args()

    success = quick_test(args.model_dir, args.index_dir)

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
