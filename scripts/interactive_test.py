#!/usr/bin/env python3
"""
Interactive testing script for fine-tuned RAG model.

This allows you to:
1. Ask questions interactively
2. See retrieved documents
3. Compare with baseline
"""

import argparse
import json
import torch
from pathlib import Path
import logging

from transformers import BartTokenizer, BartForConditionalGeneration

from rag.retrieval import DPRRetriever, FAISSIndex

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGInteractiveTester:
    """Interactive tester for RAG model."""

    def __init__(
        self,
        model_dir: str,
        index_dir: str,
        num_retrieved_docs: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.num_retrieved_docs = num_retrieved_docs

        logger.info("Loading model and index...")

        # Load FAISS index
        self.index = FAISSIndex.load(f"{index_dir}/index.faiss")

        self.passages = []
        with open(f"{index_dir}/passages.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                self.passages.append(json.loads(line))

        logger.info(f"Loaded {len(self.passages)} passages")

        # Load retriever
        question_encoder_path = f"{model_dir}/question_encoder"
        self.retriever = DPRRetriever(
            question_encoder=question_encoder_path,
            ctx_encoder="facebook/dpr-ctx_encoder-single-nq-base",
            index=self.index,
            passages=self.passages,
            device=device,
            freeze_doc_encoder=True
        )

        # Load generator
        self.generator = BartForConditionalGeneration.from_pretrained(
            f"{model_dir}/best_model"
        )
        self.tokenizer = BartTokenizer.from_pretrained(f"{model_dir}/best_model")
        self.generator = self.generator.to(device)
        self.generator.eval()

        logger.info("Model loaded successfully!")

    def answer_question(
        self,
        question: str,
        max_length: int = 50,
        num_beams: int = 4,
        show_docs: bool = True
    ):
        """Answer a question and show the process."""
        print("\n" + "="*80)
        print(f"QUESTION: {question}")
        print("="*80)

        # Retrieve documents
        print(f"\n[1/3] Retrieving top {self.num_retrieved_docs} documents...")
        retrieved = self.retriever.retrieve([question], k=self.num_retrieved_docs)
        doc_texts = [doc.text for doc in retrieved[0]]

        if show_docs:
            print("\nRetrieved Documents:")
            for i, doc in enumerate(retrieved[0], 1):
                print(f"\n  {i}. [Score: {doc.score:.4f}]")
                print(f"     {doc.text[:200]}...")

        # Prepare input
        print(f"\n[2/3] Generating answer...")
        context = f"{question} {' '.join(doc_texts[:3])}"  # Top 3 docs

        inputs = self.tokenizer(
            context,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.generator.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )

        # Decode
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"\n[3/3] Result:")
        print("-"*80)
        print(f"ANSWER: {prediction}")
        print("-"*80)

        return {
            'question': question,
            'answer': prediction,
            'retrieved_docs': [{'text': doc.text, 'score': doc.score} for doc in retrieved[0]]
        }

    def run_interactive(self):
        """Run interactive question-answering session."""
        print("\n" + "="*80)
        print("RAG Interactive Testing Mode")
        print("="*80)
        print("\nType your questions below. Type 'quit' or 'exit' to stop.")
        print("Type 'nodocs' to toggle document display.")
        print("="*80)

        show_docs = True

        while True:
            try:
                question = input("\n\nYour question: ").strip()

                if not question:
                    continue

                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break

                if question.lower() == 'nodocs':
                    show_docs = not show_docs
                    print(f"Document display: {'ON' if show_docs else 'OFF'}")
                    continue

                self.answer_question(question, show_docs=show_docs)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()

    def run_batch_questions(self, questions_file: str, output_file: str = None):
        """Run questions from a file."""
        print(f"\nProcessing questions from: {questions_file}")

        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]

        results = []

        for i, question in enumerate(questions, 1):
            print(f"\n\n{'='*80}")
            print(f"Question {i}/{len(questions)}")
            result = self.answer_question(question, show_docs=False)
            results.append(result)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n\nResults saved to: {output_file}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Interactive testing for fine-tuned RAG model"
    )
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Directory containing fine-tuned model")
    parser.add_argument("--index_dir", type=str, required=True,
                       help="Directory containing FAISS index")
    parser.add_argument("--questions_file", type=str, default=None,
                       help="File with questions (one per line) for batch mode")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file for batch mode results")
    parser.add_argument("--num_retrieved_docs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Initialize tester
    tester = RAGInteractiveTester(
        model_dir=args.model_dir,
        index_dir=args.index_dir,
        num_retrieved_docs=args.num_retrieved_docs,
        device=args.device
    )

    # Run batch or interactive mode
    if args.questions_file:
        tester.run_batch_questions(args.questions_file, args.output_file)
    else:
        tester.run_interactive()


if __name__ == "__main__":
    main()
