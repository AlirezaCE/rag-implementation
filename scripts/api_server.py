#!/usr/bin/env python3
"""
REST API server for RAG model question answering.

Provides endpoints to:
1. Ask questions and get answers
2. Check server health
3. Get model information
"""

import argparse
import json
import torch
from pathlib import Path
import logging
from typing import Dict, List, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS

from transformers import BartTokenizer, BartForConditionalGeneration
from rag.retrieval import DPRRetriever, FAISSIndex

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for model components
model_manager = None


class RAGModelManager:
    """Manages RAG model components for API serving."""

    def __init__(
        self,
        model_dir: str,
        index_dir: str,
        num_retrieved_docs: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.num_retrieved_docs = num_retrieved_docs
        self.model_dir = model_dir
        self.index_dir = index_dir

        logger.info("="*70)
        logger.info("Loading RAG Model for API Service")
        logger.info("="*70)
        logger.info(f"Model directory: {model_dir}")
        logger.info(f"Index directory: {index_dir}")
        logger.info(f"Device: {device}")

        # Load FAISS index
        logger.info("\n[1/4] Loading FAISS index...")
        self.index = FAISSIndex.load(f"{index_dir}/index.faiss")

        # Load passages
        logger.info("[2/4] Loading passages...")
        self.passages = []
        with open(f"{index_dir}/passages.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                self.passages.append(json.loads(line))
        logger.info(f"Loaded {len(self.passages)} passages")

        # Load retriever
        logger.info("[3/4] Loading retriever...")
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
        logger.info("[4/4] Loading generator...")
        self.generator = BartForConditionalGeneration.from_pretrained(
            f"{model_dir}/best_model"
        )
        self.tokenizer = BartTokenizer.from_pretrained(f"{model_dir}/best_model")
        self.generator = self.generator.to(device)
        self.generator.eval()

        logger.info("="*70)
        logger.info("Model loaded successfully! Ready to serve requests.")
        logger.info("="*70)

    def answer_question(
        self,
        question: str,
        max_length: int = 50,
        num_beams: int = 4,
        include_docs: bool = False,
        num_docs: Optional[int] = None
    ) -> Dict:
        """
        Answer a question using the RAG model.

        Args:
            question: The question to answer
            max_length: Maximum length of generated answer
            num_beams: Number of beams for beam search
            include_docs: Whether to include retrieved documents in response
            num_docs: Number of documents to retrieve (None = use default)

        Returns:
            Dictionary with answer and optionally retrieved documents
        """
        try:
            k = num_docs if num_docs else self.num_retrieved_docs

            # Retrieve documents
            retrieved = self.retriever.retrieve([question], k=k)
            doc_texts = [doc.text for doc in retrieved[0]]

            # Prepare input
            context = f"{question} {' '.join(doc_texts[:3])}"  # Top 3 docs

            inputs = self.tokenizer(
                context,
                max_length=512,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            # Generate answer
            with torch.no_grad():
                outputs = self.generator.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                )

            # Decode
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            result = {
                'question': question,
                'answer': answer,
                'success': True
            }

            # Include retrieved documents if requested
            if include_docs:
                result['retrieved_documents'] = [
                    {
                        'text': doc.text,
                        'score': float(doc.score),
                        'title': doc.metadata.get('title', '') if doc.metadata else ''
                    }
                    for doc in retrieved[0]
                ]

            return result

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            import traceback
            traceback.print_exc()
            return {
                'question': question,
                'answer': None,
                'success': False,
                'error': str(e)
            }


# API Endpoints

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_manager is not None,
        'device': model_manager.device if model_manager else None
    })


@app.route('/info', methods=['GET'])
def model_info():
    """Get model information."""
    if model_manager is None:
        return jsonify({'error': 'Model not loaded'}), 500

    return jsonify({
        'model_dir': model_manager.model_dir,
        'index_dir': model_manager.index_dir,
        'num_passages': len(model_manager.passages),
        'device': model_manager.device,
        'default_retrieved_docs': model_manager.num_retrieved_docs
    })


@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Ask a question and get an answer.

    Request JSON:
    {
        "question": "What is the capital of France?",
        "max_length": 50,  // optional, default 50
        "num_beams": 4,  // optional, default 4
        "include_docs": false,  // optional, default false
        "num_docs": 5  // optional, number of docs to retrieve
    }

    Response JSON:
    {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "success": true,
        "retrieved_documents": [...]  // only if include_docs=true
    }
    """
    if model_manager is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()

        if not data or 'question' not in data:
            return jsonify({'error': 'Missing "question" in request body'}), 400

        question = data['question']
        max_length = data.get('max_length', 50)
        num_beams = data.get('num_beams', 4)
        include_docs = data.get('include_docs', False)
        num_docs = data.get('num_docs', None)

        logger.info(f"Received question: {question}")

        result = model_manager.answer_question(
            question=question,
            max_length=max_length,
            num_beams=num_beams,
            include_docs=include_docs,
            num_docs=num_docs
        )

        logger.info(f"Answer: {result.get('answer', 'Error')}")

        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/batch', methods=['POST'])
def batch_questions():
    """
    Answer multiple questions at once.

    Request JSON:
    {
        "questions": ["Question 1?", "Question 2?", ...],
        "max_length": 50,  // optional
        "num_beams": 4,  // optional
        "include_docs": false  // optional
    }

    Response JSON:
    {
        "results": [
            {"question": "...", "answer": "...", "success": true},
            ...
        ],
        "total": 2,
        "successful": 2
    }
    """
    if model_manager is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()

        if not data or 'questions' not in data:
            return jsonify({'error': 'Missing "questions" in request body'}), 400

        questions = data['questions']
        max_length = data.get('max_length', 50)
        num_beams = data.get('num_beams', 4)
        include_docs = data.get('include_docs', False)

        if not isinstance(questions, list):
            return jsonify({'error': '"questions" must be a list'}), 400

        logger.info(f"Received batch of {len(questions)} questions")

        results = []
        for question in questions:
            result = model_manager.answer_question(
                question=question,
                max_length=max_length,
                num_beams=num_beams,
                include_docs=include_docs
            )
            results.append(result)

        successful = sum(1 for r in results if r['success'])

        return jsonify({
            'results': results,
            'total': len(results),
            'successful': successful
        }), 200

    except Exception as e:
        logger.error(f"Error processing batch request: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


def main():
    parser = argparse.ArgumentParser(
        description="REST API server for RAG model"
    )
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Directory containing fine-tuned model")
    parser.add_argument("--index_dir", type=str, required=True,
                       help="Directory containing FAISS index")
    parser.add_argument("--num_retrieved_docs", type=int, default=5,
                       help="Number of documents to retrieve")
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000,
                       help="Port to bind to")
    parser.add_argument("--debug", action="store_true",
                       help="Run in debug mode")

    args = parser.parse_args()

    # Initialize model manager
    global model_manager
    model_manager = RAGModelManager(
        model_dir=args.model_dir,
        index_dir=args.index_dir,
        num_retrieved_docs=args.num_retrieved_docs,
        device=args.device
    )

    # Start server
    logger.info(f"\nStarting API server on {args.host}:{args.port}")
    logger.info(f"Debug mode: {args.debug}")
    logger.info("\nAvailable endpoints:")
    logger.info(f"  - GET  http://{args.host}:{args.port}/health")
    logger.info(f"  - GET  http://{args.host}:{args.port}/info")
    logger.info(f"  - POST http://{args.host}:{args.port}/ask")
    logger.info(f"  - POST http://{args.host}:{args.port}/batch")
    logger.info("\n" + "="*70)

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
