#!/usr/bin/env python3
"""
Test RAG with 200k Wikipedia articles index.
"""

import json
from rag import RAGSequenceForGeneration, RAGConfig
from rag.retrieval import DPRRetriever, FAISSIndex

# Create config
config = RAGConfig(
    model_type="rag_sequence",
    num_retrieved_docs=5,
    generator_name_or_path="facebook/bart-base",
)

# Load FAISS index and create retriever
print("Loading FAISS index (200k articles)...")
index = FAISSIndex.load("./data/index_200k/index.faiss")

# Load passages
print("Loading passages...")
passages = []
with open("./data/index_200k/passages.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        passages.append(json.loads(line))

print(f"Loaded {len(passages)} passages from 200k Wikipedia articles")

print("Creating DPR retriever...")
retriever = DPRRetriever(
    question_encoder="facebook/dpr-question_encoder-single-nq-base",
    ctx_encoder="facebook/dpr-ctx_encoder-single-nq-base",
    index=index,
    passages=passages,
    device="cpu"  # Change to "cuda" if GPU is working for inference
)

print("Creating RAG model...")
model = RAGSequenceForGeneration(config=config, retriever=retriever)

# Test questions - mix of common and specific knowledge
questions = [
    "What is the capital of France?",
    "Who invented the telephone?",
    "What is photosynthesis?",
    "When did World War 2 end?",
    "Who wrote Romeo and Juliet?",
    "What is the largest planet in our solar system?",
    "Who was the first president of the United States?",
    "What is the speed of light?",
    "When was the Eiffel Tower built?",
    "What is DNA?",
]

print("\n" + "="*70)
print("Testing RAG with 200k Wikipedia articles index")
print("="*70 + "\n")

for i, question in enumerate(questions, 1):
    print(f"\n[{i}/{len(questions)}] Question: {question}")
    try:
        answer = model.generate_from_query(question, max_length=50)
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"Error: {e}")
    print("-" * 70)

print("\n" + "="*70)
print("Testing complete!")
print("="*70)
