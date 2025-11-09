#!/usr/bin/env python3
"""
Test RAG with real FAISS index.
"""

from rag import RAGSequenceForGeneration, RAGConfig
from rag.retrieval import DPRRetriever, FAISSIndex

# Create config
config = RAGConfig(
    model_type="rag_sequence",
    num_retrieved_docs=5,
    generator_name_or_path="facebook/bart-base",
)

# Load FAISS index and create retriever
print("Loading FAISS index...")
index = FAISSIndex.load("./data/index")

print("Creating DPR retriever...")
retriever = DPRRetriever(
    index=index,
    query_encoder_name="facebook/dpr-question_encoder-single-nq-base"
)

print("Creating RAG model...")
model = RAGSequenceForGeneration(config=config, retriever=retriever)

# Test questions
questions = [
    "What is the capital of France?",
    "Who invented the telephone?",
    "What is photosynthesis?",
]

print("\n" + "="*60)
print("Testing RAG with real retrieval")
print("="*60 + "\n")

for question in questions:
    print(f"Question: {question}")
    answer = model.generate_from_query(question, max_length=50)
    print(f"Answer: {answer}\n")
