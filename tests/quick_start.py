from rag import RAGSequenceForGeneration, RAGConfig
from rag.retrieval import MockRetriever

# Create config
config = RAGConfig(
    model_type="rag_sequence",
    num_retrieved_docs=5,
    generator_name_or_path="facebook/bart-base",
)

# Create model with mock retriever
retriever = MockRetriever(num_docs=100)
model = RAGSequenceForGeneration(config=config, retriever=retriever)

# Generate
question = "What is the capital of France?"
answer = model.generate_from_query(question, max_length=50)
print(answer)
