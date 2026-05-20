import chromadb
from retrieval.dense import DenseRetriever, CHROMA_DIR, COLLECTION_NAME

# ── Verify index completeness ──
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION_NAME)
count = collection.count()
EXPECTED = 8937
print(f"Index check: {count}/{EXPECTED} chunks stored", "✅" if count == EXPECTED else "❌ INCOMPLETE")
assert count == EXPECTED, f"Expected {EXPECTED} chunks, found {count}"

retriever = DenseRetriever()

queries = [
    "How does transformer attention mechanism work?",
    "What is BERT and how is it pretrained?",
    "How do scaling laws affect large language models?",
    "What are the advantages of GraphRAG over regular RAG?",
    "How is BM25 used for sparse retrieval?"
]

print("=" * 60)
for query in queries:
    print(f"\nQuery: {query}")
    print("-" * 40)
    results = retriever.retrieve(query, top_k=3)
    for i, r in enumerate(results):
        print(f"  [{i+1}] {r['metadata']['paper_id']} | score: {r['score']:.3f}")
        print(f"       {r['text'][:150]}...")
print("=" * 60)
print("Day 1 checkpoint PASSED - Dense retrieval working!")
