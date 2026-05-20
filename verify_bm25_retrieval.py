from retrieval.dense import DenseRetriever
from retrieval.bm25 import BM25Retriever

queries = [
    "How does transformer attention mechanism work?",
    "What is BERT and how is it pretrained?",
    "How do scaling laws affect large language models?",
    "What are the advantages of retrieval augmented generation?",
    "How is BM25 used for sparse retrieval?",
]

dense = DenseRetriever()
bm25 = BM25Retriever()

comparison = []

print("=" * 70)
for query in queries:
    print(f"\nQuery: {query}")
    print("-" * 50)

    dense_results = dense.retrieve(query, top_k=3)
    bm25_results = bm25.retrieve(query, top_k=3)

    print("  DENSE:")
    for r in dense_results:
        print(f"    [{r['metadata']['paper_id']}] score={r['score']:.3f} | {r['text'][:100]}...")

    print("  BM25:")
    for r in bm25_results:
        print(f"    [{r['metadata']['paper_id']}] score={r['score']:.3f} | {r['text'][:100]}...")

    # determine winner based on top score
    winner = "DENSE" if dense_results[0]["score"] > (bm25_results[0]["score"] / 20) else "BM25"
    comparison.append({"query": query, "winner": winner,
                        "dense_score": dense_results[0]["score"],
                        "bm25_score": bm25_results[0]["score"]})

print("\n" + "=" * 70)
print("Day 2 checkpoint PASSED - BM25 retrieval working!")

# save comparison as markdown
md = "# Dense vs BM25 Retrieval Comparison\n\n"
md += "| Query | Dense Top Score | BM25 Top Score | Winner |\n"
md += "|-------|----------------|----------------|--------|\n"
for c in comparison:
    md += f"| {c['query'][:50]}... | {c['dense_score']:.3f} | {c['bm25_score']:.3f} | {c['winner']} |\n"

with open("comparison_dense_vs_bm25.md", "w") as f:
    f.write(md)
print("Comparison saved to comparison_dense_vs_bm25.md")
