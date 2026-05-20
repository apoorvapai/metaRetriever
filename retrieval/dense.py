import os
import glob
import logging
logging.getLogger("chromadb").setLevel(logging.ERROR)
from pathlib import Path
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
PAPERS_DIR = Path("data/papers")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "papers_dense"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# load model once at module level
_model = SentenceTransformer(EMBEDDING_MODEL)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def ingest_papers() -> list[dict]:
    docs = []
    for filepath in sorted(glob.glob(str(PAPERS_DIR / "*.txt"))):
        text = Path(filepath).read_text(encoding="utf-8", errors="ignore")
        paper_id = Path(filepath).stem
        for idx, chunk in enumerate(chunk_text(text)):
            docs.append({
                "id": f"{paper_id}_chunk{idx}",
                "text": chunk,
                "metadata": {"paper_id": paper_id, "chunk_idx": idx}
            })
    return docs


def get_embedding(text: str) -> list[float]:
    return _model.encode(text, normalize_embeddings=True).tolist()


def build_index(docs: list[dict]):
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # get or create collection (resume-safe)
    try:
        collection = client.get_collection(COLLECTION_NAME)
        existing_ids = set(collection.get()["ids"])
        print(f"Resuming — {len(existing_ids)} chunks already indexed.")
    except Exception:
        collection = client.create_collection(COLLECTION_NAME)
        existing_ids = set()

    docs = [d for d in docs if d["id"] not in existing_ids]
    if not docs:
        print("All chunks already indexed!")
        return

    print(f"Indexing {len(docs)} remaining chunks...")
    batch_size = 100  # large batch — no rate limits with local model
    for i in tqdm(range(0, len(docs), batch_size), desc="Embedding & indexing"):
        batch = docs[i: i + batch_size]
        embeddings = _model.encode(
            [d["text"] for d in batch],
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()
        collection.add(
            ids=[d["id"] for d in batch],
            embeddings=embeddings,
            documents=[d["text"] for d in batch],
            metadatas=[d["metadata"] for d in batch]
        )
    print(f"Indexed {len(docs)} chunks into ChromaDB.")


class DenseRetriever:
    def __init__(self):
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = client.get_collection(COLLECTION_NAME)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        query_embedding = _model.encode(
            query, normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]
            })
        return chunks


if __name__ == "__main__":
    print("Ingesting papers...")
    docs = ingest_papers()
    print(f"Total chunks: {len(docs)}")
    print("Building ChromaDB index...")
    build_index(docs)
    print("Done! Index ready.")
