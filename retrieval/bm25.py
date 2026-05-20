import glob
import pickle
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from retrieval.base import RetrieverInterface
from retrieval.dense import chunk_text, PAPERS_DIR

BM25_INDEX_PATH = "bm25_index.pkl"
STOP_WORDS = set(stopwords.words("english"))


def tokenize(text: str) -> list[str]:
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha() and t not in STOP_WORDS]


def build_bm25_index() -> tuple:
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

    print(f"Tokenizing {len(docs)} chunks...")
    tokenized = [tokenize(d["text"]) for d in tqdm(docs, desc="Tokenizing")]
    bm25 = BM25Okapi(tokenized)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": docs}, f)
    print(f"BM25 index saved to {BM25_INDEX_PATH}")
    return bm25, docs


def load_bm25_index() -> tuple:
    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["docs"]


class BM25Retriever(RetrieverInterface):
    def __init__(self):
        self.bm25, self.docs = load_bm25_index()

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [{
            "id": self.docs[i]["id"],
            "text": self.docs[i]["text"],
            "metadata": self.docs[i]["metadata"],
            "score": float(scores[i])
        } for i in top_indices]


if __name__ == "__main__":
    print("Building BM25 index...")
    build_bm25_index()
    print("Done!")
