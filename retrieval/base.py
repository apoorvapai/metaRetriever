from abc import ABC, abstractmethod


class RetrieverInterface(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve top_k relevant chunks for a given query.

        Returns a list of dicts with keys:
            - id       : unique chunk identifier
            - text     : chunk text content
            - metadata : dict with paper_id, chunk_idx
            - score    : relevance score (higher = more relevant)
        """
        pass
