from typing import Dict, Iterable, List

from rla_rag.data.dataset import Document
from rla_rag.retrieval.types import RetrievalResult


def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().replace("?", " ").replace(",", " ").split() if t]


class SimpleReranker:
    def __init__(self, docs: List[Document]):
        self.doc_index: Dict[str, Document] = {d.doc_id: d for d in docs}

    def rerank(
        self, question: str, query: str, candidates: Iterable[RetrievalResult], top_k: int = 3
    ) -> List[RetrievalResult]:
        q_tokens = set(_tokenize(question) + _tokenize(query))
        rescored = []
        for c in candidates:
            doc = self.doc_index[c.doc_id]
            d_tokens = set(_tokenize(f"{doc.title} {doc.text}"))
            overlap = len(q_tokens & d_tokens) / max(1, len(q_tokens))
            new_score = 0.7 * c.score + 0.3 * overlap
            rescored.append(RetrievalResult(doc_id=c.doc_id, score=float(new_score), source=c.source))
        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored[:top_k]

