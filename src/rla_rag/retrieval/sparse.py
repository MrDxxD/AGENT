from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from rla_rag.data.dataset import Document
from rla_rag.retrieval.types import RetrievalResult


class SparseRetriever:
    def __init__(self, docs: List[Document]):
        self.docs = docs
        self.doc_id_to_index: Dict[str, int] = {d.doc_id: i for i, d in enumerate(docs)}
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        self.doc_matrix = self.vectorizer.fit_transform([f"{d.title} {d.text}" for d in docs])

    def search(self, query: str, k: int = 3) -> List[RetrievalResult]:
        query_vec = self.vectorizer.transform([query])
        scores = (self.doc_matrix @ query_vec.T).toarray().ravel()
        top_indices = np.argsort(scores)[::-1][:k]
        return [
            RetrievalResult(doc_id=self.docs[i].doc_id, score=float(scores[i]), source="sparse")
            for i in top_indices
        ]

