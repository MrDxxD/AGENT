from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from rla_rag.data.dataset import Document
from rla_rag.retrieval.types import RetrievalResult


class DenseRetriever:
    """
    Lightweight dense proxy retriever.
    Uses char n-gram TF-IDF to approximate semantic recall without large model dependencies.
    """

    def __init__(self, docs: List[Document]):
        self.docs = docs
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            min_df=1,
        )
        self.doc_matrix = self.vectorizer.fit_transform([f"{d.title} {d.text}" for d in docs])

    def search(self, query: str, k: int = 3) -> List[RetrievalResult]:
        query_vec = self.vectorizer.transform([query])
        scores = (self.doc_matrix @ query_vec.T).toarray().ravel()
        top_indices = np.argsort(scores)[::-1][:k]
        return [
            RetrievalResult(doc_id=self.docs[i].doc_id, score=float(scores[i]), source="dense")
            for i in top_indices
        ]

