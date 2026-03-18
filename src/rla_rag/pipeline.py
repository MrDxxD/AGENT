from dataclasses import dataclass
from typing import Dict, List

from rla_rag.agent.reasoner import EvidenceReasoner
from rla_rag.data.dataset import Document, QASample
from rla_rag.retrieval.dense import DenseRetriever
from rla_rag.retrieval.reranker import SimpleReranker
from rla_rag.retrieval.sparse import SparseRetriever
from rla_rag.retrieval.types import RetrievalResult


@dataclass
class Pipeline:
    docs: List[Document]
    doc_lookup: Dict[str, Document]
    sparse: SparseRetriever
    dense: DenseRetriever
    reranker: SimpleReranker
    reasoner: EvidenceReasoner

    def search_sparse(self, question: str, query: str, k: int = 3) -> List[RetrievalResult]:
        raw = self.sparse.search(query, k=max(4, k))
        return self.reranker.rerank(question, query, raw, top_k=k)

    def search_dense(self, question: str, query: str, k: int = 3) -> List[RetrievalResult]:
        raw = self.dense.search(query, k=max(4, k))
        return self.reranker.rerank(question, query, raw, top_k=k)

    def get_doc_text(self, doc_id: str) -> str:
        d = self.doc_lookup[doc_id]
        return f"{d.title}. {d.text}"

    def summarize(self, evidence_chunks: List[str], max_sentences: int = 2) -> str:
        if not evidence_chunks:
            return ""
        merged = " ".join(evidence_chunks)
        sentences = [s.strip() for s in merged.split(".") if s.strip()]
        return ". ".join(sentences[:max_sentences])


def build_pipeline(docs: List[Document], qa_samples: List[QASample]) -> Pipeline:
    return Pipeline(
        docs=docs,
        doc_lookup={d.doc_id: d for d in docs},
        sparse=SparseRetriever(docs),
        dense=DenseRetriever(docs),
        reranker=SimpleReranker(docs),
        reasoner=EvidenceReasoner(candidate_answers=[q.answer for q in qa_samples]),
    )
