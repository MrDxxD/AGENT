from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalResult:
    doc_id: str
    score: float
    source: str

