import re
from collections import Counter
from typing import List


_BRIDGE_HINTS = [
    ("capital", "country capital"),
    ("country", "country"),
    ("who", "person"),
    ("where", "location"),
    ("首都", "国家 首都"),
    ("谁", "人物"),
    ("哪里", "地点"),
    ("在哪", "地点"),
]


def _keywords_from_evidence(evidence_chunks: List[str], current_query: str, k: int = 4) -> List[str]:
    text = " ".join(evidence_chunks)

    en_tokens = re.findall(r"\b[A-Za-z][A-Za-z\-']{2,}\b", text)
    zh_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    tokens = [t.lower() for t in en_tokens] + zh_tokens

    stop = {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "into",
        "is",
        "was",
        "are",
        "it",
        "of",
        "in",
        "to",
        "a",
        "an",
    }

    query_norm = current_query.lower()
    counts = Counter(t for t in tokens if t not in stop and len(t) > 1)
    ranked = [t for t, _ in counts.most_common(20) if t not in query_norm]
    return ranked[:k]


def rewrite_query(question: str, current_query: str, evidence_chunks: List[str]) -> str:
    query = current_query.strip()
    if not evidence_chunks:
        return query

    hints = [v for k, v in _BRIDGE_HINTS if k in question.lower() or k in question]
    evidence_terms = _keywords_from_evidence(evidence_chunks, query, k=4)

    add_on = " ".join(hints + evidence_terms).strip()
    if not add_on:
        return query

    rewritten = f"{query} {add_on}".strip()
    if len(rewritten) > 220:
        return rewritten[:220].rstrip()
    return rewritten
