import re
from typing import Iterable, List, Tuple


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    return [t for t in normalize_text(text).split() if t]


class EvidenceReasoner:
    def __init__(self, candidate_answers: Iterable[str]):
        self.candidate_answers = sorted(set(candidate_answers), key=len, reverse=True)

    def answer(self, question: str, evidence_chunks: List[str], summary: str = "") -> Tuple[str, float]:
        context = " ".join(evidence_chunks + ([summary] if summary else []))
        if not context.strip():
            return "UNKNOWN", 0.0

        q_tokens = set(_tokenize(question))
        context_norm = normalize_text(context)

        best_answer = "UNKNOWN"
        best_score = -1.0

        for candidate in self.candidate_answers:
            cand_norm = normalize_text(candidate)
            cand_tokens = set(cand_norm.split())
            if not cand_tokens:
                continue
            in_context = 1.0 if cand_norm in context_norm else 0.0
            overlap_q = len(cand_tokens & q_tokens) / len(cand_tokens)
            score = 1.8 * in_context + 0.2 * overlap_q
            if score > best_score:
                best_score = score
                best_answer = candidate

        if best_score <= 0:
            cap_phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", context)
            if cap_phrases:
                best_answer = cap_phrases[0]
                best_score = 0.2
            else:
                best_answer = "UNKNOWN"
                best_score = 0.0

        confidence = max(0.0, min(1.0, best_score / 2.0))
        return best_answer, confidence
