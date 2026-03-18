import re
from collections import Counter
from typing import List, Tuple


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    chars = []
    for ch in text:
        if ch.isalnum() or ch.isspace():
            chars.append(ch)
        else:
            chars.append(" ")
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def _tokenize(text: str) -> List[str]:
    return [t for t in normalize_text(text).split() if t]


def _extract_phrases(text: str) -> List[str]:
    phrases: List[str] = []
    phrases.extend(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text))
    phrases.extend(re.findall(r"[\u4e00-\u9fff]{2,}", text))
    return [p.strip() for p in phrases if p.strip()]


class EvidenceReasoner:
    def __init__(self):
        self._stop = {
            "the",
            "a",
            "an",
            "is",
            "of",
            "in",
            "on",
            "and",
            "to",
            "for",
            "which",
            "what",
            "who",
            "where",
        }

    def _extract_capital_pairs(self, context: str):
        city_to_country = {}
        country_to_city = {}
        patterns = [
            re.compile(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+the\s+capital(?:\s+city)?\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"),
            re.compile(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"),
        ]
        for sent in re.split(r"[\.!?]\s*", context):
            s = sent.strip()
            if not s:
                continue
            for pat in patterns:
                m = pat.search(s)
                if m:
                    left = m.group(1).strip()
                    right = m.group(2).strip()
                    if "capital" in s.lower():
                        country_to_city[right] = left
                        city_to_country[left] = right
                    else:
                        city_to_country.setdefault(left, right)
        return city_to_country, country_to_city

    def _best_phrase(self, question: str, context: str) -> Tuple[str, float]:
        phrases = _extract_phrases(context)
        if not phrases:
            return "UNKNOWN", 0.0

        q_tokens = {t for t in _tokenize(question) if t not in self._stop}
        counts = Counter([normalize_text(p) for p in phrases])
        best = "UNKNOWN"
        best_score = -1.0
        for phrase in phrases:
            p_norm = normalize_text(phrase)
            p_tokens = {t for t in p_norm.split() if t and t not in self._stop}
            overlap = (len(p_tokens & q_tokens) / max(1, len(p_tokens))) if p_tokens else 0.0
            freq = counts[p_norm]
            score = overlap + 0.1 * min(freq, 5)
            if score > best_score:
                best_score = score
                best = phrase

        conf = max(0.0, min(1.0, best_score / 1.5))
        return best, conf

    def answer(self, question: str, evidence_chunks: List[str], summary: str = "") -> Tuple[str, float]:
        context = " ".join(evidence_chunks + ([summary] if summary else []))
        if not context.strip():
            return "UNKNOWN", 0.0

        q_norm = normalize_text(question)
        city_to_country, country_to_city = self._extract_capital_pairs(context)

        if "capital" in q_norm or "首都" in question:
            for country, city in country_to_city.items():
                if normalize_text(country) in q_norm:
                    return city, 0.8
            for city, country in city_to_country.items():
                if normalize_text(city) in q_norm:
                    return country, 0.75

        if any(k in q_norm for k in ["who", "谁", "created", "introduced", "developed", "designed"]):
            by_matches = re.findall(r"(?:by|introduced by|created by|designed by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", context)
            if by_matches:
                return by_matches[0].strip(), 0.75

        return self._best_phrase(question, context)
