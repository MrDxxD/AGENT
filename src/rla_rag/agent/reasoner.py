import re
from collections import Counter
from typing import Dict, List, Tuple


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
    return [p.strip() for p in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text) if p.strip()]


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
        city_to_country: Dict[str, str] = {}
        country_to_city: Dict[str, str] = {}
        patterns = [
            re.compile(
                r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+the\s+capital(?:\s+city)?\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
            ),
            re.compile(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"),
        ]
        for sent in re.split(r"[\.!?]\s*", context):
            s = sent.strip()
            if not s:
                continue
            for pat in patterns:
                m = pat.search(s)
                if not m:
                    continue
                left = m.group(1).strip()
                right = m.group(2).strip()
                if "capital" in s.lower():
                    country_to_city[right] = left
                    city_to_country[left] = right
                else:
                    city_to_country.setdefault(left, right)
        return city_to_country, country_to_city

    def _extract_birth_country_pairs(self, context: str) -> Dict[str, str]:
        result: Dict[str, str] = {}
        pattern = re.compile(
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+was\s+born\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
        )
        for sent in re.split(r"[\.!?]\s*", context):
            s = sent.strip()
            if not s:
                continue
            m = pattern.search(s)
            if m:
                result[m.group(1).strip()] = m.group(2).strip()
        return result

    def _question_mentions_phrase(self, question_norm: str, phrase: str) -> bool:
        p_norm = normalize_text(phrase)
        if not p_norm:
            return False
        if p_norm in question_norm:
            return True
        tokens = [t for t in p_norm.split() if len(t) >= 4]
        return any(t in question_norm for t in tokens)

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

        if "capital" in q_norm:
            asks_city = any(x in q_norm for x in ["which city", "what city"])
            asks_country = any(x in q_norm for x in ["which country", "what country"])

            for country, city in country_to_city.items():
                if self._question_mentions_phrase(q_norm, country):
                    return city, 0.85

            for city, country in city_to_country.items():
                if self._question_mentions_phrase(q_norm, city):
                    return country, 0.80

            birth_pairs = self._extract_birth_country_pairs(context)
            if birth_pairs and "birth country" in q_norm:
                for person, country in birth_pairs.items():
                    if self._question_mentions_phrase(q_norm, person) and country in country_to_city:
                        return country_to_city[country], 0.80

            if asks_city and country_to_city:
                return next(iter(country_to_city.values())), 0.65
            if asks_country and city_to_country:
                return next(iter(city_to_country.values())), 0.65

        if any(k in q_norm for k in ["who", "created", "introduced", "developed", "designed"]):
            by_matches = re.findall(
                r"(?:by|introduced by|created by|designed by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                context,
            )
            if by_matches:
                return by_matches[0].strip(), 0.75

        return self._best_phrase(question, context)
