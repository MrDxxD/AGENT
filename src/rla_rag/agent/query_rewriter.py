from typing import List


def rewrite_query(question: str, current_query: str, evidence_chunks: List[str]) -> str:
    """
    Simple heuristic rewrite:
    - append salient entities from evidence
    - add bridge words for multi-hop hints
    """
    query = current_query.strip()
    evidence_text = " ".join(evidence_chunks).lower()

    bridge_words = []
    if "capital" in question.lower() and "country" in question.lower():
        bridge_words.append("country capital")
    if "who" in question.lower() and "created" in question.lower():
        bridge_words.append("founder")
    if "where" in question.lower():
        bridge_words.append("located")

    entity_hints = []
    for token in ["eiffel", "einstein", "python", "guido", "berlin", "paris", "netherlands"]:
        if token in evidence_text and token not in query.lower():
            entity_hints.append(token)

    add_on = " ".join(bridge_words + entity_hints).strip()
    if not add_on:
        return query
    return f"{query} {add_on}".strip()
