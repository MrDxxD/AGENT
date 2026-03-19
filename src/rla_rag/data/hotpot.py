import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from rla_rag.data.dataset import Document, QASample


def _stable_doc_id(title: str) -> str:
    digest = hashlib.md5(title.encode("utf-8")).hexdigest()[:10]
    return f"hp_{digest}"


def _normalize_sentences(sentences: Sequence[str]) -> str:
    cleaned = [s.strip() for s in sentences if isinstance(s, str) and s.strip()]
    return " ".join(cleaned)


def load_hotpotqa_dataset(
    path: str,
    max_samples: Optional[int] = None,
) -> Tuple[List[Document], List[QASample]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"HotpotQA file not found: {path}")

    with file_path.open("r", encoding="utf-8-sig") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("HotpotQA json must be a list of samples.")

    if max_samples is not None and max_samples > 0:
        raw = raw[:max_samples]

    title_to_doc: Dict[str, Document] = {}
    qa_samples: List[QASample] = []

    for item in raw:
        if not isinstance(item, dict):
            continue

        qid = str(item.get("_id", ""))
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        context = item.get("context", [])
        supporting_facts = item.get("supporting_facts", [])

        if not qid or not question or not answer or not isinstance(context, list):
            continue

        support_titles: Set[str] = set()
        if isinstance(supporting_facts, list):
            for sf in supporting_facts:
                if isinstance(sf, list) and sf and isinstance(sf[0], str):
                    support_titles.add(sf[0])

        sample_support_doc_ids: Set[str] = set()

        for entry in context:
            if not (isinstance(entry, list) and len(entry) == 2):
                continue
            title, sentences = entry
            if not isinstance(title, str) or not isinstance(sentences, list):
                continue

            text = _normalize_sentences(sentences)
            if not text:
                continue

            existing = title_to_doc.get(title)
            if existing is None or len(text) > len(existing.text):
                title_to_doc[title] = Document(
                    doc_id=_stable_doc_id(title),
                    title=title,
                    text=text,
                )

            if title in support_titles:
                sample_support_doc_ids.add(_stable_doc_id(title))

        if not sample_support_doc_ids:
            for title in support_titles:
                if title in title_to_doc:
                    sample_support_doc_ids.add(title_to_doc[title].doc_id)

        qa_samples.append(
            QASample(
                qid=qid,
                question=question,
                answer=answer,
                support_doc_ids=sample_support_doc_ids,
            )
        )

    docs = sorted(title_to_doc.values(), key=lambda d: d.doc_id)
    return docs, qa_samples

