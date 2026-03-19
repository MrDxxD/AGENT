from typing import Dict, List, Optional, Tuple

from rla_rag.data.dataset import Document, QASample, load_toy_dataset, split_qa_samples
from rla_rag.data.hotpot import load_hotpotqa_dataset


def _merge_docs(*doc_lists: List[Document]) -> List[Document]:
    merged: Dict[str, Document] = {}
    for docs in doc_lists:
        for d in docs:
            existing = merged.get(d.doc_id)
            if existing is None or len(d.text) > len(existing.text):
                merged[d.doc_id] = d
    return sorted(merged.values(), key=lambda x: x.doc_id)


def load_project_dataset(
    dataset: str,
    hotpot_path: str = "",
    max_samples: Optional[int] = None,
) -> Tuple[List[Document], List[QASample]]:
    dataset = dataset.lower().strip()
    if dataset == "toy":
        return load_toy_dataset()
    if dataset == "hotpot":
        if not hotpot_path:
            raise ValueError("--hotpot-path is required when --dataset hotpot")
        return load_hotpotqa_dataset(hotpot_path, max_samples=max_samples)
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_project_splits(
    dataset: str,
    seed: int,
    train_ratio: float = 0.6,
    dev_ratio: float = 0.2,
    hotpot_path: str = "",
    train_path: str = "",
    dev_path: str = "",
    max_samples: Optional[int] = None,
    max_train_samples: Optional[int] = None,
    max_dev_samples: Optional[int] = None,
) -> Tuple[List[Document], List[QASample], List[QASample], List[QASample], str]:
    dataset = dataset.lower().strip()

    if dataset == "toy":
        docs, qa_samples = load_toy_dataset()
        train, dev, test = split_qa_samples(qa_samples, train_ratio=train_ratio, dev_ratio=dev_ratio, seed=seed)
        return docs, train, dev, test, "random_split"

    if dataset == "hotpot":
        if train_path or dev_path:
            if not train_path or not dev_path:
                raise ValueError("Both --train-path and --dev-path are required for official split mode")
            train_docs, train_samples = load_hotpotqa_dataset(
                train_path,
                max_samples=max_train_samples if max_train_samples is not None else max_samples,
            )
            dev_docs, dev_samples = load_hotpotqa_dataset(
                dev_path,
                max_samples=max_dev_samples if max_dev_samples is not None else max_samples,
            )
            docs = _merge_docs(train_docs, dev_docs)
            return docs, train_samples, dev_samples, [], "official_split"

        if hotpot_path:
            docs, qa_samples = load_hotpotqa_dataset(hotpot_path, max_samples=max_samples)
            train, dev, test = split_qa_samples(
                qa_samples,
                train_ratio=train_ratio,
                dev_ratio=dev_ratio,
                seed=seed,
            )
            return docs, train, dev, test, "random_split"

        raise ValueError(
            "For --dataset hotpot, provide either --hotpot-path or both --train-path and --dev-path"
        )

    raise ValueError(f"Unsupported dataset: {dataset}")
