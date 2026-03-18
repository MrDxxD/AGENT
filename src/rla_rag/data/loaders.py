from typing import List, Optional, Tuple

from rla_rag.data.dataset import Document, QASample, load_toy_dataset
from rla_rag.data.hotpot import load_hotpotqa_dataset


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
