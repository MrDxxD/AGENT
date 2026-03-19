import json
import uuid
from pathlib import Path

import pytest

from rla_rag.data.loaders import load_project_splits


def _write_json(tmp_dir: Path, rows):
    p = tmp_dir / f"hp_{uuid.uuid4().hex}.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    return p


def _sample_row(qid: str, title: str, answer: str):
    return {
        "_id": qid,
        "question": f"Q for {answer}?",
        "answer": answer,
        "supporting_facts": [[title, 0]],
        "context": [[title, [f"{answer} evidence sentence."]]],
    }


def test_hotpot_official_split_requires_both_paths():
    with pytest.raises(ValueError):
        load_project_splits(dataset="hotpot", seed=42, train_path="train.json", dev_path="")


def test_hotpot_official_split_mode_and_sizes():
    tmp_dir = Path("tests") / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    train_path = _write_json(
        tmp_dir,
        [
            _sample_row("t1", "TrainDocA", "Alpha"),
            _sample_row("t2", "TrainDocB", "Beta"),
        ],
    )
    dev_path = _write_json(
        tmp_dir,
        [
            _sample_row("d1", "DevDocA", "Gamma"),
            _sample_row("d2", "DevDocB", "Delta"),
        ],
    )

    try:
        docs, train, dev, test, mode = load_project_splits(
            dataset="hotpot",
            seed=42,
            train_path=str(train_path),
            dev_path=str(dev_path),
        )
        assert mode == "official_split"
        assert len(train) == 2
        assert len(dev) == 2
        assert len(test) == 0
        assert len(docs) == 4
    finally:
        if train_path.exists():
            train_path.unlink()
        if dev_path.exists():
            dev_path.unlink()
