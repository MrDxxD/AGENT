import json
import uuid
from pathlib import Path

import pytest

from rla_rag.data.hotpot import load_hotpotqa_dataset
from rla_rag.data.loaders import load_project_dataset


def test_hotpot_loader_parses_docs_and_support():
    data = [
        {
            "_id": "hp1",
            "question": "Which city is the capital of France?",
            "answer": "Paris",
            "supporting_facts": [["France", 0]],
            "context": [
                ["France", ["Paris is the capital of France.", "France is in Europe."]],
                ["Noise", ["Something else."]],
            ],
        },
        {
            "_id": "hp2",
            "question": "Who developed relativity?",
            "answer": "Albert Einstein",
            "supporting_facts": [["Einstein", 0]],
            "context": [["Einstein", ["Albert Einstein developed the theory of relativity."]]],
        },
    ]

    tmp_dir = Path("tests") / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    file_path = tmp_dir / f"hotpot_sample_{uuid.uuid4().hex}.json"

    try:
        file_path.write_text(json.dumps(data), encoding="utf-8")
        docs, qa = load_hotpotqa_dataset(str(file_path))

        assert len(qa) == 2
        assert len(docs) == 3
        assert qa[0].qid == "hp1"
        assert qa[1].qid == "hp2"
        assert qa[0].support_doc_ids
        assert qa[1].support_doc_ids
    finally:
        if file_path.exists():
            file_path.unlink()


def test_project_loader_requires_hotpot_path():
    with pytest.raises(ValueError):
        load_project_dataset(dataset="hotpot", hotpot_path="")
