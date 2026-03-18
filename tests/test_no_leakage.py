from rla_rag.agent.reasoner import EvidenceReasoner
from rla_rag.data.dataset import load_toy_dataset
from rla_rag.pipeline import build_pipeline


def test_pipeline_reasoner_does_not_use_qa_answer_list():
    docs, qa = load_toy_dataset()
    pipeline = build_pipeline(docs)

    assert isinstance(pipeline.reasoner, EvidenceReasoner)
    assert not hasattr(pipeline.reasoner, "candidate_answers")

    ans, _ = pipeline.reasoner.answer("What planet is mentioned?", ["Mars is a planet in the solar system."])
    assert ans.lower() in {"mars", "mars is a planet in the solar system"} or "mars" in ans.lower()
