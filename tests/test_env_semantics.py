from rla_rag.data.dataset import load_toy_dataset
from rla_rag.env.rla_rag_env import Action, RlaRagEnv
from rla_rag.pipeline import build_pipeline


def test_done_by_answer_true_when_answer_action():
    docs, qa = load_toy_dataset()
    env = RlaRagEnv(docs=docs, qa_samples=qa[:2], pipeline=build_pipeline(docs), max_steps=3, seed=7)
    env.reset()
    _, _, terminated, truncated, info = env.step(int(Action.ANSWER_NOW))
    assert terminated is True
    assert truncated is False
    assert info["done_by_answer"] is True


def test_done_by_answer_false_when_truncated():
    docs, qa = load_toy_dataset()
    env = RlaRagEnv(docs=docs, qa_samples=qa[:2], pipeline=build_pipeline(docs), max_steps=1, seed=7)
    env.reset()
    _, _, terminated, truncated, info = env.step(int(Action.RETRIEVE_SPARSE))
    assert terminated is False
    assert truncated is True
    assert info["done_by_answer"] is False
