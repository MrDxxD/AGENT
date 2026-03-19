from rla_rag.data.dataset import load_toy_dataset
from rla_rag.env.rla_rag_env import Action, RlaRagEnv
from rla_rag.pipeline import build_pipeline


def _make_env(**kwargs):
    docs, qa = load_toy_dataset()
    return RlaRagEnv(
        docs=docs,
        qa_samples=qa,
        pipeline=build_pipeline(docs),
        max_steps=3,
        seed=123,
        **kwargs,
    )


def test_disabled_dense_action_is_reported():
    env = _make_env(enable_dense_retrieval=False)
    env.reset(sample_id="q2")
    _, reward, terminated, truncated, info = env.step(int(Action.RETRIEVE_DENSE))

    assert terminated is False
    assert truncated is False
    assert info["action_enabled"] is False
    assert reward < -0.1


def test_coverage_reward_toggle_changes_reward():
    env_on = _make_env(enable_coverage_reward=True)
    env_off = _make_env(enable_coverage_reward=False)

    env_on.reset(sample_id="q7")
    env_off.reset(sample_id="q7")

    _, reward_on, *_ = env_on.step(int(Action.RETRIEVE_SPARSE))
    _, reward_off, *_ = env_off.step(int(Action.RETRIEVE_SPARSE))

    assert reward_on >= reward_off
