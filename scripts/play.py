import argparse

from stable_baselines3 import PPO

from rla_rag.data.dataset import load_toy_dataset
from rla_rag.env.rla_rag_env import Action, RlaRagEnv
from rla_rag.pipeline import build_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo_rla_rag.zip")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    docs, qa_samples = load_toy_dataset()
    pipeline = build_pipeline(docs, qa_samples)
    env = RlaRagEnv(
        docs=docs,
        qa_samples=qa_samples,
        pipeline=pipeline,
        max_steps=6,
        max_token_budget=400,
        seed=args.seed,
    )

    model = PPO.load(args.model)
    sample = env.peek_random_sample()
    obs, _ = env.reset(sample_id=sample.qid)
    done = False
    step_id = 0
    print(f"Question: {sample.question}")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        print(
            f"step={step_id}, action={Action(int(action)).name}, "
            f"reward={reward:.3f}, coverage={info.get('support_coverage', 0.0):.2f}"
        )
        step_id += 1
    print(f"Predicted: {info.get('predicted_answer')}")
    print(f"Gold: {info.get('gold_answer')}")
    print(f"Correct: {info.get('is_correct')}")


if __name__ == "__main__":
    main()
