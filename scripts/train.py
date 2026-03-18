import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rla_rag.data.dataset import load_toy_dataset
from rla_rag.env.rla_rag_env import RlaRagEnv
from rla_rag.eval.runner import evaluate_policy
from rla_rag.pipeline import build_pipeline


def make_env(seed: int = 42):
    docs, qa_samples = load_toy_dataset()
    pipeline = build_pipeline(docs, qa_samples)

    def _env():
        return RlaRagEnv(
            docs=docs,
            qa_samples=qa_samples,
            pipeline=pipeline,
            max_steps=6,
            max_token_budget=400,
            seed=seed,
        )

    return _env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", type=str, default="models/ppo_rla_rag.zip")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=200)
    args = parser.parse_args()

    model_dir = os.path.dirname(args.model_out)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    vec_env = DummyVecEnv([make_env(seed=args.seed + i) for i in range(args.n_envs)])
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=args.seed,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    model.save(args.model_out)

    eval_env = make_env(seed=args.seed + 1000)()
    metrics = evaluate_policy(model, eval_env, episodes=args.eval_episodes)
    print("Training finished. Evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
