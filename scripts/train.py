import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rla_rag.data.loaders import load_project_splits
from rla_rag.env.rla_rag_env import RlaRagEnv
from rla_rag.eval.runner import evaluate_policy
from rla_rag.pipeline import build_pipeline


def make_env(docs, qa_samples, seed: int = 42, env_kwargs=None):
    pipeline = build_pipeline(docs)
    env_kwargs = env_kwargs or {}

    def _env():
        return RlaRagEnv(
            docs=docs,
            qa_samples=qa_samples,
            pipeline=pipeline,
            max_steps=6,
            max_token_budget=400,
            seed=seed,
            **env_kwargs,
        )

    return _env


def parse_env_kwargs(args):
    return {
        "enable_dense_retrieval": not args.disable_dense,
        "enable_query_rewrite": not args.disable_rewrite,
        "enable_summarization": not args.disable_summary,
        "enable_coverage_reward": not args.disable_coverage_reward,
        "enable_token_penalty": not args.disable_token_penalty,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="toy", choices=["toy", "hotpot"])
    parser.add_argument("--hotpot-path", type=str, default="")
    parser.add_argument("--train-path", type=str, default="")
    parser.add_argument("--dev-path", type=str, default="")

    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-dev-samples", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--dev-ratio", type=float, default=0.2)

    parser.add_argument("--disable-dense", action="store_true")
    parser.add_argument("--disable-rewrite", action="store_true")
    parser.add_argument("--disable-summary", action="store_true")
    parser.add_argument("--disable-coverage-reward", action="store_true")
    parser.add_argument("--disable-token-penalty", action="store_true")

    parser.add_argument("--timesteps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", type=str, default="models/ppo_rla_rag.zip")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=200)
    args = parser.parse_args()

    model_dir = os.path.dirname(args.model_out)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    docs, train_samples, dev_samples, test_samples, split_mode = load_project_splits(
        dataset=args.dataset,
        seed=args.seed,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        hotpot_path=args.hotpot_path,
        train_path=args.train_path,
        dev_path=args.dev_path,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        max_train_samples=args.max_train_samples if args.max_train_samples > 0 else None,
        max_dev_samples=args.max_dev_samples if args.max_dev_samples > 0 else None,
    )

    eval_samples = dev_samples if dev_samples else test_samples
    env_kwargs = parse_env_kwargs(args)

    print(
        f"Dataset={args.dataset}, split_mode={split_mode}, docs={len(docs)}, "
        f"train/dev/test={len(train_samples)}/{len(dev_samples)}/{len(test_samples)}"
    )
    print(f"EnvAblation={env_kwargs}")

    vec_env = DummyVecEnv([make_env(docs, train_samples, seed=args.seed + i, env_kwargs=env_kwargs) for i in range(args.n_envs)])
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

    eval_env = make_env(docs, eval_samples, seed=args.seed + 1000, env_kwargs=env_kwargs)()
    metrics = evaluate_policy(model, eval_env, episodes=args.eval_episodes)
    print("Training finished. Evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
