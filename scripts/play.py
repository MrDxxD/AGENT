import argparse

from stable_baselines3 import PPO

from rla_rag.data.loaders import load_project_splits
from rla_rag.env.rla_rag_env import Action, RlaRagEnv
from rla_rag.pipeline import build_pipeline


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

    parser.add_argument("--model", type=str, default="models/ppo_rla_rag.zip")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test"])
    args = parser.parse_args()

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

    if args.split == "dev":
        eval_samples = dev_samples
    else:
        eval_samples = test_samples if test_samples else dev_samples

    env_kwargs = parse_env_kwargs(args)
    pipeline = build_pipeline(docs)
    env = RlaRagEnv(
        docs=docs,
        qa_samples=eval_samples,
        pipeline=pipeline,
        max_steps=6,
        max_token_budget=400,
        seed=args.seed,
        **env_kwargs,
    )

    model = PPO.load(args.model)
    sample = env.peek_random_sample()
    obs, _ = env.reset(sample_id=sample.qid)
    done = False
    step_id = 0

    print(
        f"Dataset={args.dataset}, split_mode={split_mode}, docs={len(docs)}, "
        f"train/dev/test={len(train_samples)}/{len(dev_samples)}/{len(test_samples)}, "
        f"eval_split={args.split}, eval_size={len(eval_samples)}"
    )
    print(f"EnvAblation={env_kwargs}")
    print(f"Question: {sample.question}")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        print(
            f"step={step_id}, action={Action(int(action)).name}, "
            f"enabled={info.get('action_enabled')}, reward={reward:.3f}, "
            f"coverage={info.get('support_coverage', 0.0):.2f}"
        )
        step_id += 1

    print(f"Predicted: {info.get('predicted_answer')}")
    print(f"Gold: {info.get('gold_answer')}")
    print(f"Correct: {info.get('is_correct')}")
    print(f"DoneByAnswer: {info.get('done_by_answer')}")


if __name__ == "__main__":
    main()
