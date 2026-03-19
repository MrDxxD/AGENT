import argparse

from stable_baselines3 import PPO

from rla_rag.data.loaders import load_project_splits
from rla_rag.env.rla_rag_env import Action, RlaRagEnv
from rla_rag.eval.runner import evaluate_baseline, evaluate_policy
from rla_rag.pipeline import build_pipeline


def parse_env_kwargs(args):
    return {
        "enable_dense_retrieval": not args.disable_dense,
        "enable_query_rewrite": not args.disable_rewrite,
        "enable_summarization": not args.disable_summary,
        "enable_coverage_reward": not args.disable_coverage_reward,
        "enable_token_penalty": not args.disable_token_penalty,
    }


def one_shot_sparse(step_idx: int) -> int:
    if step_idx == 0:
        return int(Action.RETRIEVE_SPARSE)
    return int(Action.ANSWER_NOW)


def make_react_style(env_kwargs):
    sequence = [Action.RETRIEVE_SPARSE]
    if env_kwargs.get("enable_query_rewrite", True):
        sequence.append(Action.REWRITE_QUERY)
    if env_kwargs.get("enable_dense_retrieval", True):
        sequence.append(Action.RETRIEVE_DENSE)
    if env_kwargs.get("enable_summarization", True):
        sequence.append(Action.SUMMARIZE_EVIDENCE)

    def _policy(step_idx: int) -> int:
        if step_idx < len(sequence):
            return int(sequence[step_idx])
        return int(Action.ANSWER_NOW)

    return _policy


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
    parser.add_argument("--episodes", type=int, default=300)
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

    print(
        f"Dataset={args.dataset}, split_mode={split_mode}, docs={len(docs)}, "
        f"train/dev/test={len(train_samples)}/{len(dev_samples)}/{len(test_samples)}, "
        f"eval_split={args.split}, eval_size={len(eval_samples)}"
    )
    print(f"EnvAblation={env_kwargs}")

    sparse_metrics = evaluate_baseline(env, one_shot_sparse, episodes=args.episodes)
    react_metrics = evaluate_baseline(env, make_react_style(env_kwargs), episodes=args.episodes)

    print(f"[Eval split] {args.split}")
    print("[Baseline] One-shot sparse RAG")
    for k, v in sparse_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n[Baseline] Fixed ReAct-like policy")
    for k, v in react_metrics.items():
        print(f"{k}: {v:.4f}")

    model = PPO.load(args.model)
    rl_metrics = evaluate_policy(model, env, episodes=args.episodes)

    print("\n[RL] PPO policy")
    for k, v in rl_metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
