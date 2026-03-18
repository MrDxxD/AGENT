import argparse

from stable_baselines3 import PPO

from rla_rag.data.dataset import split_qa_samples
from rla_rag.data.loaders import load_project_dataset
from rla_rag.env.rla_rag_env import Action, RlaRagEnv
from rla_rag.eval.runner import evaluate_baseline, evaluate_policy
from rla_rag.pipeline import build_pipeline


def one_shot_sparse(step_idx: int) -> int:
    if step_idx == 0:
        return int(Action.RETRIEVE_SPARSE)
    return int(Action.ANSWER_NOW)


def react_style(step_idx: int) -> int:
    if step_idx == 0:
        return int(Action.RETRIEVE_SPARSE)
    if step_idx == 1:
        return int(Action.REWRITE_QUERY)
    if step_idx == 2:
        return int(Action.RETRIEVE_DENSE)
    if step_idx == 3:
        return int(Action.SUMMARIZE_EVIDENCE)
    return int(Action.ANSWER_NOW)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="toy", choices=["toy", "hotpot"])
    parser.add_argument("--hotpot-path", type=str, default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--dev-ratio", type=float, default=0.2)

    parser.add_argument("--model", type=str, default="models/ppo_rla_rag.zip")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test"])
    args = parser.parse_args()

    docs, qa_samples = load_project_dataset(
        dataset=args.dataset,
        hotpot_path=args.hotpot_path,
        max_samples=args.max_samples if args.max_samples > 0 else None,
    )
    _, dev_samples, test_samples = split_qa_samples(
        qa_samples,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )
    eval_samples = dev_samples if args.split == "dev" and dev_samples else test_samples

    pipeline = build_pipeline(docs)
    env = RlaRagEnv(
        docs=docs,
        qa_samples=eval_samples,
        pipeline=pipeline,
        max_steps=6,
        max_token_budget=400,
        seed=args.seed,
    )

    print(
        f"Dataset={args.dataset}, docs={len(docs)}, qa={len(qa_samples)}, "
        f"eval_split={args.split}, eval_size={len(eval_samples)}"
    )

    sparse_metrics = evaluate_baseline(env, one_shot_sparse, episodes=args.episodes)
    react_metrics = evaluate_baseline(env, react_style, episodes=args.episodes)

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
