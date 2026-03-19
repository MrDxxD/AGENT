import argparse
import csv
import os
from statistics import mean, pstdev
from typing import Dict, List

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rla_rag.data.loaders import load_project_splits
from rla_rag.env.rla_rag_env import Action, RlaRagEnv
from rla_rag.eval.runner import evaluate_baseline, evaluate_policy
from rla_rag.pipeline import build_pipeline


def make_env(docs, qa_samples, seed: int, env_kwargs: Dict):
    pipeline = build_pipeline(docs)

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


def parse_seeds(text: str) -> List[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return [42]
    return [int(x) for x in parts]


def condition_to_env_kwargs(name: str) -> Dict:
    base = {
        "enable_dense_retrieval": True,
        "enable_query_rewrite": True,
        "enable_summarization": True,
        "enable_coverage_reward": True,
        "enable_token_penalty": True,
    }
    if name == "full":
        return base
    if name == "no_dense":
        base["enable_dense_retrieval"] = False
        return base
    if name == "no_rewrite":
        base["enable_query_rewrite"] = False
        return base
    if name == "no_summary":
        base["enable_summarization"] = False
        return base
    if name == "no_coverage_reward":
        base["enable_coverage_reward"] = False
        return base
    if name == "no_token_penalty":
        base["enable_token_penalty"] = False
        return base
    raise ValueError(f"Unknown condition: {name}")


def write_rows_csv(path: str, rows: List[Dict]):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: List[Dict]) -> List[Dict]:
    grouped = {}
    metric_keys = ["avg_reward", "accuracy", "avg_support_coverage", "avg_steps", "avg_token_cost"]

    for r in rows:
        key = (r["condition"], r["policy"])
        grouped.setdefault(key, []).append(r)

    out = []
    for (condition, policy), group in grouped.items():
        row = {"condition": condition, "policy": policy, "runs": len(group)}
        for m in metric_keys:
            vals = [float(g[m]) for g in group]
            row[f"{m}_mean"] = mean(vals)
            row[f"{m}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        out.append(row)

    out.sort(key=lambda x: (x["condition"], x["policy"]))
    return out


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

    parser.add_argument("--split", type=str, default="test", choices=["dev", "test"])
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--seeds", type=str, default="42,43,44")

    parser.add_argument(
        "--conditions",
        type=str,
        default="full,no_dense,no_rewrite,no_summary,no_coverage_reward,no_token_penalty",
    )

    parser.add_argument("--out-csv", type=str, default="results/ablations_runs.csv")
    parser.add_argument("--out-summary-csv", type=str, default="results/ablations_summary.csv")
    parser.add_argument("--save-models-dir", type=str, default="")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    rows: List[Dict] = []

    for condition in conditions:
        env_kwargs = condition_to_env_kwargs(condition)
        print(f"\n=== Condition: {condition} | env={env_kwargs} ===")

        for seed in seeds:
            docs, train_samples, dev_samples, test_samples, split_mode = load_project_splits(
                dataset=args.dataset,
                seed=seed,
                train_ratio=args.train_ratio,
                dev_ratio=args.dev_ratio,
                hotpot_path=args.hotpot_path,
                train_path=args.train_path,
                dev_path=args.dev_path,
                max_samples=args.max_samples if args.max_samples > 0 else None,
                max_train_samples=args.max_train_samples if args.max_train_samples > 0 else None,
                max_dev_samples=args.max_dev_samples if args.max_dev_samples > 0 else None,
            )

            eval_samples = dev_samples if args.split == "dev" else (test_samples if test_samples else dev_samples)
            print(
                f"seed={seed}, split_mode={split_mode}, sizes(train/dev/test)="
                f"{len(train_samples)}/{len(dev_samples)}/{len(test_samples)}, eval={len(eval_samples)}"
            )

            vec_env = DummyVecEnv(
                [make_env(docs, train_samples, seed=seed + i, env_kwargs=env_kwargs) for i in range(args.n_envs)]
            )
            model = PPO(
                "MlpPolicy",
                vec_env,
                verbose=0,
                learning_rate=3e-4,
                n_steps=512,
                batch_size=256,
                gamma=0.98,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                seed=seed,
            )
            model.learn(total_timesteps=args.timesteps, progress_bar=False)

            if args.save_models_dir:
                os.makedirs(args.save_models_dir, exist_ok=True)
                model.save(os.path.join(args.save_models_dir, f"{args.dataset}_{condition}_seed{seed}.zip"))

            eval_env = make_env(docs, eval_samples, seed=seed + 1000, env_kwargs=env_kwargs)()

            sparse_metrics = evaluate_baseline(eval_env, one_shot_sparse, episodes=args.eval_episodes)
            react_metrics = evaluate_baseline(eval_env, make_react_style(env_kwargs), episodes=args.eval_episodes)
            rl_metrics = evaluate_policy(model, eval_env, episodes=args.eval_episodes)

            for policy_name, metrics in [
                ("baseline_sparse", sparse_metrics),
                ("baseline_react", react_metrics),
                ("rl", rl_metrics),
            ]:
                row = {
                    "dataset": args.dataset,
                    "split_mode": split_mode,
                    "eval_split": args.split,
                    "condition": condition,
                    "seed": seed,
                    "policy": policy_name,
                    **metrics,
                }
                rows.append(row)

    write_rows_csv(args.out_csv, rows)
    summary_rows = aggregate_rows(rows)
    write_rows_csv(args.out_summary_csv, summary_rows)

    print(f"\nSaved run-level results: {args.out_csv}")
    print(f"Saved summary results: {args.out_summary_csv}")


if __name__ == "__main__":
    main()
