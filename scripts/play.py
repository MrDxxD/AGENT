import argparse

from stable_baselines3 import PPO

from rla_rag.data.dataset import split_qa_samples
from rla_rag.data.loaders import load_project_dataset
from rla_rag.env.rla_rag_env import Action, RlaRagEnv
from rla_rag.pipeline import build_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="toy", choices=["toy", "hotpot"])
    parser.add_argument("--hotpot-path", type=str, default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--dev-ratio", type=float, default=0.2)

    parser.add_argument("--model", type=str, default="models/ppo_rla_rag.zip")
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

    model = PPO.load(args.model)
    sample = env.peek_random_sample()
    obs, _ = env.reset(sample_id=sample.qid)
    done = False
    step_id = 0

    print(
        f"Dataset={args.dataset}, docs={len(docs)}, qa={len(qa_samples)}, "
        f"eval_split={args.split}, eval_size={len(eval_samples)}"
    )
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
    print(f"DoneByAnswer: {info.get('done_by_answer')}")


if __name__ == "__main__":
    main()
