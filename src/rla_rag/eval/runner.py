from typing import Callable, Dict

import numpy as np


def evaluate_policy(model, env, episodes: int = 200) -> Dict[str, float]:
    stats = _rollout(episodes, env, lambda obs, i: int(model.predict(obs, deterministic=True)[0]))
    return _aggregate(stats)


def evaluate_baseline(env, policy_fn: Callable[[int], int], episodes: int = 200) -> Dict[str, float]:
    stats = _rollout(episodes, env, lambda obs, i: int(policy_fn(i)))
    return _aggregate(stats)


def _rollout(episodes: int, env, action_selector: Callable):
    records = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        step_idx = 0
        ep_reward = 0.0
        final_info = {}
        while not done:
            action = action_selector(obs, step_idx)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            final_info = info
            step_idx += 1
        records.append(
            {
                "reward": ep_reward,
                "is_correct": float(final_info.get("is_correct", 0.0)),
                "support_coverage": float(final_info.get("support_coverage", 0.0)),
                "steps": float(final_info.get("steps", 0.0)),
                "token_cost": float(final_info.get("token_cost", 0.0)),
            }
        )
    return records


def _aggregate(records):
    def avg(key: str) -> float:
        return float(np.mean([r[key] for r in records])) if records else 0.0

    return {
        "avg_reward": avg("reward"),
        "accuracy": avg("is_correct"),
        "avg_support_coverage": avg("support_coverage"),
        "avg_steps": avg("steps"),
        "avg_token_cost": avg("token_cost"),
    }
