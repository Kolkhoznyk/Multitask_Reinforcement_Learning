from typing import Tuple

import numpy as np
import gymnasium as gym


def evaluate_model_on_env(model, env: gym.Env, n_eval_episodes: int) -> Tuple[float, float]:
    """
    Run *n_eval_episodes* deterministic rollouts and return mean episode reward
    and mean success rate.

    Args:
        model: Any SB3-compatible model with a ``predict(obs, deterministic)`` method.
        env: A single (non-vectorised) gymnasium environment.
        n_eval_episodes: Number of episodes to roll out.

    Returns:
        Tuple of (mean_reward, mean_success_rate).
    """
    ep_rews = []
    successes = []

    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        success = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)

            if isinstance(info, dict) and info.get("success", 0) == 1:
                success = 1

        ep_rews.append(total_reward)
        successes.append(success)

    return float(np.mean(ep_rews)), float(np.mean(successes))
