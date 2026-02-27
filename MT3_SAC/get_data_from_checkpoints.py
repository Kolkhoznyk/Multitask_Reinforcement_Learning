import os
import csv
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
from MT10_SAC.algos.sac_disentangled_alpha import SACDisentangledAlpha
from utils.wrappers import SingleTaskOneHotWrapper
from utils.checkpoint import extract_step_from_filename, list_checkpoint_files
from utils.evaluation import evaluate_model_on_env


def main():
    TASK_NAMES = ["reach-v3", "push-v3", "pick-place-v3"]
    REWARD_SCALES = [0.5, 1, 1.2]
    ALGORITHM = "SAC"

    N_EVAL_EPISODES = 20
    MAX_EPISODE_STEPS = 150
    SEED = 30

    EXP = "MT3_1"
    CHECKPOINT_DIR = "./metaworld_models/checkpoints_MT3/"
    OUTPUT_CSV = f"./analysis/checkpoints_eval_{EXP}.csv"

    if not os.path.isdir(CHECKPOINT_DIR):
        raise FileNotFoundError(f"Checkpoint dir not found: {CHECKPOINT_DIR}")
    checkpoint_files = list_checkpoint_files(CHECKPOINT_DIR)
    if not checkpoint_files:
        raise RuntimeError("No checkpoint .zip files found")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # --- 1) Create eval environments once ---
    eval_envs = []
    n_tasks = len(TASK_NAMES)
    for i, task_name in enumerate(TASK_NAMES):
        env = gym.make(
            "Meta-World/MT1",
            env_name=task_name,
            seed=SEED,
            reward_function_version="v3",
            max_episode_steps=MAX_EPISODE_STEPS,
            terminate_on_success=True,   # recommended for accurate success rate evaluation
        )
        env = SingleTaskOneHotWrapper(env=env, task_id=i, n_tasks=n_tasks, reward_scale=REWARD_SCALES[i])
        eval_envs.append(env)

    # CSV Header
    header = [
        "step",
        "reach mean reward", "reach successrate",
        "push mean reward", "push successrate",
        "pick-place mean reward", "pick-place successrate",
        "avg mean reward", "avg successrate",
    ]
    rows = []

    # For each checkpoint evaluate tasks!
    for model_path in checkpoint_files:
        step = extract_step_from_filename(model_path)
        if not os.path.exists(model_path):
            print(f"[skip] model missing: {model_path}")
            continue

        if ALGORITHM == "SAC":
            model = SACDisentangledAlpha.load(model_path)
        else:
            raise ValueError(f"Unknown algorithm: {ALGORITHM}")

        task_mean_rewards = []
        task_success_rates = []

        for env in eval_envs:
            mean_rew, mean_sr = evaluate_model_on_env(model, env, N_EVAL_EPISODES)
            task_mean_rewards.append(mean_rew)
            task_success_rates.append(mean_sr)

        avg_mean_rew = float(np.mean(task_mean_rewards))
        avg_mean_sr = float(np.mean(task_success_rates))

        row = [
            step,
            task_mean_rewards[0], task_success_rates[0],
            task_mean_rewards[1], task_success_rates[1],
            task_mean_rewards[2], task_success_rates[2],
            avg_mean_rew, avg_mean_sr
        ]
        rows.append(row)

        print(f"Step {step}: "
              f"reach SR={task_success_rates[0]:.2f}, "
              f"push SR={task_success_rates[1]:.2f}, "
              f"pick SR={task_success_rates[2]:.2f}, "
              f"avg SR={avg_mean_sr:.2f}")

    for env in eval_envs:
        env.close()

    # Write to csv
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nSaved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
