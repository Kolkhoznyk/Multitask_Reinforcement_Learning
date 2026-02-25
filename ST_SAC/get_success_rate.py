import argparse
import csv
import logging
import os
import re
from typing import Optional

import gymnasium as gym
from stable_baselines3 import SAC

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Maps algorithm name to its SB3 class; extend here to support additional algorithms.
SUPPORTED_ALGORITHMS = {
    "SAC": SAC,
}

_STEP_RE = re.compile(r"_(\d+)_steps\.zip$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SAC checkpoints on a MetaWorld task and write success rates to CSV."
    )
    parser.add_argument("--task", default="pick-place-v3", help="MetaWorld task name.")
    parser.add_argument(
        "--algorithm",
        default="SAC",
        choices=list(SUPPORTED_ALGORITHMS),
        help="RL algorithm.",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes per checkpoint.",
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=150, help="Maximum steps per episode."
    )
    parser.add_argument("--seed", type=int, default=40, help="Environment seed.")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory containing checkpoint .zip files. "
        "Defaults to ./metaworld_models/checkpoints_<task>/",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path. Defaults to ./analysis/MT1_<algorithm>_<task>_SuccessRate.csv",
    )
    parser.add_argument("--render", action="store_true", help="Enable visual rendering.")
    return parser.parse_args()


def extract_step_from_filename(fname: str) -> Optional[int]:
    m = _STEP_RE.search(fname)
    return int(m.group(1)) if m else None


def list_checkpoint_files(checkpoint_dir: str) -> list:
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")]

    def sort_key(f: str) -> tuple:
        step = extract_step_from_filename(f)
        return (0, step) if step is not None else (1, f)

    files.sort(key=sort_key)
    return [os.path.join(checkpoint_dir, f) for f in files]


def evaluate_checkpoint(
    model_path: str,
    env: gym.Env,
    task_name: str,
    n_eval_episodes: int,
    algorithm: str,
) -> float:
    model_cls = SUPPORTED_ALGORITHMS[algorithm]
    model = model_cls.load(model_path, env=env)

    success_count = 0
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_success = False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)

            # Reach task has no grasp phase; force gripper closed throughout.
            if task_name == "reach-v3":
                action = action.copy()
                action[-1] = 1.0

            obs, _, done, truncated, info = env.step(action)

            if info.get("success"):
                episode_success = True

        if episode_success:
            success_count += 1

    return success_count / n_eval_episodes


def main() -> None:
    args = parse_args()

    task = args.task
    algorithm = args.algorithm
    checkpoint_dir = args.checkpoint_dir or f"./metaworld_models/checkpoints_{task}/"
    output_csv = args.output_csv or f"./analysis/MT1_{algorithm}_{task}_SuccessRate.csv"

    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoint_files = list_checkpoint_files(checkpoint_dir)
    if not checkpoint_files:
        raise RuntimeError(f"No checkpoint .zip files found in: {checkpoint_dir}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    env = gym.make(
        "Meta-World/MT1",
        env_name=task,
        seed=args.seed,
        render_mode="human" if args.render else None,
        reward_function_version="v3",
        max_episode_steps=args.max_episode_steps,
        terminate_on_success=True,
    )

    results = []
    for model_path in checkpoint_files:
        step = extract_step_from_filename(model_path)
        success_rate = evaluate_checkpoint(
            model_path=model_path,
            env=env,
            task_name=task,
            n_eval_episodes=args.n_eval_episodes,
            algorithm=algorithm,
        )
        logger.info("Step %s | Success rate: %.3f", step, success_rate)
        results.append((step, success_rate))

    env.close()

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "success_rate"])
        writer.writerows(results)

    logger.info("Saved results to: %s", output_csv)


if __name__ == "__main__":
    main()