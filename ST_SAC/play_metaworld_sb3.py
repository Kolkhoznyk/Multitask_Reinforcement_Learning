"""
Meta-World MT1 Evaluation Script with Stable Baselines3

Updated to align with the latest Meta-World API.
Loads trained models and evaluates them with visual rendering.

Usage:
    1. Set TASK_NAME to match your trained task (e.g., 'reach-v3', 'pick-place-v3')
    2. Set ALGORITHM to match the trained algorithm ('TD3' or 'SAC')
    3. Run: python play_metaworld_sb3.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml
import gymnasium as gym
import metaworld  # registers Meta-World namespace with gymnasium
import numpy as np
from stable_baselines3 import TD3, SAC, DDPG, PPO

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(os.path.join(_script_dir, "config_ST.yaml"))

    play_cfg      = cfg["play"]
    ALGORITHM     = play_cfg["algorithm"]
    SEED          = play_cfg["seed"]
    num_episodes  = play_cfg["num_episodes"]
    RENDER_MODE   = play_cfg["render_mode"]
    TASK_NAME     = cfg["experiment"]["task_name"]
    MAX_EPISODE_STEPS = cfg["experiment"]["max_episode_steps"]
    paths         = cfg["paths"]

    # Create environment with rendering
    print(f"Creating {TASK_NAME} environment...")
    env = gym.make(
        'Meta-World/MT1',
        env_name=TASK_NAME,
        seed=SEED,
        render_mode=RENDER_MODE,
        reward_function_version='v3',
        max_episode_steps=MAX_EPISODE_STEPS,
        terminate_on_success=False,
    )

    # Load the trained model
    model_path = f"{paths['models']}/best_{TASK_NAME}_{ALGORITHM}/best_model.zip"

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Trying final model instead...")
        model_path = f"{paths['models']}/{ALGORITHM.lower()}_{TASK_NAME}_final.zip"

        if not os.path.exists(model_path):
            print(f"No trained model found!")
            print(f"Please train the model first using train_metaworld_sb3.py")
            exit(1)

    print(f"Loading model from: {model_path}")
    if ALGORITHM == "DDPG":
        model = DDPG.load(model_path, env=env)
    elif ALGORITHM == "TD3":
        model = TD3.load(model_path, env=env)
    elif ALGORITHM == "SAC":
        model = SAC.load(model_path, env=env)
    elif ALGORITHM == "PPO":
        model = PPO.load(model_path, env=env)
    else:
        raise ValueError(f"Unknown algorithm: {ALGORITHM}")

    # Run evaluation episodes
    total_rewards = []
    success_count = 0

    print(f"\nRunning {num_episodes} evaluation episodes...")
    print("=" * 60)

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        episode_success = False

        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

        while not (done or truncated):
            # Get action from policy (deterministic for evaluation)
            action, _states = model.predict(obs, deterministic=True)
            if TASK_NAME == "reach-v3":
                action = action.copy()
                action[-1] = 1.0

            # Step environment
            obs, reward, done, truncated, info = env.step(action)

            total_reward += float(reward)
            steps += 1

            # Check for success (Meta-World provides success info)
            if 'success' in info and info['success']:
                episode_success = True

            # Render
            env.render()

        print(f"Episode finished after {steps} steps")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Success: {episode_success}")

        total_rewards.append(total_reward)
        if episode_success:
            success_count += 1

    # Print summary statistics
    print("\n" + "=" * 60)
    print("=== Evaluation Complete ===")
    print(f"Task: {TASK_NAME}")
    print(f"Episodes: {num_episodes}")
    print(f"Average reward: {np.mean(total_rewards):.2f}")
    print(f"Std reward: {np.std(total_rewards):.2f}")
    print(f"Min reward: {np.min(total_rewards):.2f}")
    print(f"Max reward: {np.max(total_rewards):.2f}")
    print(f"Success rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print("=" * 60)

    # Cleanup
    env.close()
