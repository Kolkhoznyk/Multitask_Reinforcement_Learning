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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import gymnasium as gym
import metaworld  # registers Meta-World namespace with gymnasium
import numpy as np
from stable_baselines3 import TD3, SAC, DDPG, PPO

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

class SingleTaskOneHotWrapper(gym.Wrapper):
    def __init__(self, env, task_id, n_tasks, reward_scale=1.0):
        super().__init__(env)
        self.task_id = task_id
        self.n_tasks = n_tasks
        self.reward_scale = reward_scale

        obs_space = self.env.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([obs_space.low, np.zeros(self.n_tasks)]), # type: ignore
            high=np.concatenate([obs_space.high, np.ones(self.n_tasks)]), # type: ignore
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._one_hot_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._one_hot_obs(obs), reward * self.reward_scale, terminated, truncated, info # type: ignore

    def _one_hot_obs(self, obs):
        one_hot = np.zeros(self.n_tasks, dtype=np.float32)
        one_hot[self.task_id] = 1.0
        return np.concatenate([np.array(obs, dtype=np.float32), one_hot])

if __name__ == "__main__":
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(os.path.join(_script_dir, "config_MT3.yaml"))

    play_cfg       = cfg["play"]
    ALGORITHM      = play_cfg["algorithm"]
    SEED           = play_cfg["seed"]
    num_episodes   = play_cfg["num_episodes"]
    RENDER_MODE    = play_cfg["render_mode"]
    MAX_EPISODE_STEPS = cfg["experiment"]["max_episode_steps"]
    paths          = cfg["paths"]

    TRAINING_TASKS = cfg["tasks"]["training"]
    UNIQUE_TASKS   = list(dict.fromkeys(TRAINING_TASKS))
    TASK_NAMES     = UNIQUE_TASKS
    TASK_ID        = {name: i for i, name in enumerate(UNIQUE_TASKS)}

    for task_name in TASK_NAMES:
        # Create environment with rendering
        print(f"Creating {task_name} environment...")
        env = gym.make(
            'Meta-World/MT1',
            env_name=task_name,
            seed=SEED,
            render_mode=RENDER_MODE,
            reward_function_version='v3',
            max_episode_steps=MAX_EPISODE_STEPS,
            terminate_on_success=False,
        )
        env = SingleTaskOneHotWrapper(env=env, task_id=TASK_ID[task_name], n_tasks=len(UNIQUE_TASKS))

        # Load the trained model
        model_path = paths["best_model"]

        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            print("Trying final model instead...")
            model_path = f"{paths['models']}/{ALGORITHM.lower()}_{task_name}_final.zip"

            if not os.path.exists(model_path):
                print(f"No trained model found!")
                print(f"Please train the model first using train_metaworld_sb3_MT3_v2.py")
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

            while not (done or truncated):
                # Get action from policy (deterministic for evaluation)
                action, _states = model.predict(obs, deterministic=True)
                if task_name == "reach-v3":
                    action = action.copy()
                    action[-1] = 1.0

                # Step environment
                obs, reward, done, truncated, info = env.step(action)

                total_reward += reward
                steps += 1

                # Check for success (Meta-World provides success info)
                if 'success' in info and info['success']:
                    episode_success = True

                # Render
                env.render()
            
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            # print(f"Episode finished after {steps} steps")
            # print(f"Total reward: {total_reward:.2f}")
            # print(f"Success: {episode_success}")

            total_rewards.append(total_reward)
            if episode_success:
                success_count += 1

        # Print summary statistics
        print("\n" + "=" * 60)
        print("=== Evaluation Complete ===")
        print(f"Task: {task_name}")
        print(f"Episodes: {num_episodes}")
        print(f"Average reward: {np.mean(total_rewards):.2f}")
        print(f"Std reward: {np.std(total_rewards):.2f}")
        print(f"Min reward: {np.min(total_rewards):.2f}")
        print(f"Max reward: {np.max(total_rewards):.2f}")
        print(f"Success rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
        print("=" * 60)

        # Cleanup
        env.close()
