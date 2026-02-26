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
import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3, SAC, DDPG, PPO

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
    # Configuration
    TASK_NAMES = ["reach-v3", "push-v3", "pick-place-v3"]  # Must match the task used for training (v3, not v2!)
    TASK_ID = {"reach-v3": 0, "push-v3": 1, "pick-place-v3": 2}
    ALGORITHM = "SAC"  # SAC or PPO
    SEED = 40
    MAX_EPISODE_STEPS = 150  # Must match training configuration
    num_episodes = 20

    for task_name in TASK_NAMES:
        # Create environment with rendering
        print(f"Creating {task_name} environment...")
        env = gym.make(
            'Meta-World/MT1',
            env_name=task_name,
            seed=SEED,
            render_mode='human', # Enable visual rendering
            reward_function_version='v3',  # Use v2 reward (same as training)
            max_episode_steps=MAX_EPISODE_STEPS,  # Episode length
            terminate_on_success=False,  # Don't terminate early (for consistent evaluation)
        )
        env = SingleTaskOneHotWrapper(env=env, task_id=TASK_ID[task_name], n_tasks=3)

        # Load the trained model
        # model_path = f"./metaworld_models/best_{TASK_NAME}_{ALGORITHM}/best_model.zip"
        model_path = f"./metaworld_models/best_MT3_model.zip"

        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            print("Trying final model instead...")
            model_path = f"./metaworld_models/{ALGORITHM.lower()}_{task_name}_final.zip"

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
