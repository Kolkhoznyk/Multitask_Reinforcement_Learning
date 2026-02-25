"""
Meta-World MT1 Training Script with Stable Baselines3

Updated to align with the latest Meta-World API:
- Uses official gymnasium.make('Meta-World/MT1', ...) registration
- Supports reward_function_version parameter (v1/v2)
- Configurable max_episode_steps
- Optional reward normalization
- Parallel training with SubprocVecEnv (spawn method)
- Comprehensive evaluation and checkpointing

For documentation, see: METAWORLD_README.md
For hyperparameter tuning guide, see: METAWORLD_TUNING.md
"""

import os
import warnings

import gymnasium as gym
import metaworld
import numpy as np
import torch
from stable_baselines3 import TD3, DDPG, SAC, PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv



def make_env(task_name='reach-v3', rank=0, seed=0, max_episode_steps=500, normalize_reward=False):
    """
    Create and wrap the Meta-World MT1 environment.

    Args:
        task_name: Name of the Meta-World task (e.g., 'reach-v3', 'push-v3')
        rank: Index of the subprocess (for parallel envs)
        seed: Random seed
        max_episode_steps: Maximum steps per episode (default: 500)
        normalize_reward: Whether to normalize rewards (optional, can improve learning)
    """
    def _init():
        # Create Meta-World MT1 environment
        # Note: Meta-World uses v3 suffix (not v2)
        env = gym.make(
            'Meta-World/MT1',
            env_name=task_name,
            seed=seed + rank,  # Different seed for each parallel env
            reward_function_version='v3',  # Use v2 reward (default, more stable)
            max_episode_steps=max_episode_steps,  # Episode length
            terminate_on_success=False,  # Don't terminate early on success (for training)
        )

        # Optional: Normalize rewards for more stable learning
        # Uncomment if experiencing training instability
        # if normalize_reward:
        #     env = gym.wrappers.NormalizeReward(env)

        # Monitor wrapper for logging episode statistics
        # This automatically tracks episode rewards, lengths, and success rates
        env = Monitor(env)

        return env

    return _init


if __name__ == "__main__":
    # ==================== CONFIGURATION ====================
    # Task Selection
    TASK_NAME = "pick-place-v3"  # MT1 tasks: "reach-v3", "push-v3", "pick-place-v3", etc.

    # Algorithm Selection
    ALGORITHM = "SAC"  # "SAC" or "PPO"

    # Environment Settings
    USE_PARALLEL = False  # Set to False for single environment
    N_ENVS = 1 if USE_PARALLEL else 1
    SEED = 42

    # Training Settings
    TOTAL_TIMESTEPS = 1_200_000  # Increased for better convergence
    MAX_EPISODE_STEPS = 150  # Maximum steps per episode
    NORMALIZE_REWARD = False  # Set to True if experiencing training instability

    # Evaluation Settings
    EVAL_FREQ = 15_000  # Evaluate every N steps
    N_EVAL_EPISODES = 20  # Number of episodes for evaluation
    CHECKPOINT_FREQ = 20_000  # Save checkpoint every N steps
    # ======================================================

    # Create output directories
    os.makedirs("./metaworld_models", exist_ok=True)
    os.makedirs("./metaworld_logs", exist_ok=True)

    print(f"=" * 60)
    print(f"Meta-World MT1 Training: {TASK_NAME}")
    print(f"Algorithm: {ALGORITHM}")
    print(f"=" * 60)

    # Create vectorized training environments (parallel)
    if USE_PARALLEL:
        print(f"Creating {N_ENVS} parallel training environments...")
        env = SubprocVecEnv(
            [make_env(TASK_NAME, i, SEED, MAX_EPISODE_STEPS, NORMALIZE_REWARD) for i in range(N_ENVS)],
            start_method='spawn'
        )
    else:
        print("Creating single training environment...")
        env = make_env(TASK_NAME, 0, SEED, MAX_EPISODE_STEPS, NORMALIZE_REWARD)()

    # Create evaluation environment (without reward normalization for accurate eval)
    print("Creating evaluation environment...")
    eval_env = make_env(TASK_NAME, 0, SEED + 1000, MAX_EPISODE_STEPS, normalize_reward=False)()

    # Get action space dimensions
    n_actions = env.action_space.shape[0]

    # Initialize the RL algorithm
    print(f"\nInitializing {ALGORITHM} agent...")

    if ALGORITHM == "SAC":
        # SAC - Recommended for Meta-World (better exploration)
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=2.5e-4,
            buffer_size=300_000,
            learning_starts=10_000,  # Start training sooner
            batch_size=256,
            tau=0.0042,
            gamma=0.95,  # Higher gamma for multi-step tasks
            train_freq=1,
            gradient_steps=1,  # Train on all available data
            ent_coef='auto',  # Automatic entropy tuning - crucial for SAC
            target_entropy='auto',  # Automatically set target entropy
            use_sde=False,  # State-dependent exploration (can be enabled for more exploration)
            policy_kwargs=dict(
                net_arch=[256, 256, 256],  # Deeper network
                activation_fn=torch.nn.ReLU,
                log_std_init=-3,  # Initial exploration level
            ),
            tensorboard_log=f"./metaworld_logs/{TASK_NAME}/",
            verbose=1,
            device="auto",
            seed=SEED,
        )
    else:
        raise ValueError(f"Unknown algorithm: {ALGORITHM}")

    # Callbacks
    # Save checkpoint every CHECKPOINT_FREQ steps
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=f"./metaworld_models/checkpoints_{TASK_NAME}/",
        name_prefix=f"{ALGORITHM.lower()}_{TASK_NAME}",
        verbose=1
    )

    # Evaluate every EVAL_FREQ steps
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./metaworld_models/best_{TASK_NAME}_{ALGORITHM}/",
        log_path=f"./metaworld_logs/eval_{TASK_NAME}/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,  # More episodes for robust evaluation
        deterministic=True,
        render=False,
        verbose=1,
        warn=False
    )

    # Train the agent
    total_timesteps = TOTAL_TIMESTEPS
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print("=" * 60)
    print("Training configuration:")
    print(f"  - Task: {TASK_NAME}")
    print(f"  - Algorithm: {ALGORITHM}")
    print(f"  - Parallel environments: {N_ENVS}")
    print(f"  - Learning rate: {model.learning_rate}")
    print(f"  - Batch size: {model.batch_size}")
    print(f"  - Gamma: {model.gamma}")
    #print(f"  - Learning starts: {model.learning_starts}")
    #print(f"  - Buffer size: {model.buffer_size:,}")
    print(f"  - Network architecture: [256, 256, 256]")
    print(f"  - Gradient steps: -1 (train on all data)")
    print(f"  - Seed: {SEED}")
    print(f"  - Max episode steps: {MAX_EPISODE_STEPS}")
    print(f"  - Reward function: v2 (more stable)")
    print(f"  - Normalize reward: {NORMALIZE_REWARD}")
    print(f"  - Eval frequency: {EVAL_FREQ} steps")
    print(f"  - Eval episodes: {N_EVAL_EPISODES}")
    print(f"  - Checkpoint frequency: {CHECKPOINT_FREQ} steps")
    if ALGORITHM == "TD3":
        print(f"  - Exploration noise: σ=0.1")
        print(f"  - Target policy noise: 0.1 (clip: 0.3)")
    elif ALGORITHM == "SAC":
        print(f"  - Entropy tuning: Automatic")
        print(f"  - Target entropy: Automatic")
    elif ALGORITHM == "PPO":
        print(f"  - On-policy: PPO with GAE")
        print(f"  - n_steps: 512, n_epochs: 10, clip_range: 0.2")
    print("=" * 60)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True
    )

    # Save the final model
    print("\nSaving final model...")
    model.save(f"./metaworld_models/{ALGORITHM.lower()}_{TASK_NAME}_final")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final model saved to: ./metaworld_models/{ALGORITHM.lower()}_{TASK_NAME}_final.zip")
    print(f"Best model saved to: ./metaworld_models/best_{TASK_NAME}/best_model.zip")
    print(f"Checkpoints saved to: ./metaworld_models/checkpoints_{TASK_NAME}/")
    print(f"\nTo monitor training, run: tensorboard --logdir=./metaworld_logs/")
    print("=" * 60)

    # Cleanup
    env.close()
    eval_env.close()
