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
import sys
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_script_dir)
sys.path.insert(0, _script_dir)
sys.path.insert(0, _root_dir)

import gymnasium as gym
import metaworld  # registers Meta-World namespace with gymnasium
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from utils.config import load_config, resolve_policy_kwargs



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
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(os.path.join(_script_dir, "config_ST.yaml"))

    exp               = cfg["experiment"]
    TASK_NAME         = exp["task_name"]
    ALGORITHM         = exp["algorithm"]
    SEED              = exp["seed"]
    TOTAL_TIMESTEPS   = exp["total_timesteps"]
    MAX_EPISODE_STEPS = exp["max_episode_steps"]
    NORMALIZE_REWARD  = exp["normalize_reward"]
    USE_PARALLEL      = exp["use_parallel"]
    N_ENVS            = exp["n_envs"] if USE_PARALLEL else 1

    EVAL_FREQ       = cfg["eval"]["freq"]
    N_EVAL_EPISODES = cfg["eval"]["n_episodes"]
    CHECKPOINT_FREQ = cfg["checkpoint"]["freq"]
    paths           = cfg["paths"]

    # Create output directories
    os.makedirs(paths["models"], exist_ok=True)
    os.makedirs(paths["logs"], exist_ok=True)

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
    assert env.action_space.shape is not None, "Action space shape is None"
    n_actions = env.action_space.shape[0]

    # Initialize the RL algorithm
    print(f"\nInitializing {ALGORITHM} agent...")

    if ALGORITHM == "SAC":
        sac_cfg = cfg["sac"]
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=sac_cfg["learning_rate"],
            buffer_size=sac_cfg["buffer_size"],
            learning_starts=sac_cfg["learning_starts"],
            batch_size=sac_cfg["batch_size"],
            tau=sac_cfg["tau"],
            gamma=sac_cfg["gamma"],
            train_freq=sac_cfg["train_freq"],
            gradient_steps=sac_cfg["gradient_steps"],
            ent_coef=sac_cfg["ent_coef"],
            target_entropy=sac_cfg["target_entropy"],
            use_sde=sac_cfg["use_sde"],
            policy_kwargs=resolve_policy_kwargs(sac_cfg["policy_kwargs"]),
            tensorboard_log=f"{paths['logs']}/{TASK_NAME}/",
            verbose=1,
            device="auto",
            seed=SEED,
        )
    else:
        raise ValueError(f"Unknown algorithm: {ALGORITHM}")

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=f"{paths['models']}/checkpoints_{TASK_NAME}/",
        name_prefix=f"{ALGORITHM.lower()}_{TASK_NAME}",
        verbose=1
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{paths['models']}/best_{TASK_NAME}_{ALGORITHM}/",
        log_path=f"{paths['logs']}/eval_{TASK_NAME}/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
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
    model.save(f"{paths['models']}/{ALGORITHM.lower()}_{TASK_NAME}_final")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final model saved to: {paths['models']}/{ALGORITHM.lower()}_{TASK_NAME}_final.zip")
    print(f"Best model saved to: {paths['models']}/best_{TASK_NAME}_{ALGORITHM}/best_model.zip")
    print(f"Checkpoints saved to: {paths['models']}/checkpoints_{TASK_NAME}/")
    print(f"\nTo monitor training, run: tensorboard --logdir={paths['logs']}/")
    print("=" * 60)

    # Cleanup
    env.close()
    eval_env.close()
