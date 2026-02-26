import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from MT10_SAC.algos.sac_disentangled_alpha import SACDisentangledAlpha
from get_data_from_checkpoints import SingleTaskOneHotWrapper
import metaworld  # registers Meta-World namespace with gymnasium in each subprocess

_ACTIVATION_FNS = {
    "ReLU": torch.nn.ReLU,
    "Tanh": torch.nn.Tanh,
    "ELU": torch.nn.ELU,
}

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def resolve_policy_kwargs(policy_kwargs: dict) -> dict:
    kwargs = dict(policy_kwargs)
    if "activation_fn" in kwargs:
        kwargs["activation_fn"] = _ACTIVATION_FNS[kwargs["activation_fn"]]
    if "net_arch" in kwargs:
        kwargs["net_arch"] = list(kwargs["net_arch"])
    return kwargs
       
def make_env(task_name, task_id, n_tasks, rew_scale, rank, seed, max_steps, terminate_on_success=False):
    def _init():      
        env = gym.make(
            'Meta-World/MT1',
            env_name=task_name,
            seed=seed + rank,
            reward_function_version='v3',
            max_episode_steps=max_steps,
            terminate_on_success=terminate_on_success
        )
        env = Monitor(env)
        return SingleTaskOneHotWrapper(env, task_id, n_tasks, rew_scale)
    return _init

class MultiTaskEvalCallback(BaseCallback):
    def __init__(self, unique_tasks, n_eval_episodes=10, eval_freq=10000, save_path=None, seed=42, max_steps=500, terminate_on_success=True):
        super().__init__()
        self.unique_tasks = unique_tasks
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.next_eval = eval_freq
        self.best_mean_sr = -np.inf
        self.save_path = save_path
        self.seed = seed
        self.max_steps = max_steps
        self.terminate_on_success = terminate_on_success

        self.eval_envs: list = []

    def _on_training_start(self):
        print("!Create Evaluation Environments!")
        n_tasks = len(self.unique_tasks)
        self.eval_envs = []
        for i, task_name in enumerate(self.unique_tasks):
            env = make_env(task_name, i, n_tasks, 1.0, 999, self.seed, self.max_steps,
                           terminate_on_success=self.terminate_on_success)()
            self.eval_envs.append(env)

    def _on_training_end(self):
        if self.eval_envs is not None:
            for env in self.eval_envs:
                env.close()

    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.eval_freq
            all_tasks_rewards = []
            all_tasks_success_rates = []
            # n_tasks = len(self.unique_tasks)

            for _, (task_name, env) in enumerate(zip(self.unique_tasks, self.eval_envs)):

                ep_rews, ep_lens, successes = [], [], []

                for _ in range(self.n_eval_episodes):
                    obs, _ = env.reset()
                    done = False
                    total_reward = 0.0
                    length = 0
                    success = 0

                    while not done:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated
                        total_reward += reward
                        length += 1
                        
                        if isinstance(info, list):
                            if info[0].get('success', 0) == 1:
                                success = 1
                        else:
                            if info.get('success', 0) == 1:
                                success = 1

                    ep_rews.append(total_reward)
                    ep_lens.append(length)
                    successes.append(success)

                mean_rew = np.mean(ep_rews)
                mean_len = np.mean(ep_lens)
                mean_success = np.mean(successes)
                all_tasks_rewards.append(mean_rew)
                all_tasks_success_rates.append(mean_success)

                print(f"Task: {task_name}")
                print("-" * 30)
                print("| rollout/           |          |")
                print(f"|    ep_len_mean     | {mean_len:.0f}      |")
                print(f"|    ep_rew_mean     | {mean_rew:.2f}    |")
                print(f"|    success_mean    | {mean_success:.2f}    |")
                print("-" * 30)

                self.logger.record(f"eval/{task_name}/ep_rew_mean", mean_rew)
                self.logger.record(f"eval/{task_name}/ep_len_mean", mean_len)
                self.logger.record(f"eval/{task_name}/success_rate", mean_success)
                

            mean_all_tasks_success_rates = np.mean(all_tasks_success_rates)
            mean_all_tasks_rewards = np.mean(all_tasks_rewards)

            print(f"|    ep_rew_mean of all tasks     | {mean_all_tasks_rewards:.2f}    |")
            print(f"|    success_mean of all tasks    | {mean_all_tasks_success_rates:.2f}    |")

            self.logger.record("eval/mean_ep_rew_all_tasks", mean_all_tasks_rewards)
            self.logger.record("eval/mean_success_rates_all_tasks", mean_all_tasks_success_rates)

            if self.save_path is not None and mean_all_tasks_success_rates > self.best_mean_sr:
                self.best_mean_sr = mean_all_tasks_success_rates
                self.model.save(self.save_path) 
                print(f"New best model saved with mean success rate over all tasks = {mean_all_tasks_success_rates:.2f}")
                print(f"Mean ep_rew of all tasks = {mean_all_tasks_rewards:.2f}")

            self.logger.dump(self.num_timesteps)
            
        return True

if __name__ == "__main__":
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(os.path.join(_script_dir, "config_MT3.yaml"))

    TRAINING_TASKS = cfg["tasks"]["training"]
    REWARD_SCALES  = cfg["tasks"]["reward_scales"]
    UNIQUE_TASKS   = list(dict.fromkeys(TRAINING_TASKS))

    exp             = cfg["experiment"]
    ALGORITHM       = exp["algorithm"]
    SEED            = exp["seed"]
    TOTAL_TIMESTEPS = exp["total_timesteps"]
    MAX_EPISODE_STEPS = exp["max_episode_steps"]

    EVAL_FREQ       = cfg["eval"]["freq"]
    N_EVAL_EPISODES = cfg["eval"]["n_episodes"]
    EVAL_TERMINATE  = cfg["eval"]["terminate_on_success"]
    CHECKPOINT_FREQ = cfg["checkpoint"]["freq"]

    paths = cfg["paths"]

    print(TRAINING_TASKS)

    os.makedirs(paths["models"], exist_ok=True)
    os.makedirs(paths["logs"], exist_ok=True)

    print(f"Creating {len(TRAINING_TASKS)} parallel environments...")
    env_fns = [
        make_env(name, UNIQUE_TASKS.index(name), len(UNIQUE_TASKS), REWARD_SCALES[name], i, SEED, MAX_EPISODE_STEPS, terminate_on_success=False)
        for i, name in enumerate(TRAINING_TASKS)
    ]
    env = SubprocVecEnv(env_fns, start_method='spawn') # type: ignore

    if ALGORITHM in ("SAC", "SAC_DA"):
        sac_cfg = cfg["sac"]
        policy_kwargs = resolve_policy_kwargs(sac_cfg["policy_kwargs"])
        sac_params = dict(
            policy=sac_cfg["policy"],
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
            policy_kwargs=policy_kwargs,
            tensorboard_log=f"{paths['logs']}/{ALGORITHM}/",
            verbose=1,
            device="auto",
            seed=SEED,
        )
        if ALGORITHM == "SAC":
            model = SAC(**sac_params)  # type: ignore[arg-type]
        else:
            model = SACDisentangledAlpha(num_tasks=len(UNIQUE_TASKS), **sac_params)  # type: ignore[arg-type]

    elif ALGORITHM == "PPO":
        ppo_cfg = cfg["ppo"]
        policy_kwargs = resolve_policy_kwargs(ppo_cfg["policy_kwargs"])
        model = PPO(
            policy=ppo_cfg["policy"],
            env=env,
            learning_rate=ppo_cfg["learning_rate"],
            batch_size=ppo_cfg["batch_size"],
            gamma=ppo_cfg["gamma"],
            gae_lambda=ppo_cfg["gae_lambda"],
            clip_range=ppo_cfg["clip_range"],
            ent_coef=ppo_cfg["ent_coef"],
            vf_coef=ppo_cfg["vf_coef"],
            policy_kwargs=policy_kwargs,
            tensorboard_log=f"{paths['logs']}/{ALGORITHM}/",
            verbose=1,
            device="auto",
            seed=SEED,
        )
    else:
        raise ValueError(f"Unknown algorithm: {ALGORITHM}")

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=paths["checkpoints"],
        name_prefix=f"{ALGORITHM.lower()}_MT3",
        verbose=1
    )

    multi_eval_cb = MultiTaskEvalCallback(
        UNIQUE_TASKS,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=EVAL_FREQ,
        save_path=paths["best_model"],
        seed=SEED,
        max_steps=MAX_EPISODE_STEPS,
        terminate_on_success=EVAL_TERMINATE,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, multi_eval_cb],
        log_interval=10,
        progress_bar=True,
    )

    model.save(f"{paths['models']}/{ALGORITHM.lower()}_MT3_final")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final model saved to: {paths['models']}/{ALGORITHM.lower()}_MT3_final.zip")
    print(f"Best model saved to: {paths['best_model']}")
    print(f"Checkpoints saved to: {paths['checkpoints']}")
    print(f"\nTo monitor training, run: tensorboard --logdir={paths['logs']}/")
    print("=" * 60)

    env.close()