import optuna
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from MT3_SAC.train_metaworld_sb3_MT3_v2 import make_env
from utils.callbacks import MT10TaskMetricsCallback  # noqa: F401  (re-exported for callers)


class OptunaEvalCallback(BaseCallback):
    """
    Evaluates all eval_freq steps, reported in Optuna and pruned if necessary.
    """
    def __init__(self, eval_env, trial: optuna.Trial, n_eval_episodes: int, eval_freq: int, deterministic: bool = True):
        super().__init__()
        self.eval_env = eval_env
        self.trial = trial
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward, _ = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                return_episode_rewards=False,
            )
            # report an Optuna
            self.trial.report(float(np.mean(mean_reward)), step=self.n_calls)

            # # prune?
            # if self.trial.should_prune():
            #     return False

        return True


class MultiTaskEvalCallback(BaseCallback):
    """
    Evaluates the current policy on every unique training task at a fixed frequency
    and saves the best model based on the mean success rate across all tasks.
    """
    def __init__(self, unique_tasks, n_eval_episodes=10, eval_freq=10000, save_path=None, seed=42, max_steps=500, terminate_on_success=True):
        super().__init__()
        self.unique_tasks = unique_tasks
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_sr = -np.inf
        self.save_path = save_path
        self.seed = seed
        self.max_steps = max_steps
        self.terminate_on_success = terminate_on_success

    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0:
            all_tasks_rewards = []
            all_tasks_success_rates = []
            n_tasks = len(self.unique_tasks)

            for i, task_name in enumerate(self.unique_tasks):
                eval_env_fn = make_env(task_name, i, n_tasks, 1.0, 999, self.seed, self.max_steps, terminate_on_success=self.terminate_on_success)
                env = eval_env_fn()

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
                            if info[0].get('success', 0) == 1: # type: ignore
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
                
                env.close()

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