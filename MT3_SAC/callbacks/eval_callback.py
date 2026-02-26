import optuna
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from MT3_SAC.train_metaworld_sb3_MT3_v2 import make_env

class OptunaEvalCallback(BaseCallback):
    """
    Evaluiert alle eval_freq steps, reported an Optuna und pruned ggf.
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
    
class MT10TaskMetricsCallback(BaseCallback):
    """
    Log episode return and success rate per Meta-World MT10 task_id.
    Works with DummyVecEnv and SubprocVecEnv using `infos` from env.step().

    Requirements:
      - info["mt_task_id"] each step (you provide it)
      - optionally info["success"] in {0.0, 1.0}
    """

    def __init__(self, num_tasks: int = 10, verbose: int = 0, max_hist: int = 100):
        super().__init__(verbose)
        self.num_tasks = int(num_tasks)
        self.max_hist = int(max_hist)

        # Per-env episode accumulators
        self._ep_rew = None           # (n_envs,)
        self._last_task_id = None     # (n_envs,)
        self._ep_success_any = None   # (n_envs,) bool

        # Per-task history buffers
        self._task_returns = [[] for _ in range(self.num_tasks)]
        self._task_success = [[] for _ in range(self.num_tasks)]

        # sampling stats window
        self.sample_window_steps = 10_000  # VecEnv-steps (not transitions)
        self._sample_counts = None
        self._sample_total = 0
        self._window_vecenv_steps = 0

    def _init_callback(self) -> None:
        n_envs = self.training_env.num_envs
        self._ep_rew = np.zeros((n_envs,), dtype=np.float64)
        self._last_task_id = -np.ones((n_envs,), dtype=np.int64)
        self._ep_success_any = np.zeros((n_envs,), dtype=bool)

        # sampling stats window
        self._sample_counts = np.zeros((self.num_tasks,), dtype=np.int64)
        self._sample_total = 0
        self._window_vecenv_steps = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        dones = self.locals.get("dones", None)
        rewards = self.locals.get("rewards", None)

        if infos is None or dones is None or rewards is None:
            return True

        # accumulate per env
        for env_idx, (info, done, r) in enumerate(zip(infos, dones, rewards)):
            self._ep_rew[env_idx] += float(r) # type: ignore

            tid = info.get("mt_task_id", None)
            if tid is not None:
                self._last_task_id[env_idx] = int(tid) # type: ignore

            if "success" in info:
                if float(info["success"]) >= 1.0:
                    self._ep_success_any[env_idx] = True # type: ignore

            if done:
                task_id = int(self._last_task_id[env_idx]) # type: ignore
                if 0 <= task_id < self.num_tasks:
                    # store episode return
                    self._task_returns[task_id].append(float(self._ep_rew[env_idx])) # type: ignore
                    if len(self._task_returns[task_id]) > self.max_hist:
                        self._task_returns[task_id] = self._task_returns[task_id][-self.max_hist:]

                    # store episode success if available
                    if "success" in info:
                        self._task_success[task_id].append(float(self._ep_success_any[env_idx])) # type: ignore
                        if len(self._task_success[task_id]) > self.max_hist:
                            self._task_success[task_id] = self._task_success[task_id][-self.max_hist:]

                # reset env accumulators
                self._ep_rew[env_idx] = 0.0 # type: ignore
                self._ep_success_any[env_idx] = False # type: ignore
                self._last_task_id[env_idx] = -1 # type: ignore

        # Sampling stats
        # One VecEnv step contains n_envs transitions
        # Count how many transitions per task were sampled in the last window
        self._window_vecenv_steps += 1

        for env_idx, (info, done, r) in enumerate(zip(infos, dones, rewards)):
            self._ep_rew[env_idx] += float(r) # type: ignore

            task_id = info.get("mt_task_id", None)
            if task_id is not None:
                tid = int(task_id)
                self._last_task_id[env_idx] = tid # type: ignore

                # Count this transition towards sampling distribution
                if 0 <= tid < self.num_tasks:
                    self._sample_counts[tid] += 1 # type: ignore
                    self._sample_total += 1

            if "success" in info:
                self._ep_success_any[env_idx] = self._ep_success_any[env_idx] or (float(info["success"]) >= 1.0) # type: ignore

            if done:
                tid = int(self._last_task_id[env_idx]) # type: ignore
                if 0 <= tid < self.num_tasks:
                    self._task_returns[tid].append(float(self._ep_rew[env_idx])) # type: ignore
                    if "success" in info:
                        self._task_success[tid].append(float(self._ep_success_any[env_idx])) # type: ignore

                self._ep_rew[env_idx] = 0.0 # type: ignore
                self._ep_success_any[env_idx] = False # type: ignore

        # log per-task means + mean across tasks (only tasks with data)
        task_reward_means = []
        task_success_means = []

        for k in range(self.num_tasks):
            if len(self._task_returns[k]) > 0:
                rew_mean = float(np.mean(self._task_returns[k]))
                self.logger.record(f"task/ep_rew_mean_task_{k}", rew_mean)
                task_reward_means.append(rew_mean)

            if len(self._task_success[k]) > 0:
                succ_mean = float(np.mean(self._task_success[k]))
                self.logger.record(f"task/ep_success_rate_task_{k}", succ_mean)
                task_success_means.append(succ_mean)

        if len(task_reward_means) > 0:
            self.logger.record("task/ep_rew_mean_mean", float(np.mean(task_reward_means)))

        if len(task_success_means) > 0:
            self.logger.record("task/ep_success_rate_mean", float(np.mean(task_success_means)))

    # Log sampling fractions every window steps
        # sample_window_steps counts VecEnv-steps, not transitions
        if self._window_vecenv_steps >= self.sample_window_steps and self._sample_total > 0 and self._sample_counts is not None:
            fracs = self._sample_counts.astype(np.float64) / float(self._sample_total)
            for k in range(self.num_tasks):
                self.logger.record(f"task/sample_frac_task_{k}", float(fracs[k]))

            # One scalar to summarize deviation from uniform
            uniform = 1.0 / float(self.num_tasks)
            mad = float(np.mean(np.abs(fracs - uniform)))
            self.logger.record("task/sample_frac_mean_abs_dev", mad)

            # reset window
            self._sample_counts[:] = 0 # type: ignore
            self._sample_total = 0
            self._window_vecenv_steps = 0

        return True
    

class MultiTaskEvalCallback(BaseCallback):
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