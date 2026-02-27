import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MT10TaskMetricsCallback(BaseCallback):
    """
    Log episode return and success rate per Meta-World task_id during training.

    Works with DummyVecEnv and SubprocVecEnv by reading ``infos`` from each
    VecEnv step. Sampling distribution statistics (fraction of transitions per
    task) are reported every *sample_window_steps* VecEnv steps.

    Requirements:
      - ``info["mt_task_id"]`` present each step.
      - ``info["success"]`` in {0.0, 1.0} (optional).

    Args:
        num_tasks: Total number of tasks.
        verbose: SB3 verbosity level.
        max_hist: Maximum number of recent episodes to keep per task for
            computing rolling statistics.
    """

    def __init__(self, num_tasks: int = 10, verbose: int = 0, max_hist: int = 100):
        super().__init__(verbose)
        self.num_tasks = int(num_tasks)
        self.max_hist = int(max_hist)

        # Per-env episode accumulators (initialised in _init_callback)
        self._ep_rew = None           # (n_envs,)
        self._last_task_id = None     # (n_envs,)
        self._ep_success_any = None   # (n_envs,) bool

        # Per-task rolling history buffers
        self._task_returns: list = [[] for _ in range(self.num_tasks)]
        self._task_success: list = [[] for _ in range(self.num_tasks)]

        # Sampling-distribution window
        self.sample_window_steps = 10_000  # VecEnv steps (not individual transitions)
        self._sample_counts = None
        self._sample_total = 0
        self._window_vecenv_steps = 0

    def _init_callback(self) -> None:
        n_envs = self.training_env.num_envs
        self._ep_rew = np.zeros((n_envs,), dtype=np.float64)
        self._last_task_id = -np.ones((n_envs,), dtype=np.int64)
        self._ep_success_any = np.zeros((n_envs,), dtype=bool)

        self._sample_counts = np.zeros((self.num_tasks,), dtype=np.int64)
        self._sample_total = 0
        self._window_vecenv_steps = 0

    def _on_step(self) -> bool:
        """
        Called after each VecEnv step. Accumulates per-env episode returns and
        success flags, flushes completed episodes into per-task history buffers,
        and logs per-task mean reward, success rate, and task sampling fractions
        to TensorBoard. Sampling distribution statistics are reported every
        *sample_window_steps* VecEnv steps.
        """
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        rewards = self.locals.get("rewards")

        if infos is None or dones is None or rewards is None:
            return True

        # --- Accumulate episode returns and success flags ---
        for env_idx, (info, done, r) in enumerate(zip(infos, dones, rewards)):
            self._ep_rew[env_idx] += float(r)  # type: ignore[index]

            tid = info.get("mt_task_id")
            if tid is not None:
                self._last_task_id[env_idx] = int(tid)  # type: ignore[index]

            if "success" in info and float(info["success"]) >= 1.0:
                self._ep_success_any[env_idx] = True  # type: ignore[index]

            if done:
                task_id = int(self._last_task_id[env_idx])  # type: ignore[index]
                if 0 <= task_id < self.num_tasks:
                    self._task_returns[task_id].append(float(self._ep_rew[env_idx]))  # type: ignore[index]
                    if len(self._task_returns[task_id]) > self.max_hist:
                        self._task_returns[task_id] = self._task_returns[task_id][-self.max_hist:]

                    if "success" in info:
                        self._task_success[task_id].append(float(self._ep_success_any[env_idx]))  # type: ignore[index]
                        if len(self._task_success[task_id]) > self.max_hist:
                            self._task_success[task_id] = self._task_success[task_id][-self.max_hist:]

                self._ep_rew[env_idx] = 0.0        # type: ignore[index]
                self._ep_success_any[env_idx] = False  # type: ignore[index]
                self._last_task_id[env_idx] = -1   # type: ignore[index]

        # --- Update sampling-distribution window ---
        self._window_vecenv_steps += 1

        for env_idx, (info, _done, _r) in enumerate(zip(infos, dones, rewards)):
            tid = info.get("mt_task_id")
            if tid is not None:
                tid = int(tid)
                if 0 <= tid < self.num_tasks:
                    self._sample_counts[tid] += 1  # type: ignore[index]
                    self._sample_total += 1

        # --- Log per-task metrics ---
        task_reward_means = []
        task_success_means = []

        for k in range(self.num_tasks):
            if self._task_returns[k]:
                rew_mean = float(np.mean(self._task_returns[k]))
                self.logger.record(f"task/ep_rew_mean_task_{k}", rew_mean)
                task_reward_means.append(rew_mean)

            if self._task_success[k]:
                succ_mean = float(np.mean(self._task_success[k]))
                self.logger.record(f"task/ep_success_rate_task_{k}", succ_mean)
                task_success_means.append(succ_mean)

        if task_reward_means:
            self.logger.record("task/ep_rew_mean_mean", float(np.mean(task_reward_means)))
        if task_success_means:
            self.logger.record("task/ep_success_rate_mean", float(np.mean(task_success_means)))

        # --- Log sampling fractions once per window ---
        if (
            self._window_vecenv_steps >= self.sample_window_steps
            and self._sample_total > 0
            and self._sample_counts is not None
        ):
            fracs = self._sample_counts.astype(np.float64) / float(self._sample_total)  # type: ignore[union-attr]
            for k in range(self.num_tasks):
                self.logger.record(f"task/sample_frac_task_{k}", float(fracs[k]))

            uniform = 1.0 / float(self.num_tasks)
            mad = float(np.mean(np.abs(fracs - uniform)))
            self.logger.record("task/sample_frac_mean_abs_dev", mad)

            self._sample_counts[:] = 0  # type: ignore[index]
            self._sample_total = 0
            self._window_vecenv_steps = 0

        return True
