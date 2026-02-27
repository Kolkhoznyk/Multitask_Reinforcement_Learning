import numpy as np
import gymnasium as gym


class SingleTaskOneHotWrapper(gym.Wrapper):
    """
    Gym wrapper that appends a one-hot task encoding to every observation and
    optionally scales the reward. Used to condition a shared policy on the
    current task identity without modifying the underlying environment.

    Args:
        env: The environment to wrap.
        task_id: Zero-based index of this task in the task set.
        n_tasks: Total number of tasks (length of the one-hot vector).
        reward_scale: Scalar multiplied to every reward (default: 1.0).
    """

    def __init__(self, env: gym.Env, task_id: int, n_tasks: int, reward_scale: float = 1.0):
        super().__init__(env)
        self.task_id = task_id
        self.n_tasks = n_tasks
        self.reward_scale = reward_scale

        obs_space = self.env.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([obs_space.low, np.zeros(self.n_tasks)]),   # type: ignore[arg-type]
            high=np.concatenate([obs_space.high, np.ones(self.n_tasks)]),  # type: ignore[arg-type]
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._one_hot_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._one_hot_obs(obs), reward * self.reward_scale, terminated, truncated, info  # type: ignore[return-value]

    def _one_hot_obs(self, obs: np.ndarray) -> np.ndarray:
        one_hot = np.zeros(self.n_tasks, dtype=np.float32)
        one_hot[self.task_id] = 1.0
        return np.concatenate([np.array(obs, dtype=np.float32), one_hot])
