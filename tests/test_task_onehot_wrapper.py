"""
Tests for MT10_SAC/metaworld_envs/task_onehot_wrapper.py

Covers:
  - Observation-space extension on construction
  - One-hot correctness (sum, hot index, cross-reset update)
  - Info dict enrichment
  - Error cases (missing attributes, invalid task_id, non-Box space)
  - Step-level behaviour (shape, one-hot stability)
"""
import numpy as np
import pytest
import gymnasium as gym

from MT10_SAC.metaworld_envs.task_onehot_wrapper import TaskOneHotObsWrapper


# ── Fake environments ─────────────────────────────────────────────────────────

class _FakeMultiTaskEnv(gym.Env):
    """Minimal gym env that mimics the MetaWorldMT10Env interface."""

    def __init__(self, obs_dim: int = 5, num_tasks: int = 3, initial_task_id: int = 0):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_tasks = num_tasks
        self.current_task_id = initial_task_id
        self.observation_space = gym.spaces.Box(
            low=-np.ones(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        return np.zeros(self.obs_dim, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self.obs_dim, dtype=np.float32), 0.0, False, False, {}


class _FakeEnvNoNumTasks(gym.Env):
    """Env missing the required ``num_tasks`` attribute."""

    def __init__(self):
        super().__init__()
        self.current_task_id = 0
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def reset(self, **kwargs):
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(4, dtype=np.float32), 0.0, False, False, {}


class _FakeEnvNoCurrentTaskId(gym.Env):
    """Env missing the required ``current_task_id`` attribute."""

    def __init__(self):
        super().__init__()
        self.num_tasks = 3
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def reset(self, **kwargs):
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(4, dtype=np.float32), 0.0, False, False, {}


class _FakeEnvDiscreteObs(gym.Env):
    """Env with a Discrete (non-Box) observation space."""

    def __init__(self):
        super().__init__()
        self.num_tasks = 3
        self.current_task_id = 0
        self.observation_space = gym.spaces.Discrete(8)
        self.action_space = gym.spaces.Discrete(4)

    def reset(self, **kwargs):
        return 0, {}

    def step(self, action):
        return 0, 0.0, False, False, {}


class _FakeEnvInvalidTaskId(_FakeMultiTaskEnv):
    """Env whose ``current_task_id`` is out of [0, num_tasks) on reset."""

    def __init__(self):
        super().__init__(obs_dim=4, num_tasks=3, initial_task_id=99)


# ── Construction tests ────────────────────────────────────────────────────────

class TestTaskOneHotObsWrapperInit:

    def test_obs_space_extended_by_num_tasks(self):
        """Wrapper must grow the observation space by exactly num_tasks dimensions."""
        env = _FakeMultiTaskEnv(obs_dim=5, num_tasks=3)
        wrapped = TaskOneHotObsWrapper(env)
        assert wrapped.observation_space.shape == (8,)

    def test_obs_space_onehot_bounds(self):
        """The appended one-hot dimensions must have low=0 and high=1."""
        env = _FakeMultiTaskEnv(obs_dim=5, num_tasks=4)
        wrapped = TaskOneHotObsWrapper(env)
        assert np.all(wrapped.observation_space.low[-4:] == 0.0) # type: ignore
        assert np.all(wrapped.observation_space.high[-4:] == 1.0) # type: ignore

    def test_raises_without_num_tasks_attr(self):
        """Missing ``num_tasks`` on the inner env must raise AssertionError."""
        with pytest.raises(AssertionError, match="num_tasks"):
            TaskOneHotObsWrapper(_FakeEnvNoNumTasks())

    def test_raises_without_current_task_id_attr(self):
        """Missing ``current_task_id`` on the inner env must raise AssertionError."""
        with pytest.raises(AssertionError, match="current_task_id"):
            TaskOneHotObsWrapper(_FakeEnvNoCurrentTaskId())

    def test_raises_for_non_box_observation_space(self):
        """Wrapping an env with a Discrete obs space must raise AssertionError."""
        with pytest.raises(AssertionError):
            TaskOneHotObsWrapper(_FakeEnvDiscreteObs())


# ── Reset tests ───────────────────────────────────────────────────────────────

class TestTaskOneHotObsWrapperReset:

    def test_reset_obs_shape(self):
        """reset() must return an observation of length obs_dim + num_tasks."""
        env = _FakeMultiTaskEnv(obs_dim=5, num_tasks=3)
        wrapped = TaskOneHotObsWrapper(env)
        obs, _ = wrapped.reset()
        assert obs.shape == (8,)

    def test_onehot_sums_to_one(self):
        """The one-hot suffix must sum to exactly 1.0 after reset."""
        env = _FakeMultiTaskEnv(obs_dim=5, num_tasks=3, initial_task_id=1)
        wrapped = TaskOneHotObsWrapper(env)
        obs, _ = wrapped.reset()
        assert float(obs[-3:].sum()) == pytest.approx(1.0)

    @pytest.mark.parametrize("task_id", [0, 1, 2])
    def test_hot_index_matches_task_id(self, task_id):
        """The index of the '1' in the one-hot must equal current_task_id."""
        env = _FakeMultiTaskEnv(obs_dim=5, num_tasks=3, initial_task_id=task_id)
        wrapped = TaskOneHotObsWrapper(env)
        obs, _ = wrapped.reset()
        assert int(np.argmax(obs[-3:])) == task_id

    def test_onehot_updates_on_new_task(self):
        """If current_task_id changes between resets, the one-hot must reflect it."""
        env = _FakeMultiTaskEnv(obs_dim=5, num_tasks=3, initial_task_id=0)
        wrapped = TaskOneHotObsWrapper(env)

        obs0, _ = wrapped.reset()
        assert np.argmax(obs0[-3:]) == 0

        env.current_task_id = 2
        obs2, _ = wrapped.reset()
        assert np.argmax(obs2[-3:]) == 2

    def test_reset_info_contains_task_onehot_id(self):
        """reset() info dict must include the key ``task_onehot_id``."""
        env = _FakeMultiTaskEnv(obs_dim=5, num_tasks=3, initial_task_id=1)
        wrapped = TaskOneHotObsWrapper(env)
        _, info = wrapped.reset()
        assert "task_onehot_id" in info
        assert info["task_onehot_id"] == 1

    def test_raises_on_out_of_range_task_id(self):
        """reset() must raise ValueError when task_id >= num_tasks."""
        env = _FakeEnvInvalidTaskId()
        wrapped = TaskOneHotObsWrapper(env)
        with pytest.raises(ValueError, match="Invalid task_id"):
            wrapped.reset()


# ── Step tests ────────────────────────────────────────────────────────────────

class TestTaskOneHotObsWrapperStep:

    def test_step_obs_shape(self):
        """step() must return an observation of length obs_dim + num_tasks."""
        env = _FakeMultiTaskEnv(obs_dim=5, num_tasks=3, initial_task_id=0)
        wrapped = TaskOneHotObsWrapper(env)
        wrapped.reset()
        obs, _, _, _, _ = wrapped.step(env.action_space.sample())
        assert obs.shape == (8,)

    def test_step_preserves_onehot_from_reset(self):
        """The one-hot appended during step() must match what was set in reset()."""
        env = _FakeMultiTaskEnv(obs_dim=5, num_tasks=3, initial_task_id=2)
        wrapped = TaskOneHotObsWrapper(env)
        wrapped.reset()
        obs, _, _, _, _ = wrapped.step(env.action_space.sample())
        assert int(np.argmax(obs[-3:])) == 2
