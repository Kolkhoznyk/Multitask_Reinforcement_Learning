"""
Tests for MT10_SAC/metaworld_envs/mt10_env.py

MetaWorldMT10Env wraps Mujoco-based Meta-World environments, so every test
patches ``metaworld.MT10`` with a lightweight mock to avoid requiring a GPU or
full Meta-World installation in CI.  The mock faithfully reproduces the interface
that MetaWorldMT10Env depends on.

Covers:
  - Deterministic task-id mapping (sorted env names)
  - reset(): valid current_task_id, required info keys, elapsed-step reset
  - step(): elapsed-step increment, max-episode truncation, terminate-on-success,
            required info keys
  - close(): graceful cleanup with multiple cached envs
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import gymnasium as gym


# ── Constants ─────────────────────────────────────────────────────────────────

OBS_DIM = 9
ACTION_DIM = 4
TASK_NAMES = ["task-c-v2", "task-a-v2", "task-b-v2"]   # intentionally unsorted


# ── Mock helpers ──────────────────────────────────────────────────────────────

def _make_mock_inner_env(success_val: float = 0.0) -> MagicMock:
    """Return a mock that quacks like a Meta-World Mujoco env."""
    mock = MagicMock()
    mock.observation_space = gym.spaces.Box(
        low=-np.ones(OBS_DIM, dtype=np.float32),
        high=np.ones(OBS_DIM, dtype=np.float32),
        dtype=np.float32,
    )
    mock.action_space = gym.spaces.Box(
        low=-np.ones(ACTION_DIM, dtype=np.float32),
        high=np.ones(ACTION_DIM, dtype=np.float32),
        dtype=np.float32,
    )
    mock.reset.return_value = (np.zeros(OBS_DIM, dtype=np.float32), {})
    mock.step.return_value = (
        np.zeros(OBS_DIM, dtype=np.float32),
        1.0,
        False,
        False,
        {"success": success_val},
    )
    return mock


def _build_mock_benchmark(task_names: list, success_val: float = 0.0) -> MagicMock:
    """
    Build a mock ``metaworld.MT10()`` object.

    Each entry in ``train_classes`` is a callable (mock class) that always
    returns the same mock env instance so space-consistency checks pass.
    """
    benchmark = MagicMock()

    classes = {}
    for name in task_names:
        instance = _make_mock_inner_env(success_val=success_val)
        env_class = MagicMock(return_value=instance)
        classes[name] = env_class

    tasks = []
    for name in task_names:
        task = MagicMock()
        task.env_name = name
        tasks.append(task)

    benchmark.train_classes = classes
    benchmark.train_tasks = tasks
    benchmark.test_classes = {}
    benchmark.test_tasks = []
    return benchmark


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def env():
    """
    A fresh MetaWorldMT10Env backed by mock inner envs.
    max_episode_steps=10 keeps step-limit tests concise.
    """
    from MT10_SAC.metaworld_envs.mt10_env import MetaWorldMT10Env

    benchmark = _build_mock_benchmark(TASK_NAMES)
    with patch("MT10_SAC.metaworld_envs.mt10_env.metaworld.MT10", return_value=benchmark):
        mt_env = MetaWorldMT10Env(seed=0, max_episode_steps=10, terminate_on_success=False)
    return mt_env


# ── Initialisation tests ──────────────────────────────────────────────────────

class TestMetaWorldMT10EnvInit:

    def test_env_names_are_sorted(self, env):
        """Sorted env_names guarantees the same task_id mapping across runs."""
        assert env._env_names == sorted(env._env_names)

    def test_num_tasks_matches_class_count(self, env):
        """num_tasks must equal the number of unique environment classes."""
        assert env.num_tasks == len(TASK_NAMES)

    def test_task_id_mapping_covers_all_tasks(self, env):
        """Every env_name must map to a unique id in [0, num_tasks)."""
        ids = list(env._env_name_to_id.values())
        assert sorted(ids) == list(range(env.num_tasks))

    def test_task_id_mapping_is_deterministic(self):
        """Two envs created with the same task names must produce identical mappings."""
        from MT10_SAC.metaworld_envs.mt10_env import MetaWorldMT10Env

        b1 = _build_mock_benchmark(TASK_NAMES)
        b2 = _build_mock_benchmark(TASK_NAMES)
        with patch("MT10_SAC.metaworld_envs.mt10_env.metaworld.MT10", return_value=b1):
            e1 = MetaWorldMT10Env(seed=0, max_episode_steps=5)
        with patch("MT10_SAC.metaworld_envs.mt10_env.metaworld.MT10", return_value=b2):
            e2 = MetaWorldMT10Env(seed=99, max_episode_steps=5)

        assert e1._env_name_to_id == e2._env_name_to_id


# ── Reset tests ───────────────────────────────────────────────────────────────

class TestMetaWorldMT10EnvReset:

    def test_reset_returns_obs_and_info(self, env):
        """reset() must return (obs, info) with the correct observation shape."""
        obs, info = env.reset()
        assert obs.shape == (OBS_DIM,)
        assert isinstance(info, dict)

    def test_reset_current_task_id_in_valid_range(self, env):
        """current_task_id must be in [0, num_tasks) after every reset."""
        for _ in range(15):          # sample several episodes
            env.reset()
            assert 0 <= env.current_task_id < env.num_tasks

    def test_reset_info_contains_mt_task_id(self, env):
        """info from reset() must include ``mt_task_id``."""
        _, info = env.reset()
        assert "mt_task_id" in info

    def test_reset_info_contains_mt_task_env_name(self, env):
        """info from reset() must include ``mt_task_env_name``."""
        _, info = env.reset()
        assert "mt_task_env_name" in info

    def test_reset_info_task_id_matches_attribute(self, env):
        """info['mt_task_id'] must equal env.current_task_id."""
        _, info = env.reset()
        assert info["mt_task_id"] == env.current_task_id

    def test_reset_zeroes_elapsed_steps(self, env):
        """reset() must zero the elapsed-step counter regardless of prior steps."""
        env.reset()
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        env.step(action)
        env.step(action)
        env.reset()
        assert env._elapsed_steps == 0


# ── Step tests ────────────────────────────────────────────────────────────────

class TestMetaWorldMT10EnvStep:

    def test_step_increments_elapsed_steps(self, env):
        """Each step() call must increment _elapsed_steps by exactly 1."""
        env.reset()
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        for expected in range(1, 5):
            env.step(action)
            assert env._elapsed_steps == expected

    def test_step_info_contains_task_metadata(self, env):
        """step() info must include both ``mt_task_id`` and ``mt_task_env_name``."""
        env.reset()
        _, _, _, _, info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
        assert "mt_task_id" in info
        assert "mt_task_env_name" in info

    def test_max_episode_steps_triggers_truncation(self):
        """truncated must become True exactly at the max_episode_steps boundary."""
        from MT10_SAC.metaworld_envs.mt10_env import MetaWorldMT10Env

        max_steps = 3
        benchmark = _build_mock_benchmark(TASK_NAMES)
        with patch("MT10_SAC.metaworld_envs.mt10_env.metaworld.MT10", return_value=benchmark):
            mt_env = MetaWorldMT10Env(seed=0, max_episode_steps=max_steps)

        mt_env.reset()
        action = np.zeros(ACTION_DIM, dtype=np.float32)

        # Steps before the limit must not truncate
        for _ in range(max_steps - 1):
            _, _, _, truncated, _ = mt_env.step(action)
            assert not truncated, "Truncation should not happen before max_episode_steps"

        # The final step must truncate
        _, _, _, truncated, _ = mt_env.step(action)
        assert truncated

    def test_terminate_on_success_flag(self):
        """terminated must be True on a success step when terminate_on_success=True."""
        from MT10_SAC.metaworld_envs.mt10_env import MetaWorldMT10Env

        benchmark = _build_mock_benchmark(TASK_NAMES, success_val=1.0)
        with patch("MT10_SAC.metaworld_envs.mt10_env.metaworld.MT10", return_value=benchmark):
            mt_env = MetaWorldMT10Env(
                seed=0, max_episode_steps=100, terminate_on_success=True
            )

        mt_env.reset()
        _, _, terminated, _, _ = mt_env.step(np.zeros(ACTION_DIM, dtype=np.float32))
        assert terminated

    def test_no_terminate_without_flag(self):
        """terminated must stay False on success when terminate_on_success=False."""
        from MT10_SAC.metaworld_envs.mt10_env import MetaWorldMT10Env

        benchmark = _build_mock_benchmark(TASK_NAMES, success_val=1.0)
        with patch("MT10_SAC.metaworld_envs.mt10_env.metaworld.MT10", return_value=benchmark):
            mt_env = MetaWorldMT10Env(
                seed=0, max_episode_steps=100, terminate_on_success=False
            )

        mt_env.reset()
        _, _, terminated, _, _ = mt_env.step(np.zeros(ACTION_DIM, dtype=np.float32))
        assert not terminated


# ── Cleanup tests ─────────────────────────────────────────────────────────────

class TestMetaWorldMT10EnvClose:

    def test_close_clears_env_cache(self, env):
        """close() must empty the internal env cache."""
        env.reset()                    # forces at least one env into the cache
        assert len(env._env_cache) > 0
        env.close()
        assert len(env._env_cache) == 0

    def test_close_sets_current_env_to_none(self, env):
        """close() must set _current_env to None."""
        env.reset()
        env.close()
        assert env._current_env is None
