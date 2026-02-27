"""
Tests for MT10_SAC/algos/sac_disentangled_alpha.py

Covers:
  - Model initialisation: log_ent_coef_vec shape, type, and optimizer
  - Validation: ValueError when ent_coef != 'auto'
  - _task_id_from_obs: single sample and batched extraction
  - _alpha_from_task_id: output shape, positivity, and consistency with parameters
"""
import numpy as np
import pytest
import torch
import gymnasium as gym
from gymnasium.spaces import Box

from MT10_SAC.algos.sac_disentangled_alpha import SACDisentangledAlpha


# ── Constants ─────────────────────────────────────────────────────────────────

NUM_TASKS = 4
BASE_OBS_DIM = 6           # dimensions from the environment state
TOTAL_OBS_DIM = BASE_OBS_DIM + NUM_TASKS   # 10  (state + one-hot task suffix)
ACTION_DIM = 3


# ── Minimal multi-task environment ────────────────────────────────────────────

class _DummyMTEnv(gym.Env):
    """
    Bare-minimum continuous-control env whose observation space already includes
    the one-hot task suffix, matching the layout expected by SACDisentangledAlpha.
    """

    def __init__(self):
        super().__init__()
        self.observation_space = Box(
            low=-1.0, high=1.0, shape=(TOTAL_OBS_DIM,), dtype=np.float32
        )
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        obs = np.zeros(TOTAL_OBS_DIM, dtype=np.float32)
        obs[BASE_OBS_DIM] = 1.0          # one-hot for task 0
        return obs, {}

    def step(self, action):
        obs = np.zeros(TOTAL_OBS_DIM, dtype=np.float32)
        obs[BASE_OBS_DIM] = 1.0
        return obs, 0.0, False, False, {}


# ── Shared model fixture ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model() -> SACDisentangledAlpha:
    """Create one SACDisentangledAlpha instance shared across the module."""
    env = _DummyMTEnv()
    return SACDisentangledAlpha(
        "MlpPolicy",
        env,
        num_tasks=NUM_TASKS,
        learning_starts=0,
        verbose=0,
    )


# ── Initialisation tests ──────────────────────────────────────────────────────

class TestSACDisentangledAlphaInit:

    def test_log_ent_coef_vec_exists(self, model):
        """Model must expose log_ent_coef_vec after __init__."""
        assert hasattr(model, "log_ent_coef_vec")

    def test_log_ent_coef_vec_shape(self, model):
        """log_ent_coef_vec must have shape (num_tasks,)."""
        assert tuple(model.log_ent_coef_vec.shape) == (NUM_TASKS,)

    def test_log_ent_coef_vec_is_nn_parameter(self, model):
        """log_ent_coef_vec must be a learnable nn.Parameter."""
        assert isinstance(model.log_ent_coef_vec, torch.nn.Parameter)

    def test_ent_coef_optimizer_is_adam(self, model):
        """Entropy-coefficient optimizer must be Adam."""
        assert isinstance(model.ent_coef_optimizer, torch.optim.Adam)

    def test_ent_coef_optimizer_tracks_log_ent_coef_vec(self, model):
        """Adam optimizer must own log_ent_coef_vec as its single parameter group."""
        param_ids = {
            id(p)
            for group in model.ent_coef_optimizer.param_groups
            for p in group["params"]
        }
        assert id(model.log_ent_coef_vec) in param_ids

    def test_raises_when_ent_coef_is_fixed(self):
        """Passing a fixed ent_coef (not 'auto') must raise ValueError."""
        env = _DummyMTEnv()
        with pytest.raises(ValueError, match="ent_coef='auto'"):
            SACDisentangledAlpha(
                "MlpPolicy", env, num_tasks=NUM_TASKS, ent_coef=0.1
            )


# ── Task-id extraction tests ──────────────────────────────────────────────────

class TestTaskIdExtraction:

    @pytest.mark.parametrize("task_id", range(NUM_TASKS))
    def test_single_sample_extraction(self, model, task_id):
        """_task_id_from_obs must return the argmax of the one-hot suffix."""
        obs = np.zeros((1, TOTAL_OBS_DIM), dtype=np.float32)
        obs[0, BASE_OBS_DIM + task_id] = 1.0
        obs_t = torch.tensor(obs, device=model.device)
        result = model._task_id_from_obs(obs_t)
        assert result.item() == task_id

    def test_batch_extraction_shape(self, model):
        """_task_id_from_obs must return a 1-D tensor of length batch_size."""
        batch_size = 8
        obs = np.zeros((batch_size, TOTAL_OBS_DIM), dtype=np.float32)
        for i in range(batch_size):
            obs[i, BASE_OBS_DIM + (i % NUM_TASKS)] = 1.0
        obs_t = torch.tensor(obs, device=model.device)
        result = model._task_id_from_obs(obs_t)
        assert result.shape == (batch_size,)

    def test_batch_extraction_values(self, model):
        """_task_id_from_obs must return the correct task id for each sample."""
        batch_size = 8
        expected = [i % NUM_TASKS for i in range(batch_size)]
        obs = np.zeros((batch_size, TOTAL_OBS_DIM), dtype=np.float32)
        for i, tid in enumerate(expected):
            obs[i, BASE_OBS_DIM + tid] = 1.0
        obs_t = torch.tensor(obs, device=model.device)
        assert model._task_id_from_obs(obs_t).tolist() == expected


# ── Alpha computation tests ───────────────────────────────────────────────────

class TestAlphaComputation:

    def test_alpha_output_shape(self, model):
        """_alpha_from_task_id must return a tensor of shape (batch, 1)."""
        task_ids = torch.zeros(8, dtype=torch.long, device=model.device)
        alpha = model._alpha_from_task_id(task_ids)
        assert alpha.shape == (8, 1)

    def test_alpha_is_strictly_positive(self, model):
        """Alpha = exp(log_alpha) must always be > 0 for every task."""
        task_ids = torch.arange(NUM_TASKS, dtype=torch.long, device=model.device)
        alpha = model._alpha_from_task_id(task_ids)
        assert (alpha > 0).all()

    @pytest.mark.parametrize("k", range(NUM_TASKS))
    def test_alpha_equals_exp_log_alpha(self, model, k):
        """alpha_k must equal exp(log_ent_coef_vec[k]) exactly."""
        task_id = torch.tensor([k], dtype=torch.long, device=model.device)
        alpha = model._alpha_from_task_id(task_id).item()
        expected = torch.exp(model.log_ent_coef_vec[k]).item()
        assert alpha == pytest.approx(expected, rel=1e-5)
