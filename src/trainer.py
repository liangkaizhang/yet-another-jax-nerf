from typing import Any, Callable
import attr
import functools

from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state
from flax import linen as nn
import jax
from jax import jit, random

from src.dataset import DatasetConfig
from src.nerf import NerfConfig


@attr.s(frozen=True, auto_attribs=True)
class OptimizerConfig:
    init_lr: float


@attr.s(frozen=True, auto_attribs=True)
class TrainerConfig:
    train_dir: str = ""
    max_steps: int = 250
    model_config: NerfConfig = NerfConfig()
    dataset_config: DatasetConfig = DatasetConfig()
    optimizer_config: OptimizerConfig = OptimizerConfig()


class Trainer:

    def __init__(self, config: TrainerConfig):
        self._train_dir = config.train_dir
        self._max_steps = config.max_steps
        self._state = None

    def train_and_evaluate(self):
        pass

    def train_step(self, state, batch, learning_rate_fn):
        """Perform a single training step."""
        def loss_fn(params):
            return

        return new_state, stats

    def eval_step(self, state, batch):
        pass

    def create_train_state(self):
        pass

    def restore_checkpoint(self):
        return checkpoints.restore_checkpoint(
            self._train_dir, self._state)

    def save_checkpoint(self):
        if jax.process_index() != 0:
            return
        # get train state from the first replica
        cur_state = jax.device_get(jax.tree_map(lambda x: x[0], self._state))
        cur_step = int(cur_state.step)
        checkpoints.save_checkpoint(self._train_dir, cur_state, cur_step, keep=3)

    def write_summary(self):
        pass



