import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used

from typing import Any, Callable
import attr
import functools

from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state
from flax import linen as nn
from flax import jax_utils
import jax
from jax import jit, random
import optax

from dataset import DatasetConfig, DatasetBuilder
from nerf import NerfConfig, nerf_builder
from geometry import Rays


@attr.s(frozen=True, auto_attribs=True)
class OptimizerConfig:
    max_steps: int = 250
    init_lr: float = 5e-4
    final_lr: float = 5e-6
    lr_delay_steps: int = 0
    lr_delay_mult: float = 1.

@attr.s(frozen=True, auto_attribs=True)
class TrainerConfig:
    train_dir: str = ""
    model_config: NerfConfig = NerfConfig()
    dataset_config: DatasetConfig = DatasetConfig()
    optimizer_config: OptimizerConfig = OptimizerConfig()

class Trainer:

    def __init__(self, config: TrainerConfig):
        self._config = config
        self._rng = random.PRNGKey(0)
        self._state = None

    def train_and_evaluate(self):
        # Build dataset.
        train_iter, eval_iter = self.create_dataset()

        # Build model and state.
        self._state, _ = self.create_train_state()

        # Load from checkpoint.

        # Train loop.


    def create_dataset(self, to_device=True):
        local_device_count = jax.local_device_count()

        def _prepare_tf_data(x):
            x = x._numpy()
            return x.reshape((local_device_count, -1) + x.shape[1:])

        def _preprocess(data):
            rays = Rays(*data)
            return jax.tree_map(_prepare_tf_data, rays)

        ds_builder = DatasetBuilder(self._config.dataset_config)
        train_ds = ds_builder.build_train_dataset()
        eval_ds = ds_builder.build_test_dataset()

        train_iter = map(_preprocess, train_ds)
        if to_device:
            train_iter = jax_utils.prefetch_to_device(train_iter, 2)

        eval_iter = map(_preprocess, eval_ds)
        if to_device:
            eval_iter = jax_utils.prefetch_to_device(eval_iter, 2)
        return train_iter, eval_iter


    def create_train_state(self, init_batch: Rays):
        model, params = nerf_builder(self._rng, self._config.model_config, init_batch)
        learning_rate_fn = None
        tx = optax.adam(learning_rate=self._config.optimizer_config.init_lr)
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx)
        return state

    # def train_step(self, state, batch, learning_rate_fn):
    #     """Perform a single training step."""
    #     def loss_fn(params):
    #         return

    #     return new_state, stats

    def eval_step(self, state, batch):
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



