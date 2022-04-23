import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used

from absl import app
from absl import flags

from sklearn import metrics
from typing import Any, Callable, Dict
import attr
import functools

import gin
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state
from flax import linen as nn
from flax import jax_utils
import jax
import jax.numpy as jnp
from jax import lax, jit, random
import optax

from dataset import DatasetConfig, DatasetBuilder
from nerf import NerfConfig, nerf_builder
from geometry import Rays
import trainer_utils

FLAGS = flags.FLAGS
trainer_utils.define_flags()


#@gin.configurable
@attr.s(frozen=True, auto_attribs=True)
class OptimizerConfig:
    max_steps: int = int(1e6)
    init_lr: float = 5e-4
    final_lr: float = 5e-6
    lr_delay_steps: int = 0
    lr_delay_mult: float = 1.
    weight_decay: float = 0.


#@gin.configurable
@attr.s(frozen=True, auto_attribs=True)
class TrainerConfig:
    train_dir: str = ""
    model_config: NerfConfig = NerfConfig()
    dataset_config: DatasetConfig = DatasetConfig()
    optimizer_config: OptimizerConfig = OptimizerConfig()


class Trainer:
    """Class for training NeRF model."""

    def __init__(self, config: TrainerConfig):
        self._config = config
        self._rng = random.PRNGKey(20200823)
        self._state = None

    def train_and_evaluate(self):
        # Build dataset.
        train_iter, eval_iter = self.create_dataset()
        # Build model and state.
        self._state, _ = self.create_train_state()
        # Load from checkpoint.

        # Train loop.
        self._state = jax_utils.replicate(self._state)
        p_train_step = self.create_train_step()
        keys = random.split(self._rng, 1)
        for step, rays in zip(range(self._config.max_steps), train_iter):
            self._state, metrics, keys = p_train_step(keys, self._state, rays)
            if self.should_log(step):
                self.write_summary(metrics)
            if self.should_eval(step):
                pass
            
    def should_log(self, step: int):
        return True

    def should_eval(self, step: int):
        return True  

    def create_train_step(self):
        return jax.pmap(
            functools.partial(self._train_step_impl, learning_rate_fn=None),
            axis_name='batch')

    def _train_step_impl(self, rng, state, rays, learning_rate_fn):
        """Perform a single training step."""
        rng, key = random.split(rng)

        def loss_fn(params):
            coarse_pred, fine_pred = state.apply_fn(params, key, rays)
            loss_coarse = trainer_utils.photometric_loss(coarse_pred, rays.colors)
            # loss_fine = trainer_utils.photometric_loss(fine_pred, rays.colors)
            loss = loss_coarse # + loss_fine
            metrics = {"loss_coarse": loss_coarse,
                       # "loss_fine": loss_fine,
                       "loss": loss}
            weight_penalty_params = jax.tree_leaves(params)
            weight_decay = 0.0001
            weight_l2 = sum([jnp.sum(x ** 2)
                            for x in weight_penalty_params
                            if x.ndim > 1])
            weight_penalty = weight_decay * 0.5 * weight_l2
            loss = loss + weight_penalty
            return loss, metrics

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, metrics), grads = grad_fn(state.params)
        # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
        grads = lax.pmean(grads, axis_name='batch')
        new_state = state.apply_gradients(grads=grads)
        return new_state, metrics, rng

    def eval_step(self, state, batch):
        pass

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

    def create_train_state(self):
        self._rng, key = random.split(self._rng)
        model, params = nerf_builder(key, self._config.model_config.coarse_module_config)
        learning_rate_fn = None
        tx = optax.adam(learning_rate=self._config.optimizer_config.init_lr)
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx)
        return state

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

    def write_summary(self, metrics: Dict):
        pass


def main(unused_argv):
    gin.parse_config_file(FLAGS.config)
    trainer = Trainer(TrainerConfig())
    trainer.train_and_evaluate()


if __name__ == "__main__":
  app.run(main)