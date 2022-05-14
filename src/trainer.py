import os

from matplotlib.pyplot import step
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used

from absl import app
from absl import flags

from sklearn import metrics
from typing import Any, Callable, Dict
import attr
import functools

import gin
import flax
import optax
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state
from flax import jax_utils, struct
import jax
import jax.numpy as jnp
from jax import lax, jit, random

from dataset import DatasetConfig, DatasetBuilder
from nerf import NerfConfig, nerf_builder
from geometry import Rays
import trainer_utils

FLAGS = flags.FLAGS
trainer_utils.define_flags()


@gin.configurable
@attr.s(frozen=True, auto_attribs=True)
class OptimizerConfig:
    max_steps: int = int(1e6)
    init_lr: float = 5e-4
    final_lr: float = 5e-6
    lr_delay_rate: float = 0.1
    weight_decay: float = 1e-3


@gin.configurable
@attr.s(frozen=True, auto_attribs=True)
class TrainerConfig:
    train_dir: str = ""
    model_config: NerfConfig = NerfConfig()
    dataset_config: DatasetConfig = DatasetConfig()
    optimizer_config: OptimizerConfig = OptimizerConfig()


class Trainer:
    """Class for training NeRF model."""

    def __init__(self, config: TrainerConfig):
        self.config = config

    def create_train_state(self, rng):
        model, params = nerf_builder(rng, self.config.model_config)
        init_lr = self.config.optimizer_config.init_lr
        max_steps = self.config.optimizer_config.max_steps
        exponential_decay_scheduler = optax.exponential_decay(
            init_value=init_lr,
            transition_steps=max_steps,
            decay_rate=0.1)
        tx = optax.adam(learning_rate=exponential_decay_scheduler)
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx)
        return state

    def create_train_step(self):
        return jax.pmap(
            self._train_step_impl,
            axis_name='batch')

    def _train_step_impl(self, key, state, rays):
        """Perform a single training step."""
        def loss_fn(params):
            coarse_rgb, _, fine_rgb, fine_depth = state.apply_fn(params, key, rays)

            # Compute RGB loss.
            loss_coarse_rgb = trainer_utils.mse_loss(coarse_rgb, rays.colors)
            loss_fine_rgb = trainer_utils.mse_loss(fine_rgb, rays.colors)
            loss_rgb = loss_coarse_rgb + loss_fine_rgb
            
            # Mask out invalid depth values.
            mask = ~jnp.isnan(rays.depths)
            depths_gt = jnp.where(mask, rays.depths, 0.)
            weights_depth = jnp.where(mask, rays.weights, 0.)

            # Compute depth loss.
            loss_depth = trainer_utils.mse_loss(fine_depth, depths_gt, weights_depth)
            loss_depth *= 0.1
        
            # Final loss.
            loss = loss_rgb + loss_depth
            metrics = {"loss_coarse_rgb": loss_coarse_rgb,
                       "loss_fine_rgb": loss_fine_rgb,
                       "loss_depth": loss_depth,
                       "loss": loss}

            # Add weight decay.
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
        grads = lax.pmean(grads, axis_name='batch')
        metrics = lax.pmean(metrics, axis_name='batch')
        new_state = state.apply_gradients(grads)
        return new_state, metrics

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

        ds_builder = DatasetBuilder(self.config.dataset_config)
        train_ds = ds_builder.build_train_dataset()
        eval_ds = ds_builder.build_test_dataset()

        train_iter = map(_preprocess, train_ds)
        if to_device:
            train_iter = jax_utils.prefetch_to_device(train_iter, 2)

        eval_iter = map(_preprocess, eval_ds)
        if to_device:
            eval_iter = jax_utils.prefetch_to_device(eval_iter, 2)
        return train_iter, eval_iter

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

    def train_and_evaluate(self):
        rng = random.PRNGKey(20200823)
        n_local_devices = jax.local_device_count()
        rng = rng + jax.host_id()  # Make random seed separate across hosts.
        # Build dataset.
        train_iter, eval_iter = self.create_dataset()
        # Build model and state.
        key, rng = random.split(rng)
        state = self.create_train_state(key)
        state = jax_utils.replicate(state)
        # Load from checkpoint.

        # Create train step.
        p_train_step = self.create_train_step()

        # Train loop.
        for step, rays in zip(range(self.config.max_steps), train_iter):
            key, rng = random.split(rng)
            keys = random.split(key, n_local_devices)
            state, metrics = p_train_step(keys, state, rays)
            if self.should_log(step):
                self.write_summary(metrics)
            if self.should_eval(step):
                pass
            
    def should_log(self, step: int):
        return True

    def should_eval(self, step: int):
        return True  