from typing import Any, Callable
from absl import flags

import jax.numpy as jnp
import jax
from jax import jit, random

from geometry import Rays, Camera


def define_flags():
    flags.DEFINE_string("config", None,  "using config files to set train parameters.")


def weight_decay_l2(parameters: jnp.ndarray):
    def tree_sum_fn(fn):
      return jax.tree_util.tree_reduce(
          lambda x, y: x + fn(y), parameters, initializer=0)

    weight_l2 = (
        tree_sum_fn(lambda z: jnp.sum(z**2)) /
        tree_sum_fn(lambda z: jnp.prod(jnp.array(z.shape))))
    return weight_l2


def mae_loss(pred: jnp.ndarray, gd: jnp.ndarray, weights: jnp.ndarray=1.0):
    errors = jnp.abs(pred - gd) * weights
    return errors.mean()

def mse_loss(pred: jnp.ndarray, gd: jnp.ndarray, weights: jnp.ndarray=1.0):
    errors = jnp.square(pred - gd) * weights
    return errors.mean()

# def compute_metrics(logits, labels):
#   loss = cross_entropy_loss(logits, labels)
#   accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
#   metrics = {
#       'loss': loss,
#       'accuracy': accuracy,
#   }
#   metrics = lax.pmean(metrics, axis_name='batch')
#   return metrics


# def create_learning_rate_fn(
#     config: ml_collections.ConfigDict,
#     base_learning_rate: float,
#     steps_per_epoch: int):
#   """Create learning rate schedule."""
#   warmup_fn = optax.linear_schedule(
#       init_value=0., end_value=base_learning_rate,
#       transition_steps=config.warmup_epochs * steps_per_epoch)
#   cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
#   cosine_fn = optax.cosine_decay_schedule(
#       init_value=base_learning_rate,
#       decay_steps=cosine_epochs * steps_per_epoch)
#   schedule_fn = optax.join_schedules(
#       schedules=[warmup_fn, cosine_fn],
#       boundaries=[config.warmup_epochs * steps_per_epoch])
#   return schedule_fn


def render_image(image: jnp.ndarray, camera: Camera):
    pass
