from typing import Any, Callable
from absl import flags

import jax.numpy as jnp
import jax
from jax import jit, random
import optax

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

def learning_rate_decay(step,
                        lr_init,
                        lr_final,
                        max_steps,
                        lr_delay_steps=0,
                        lr_delay_mult=1):
  """Continuous learning rate decay function.
  The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
  is log-linearly interpolated elsewhere (equivalent to exponential decay).
  If lr_delay_steps>0 then the learning rate will be scaled by some smooth
  function of lr_delay_mult, such that the initial learning rate is
  lr_init*lr_delay_mult at the beginning of optimization but will be eased back
  to the normal learning rate when steps>lr_delay_steps.
  Args:
    step: int, the current optimization step.
    lr_init: float, the initial learning rate.
    lr_final: float, the final learning rate.
    max_steps: int, the number of steps during optimization.
    lr_delay_steps: int, the number of steps to delay the full learning rate.
    lr_delay_mult: float, the multiplier on the rate when delaying it.
  Returns:
    lr: the learning for current step 'step'.
  """
  if lr_delay_steps > 0:
    # A kind of reverse cosine decay.
    delay_rate = lr_delay_mult + (1 - lr_delay_mult) * jnp.sin(
        0.5 * jnp.pi * jnp.clip(step / lr_delay_steps, 0, 1))
  else:
    delay_rate = 1.
  t = jnp.clip(step / max_steps, 0, 1)
  log_lerp = jnp.exp(jnp.log(lr_init) * (1 - t) + jnp.log(lr_final) * t)
  return delay_rate * log_lerp

def render_image(image: jnp.ndarray, camera: Camera):
    pass
