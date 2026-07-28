"""A two-layer MLP forward pass.

The smallest example worth looking at: the optimized HLO shows XLA fusing the bias add
and the activation into the dot.
"""

import jax
import jax.numpy as jnp


def f(x, w1, b1, w2, b2):
    h = jnp.tanh(x @ w1 + b1)
    return h @ w2 + b2


args = (
    jax.ShapeDtypeStruct((32, 128), jnp.float32),
    jax.ShapeDtypeStruct((128, 256), jnp.float32),
    jax.ShapeDtypeStruct((256,), jnp.float32),
    jax.ShapeDtypeStruct((256, 10), jnp.float32),
    jax.ShapeDtypeStruct((10,), jnp.float32),
)
