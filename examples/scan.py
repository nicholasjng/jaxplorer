"""A cumulative RNN-ish step under `lax.scan`.

Control flow is where the jaxpr earns its keep: `scan` stays a single primitive with a
nested jaxpr rather than being unrolled, and you can watch that survive into the HLO as
a while loop.
"""

import jax
import jax.numpy as jnp
from jax import lax


def f(h0, xs, w):
    def step(h, x):
        h = jnp.tanh(h @ w + x)
        return h, h.sum()

    final, outs = lax.scan(step, h0, xs)
    return final, outs


args = (
    jax.ShapeDtypeStruct((16,), jnp.float32),
    jax.ShapeDtypeStruct((64, 16), jnp.float32),
    jax.ShapeDtypeStruct((16, 16), jnp.float32),
)
