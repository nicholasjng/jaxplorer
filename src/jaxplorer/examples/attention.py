"""Single-head causal self-attention.

A heavier example. `static_argnums` keeps `causal` out of the traced arguments, and the
Analysis pane is the interesting one here: compare its flops and bytes accessed against
the MLP.
"""

import jax
import jax.numpy as jnp


def f(q, k, v, causal):
    scores = q @ k.T / jnp.sqrt(q.shape[-1]).astype(q.dtype)
    if causal:
        length = scores.shape[-1]
        mask = jnp.tril(jnp.ones((length, length), dtype=bool))
        scores = jnp.where(mask, scores, -jnp.inf)
    return jax.nn.softmax(scores, axis=-1) @ v


args = (
    jax.ShapeDtypeStruct((128, 64), jnp.float32),
    jax.ShapeDtypeStruct((128, 64), jnp.float32),
    jax.ShapeDtypeStruct((128, 64), jnp.float32),
    True,
)

static_argnums = (3,)
