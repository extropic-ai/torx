from abc import abstractmethod

import jax
import jax.numpy as jnp
from ihoop.eqx import AbstractStrictModule
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int, Key


class AbstractSampler(AbstractStrictModule):
    """Produces random draws used by sample-based simulators."""

    @abstractmethod
    def bernoulli(
        self,
        key: Key[Array, ""],
        p: Float[Array, "..."],
        shape: tuple[int, ...] | None = None,
    ) -> Int[Array, "..."]:
        raise NotImplementedError

    @abstractmethod
    def categorical(
        self,
        key: Key[Array, ""],
        logits: Float[Array, "... K"],
        axis: int = -1,
        shape: tuple[int, ...] | None = None,
    ) -> Int[Array, "..."]:
        raise NotImplementedError

    @abstractmethod
    def normal(
        self,
        key: Key[Array, ""],
        shape: tuple[int, ...] = (),
        dtype: DTypeLike = float,
    ) -> Float[Array, "..."]:
        raise NotImplementedError


class JaxPRNGSampler(AbstractSampler):
    """Default sampler drawing from `jax.random` with a JAX PRNG key."""

    def bernoulli(
        self,
        key: Key[Array, ""],
        p: Float[Array, "..."],
        shape: tuple[int, ...] | None = None,
    ) -> Int[Array, "..."]:
        return jax.random.bernoulli(key, p, shape=shape).astype(jnp.int32)

    def categorical(
        self,
        key: Key[Array, ""],
        logits: Float[Array, "... K"],
        axis: int = -1,
        shape: tuple[int, ...] | None = None,
    ) -> Int[Array, "..."]:
        return jax.random.categorical(key, logits, axis=axis, shape=shape)

    def normal(
        self,
        key: Key[Array, ""],
        shape: tuple[int, ...] = (),
        dtype: DTypeLike = float,
    ) -> Float[Array, "..."]:
        # `float` honors jax_enable_x64: float64 draws under x64, else float32,
        # matching the affine-Gaussian simulator's dtype policy.
        return jax.random.normal(key, shape, dtype=dtype)


DEFAULT_SAMPLER: AbstractSampler = JaxPRNGSampler()


def _resolve(sampler: AbstractSampler | None) -> AbstractSampler:
    return DEFAULT_SAMPLER if sampler is None else sampler
