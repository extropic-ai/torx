from abc import abstractmethod
from typing import Any, Callable, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Key, PyTree

from ._utils import same_pytree_spec
from .factor import (
    _SampleOutput,
    AbstractFactor,
    AbstractReferenceFactor,
    InfoTree,
    ParamsTree,
    PortSpec,
)


def _state_to_index(
    query: PyTree[Array],
    stacked_states: PyTree[Array],
    spec: PortSpec,
) -> Float[Array, ""]:
    """Index of the first state equal to ``query``.

    ``stacked_states`` is the canonical ordering with a leading axis of size
    ``n_states``; ``query`` is a single state. Returns the matching
    index, or `NaN` if ``query`` is not among the canonical states.
    """
    if not same_pytree_spec(query, spec):
        raise ValueError(f"query {query} does not match the expected spec {spec}.")
    s_leaves = jax.tree.leaves(stacked_states)
    if len(s_leaves) == 0:
        return jnp.array(0.0)
    n = s_leaves[0].shape[0]
    q_leaves = jax.tree.leaves(query)
    matches = jnp.ones((n,), dtype=bool)
    for q, s in zip(q_leaves, s_leaves):
        matches = matches & (s == q[None]).reshape(n, -1).all(axis=1)
    return jnp.where(jnp.any(matches), jnp.argmax(matches).astype(jnp.float32), jnp.nan)


def _n_states_from_stacked(stacked: PyTree[Array], name: str) -> int:
    """Leading-axis size shared by every leaf of a stacked state pytree."""
    leaves = jax.tree.leaves(stacked)
    if len(leaves) == 0:
        return 1
    n = leaves[0].shape[0]
    for i, leaf in enumerate(leaves):
        if leaf.shape[0] != n:
            raise ValueError(
                f"{name} leaf {i} has shape {leaf.shape}, expected leading dim {n}"
            )
    return n


def _state_spec(leaf: Array) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(shape=leaf.shape[1:], dtype=leaf.dtype)


class AbstractHasLogProbability(AbstractFactor):
    """Capability mixin for factors with a tractable, analytic
    `log_probability(outputs | inputs)`.
    """

    @abstractmethod
    def log_probability(
        self,
        inputs: Mapping[str, PyTree[Array]],
        outputs: PyTree[Array],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> Float[Array, ""] | tuple[Float[Array, ""], PyTree[Array]]:
        """Scalar log-probability of `outputs` given `inputs`.

        **Arguments:**

        - `inputs`: Per-port pytree inputs, as in `sample`.
        - `outputs`: An output pytree matching `output_spec`.
        - `params`: Parameter pytree for this factor.
        - `info`: Runtime auxiliary info, as in `sample`.
        - `site_info`: Static per-site metadata, as in `sample`.
        - `return_aux`: Whether to additionally return a factor-defined
            `aux` pytree.

        **Returns:**

        The scalar `log P(outputs | inputs)`, or `(log_p, aux)` when
        `return_aux=True`.
        """
        raise NotImplementedError


class AbstractEnumerableOutputFactor(AbstractFactor):
    """Factor whose output state space is finite, with a canonical ordering."""

    @property
    @abstractmethod
    def n_output_states(self) -> int:
        """Number of output states in the canonical ordering."""
        raise NotImplementedError

    @abstractmethod
    def get_nth_output_state(self, n: int | Int[Array, ""]) -> PyTree[Array]:
        """Return the `n`-th output state in the canonical ordering.

        The returned pytree has the same structure as the output of
        [`torx.AbstractFactor.sample`][].
        """
        raise NotImplementedError

    @abstractmethod
    def output_state_to_index(self, outputs: PyTree[Array]) -> Float[Array, ""]:
        """Index of `outputs` in the canonical ordering, as a scalar float."""
        raise NotImplementedError


class AbstractFiniteStateSpaceFactor(AbstractEnumerableOutputFactor):
    """Factor whose input and output state spaces are both finite."""

    @property
    @abstractmethod
    def n_input_states(self) -> int:
        """Number of input states in the canonical ordering."""
        raise NotImplementedError

    @abstractmethod
    def get_nth_input_state(
        self, n: int | Int[Array, ""]
    ) -> Mapping[str, PyTree[Array]]:
        """Return the `n`-th input state in the canonical ordering.

        The returned pytree has the same structure as the `inputs` argument
        to [`torx.AbstractFactor.sample`][].
        """
        raise NotImplementedError

    @abstractmethod
    def input_state_to_index(
        self, inputs: Mapping[str, PyTree[Array]]
    ) -> Float[Array, ""]:
        """Index of `inputs` in the canonical ordering, as a scalar float."""
        raise NotImplementedError


class AbstractHasExplicitOutputDistribution(
    AbstractEnumerableOutputFactor, AbstractHasLogProbability
):
    """Factor whose conditional is available in closed form as an explicit
    log-prob vector over the enumerable output states.
    """

    @abstractmethod
    def get_log_output_distribution(
        self,
        inputs: Mapping[str, PyTree[Array]],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> (
        Float[Array, " n_output_states"]
        | tuple[Float[Array, " n_output_states"], PyTree[Array]]
    ):
        """Log-probabilities of all `n_output_states` outputs given `inputs`.

        Returns a vector `v` of shape `(n_output_states,)` with
        `v[j] = log P(get_nth_output_state(j) | inputs)`.

        **Arguments:**

        - `inputs`: Per-port pytree inputs, as in `sample`.
        - `params`: Parameter pytree for this factor.
        - `info`: Runtime auxiliary info, as in `sample`.
        - `site_info`: Static per-site metadata, as in `sample`.
        - `return_aux`: Whether to additionally return a factor-defined `aux`.

        **Returns:**

        The length-`n_output_states` log-probability vector, or `(v, aux)` when
        `return_aux=True`.
        """
        raise NotImplementedError

    def log_probability(
        self,
        inputs: Mapping[str, PyTree[Array]],
        outputs: PyTree[Array],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> Float[Array, ""] | tuple[Float[Array, ""], PyTree[Array]]:
        """Index `get_log_output_distribution` at the queried `outputs`."""
        log_dist, aux = self.get_log_output_distribution(
            inputs, params, info, site_info, return_aux=True
        )

        n_out = self.output_state_to_index(outputs)
        out_valid = ~jnp.isnan(n_out)
        j = jnp.where(out_valid, n_out, 0.0).astype(jnp.int32)
        log_prob = jnp.where(out_valid, log_dist[j], -jnp.inf)
        log_prob = jnp.where(jnp.all(jnp.isnan(log_dist)), jnp.nan, log_prob)

        if return_aux:
            return log_prob, aux
        return log_prob


class AbstractMatrixFactor(
    AbstractHasExplicitOutputDistribution,
    AbstractFiniteStateSpaceFactor,
    AbstractReferenceFactor,
):
    """A finite-state factor whose conditional is given explicitly as a matrix."""

    input_states: eqx.AbstractVar[dict[str, PyTree[Array]]]
    output_states: eqx.AbstractVar[PyTree[Array]]

    @property
    def input_ports(self) -> Mapping[str, PortSpec]:  # type: ignore[override]
        return jax.tree.map(_state_spec, self.input_states)

    @property
    def output_spec(self) -> PortSpec:
        return jax.tree.map(_state_spec, self.output_states)

    @property
    def n_input_states(self) -> int:
        return _n_states_from_stacked(self.input_states, "input_states")

    @property
    def n_output_states(self) -> int:
        return _n_states_from_stacked(self.output_states, "output_states")

    def get_nth_input_state(
        self, n: int | Int[Array, ""]
    ) -> Mapping[str, PyTree[Array]]:
        """See [`torx.AbstractFiniteStateSpaceFactor.get_nth_input_state`][]."""
        return jax.tree.map(lambda x: x[n], self.input_states)

    def get_nth_output_state(self, n: int | Int[Array, ""]) -> PyTree[Array]:
        """See [`torx.AbstractFiniteStateSpaceFactor.get_nth_output_state`][]."""
        return jax.tree.map(lambda x: x[n], self.output_states)

    def input_state_to_index(
        self, inputs: Mapping[str, PyTree[Array]]
    ) -> Float[Array, ""]:
        """See [`torx.AbstractFiniteStateSpaceFactor.input_state_to_index`][]."""
        return _state_to_index(inputs, self.input_states, self.input_ports)

    def output_state_to_index(self, outputs: PyTree[Array]) -> Float[Array, ""]:
        """See [`torx.AbstractEnumerableOutputFactor.output_state_to_index`][]."""
        return _state_to_index(outputs, self.output_states, self.output_spec)

    @abstractmethod
    def get_log_probability_matrix(
        self,
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
    ) -> Float[Array, "n_input_states n_output_states"]:
        """Return the `(n_input_states, n_output_states)` log-probability matrix.

        Entry `[i, j]` is `log P(output_states[j] | input_states[i])`.

        **Arguments:**

        - `params`: Parameter pytree for this factor.
        - `info`: Runtime auxiliary info, as in `sample`.
        - `site_info`: Static per-site metadata, as in `sample`.

        **Returns:**

        The matrix of log-probabilities.
        """
        raise NotImplementedError

    def get_log_output_distribution(
        self,
        inputs: Mapping[str, PyTree[Array]],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> (
        Float[Array, " n_output_states"]
        | tuple[Float[Array, " n_output_states"], PyTree[Array]]
    ):
        """The matrix row selected by the input's canonical index."""
        n_in = self.input_state_to_index(inputs)
        matrix = self.get_log_probability_matrix(params, info, site_info)

        in_valid = ~jnp.isnan(n_in)
        i = jnp.where(in_valid, n_in, 0.0).astype(jnp.int32)
        log_dist = jnp.where(in_valid, matrix[i], jnp.nan)

        if return_aux:
            return log_dist, None
        return log_dist


class DeterministicFactor(AbstractHasLogProbability):
    """A factor whose output is a deterministic function of its inputs.

    **Arguments:**

    - `fn`: A pure function `fn(inputs, site_info) -> output` mapping the
        per-port `inputs` dict (and the surrounding `Site`'s static
        `site_info`) to an output pytree matching `output_spec`.
    - `input_ports`: The factor's input-port specs.
    - `output_spec`: The spec of `fn`'s output.
    """

    fn: Callable[[Mapping[str, PyTree[Array]], Any], PyTree[Array]] = eqx.field(
        static=True
    )
    input_ports: Mapping[str, PortSpec] = eqx.field(static=True, converter=dict)
    output_spec: PortSpec = eqx.field(static=True)

    def sample(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> _SampleOutput:
        """Return `fn(inputs, site_info)`."""
        output = self.fn(inputs, site_info)
        if return_aux:
            return output, None
        return output

    def sample_with_references(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        n_references: int = 1,
    ) -> tuple[PyTree[Array], PyTree[Array]]:
        """Evaluate `fn` once."""
        return self.fn(inputs, site_info), None

    def init_params(self, key: Key[Array, ""]) -> ParamsTree:
        return None

    def log_probability(
        self,
        inputs: Mapping[str, PyTree[Array]],
        outputs: PyTree[Array],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> Float[Array, ""] | tuple[Float[Array, ""], PyTree[Array]]:
        """`0` if `outputs == fn(inputs, site_info)`, else `-inf`."""
        equal = eqx.tree_equal(self.fn(inputs, site_info), outputs)
        log_prob = jnp.where(equal, 0.0, -jnp.inf)

        if return_aux:
            return log_prob, None
        return log_prob
