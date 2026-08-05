"""Abstract base classes for probabilistic gates."""

import itertools
from abc import abstractmethod
from typing import Any, ClassVar, Generic, Mapping, TypeAlias, TypedDict, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jaxtyping import Array, Float, Int, Key, PyTree

from ...factor import (
    _SampleOutput,
    AbstractReferenceFactor,
    InfoTree,
    ParamsTree,
    PortSpec,
)
from ...tractable_prob_factors import AbstractMatrixFactor


class HybridSites(TypedDict):
    discrete: list[int]
    continuous: list[int]


DiscreteSites: TypeAlias = int | list[int]
_SiteType = TypeVar("_SiteType", bound=HybridSites | DiscreteSites)
_DiscreteSiteType = TypeVar("_DiscreteSiteType", bound=DiscreteSites)
_ThetaType = TypeVar("_ThetaType", bound=PyTree[Array])
_DimsType = TypeVar("_DimsType", bound=tuple[int, ...])


class AbstractPGate(AbstractReferenceFactor, Generic[_SiteType, _ThetaType, _DimsType]):
    """Base class for all probabilistic gates.

    For visualization, there are additional properties:

    - `_control_indices`: Indices of control sites for drawing.
        - `()` empty tuple: no controls, draw as spanning box (default)
        - `(0,)` tuple of ints: specific site indices are controls
        - `"all_but_last"`: all sites except the last are controls
    - `_draw_label`: Optional short label for visualization (defaults to class name).
    """

    _control_indices: ClassVar[tuple[int, ...] | str] = ()
    _draw_label: ClassVar[str | None] = None

    sites: eqx.AbstractVar[_SiteType]
    dims: eqx.AbstractVar[_DimsType]

    @abstractmethod
    def init_params(self, key: Key[Array, ""]) -> _ThetaType:
        """Return a freshly-initialised parameter pytree (``theta``).

        **Arguments:**

        - `key`: PRNG key.

        **Returns:**

        An initial ``theta`` pytree for this gate.
        """
        raise NotImplementedError


class AbstractDiscreteGate(
    AbstractMatrixFactor,
    AbstractPGate[_DiscreteSiteType, _ThetaType, _DimsType],
):
    """A discrete probabilistic gate, a finite-state matrix factor."""

    @abstractmethod
    def get_matrix(self, theta: _ThetaType) -> Float[Array, "dim1 dim2"]:
        """
        Get the matrix representation of the gate.

        **Arguments:**

        - `theta`: the gate's parameters.

        **Returns:**

        The column-stochastic transition matrix ``P[out, in]`` of the gate.
        """
        raise NotImplementedError

    @property
    def _basis(self) -> Int[Array, "n_states n_sites"]:
        """The ``prod(dims)`` basis configurations, row-major over `dims`."""
        return jnp.array(list(itertools.product(*(range(d) for d in self.dims))))

    @property
    def input_states(self) -> dict[str, PyTree[Array]]:  # type: ignore[override]
        return {"in": self._basis}

    @property
    def output_states(self) -> PyTree[Array]:
        return self._basis

    @property
    def input_ports(self) -> Mapping[str, PortSpec]:  # type: ignore[override]
        return super().input_ports

    def get_log_probability_matrix(
        self,
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
    ) -> Float[Array, "n_input_states n_output_states"]:
        """Row-stochastic ``log P[in, out] = log(get_matrix(params).T)``."""
        return jnp.log(self.get_matrix(params).T)

    def sample(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> _SampleOutput:
        """Draw the gate's output configuration from its transition matrix."""
        i = self.input_state_to_index(inputs).astype(jnp.int32)
        row = self.get_log_probability_matrix(params)[i]
        output = self.get_nth_output_state(jax.random.categorical(key, row))
        if return_aux:
            return output, None
        return output


class AbstractGeneratorGate(
    AbstractDiscreteGate[_DiscreteSiteType, _ThetaType, _DimsType]
):
    r"""
    Discrete gate defined by a continuous-time Markov generator $Q$.

    Subclasses provide a rate matrix $Q$ via `get_generator` and a time
    step via `dt`. The transition matrix is then

    $$P = \exp(Q \cdot \Delta t).$$
    """

    @abstractmethod
    def dt(self, theta: _ThetaType) -> Float[Array, ""]:
        r"""The continuous-time step $\Delta t$, extracted from ``theta``."""
        raise NotImplementedError

    @abstractmethod
    def get_generator(self, theta: _ThetaType) -> Float[Array, "dim1 dim2"]:
        r"""
        Return the continuous-time Markov generator $Q$.

        $Q$ must be a rate matrix, which means off-diagonal entries are non-negative
        and column sums equal to zero.

        **Arguments:**

        - `theta`: the gate's parameters.

        **Returns:**

        The Markov generator $Q$.
        """
        raise NotImplementedError

    def get_matrix(self, theta: _ThetaType) -> Float[Array, "dim1 dim2"]:
        r"""Column-stochastic transition matrix $P = \exp(Q \cdot \Delta t)$.

        See [`torx.psc.AbstractDiscreteGate.get_matrix`][] for documentation.
        """
        return expm(self.get_generator(theta) * self.dt(theta))


class AbstractHybridGate(AbstractPGate[HybridSites, _ThetaType, _DimsType]):
    """Base class for hybrid gates.

    Gates specify which discrete and continuous sites they act on via the `sites`
    dict. As a `Factor`, the gate's `sample` takes the relevant substate as its
    `inputs` (a dict with ``"discrete"`` / ``"continuous"`` values at the gate's
    sites) and returns the new continuous substate.

    !!! note

        `torx` currently only supports discrete controlled continuous gates.
    """

    @property
    def discrete_dims(self) -> tuple[int, ...]:
        """Dimensions of discrete control sites, aligned with ``sites["discrete"]``."""
        return (2,) * len(self.sites.get("discrete", []))

    @property
    def input_ports(self) -> Mapping[str, PortSpec]:  # type: ignore[override]
        n_discrete = len(self.sites.get("discrete", []))
        cont_dim = sum(self.dims)
        return {
            "discrete": jax.ShapeDtypeStruct((n_discrete,), jnp.int32),
            "continuous": jax.ShapeDtypeStruct((cont_dim,), jnp.float32),
        }

    @property
    def output_spec(self) -> PortSpec:
        return jax.ShapeDtypeStruct((sum(self.dims),), jnp.float32)

    @abstractmethod
    def sample(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> _SampleOutput:
        """Sample the new continuous substate.

        **Arguments:**

        - `key`: JAX random key.
        - `inputs`: Substate dict with ``"discrete"`` / ``"continuous"`` keys
          holding the values at the gate's sites.
        - `params`: the gate's ``theta``.
        - `info`: optional runtime information.
        - `site_info`: unused.
        - `return_aux`: if ``True``, return ``(output, None)``.

        **Returns:**

        New values for the continuous sites (same shape as
        ``inputs["continuous"]``), or ``(output, None)`` when ``return_aux``.
        """
        raise NotImplementedError


class AbstractContinuousGate(AbstractHybridGate[_ThetaType, _DimsType]):
    """Gate that only touches continuous sites.

    Subclasses set sites to specify which continuous sites they act on.
    """

    pass


class AbstractAffineGaussianGate(AbstractContinuousGate[_ThetaType, _DimsType]):
    r"""Continuous gate with an exact affine-Gaussian channel.

    Subclasses expose their local continuous-state transition as

    $$
    x' = A x + b + \epsilon,\qquad
    \epsilon \sim \mathcal{N}(0, \operatorname{diag}(\exp(\text{log\_var}))),
    $$

    by implementing `affine_parameters`. The default `sample` draws from this
    channel, but exposing $(A, b, \log\text{var})$ also lets simulators reason
    about the exact Gaussian transition rather than only drawing samples.
    """

    @abstractmethod
    def affine_parameters(
        self,
        theta: _ThetaType,
    ) -> tuple[
        Float[Array, " local_dim local_dim"],
        Float[Array, " local_dim"],
        Float[Array, " local_dim"],
    ]:
        """Return ``(A, b, log_var)`` for the local affine-Gaussian channel.

        **Arguments:**

        - `theta`: the gate's parameters.

        **Returns:**

        A tuple ``(A, b, log_var)`` describing ``x' = A x + b + noise`` with
        ``noise ~ Normal(0, diag(exp(log_var)))``.
        """
        raise NotImplementedError

    def sample(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> _SampleOutput:
        """Sample the affine-Gaussian channel exposed by `affine_parameters`."""
        x = inputs["continuous"]
        A, b, log_var = self.affine_parameters(params)
        noise = jax.random.normal(key, x.shape, dtype=log_var.dtype)
        mean = jnp.einsum("ij,...j->...i", A, x) + b
        output = mean + noise * jnp.sqrt(jnp.exp(log_var))
        return (output, None) if return_aux else output


class AbstractControlledContinuousGate(AbstractHybridGate[_ThetaType, _DimsType]):
    """Gate where discrete sites control the continuous transformation.

    !!! note

        Currently we only support control by a single discrete site (pbit/pdit),
        so dims only specifies continuous dimensions. Multi-site discrete control
        would require tracking discrete dims as well.
    """

    def _get_discrete_control(self, inputs: Mapping[str, PyTree[Array]]) -> Array:
        """Extract single discrete control value with validation."""
        disc = inputs["discrete"]
        if disc.shape != (1,):
            raise ValueError(
                f"{self.__class__.__name__} expects exactly 1 discrete control "
                f"site, got shape {disc.shape}"
            )
        return disc[0]


class AbstractKBranchGate(
    AbstractDiscreteGate[
        _DiscreteSiteType, Float[Array, " num_branch_params"], _DimsType
    ]
):
    """
    Gates with K branches selected via softmax over K-1 parameters.

    The probability of branch k is softmax([0, theta])[k], where theta has K-1
    learnable parameters for K branches. For K = 2, this reduces to the sigmoid
    parameterization: softmax([0, theta]) = [1 - sigmoid(theta), sigmoid(theta)].

    Subclasses must implement:

    - num_branches: The number of branches K (>= 2)

    - branches: A list of K lookup tables, one per branch
    """

    @property
    @abstractmethod
    def num_branches(self) -> int:
        """Return the number of branches K (must be >= 2)."""
        raise NotImplementedError

    def init_params(self, key: Key[Array, ""]) -> Float[Array, " num_branch_params"]:
        """Initial ``theta`` of shape ``(num_branches - 1,)`` (zeros).

        See [`torx.psc.AbstractPGate.init_params`][].
        """
        return jnp.zeros((self.num_branches - 1,))

    def probs(self, theta: Float[Array, " num_branch_params"]) -> Float[Array, " K"]:
        """Branch probabilities for parameters ``theta``."""
        theta = jnp.atleast_1d(theta)

        if self.num_branches == 2:
            sig = jax.nn.sigmoid(theta[0])
            return jnp.stack([1 - sig, sig])

        padded = jnp.concatenate([jnp.zeros(1), theta])
        return jax.nn.softmax(padded)

    @property
    @abstractmethod
    def branches(self) -> Int[Array, "K basis_size num_sites"]:
        """
        Get the K lookup tables for all branches as a stacked array.

        Each lookup table (along axis 0) maps computational basis state index to
        output state values for each site. For example, for a PCNOT gate the
        operation branch is a CNOT gate. A CNOT gate's action on the computational
        basis states is 00 -> 00, 01 -> 01, 10 -> 11, and 11 -> 10, so the lookup
        table is [[0, 0], [0, 1], [1, 1], [1, 0]].

        The identity branch (often branches[0]) maps each state to itself.
        For the PCNOT example, the identity maps 00 -> 00, 01 -> 01, 10 -> 10,
        11 -> 11, so the lookup table is [[0, 0], [0, 1], [1, 0], [1, 1]].

        For binary gates, basis_size = 2**num_sites.
        For pdit gates, basis_size = prod(dims).

        **Returns:**

        A stacked array of shape (K, basis_size, num_sites) containing all branch
        lookup tables.
        """
        raise NotImplementedError

    # todo: do we want to remove this?
    def prob(self, theta: Float[Array, " num_branch_params"]) -> Float[Array, ""]:
        """Probability of applying the operation (branches[1]); K=2 only."""
        probs = self.probs(theta)
        if len(probs) > 2:
            raise ValueError("prob is undefined for K>2 branches.")
        return probs[1]


class AbstractSingleBinaryPGate(AbstractKBranchGate[int, tuple[int]]):
    """Abstract base class for single-site binary gates (dimension 2)."""

    sites: eqx.AbstractVar[int]

    @property
    def dims(self) -> tuple[int]:  # type: ignore[override]
        return (2,)


class AbstractMultiBinaryPGate(AbstractKBranchGate[list[int], tuple[int, ...]]):
    """Abstract base class for multi-site binary gates (dims = 2)."""

    sites: eqx.AbstractVar[list[int]]

    @property
    def dims(self) -> tuple[int, ...]:  # type: ignore[override]
        return (2,) * len(self.sites)


class AbstractSinglePditGate(AbstractKBranchGate[int, tuple[int, ...]]):
    """Abstract base class for single-site pdit gates (arbitrary dimension k)."""

    pass


class AbstractMultiPditGate(AbstractKBranchGate[list[int], tuple[int, ...]]):
    """Abstract base class for multi-site pdit gates (same dimension k per site)."""

    pass


AbstractSingleBinaryPGate.__init__.__doc__ = """**Arguments:**

- `sites`: The index of the site that this gate acts on.
"""

AbstractMultiBinaryPGate.__init__.__doc__ = """**Arguments:**

- `sites`: A list of site indices that this gate acts on.
"""

AbstractContinuousGate.__init__.__doc__ = """**Arguments:**

- `sites`: List of continuous site indices.
- `dims`: Tuple of continuous dimensions for each site.
"""

AbstractAffineGaussianGate.__init__.__doc__ = """**Arguments:**

- `sites`: List of continuous site indices.
- `dims`: Tuple of continuous dimensions for each site.
"""

AbstractControlledContinuousGate.__init__.__doc__ = """**Arguments:**

- `sites`: Tuple of (discrete_site, continuous_sites).
- `dims`: Tuple of continuous dimensions.
"""

AbstractSinglePditGate.__init__.__doc__ = """**Arguments:**

- `sites`: The index of the site that this gate acts on.
- `dims`: Tuple containing the dimension of the pdit.
"""

AbstractMultiPditGate.__init__.__doc__ = """**Arguments:**

- `sites`: A list of site indices that this gate acts on.
- `dims`: Tuple containing the dimensions of each pdit.
"""
