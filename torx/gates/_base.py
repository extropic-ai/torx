"""Abstract base classes for probabilistic gates."""

from abc import abstractmethod
from typing import ClassVar, Generic, TypeAlias, TypedDict, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Key, PyTree

from .._custom_meta import AbstractStrictModule


class HybridSites(TypedDict):
    discrete: list[int]
    continuous: list[int]


DiscreteSites: TypeAlias = int | list[int]
_SiteType = TypeVar("_SiteType", bound=HybridSites | DiscreteSites)
_DiscreteSiteType = TypeVar("_DiscreteSiteType", bound=DiscreteSites)
_ThetaType = TypeVar("_ThetaType", bound=PyTree[Array])
_DimsType = TypeVar("_DimsType", bound=tuple[int, ...])


class AbstractPGate(AbstractStrictModule, Generic[_SiteType, _ThetaType, _DimsType]):
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

    theta: eqx.AbstractVar[_ThetaType]
    sites: eqx.AbstractVar[_SiteType]
    dims: eqx.AbstractVar[_DimsType]


class AbstractDiscreteGate(AbstractPGate[_DiscreteSiteType, _ThetaType, _DimsType]):
    @abstractmethod
    def get_matrix(self) -> Float[Array, "dim1 dim2"]:
        """
        Get the matrix representation of the gate.

        **Returns:**

        The matrix representation of the probabilistic gate.
        """
        raise NotImplementedError


class AbstractHybridGate(AbstractPGate[HybridSites, _ThetaType, _DimsType]):
    """Base class for hybrid gates.

    Gates specify which discrete and continuous sites they act on via the `sites`
    dict. The `sample()` method receives the relevant substates and returns the
    new continuous substate.

    !!! note

        `torx` currently only supports discrete controlled continuous gates.
    """

    @abstractmethod
    def sample(
        self,
        substate: dict[str, Array],
        key: Key[Array, ""],
    ) -> Float[Array, "..."]:
        """Sample new continuous substate.

        **Arguments:**

        - `substate`: Dict with "discrete" and/or "continuous" keys containing
          the values at the gate's sites.
        - `key`: JAX random key.

        **Returns:**

        New values for sites["continuous"], same shape as substate["continuous"].
        """
        raise NotImplementedError


class AbstractContinuousGate(AbstractHybridGate[_ThetaType, _DimsType]):
    """Gate that only touches continuous sites.

    Subclasses set sites to specify which continuous sites they act on.
    """

    pass


class AbstractControlledContinuousGate(AbstractHybridGate[_ThetaType, _DimsType]):
    """Gate where discrete sites control the continuous transformation.

    !!! note

        Currently we only support control by a single discrete site (pbit/pdit),
        so dims only specifies continuous dimensions. Multi-site discrete control
        would require tracking discrete dims as well.
    """

    def _get_discrete_control(self, substate: dict[str, Array]) -> Array:
        """Extract single discrete control value with validation."""
        disc = substate["discrete"]
        if disc.shape != (1,):
            raise ValueError(
                f"{self.__class__.__name__} expects exactly 1 discrete control "
                f"site, got shape {disc.shape}"
            )
        return disc[0]


# todo: type as k-1
class AbstractKBranchGate(
    AbstractDiscreteGate[_DiscreteSiteType, Float[Array, "..."], _DimsType]
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

    theta: eqx.AbstractVar[Float[Array, "..."]]

    @property
    @abstractmethod
    def num_branches(self) -> int:
        """Return the number of branches K (must be >= 2)."""
        raise NotImplementedError

    @property
    def probs(self) -> Float[Array, " K"]:
        """Branch probabilities."""
        theta = jnp.atleast_1d(self.theta)

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
    @property
    def prob(self) -> Float[Array, ""]:
        """The probability of applying the operation (branches[1]). For K=2 only."""
        if len(self.probs) > 2:
            raise ValueError("prob is undefined for K>2 branches.")
        return self.probs[1]


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

- `theta`: The gate parameter. The probability of applying the gate is `sigmoid(theta)`.
- `sites`: The index of the site that this gate acts on.
"""

AbstractMultiBinaryPGate.__init__.__doc__ = """**Arguments:**

- `theta`: The gate parameter. The probability of applying the gate is `sigmoid(theta)`.
- `sites`: A list of site indices that this gate acts on.
"""

AbstractContinuousGate.__init__.__doc__ = """**Arguments:**

- `theta`: Gate parameters (structure specified in subclass docstring).
- `sites`: List of continuous site indices.
- `dims`: Tuple of continuous dimensions for each site.
"""

AbstractControlledContinuousGate.__init__.__doc__ = """**Arguments:**

- `theta`: Gate parameters (structure specified in subclass docstring).
- `sites`: Tuple of (discrete_site, continuous_sites).
- `dims`: Tuple of continuous dimensions.
"""

AbstractSinglePditGate.__init__.__doc__ = """**Arguments:**

- `theta`: The gate parameter. The probability of applying the gate is `sigmoid(theta)`.
- `sites`: The index of the site that this gate acts on.
- `dims`: Tuple containing the dimension of the pdit.
"""

AbstractMultiPditGate.__init__.__doc__ = """**Arguments:**

- `theta`: The gate parameter. The probability of applying the gate is `sigmoid(theta)`.
- `sites`: A list of site indices that this gate acts on.
- `dims`: Tuple containing the dimensions of each pdit.
"""
