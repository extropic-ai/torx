"""Exact Gaussian moment simulator for affine continuous hybrid circuits."""

from collections.abc import Mapping, Sequence
from typing import ClassVar, Type

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jaxtyping import Array, Float, Int, PyTree
from typing_extensions import Self

from .._circuit import _HybridGateType, HybridPCircuit
from ..gates import AbstractAffineGaussianGate
from .base import AbstractCompiledPCircuit, AbstractSimulator


class GaussianMoments(eqx.Module):
    """Joint Gaussian moments over continuous sites.

    `mean` and `covariance` follow the order given by `sites`. `site_offsets`
    maps each site to its `(start, stop)` slice into them.
    """

    mean: Float[Array, " continuous_dim"]
    covariance: Float[Array, "continuous_dim continuous_dim"]
    site_offsets: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    observed_sites: tuple[int, ...] = eqx.field(static=True, default=())
    _lookup: dict[int, tuple[int, int]] = eqx.field(static=True, init=False)

    def __post_init__(self):
        # precompute the site -> (start, stop) map once
        self._lookup = _offset_lookup(self.site_offsets)

    @property
    def sites(self) -> tuple[int, ...]:
        """Sites covered by this state, in `mean`/`covariance` order."""
        return tuple(site for site, _, _ in self.site_offsets)

    def site_indices(self, site: int) -> Int[Array, " d"]:
        """Return flat state-vector indices for one continuous site."""
        start, stop = self._lookup[int(site)]
        return jnp.arange(start, stop, dtype=jnp.int32)

    def site_moments(self, site: int) -> tuple[Float[Array, " d"], Float[Array, "d d"]]:
        """Return marginal mean and covariance for one continuous site."""
        idx = self.site_indices(site)
        return self.mean[idx], self.covariance[jnp.ix_(idx, idx)]

    def _indices(self, sites: Sequence[int]) -> Int[Array, " k"]:
        """Flat state-vector indices spanning `sites`, in order."""
        return _flatten_indices(self._lookup, sites)

    def _observation_vector(
        self, sites: Sequence[int], observations: Mapping[int, Array]
    ) -> Float[Array, " observed_dim"]:
        """Concatenate observation values for `sites`, in order.

        Each observation must match its site's dimension. A scalar or otherwise
        mis-shaped value is rejected rather than silently flattened/broadcast,
        which would condition on the wrong vector and return a bogus posterior.
        """
        values = []
        for site in sites:
            start, stop = self._lookup[int(site)]
            value = jnp.asarray(observations[site], dtype=self.mean.dtype).reshape(-1)
            expected = stop - start
            if value.shape[0] != expected:
                raise ValueError(
                    f"observation for site {site} has {value.shape[0]} "
                    f"value(s) but the site spans {expected} dimension(s)."
                )
            values.append(value)
        if not values:
            return jnp.zeros(0, dtype=self.mean.dtype)
        return jnp.concatenate(values)

    def _posterior_offsets(
        self, query_sites: Sequence[int]
    ) -> tuple[tuple[int, int, int], ...]:
        """Re-pack `query_sites` into contiguous `(site, start, stop)` slices."""
        lookup = self._lookup
        offsets = []
        start = 0
        for site in query_sites:
            site_start, site_stop = lookup[int(site)]
            stop = start + site_stop - site_start
            offsets.append((site, start, stop))
            start = stop
        return tuple(offsets)


class CompiledAffineGaussianPCircuit(AbstractCompiledPCircuit[HybridPCircuit]):
    """Compiled hybrid circuit for the affine Gaussian simulator."""

    gates: list[_HybridGateType]
    thetas: list[PyTree[Array]]
    site_offsets: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    reps: int

    @classmethod
    def from_pcircuit(
        cls, circuit: HybridPCircuit, thetas: list[PyTree[Array]]
    ) -> Self:
        """Compile `circuit` for exact affine Gaussian moment propagation.

        **Arguments:**

        - `circuit`: The hybrid circuit to compile.
        - `thetas`: Per-gate parameters aligned with `circuit.gates`.

        **Returns:**

        The compiled circuit.
        """
        if len(thetas) != len(circuit.gates):
            raise ValueError(
                f"expected {len(circuit.gates)} thetas aligned with "
                f"circuit.gates, got {len(thetas)}."
            )
        return cls(circuit.gates, thetas, circuit.continuous_offsets, circuit.reps)

    def to_pcircuit(self, structure: HybridPCircuit) -> HybridPCircuit:
        """Return a `HybridPCircuit` with `structure`'s gates and compiled reps."""
        return HybridPCircuit(structure.gates, self.reps)


class AffineGaussianSimulator(
    AbstractSimulator[HybridPCircuit, CompiledAffineGaussianPCircuit]
):
    """Exact moment simulator for the affine Gaussian fragment of hybrid circuits.

    Propagates the joint mean and covariance in closed form through each gate's
    affine Gaussian channel `(A, b, log_var)`, exposed via
    [`AbstractAffineGaussianGate`][torx.psc.AbstractAffineGaussianGate], and
    conditions on observed sites via the Schur complement. Only supports
    Gaussian gates.

    Dense reference implementation: `O(D^3)` per gate in the total continuous
    dimension `D`; intended for small affine-Gaussian circuits.
    """

    circuit_backend: ClassVar[Type[AbstractCompiledPCircuit]] = (
        CompiledAffineGaussianPCircuit
    )

    def build_circuit(
        self,
        circuit: HybridPCircuit,
        thetas: list[PyTree[Array]],
    ) -> CompiledAffineGaussianPCircuit:
        """Compile `circuit` for exact affine Gaussian moment propagation."""
        return CompiledAffineGaussianPCircuit.from_pcircuit(circuit, thetas)

    def propagate(
        self,
        circuit: CompiledAffineGaussianPCircuit,
        initial_continuous: Float[Array, " continuous_dim"],
    ) -> GaussianMoments:
        """Propagate joint Gaussian moments through an affine-Gaussian circuit.

        **Arguments:**

        - `circuit`: The compiled affine Gaussian circuit to execute.
        - `initial_continuous`: Flat initial continuous state.

        **Returns:**

        `GaussianMoments` over every continuous site.
        """
        site_offsets = circuit.site_offsets
        lookup = _offset_lookup(site_offsets)
        dim = site_offsets[-1][2] if site_offsets else 0
        initial = jnp.asarray(initial_continuous)
        default_dtype = _infer_default_dtype(circuit)
        with jax.numpy_dtype_promotion("standard"):
            state_dtype = jnp.result_type(initial.dtype, default_dtype)
        mean = initial.astype(state_dtype)
        covariance = jnp.zeros((dim, dim), dtype=mean.dtype)

        def body_fn(_: int, state: tuple) -> tuple:
            mean, covariance = state
            for gate, theta in zip(circuit.gates, circuit.thetas, strict=True):
                if isinstance(gate, AbstractAffineGaussianGate):
                    indices = _flatten_indices(lookup, gate.sites["continuous"])
                    A, b, log_var = gate.affine_parameters(theta)
                    mean, covariance = _apply_linear_gaussian(
                        mean,
                        covariance,
                        indices,
                        A,
                        b,
                        log_var,
                    )
                else:
                    msg = (
                        "AffineGaussianSimulator supports continuous gates "
                        "that implement AbstractAffineGaussianGate, got "
                        f"{type(gate).__name__}."
                    )
                    raise ValueError(msg)
            return mean, covariance

        mean, covariance = jax.lax.fori_loop(
            0, circuit.reps, body_fn, (mean, covariance)
        )
        return GaussianMoments(mean, _symmetrize(covariance), site_offsets)

    def condition(
        self,
        circuit: CompiledAffineGaussianPCircuit,
        observations: Mapping[int, Array] | None = None,
        *,
        initial_continuous: Float[Array, " continuous_dim"],
        query_sites: Sequence[int] | None = None,
        jitter: Float[Array, ""] | float = 0.0,
    ) -> GaussianMoments:
        """Condition final Gaussian moments on continuous-site observations.

        The circuit is propagated to a joint Gaussian over the final continuous
        state, then the queried sites are conditioned on the observed ones via
        the Schur complement. The solve uses a Cholesky factorization of the
        observed covariance plus `jitter` on the diagonal.
        Observed/query site membership must be static Python-level dict keys or
        `Sequence` entries; only observation values and `jitter` may be traced,
        so JIT/vmap over which sites are observed is unsupported by design.

        **Arguments:**

        - `circuit`: The compiled affine Gaussian circuit to execute.
        - `observations`: Mapping from each observed continuous site to its
          value.
        - `initial_continuous`: Flat initial continuous state.
        - `query_sites`: Sites to return. If `None`, all unobserved sites are
          queried.
        - `jitter`: Nonnegative diagonal regularizer for the observed
          covariance.

        **Returns:**

        `GaussianMoments` over the queried sites, with `observed_sites` set.
        """
        jitter = jnp.asarray(jitter)
        jitter = eqx.error_if(
            jitter, jnp.any(jitter < 0), "jitter must be nonnegative."
        )

        prior = self.propagate(circuit, initial_continuous)
        observations = {} if observations is None else observations
        observed_sites = tuple(sorted(int(site) for site in observations))
        if query_sites is None:
            query_sites = tuple(
                site for site in prior.sites if site not in observed_sites
            )
        else:
            query_sites = tuple(int(site) for site in query_sites)

        known = set(prior.sites)
        unknown = sorted((set(observed_sites) | set(query_sites)) - known)
        if unknown:
            raise ValueError(
                f"sites {unknown} are not continuous sites of this circuit; "
                f"available sites are {prior.sites}."
            )

        overlap = sorted(set(query_sites) & set(observed_sites))
        if overlap:
            raise ValueError(
                "query_sites and observed sites must be disjoint, "
                f"got overlap on {overlap}."
            )

        query_idx = prior._indices(query_sites)
        query_offsets = prior._posterior_offsets(query_sites)
        y = prior._observation_vector(observed_sites, observations)

        if len(query_sites) == 0 or len(observed_sites) == 0:
            mean = prior.mean[query_idx]
            covariance = prior.covariance[jnp.ix_(query_idx, query_idx)]
            return GaussianMoments(
                mean,
                _symmetrize(covariance),
                query_offsets,
                observed_sites,
            )

        observed_idx = prior._indices(observed_sites)
        mu_q = prior.mean[query_idx]
        mu_o = prior.mean[observed_idx]
        cov_qq = prior.covariance[jnp.ix_(query_idx, query_idx)]
        cov_qo = prior.covariance[jnp.ix_(query_idx, observed_idx)]
        cov_oo = prior.covariance[jnp.ix_(observed_idx, observed_idx)]

        # Standard Gaussian conditioning via the Schur complement. Cholesky-
        # factor cov_oo (+ jitter) once; `cho_solve((L, True), cov_qo.T)` gives
        # gain^T = cov_oo^{-1} cov_oq, so the posterior is
        #   mean = mu_q + gain @ (y - mu_o),  cov = cov_qq - cov_qo @ gain^T.
        eye = jnp.eye(cov_oo.shape[0], dtype=cov_oo.dtype)
        cov_oo_reg = cov_oo + jnp.asarray(jitter, dtype=cov_oo.dtype) * eye
        L = jnp.linalg.cholesky(cov_oo_reg)
        L = eqx.error_if(
            L,
            jnp.any(jnp.isnan(L)),
            "observed covariance is not positive definite, "
            "pass positive jitter to regularize",
        )
        gain_t = jsl.cho_solve((L, True), cov_qo.T)
        mean = mu_q + gain_t.T @ (y - mu_o)
        covariance = cov_qq - cov_qo @ gain_t
        return GaussianMoments(
            mean,
            _symmetrize(covariance),
            query_offsets,
            observed_sites,
        )

    def expval(
        self,
        circuit: CompiledAffineGaussianPCircuit,
        initial_continuous: Float[Array, " continuous_dim"],
        site: int = 0,
    ) -> Float[Array, " d"]:
        """Return the marginal mean of one continuous site.

        **Arguments:**

        - `circuit`: The compiled affine Gaussian circuit to execute.
        - `initial_continuous`: Flat initial continuous state.
        - `site`: The continuous site whose marginal mean to return.

        **Returns:**

        The marginal mean of `site`.
        """
        return self.propagate(circuit, initial_continuous).site_moments(site)[0]

    def expval_all(
        self,
        circuit: CompiledAffineGaussianPCircuit,
        initial_continuous: Float[Array, " continuous_dim"],
    ) -> Float[Array, " continuous_dim"]:
        """Return the joint mean over all continuous sites."""
        return self.propagate(circuit, initial_continuous).mean


def _infer_default_dtype(circuit: CompiledAffineGaussianPCircuit) -> jnp.dtype:
    """Promote affine gate dtypes to set the default state dtype.

    Uses `jax.eval_shape` to read parameter dtypes without materializing the
    `(A, b, log_var)` arrays. With no affine gates, falls back to the default
    float type, which honors `jax_enable_x64` (float64 under x64, else float32).
    """
    dtypes: list[jnp.dtype] = []
    for gate, theta in zip(circuit.gates, circuit.thetas, strict=True):
        if isinstance(gate, AbstractAffineGaussianGate):
            shapes = jax.eval_shape(gate.affine_parameters, theta)
            dtypes.extend(s.dtype for s in shapes)
    if not dtypes:
        return jnp.result_type(float)
    # Mixed fp32/fp64 gate params promote under standard rules (propagate casts
    # everything to the state dtype anyway); strict promotion would reject the
    # mix here before propagation gets the chance.
    with jax.numpy_dtype_promotion("standard"):
        return jnp.result_type(*dtypes)


def _offset_lookup(
    site_offsets: tuple[tuple[int, int, int], ...],
) -> dict[int, tuple[int, int]]:
    """Build the `site -> (start, stop)` slice map once, for shared reuse."""
    return {site: (start, stop) for site, start, stop in site_offsets}


def _flatten_indices(
    lookup: dict[int, tuple[int, int]],
    sites: Sequence[int],
) -> Int[Array, " k"]:
    """Concatenate the flat state-vector indices spanning `sites`, in order."""
    parts = []
    for site in sites:
        start, stop = lookup[int(site)]
        parts.append(jnp.arange(start, stop, dtype=jnp.int32))
    if not parts:
        return jnp.array([], dtype=jnp.int32)
    return jnp.concatenate(parts)


def _apply_linear_gaussian(
    mean: Float[Array, " continuous_dim"],
    covariance: Float[Array, "continuous_dim continuous_dim"],
    indices: Int[Array, " local_dim"],
    A: Float[Array, "local_dim local_dim"],
    b: Float[Array, " local_dim"],
    log_var: Float[Array, " local_dim"],
) -> tuple[
    Float[Array, " continuous_dim"],
    Float[Array, "continuous_dim continuous_dim"],
]:
    """Apply a local affine Gaussian channel by lifting it to dense state space."""
    # Gate params may be narrower than the promoted state dtype under strict mode.
    A = A.astype(mean.dtype)
    b = b.astype(mean.dtype)
    log_var = log_var.astype(mean.dtype)
    # dense lift: A on this gate's sites, identity elsewhere, plus bias and
    # diagonal noise
    dim = mean.shape[0]
    transform = jnp.eye(dim, dtype=mean.dtype).at[indices, :].set(0.0)
    transform = transform.at[jnp.ix_(indices, indices)].set(A)
    bias = jnp.zeros(dim, dtype=mean.dtype).at[indices].set(b)
    noise = jnp.zeros((dim, dim), dtype=mean.dtype)
    noise = noise.at[jnp.ix_(indices, indices)].set(jnp.diag(jnp.exp(log_var)))
    new_mean = transform @ mean + bias
    new_covariance = transform @ covariance @ transform.T + noise
    return new_mean, _symmetrize(new_covariance)


def _symmetrize(matrix: Float[Array, "d d"]) -> Float[Array, "d d"]:
    return 0.5 * (matrix + matrix.T)
