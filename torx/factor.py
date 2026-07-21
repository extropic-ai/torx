"""Abstract base classes for probabilistic factors."""

from abc import abstractmethod
from typing import Any, Mapping, TypeAlias

import equinox as eqx
import jax
from ihoop.eqx import AbstractStrictModule
from jaxtyping import Array, Key, PyTree

PortSpec: TypeAlias = PyTree[jax.ShapeDtypeStruct]
ParamsTree: TypeAlias = PyTree[Array]
InfoTree: TypeAlias = PyTree
_PortSpec: TypeAlias = PortSpec
_ParamsTree: TypeAlias = ParamsTree
_InfoTree: TypeAlias = InfoTree
_SampleOutput: TypeAlias = PyTree[Array] | tuple[PyTree[Array], PyTree[Array]]


class AbstractFactor(AbstractStrictModule):
    """Base class for probabilistic factors.

    A directed factor is a conditional distribution
    $P(\\text{output} \\mid \\text{inputs})$.

    Factor methods operate with the following:

    - `params`: the factor's parameter pytree.
    - `info`: optional runtime auxiliary info.
    - `site_info`: optional per-site metadata supplied by the
        surrounding `Site`.
    - `return_aux`: when `True`, methods additionally return a
        factor-defined `aux` pytree.
    """

    input_ports: eqx.AbstractVar[Mapping[str, _PortSpec]]
    output_spec: eqx.AbstractVar[_PortSpec]

    @abstractmethod
    def sample(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: _ParamsTree,
        info: _InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> _SampleOutput:
        """Draw a single sample from this factor.

        **Arguments:**

        - `key`: PRNG key.
        - `inputs`: Per-port pytree inputs. Keys must equal the
            `self.input_ports` keys.
        - `params`: Parameter pytree for this factor.
        - `info`: Runtime auxiliary info.
        - `site_info`: Static per-site metadata supplied by the
            surrounding `Site`.
        - `return_aux`: If `True`, return `(output, aux)`.

        **Returns:**

        The sampled output (a pytree matching `output_spec`), or
        `(output, aux)` when `return_aux=True`.
        """
        raise NotImplementedError

    @abstractmethod
    def init_params(self, key: Key[Array, ""]) -> _ParamsTree:
        """Return a freshly-initialised parameter pytree.

        **Arguments:**

        - `key`: PRNG key.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_with_references(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: _ParamsTree,
        info: _InfoTree = None,
        site_info: Any = None,
        n_references: int = 1,
    ) -> tuple[PyTree[Array], PyTree[Array]]:
        r"""Draw a "main" sample plus `n_references` additional samples.

        Always returns `(main_output, aux)`. `main_output` is a single
        sample. Every `aux` leaf gains a leading axis of size
        `n_references + 1`: position `0` is the main sample's
        aux, positions `1 .. n_references` are the references' auxes.

        For a sample with no references and no aux, use `sample`.

        **Arguments:**

        - `key`: PRNG key.
        - `inputs`: Per-port pytree inputs, as in `sample`.
        - `params`: Parameter pytree for this factor.
        - `info`: Runtime auxiliary info, as in `sample`.
        - `site_info`: Static per-site metadata, as in `sample`.
        - `n_references`: Number of reference samples to draw alongside
            the main sample.

        **Returns:**

        `(main_output, aux)`.
        """
        raise NotImplementedError

    def sample_multiple(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: _ParamsTree,
        info: _InfoTree = None,
        site_info: Any = None,
        n_samples: int = 1,
        return_aux: bool = False,
    ) -> _SampleOutput:
        """Draw `n_samples` samples by `vmap`-ing `sample`.

        **Arguments:**

        - `key`: PRNG key, split `n_samples` ways.
        - `inputs`: Per-port pytree inputs, as in `sample`.
        - `params`: Parameter pytree for this factor.
        - `info`: Runtime auxiliary info, as in `sample`.
        - `site_info`: Static per-site metadata, as in `sample`.
        - `n_samples`: Number of independent samples to draw.
        - `return_aux`: Whether to return the `aux` pytree, as in `sample`.

        **Returns:**

        The stacked outputs, or `(outputs, auxes)` when `return_aux=True`,
        each with a leading axis of size `n_samples`.
        """
        keys = jax.random.split(key, n_samples)

        def _single(
            k: Key[Array, ""],
        ) -> _SampleOutput:
            return self.sample(
                k, inputs, params, info, site_info, return_aux=return_aux
            )

        return jax.vmap(_single)(keys)


class AbstractReferenceFactor(AbstractFactor):
    """Abstract factor + a concrete `sample_with_references`."""

    def sample_with_references(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: _ParamsTree,
        info: _InfoTree = None,
        site_info: Any = None,
        n_references: int = 1,
    ) -> tuple[PyTree[Array], PyTree[Array]]:
        """Generic reference-sampling default.

        Draws `n_references + 1` samples via `sample_multiple` and returns
        the first as the main sample, with all of their auxes stacked.
        """
        samples, aux = self.sample_multiple(
            key,
            inputs,
            params,
            info,
            site_info,
            n_samples=n_references + 1,
            return_aux=True,
        )
        main = jax.tree.map(lambda x: x[0], samples)
        return main, aux
