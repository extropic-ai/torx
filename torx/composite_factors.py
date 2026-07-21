from typing import Any, Callable, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key, PyTree

from ._utils import same_pytree_spec
from .factor import _InfoTree, _ParamsTree, _PortSpec, _SampleOutput, AbstractFactor


def _prepend_axis(spec: _PortSpec, n: int) -> _PortSpec:
    """Prepend a leading axis of size `n` to every `ShapeDtypeStruct` in
    `spec`.
    """
    return jax.tree.map(
        lambda s: jax.ShapeDtypeStruct((n,) + tuple(s.shape), s.dtype),
        spec,
    )


class AbstractTiledFactor(AbstractFactor):
    """Base implementation for factors tiled across independent replicas.

    `batch_size` chooses the mapping mode: `None`, `0`, or any value
    `>= n_tiles` uses `jax.vmap` (all tiles at once); a smaller value uses
    `jax.lax.map` with chunks of that size.

    The `info` argument is the runtime info forwarded to each tile's
    `base.sample`. By default the same `info` is broadcast to every tile; set
    `slice_info=True` to instead pass per-tile info.

    **Arguments:**

    - `base`: The factor to replicate.
    - `n_tiles`: Number of tiles.
    - `weight_tied`: Whether all tiles share one parameter set (see above).
    - `batch_size`: `vmap` vs `lax.map` execution strategy (see above).
    - `slice_info`: Whether `info` is sliced per tile (see above).
    """

    base: eqx.AbstractVar[AbstractFactor]
    n_tiles: eqx.AbstractVar[int]
    weight_tied: eqx.AbstractVar[bool]
    batch_size: eqx.AbstractVar[int | None]
    slice_info: eqx.AbstractVar[bool]
    input_ports: eqx.AbstractVar[Mapping[str, _PortSpec]]
    output_spec: eqx.AbstractVar[_PortSpec]

    def __init__(
        self,
        base: AbstractFactor,
        n_tiles: int,
        weight_tied: bool,
        *,
        batch_size: int | None = None,
        slice_info: bool = False,
    ) -> None:
        self.base = base
        self.n_tiles = n_tiles
        self.weight_tied = weight_tied
        self.batch_size = batch_size
        self.slice_info = slice_info
        self.input_ports = {
            name: _prepend_axis(spec, n_tiles)
            for name, spec in base.input_ports.items()
        }
        self.output_spec = _prepend_axis(base.output_spec, n_tiles)

    def sample(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: _ParamsTree,
        info: _InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> _SampleOutput:
        """Sample every tile and stack the results along a leading `n_tiles`
        axis. With `return_aux=True` each tile's aux is stacked likewise.
        """
        main, aux = self._run(
            key,
            inputs,
            params,
            info,
            site_info,
            n_references=None,
            with_aux=return_aux,
        )
        return (main, aux) if return_aux else main

    def sample_with_references(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: _ParamsTree,
        info: _InfoTree = None,
        site_info: Any = None,
        n_references: int = 1,
    ) -> tuple[PyTree[Array], PyTree[Array]]:
        """Per-tile [`torx.AbstractFactor.sample_with_references`][].

        Each tile calls `base.sample_with_references(n_references)` and the
        results are stacked across tiles. The `main` output leaves have shape
        `(n_tiles, *base_output_shape)` and aux leaves have shape
        `(n_references + 1, n_tiles, *base_aux_leaf_shape)`.
        """
        main, aux = self._run(
            key,
            inputs,
            params,
            info,
            site_info,
            n_references=n_references,
            with_aux=True,
        )
        # `_run` stacks per-tile results tile-first, giving aux leaves
        # `(n_tiles, n_references + 1, ...)`.
        aux = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), aux)
        return main, aux

    def _run(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: _ParamsTree,
        info: _InfoTree,
        site_info: Any,
        n_references: int | None,
        with_aux: bool,
    ) -> tuple[PyTree[Array], PyTree[Array]]:
        """`vmap` or `lax.map` the base factor across tiles.

        Returns `(stacked_main, stacked_aux_or_None_tree)`.
        """
        keys = jax.random.split(key, self.n_tiles)
        base = self.base
        slice_info = self.slice_info
        batch_size = self.batch_size
        weight_tied = self.weight_tied

        def single(args):
            # `args[2]` is the per-tile `params` slice when untied (else `None`,
            # tied `params` are used); `args[3]` is the per-tile `info` slice when
            # `slice_info` (else `None`, and the broadcast `info` is used).
            k, inp = args[0], args[1]
            par = params if weight_tied else args[2]
            inf = args[3] if slice_info else info
            if n_references is None:
                if with_aux:
                    return base.sample(k, inp, par, inf, site_info, return_aux=True)
                return (
                    base.sample(k, inp, par, inf, site_info, return_aux=False),
                    None,
                )
            return base.sample_with_references(
                k, inp, par, inf, site_info, n_references=n_references
            )

        param_xs = None if weight_tied else params
        info_xs = info if slice_info else None
        xs = (keys, dict(inputs), param_xs, info_xs)
        if batch_size is None or batch_size == 0 or batch_size >= self.n_tiles:
            return jax.vmap(single)(xs)
        return jax.lax.map(single, xs, batch_size=batch_size)

    def init_params(self, key: Key[Array, ""]) -> _ParamsTree:
        """`base.init_params` when `weight_tied`, else `vmap`-ed over `n_tiles`."""
        if self.weight_tied:
            return self.base.init_params(key)
        keys = jax.random.split(key, self.n_tiles)
        return jax.vmap(self.base.init_params)(keys)


class TiledFactor(AbstractTiledFactor):
    """Replication of a base factor across `n_tiles` independent tiles."""

    base: AbstractFactor
    n_tiles: int = eqx.field(static=True)
    weight_tied: bool = eqx.field(static=True)
    batch_size: int | None = eqx.field(static=True)
    slice_info: bool = eqx.field(static=True)
    input_ports: Mapping[str, _PortSpec] = eqx.field(static=True)
    output_spec: _PortSpec = eqx.field(static=True)


class AbstractChainFactor(AbstractFactor):
    """Base implementation for factors chained via `jax.lax.scan`.

    At each step the base's input ports split into two disjoint sets:

    - Feedback ports: receive `feedback_porting_fn(previous step's main
        output)`; at step 0 they take the chain's initial state.
    - Broadcast ports: get the caller's value unchanged at every step.

    `feedback_porting_fn` is either a port-name `str` (for `lambda main: {str: main}`)
    or a `Callable` mapping the previous main output to a dict of feedback-port values.

    **Arguments:**

    - `base`: The factor applied at every step.
    - `n_steps`: Number of steps.
    - `feedback_porting_fn`: Port-name `str` or `Callable` routing each step's
        main output into the next step.
    - `weight_tied`: Whether all steps share one parameter set.
    - `slice_info`: Whether `info` is sliced per step.
    """

    base: eqx.AbstractVar[AbstractFactor]
    n_steps: eqx.AbstractVar[int]
    # `None` when built from the port-name `str` shorthand (the single feedback
    # port is then `feedback_ports[0]`); the user's callable otherwise.
    feedback_porting_fn: eqx.AbstractVar[
        Callable[[PyTree[Array]], Mapping[str, PyTree[Array]]] | None
    ]
    feedback_ports: eqx.AbstractVar[tuple[str, ...]]
    all_step_input_ports: eqx.AbstractVar[tuple[str, ...]]
    weight_tied: eqx.AbstractVar[bool]
    slice_info: eqx.AbstractVar[bool]
    input_ports: eqx.AbstractVar[Mapping[str, _PortSpec]]
    output_spec: eqx.AbstractVar[_PortSpec]

    def __init__(
        self,
        base: AbstractFactor,
        n_steps: int,
        feedback_porting_fn: (
            Callable[[PyTree[Array]], Mapping[str, PyTree[Array]]] | str
        ),
        weight_tied: bool,
        *,
        slice_info: bool = False,
    ) -> None:
        if isinstance(feedback_porting_fn, str):
            fb_port = feedback_porting_fn
            if fb_port not in base.input_ports:
                raise ValueError(
                    f"feedback_porting_fn '{fb_port}' is not an input port of "
                    f"base factor (available: {base.input_ports.keys()})."
                )
            feedback_ports = (fb_port,)
            feedback_struct = {fb_port: base.output_spec}
            stored_porting_fn = None
        elif callable(feedback_porting_fn):
            # dict from feedback-port name to spec
            feedback_struct = dict(
                eqx.filter_eval_shape(feedback_porting_fn, base.output_spec)
            )
            feedback_ports = tuple(feedback_struct.keys())
            stored_porting_fn = feedback_porting_fn

        missing = set(feedback_ports) - set(base.input_ports.keys())
        if missing:
            raise ValueError(
                f"feedback_porting_fn produces port(s) {sorted(missing)} not in "
                f"base.input_ports {base.input_ports.keys()}."
            )

        fb_in_spec = {k: base.input_ports[k] for k in feedback_ports}
        if not same_pytree_spec(feedback_struct, fb_in_spec):
            raise ValueError(
                f"feedback_porting_fn output spec "
                f"{eqx.tree_pformat(feedback_struct, struct_as_array=True)} does "
                f"not match base.input_ports spec "
                f"{eqx.tree_pformat(fb_in_spec, struct_as_array=True)}."
            )

        all_step_input_ports = tuple(
            sorted(set(base.input_ports.keys()) - set(feedback_ports))
        )

        self.base = base
        self.n_steps = n_steps
        self.feedback_porting_fn = stored_porting_fn
        self.feedback_ports = feedback_ports
        self.all_step_input_ports = all_step_input_ports
        self.weight_tied = weight_tied
        self.slice_info = slice_info
        self.input_ports = dict(base.input_ports)
        self.output_spec = base.output_spec

    def sample(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: _ParamsTree,
        info: _InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> _SampleOutput:
        """Run the scan and return the final step's output. With
        `return_aux=True` additionally returns the per-step aux stacked along
        a leading axis of size `n_steps`.
        """
        final_main, aux_trace = self._run(
            key,
            inputs,
            params,
            info,
            site_info,
            n_references=None,
            with_aux=return_aux,
        )
        return (final_main, aux_trace) if return_aux else final_main

    def sample_with_references(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: _ParamsTree,
        info: _InfoTree = None,
        site_info: Any = None,
        n_references: int = 1,
    ) -> tuple[PyTree[Array], PyTree[Array]]:
        """Per-step [`torx.AbstractFactor.sample_with_references`][] along the
        scan.

        Each step calls `base.sample_with_references(n_references)`; the main
        output of the step is threaded as the scan carry.The returned `aux_trace`
        leaves have shape `(n_references + 1, n_steps, *base_aux_leaf_shape)`.
        """
        final_main, aux_trace = self._run(
            key,
            inputs,
            params,
            info,
            site_info,
            n_references=n_references,
            with_aux=True,
        )
        aux_trace = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), aux_trace)
        return final_main, aux_trace

    def _run(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: _ParamsTree,
        info: _InfoTree,
        site_info: Any,
        n_references: int | None,
        with_aux: bool,
    ) -> tuple[PyTree[Array], PyTree[Array]]:
        """Run the scan over base steps. Returns `(final_main, aux_trace)`."""
        keys = jax.random.split(key, self.n_steps)
        all_step_inputs = {k: inputs[k] for k in self.all_step_input_ports}
        initial_feedback = {k: inputs[k] for k in self.feedback_ports}
        initial_step_main = jax.tree.map(
            lambda s: jnp.zeros(s.shape, s.dtype), self.base.output_spec
        )
        base = self.base
        feedback_porting_fn = self.feedback_porting_fn
        feedback_ports = self.feedback_ports
        tied = self.weight_tied
        slice_info = self.slice_info

        def step_fn(carry, scan_input):
            step_key, step_params_in, step_info_in = scan_input
            step_params = params if tied else step_params_in
            step_info = step_info_in if slice_info else info
            feedback, _prev_main = carry
            step_inputs = feedback | all_step_inputs
            if n_references is None:
                if with_aux:
                    step_main, step_aux = base.sample(
                        step_key,
                        step_inputs,
                        step_params,
                        step_info,
                        site_info,
                        return_aux=True,
                    )
                else:
                    step_main = base.sample(
                        step_key,
                        step_inputs,
                        step_params,
                        step_info,
                        site_info,
                        return_aux=False,
                    )
                    step_aux = None
            else:
                step_main, step_aux = base.sample_with_references(
                    step_key,
                    step_inputs,
                    step_params,
                    step_info,
                    site_info,
                    n_references=n_references,
                )
            if feedback_porting_fn is not None:
                next_feedback = feedback_porting_fn(step_main)
            else:
                next_feedback = {feedback_ports[0]: step_main}
            return (next_feedback, step_main), step_aux

        params_xs = None if tied else params
        info_xs = info if slice_info else None
        scan_xs = (keys, params_xs, info_xs)
        initial_carry = (initial_feedback, initial_step_main)
        (_, final_main), aux_trace = jax.lax.scan(step_fn, initial_carry, scan_xs)
        return final_main, aux_trace

    def init_params(self, key: Key[Array, ""]) -> _ParamsTree:
        """`vmap(base.init_params)` per step, or one shared init when
        `weight_tied`.
        """
        if self.weight_tied:
            return self.base.init_params(key)
        keys = jax.random.split(key, self.n_steps)
        return jax.vmap(self.base.init_params)(keys)


class ChainFactor(AbstractChainFactor):
    """Sequential composition of a base factor via `jax.lax.scan`."""

    base: AbstractFactor
    n_steps: int = eqx.field(static=True)
    feedback_porting_fn: (
        Callable[[PyTree[Array]], Mapping[str, PyTree[Array]]] | None
    ) = eqx.field(static=True)
    feedback_ports: tuple[str, ...] = eqx.field(static=True)
    all_step_input_ports: tuple[str, ...] = eqx.field(static=True)
    weight_tied: bool = eqx.field(static=True)
    slice_info: bool = eqx.field(static=True)
    input_ports: Mapping[str, _PortSpec] = eqx.field(static=True)
    output_spec: _PortSpec = eqx.field(static=True)
