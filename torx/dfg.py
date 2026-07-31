from collections import deque
from typing import Any, Callable, Mapping, Sequence, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key, PyTree

from .factor import AbstractFactor, InfoTree, ParamsTree, PortSpec


def _convert_parents(parents):
    if isinstance(parents, str):
        raise TypeError(
            f"parents={parents!r} is a bare string; use [{parents!r}] for a "
            f"single parent."
        )
    return tuple(parents)


def _convert_porting_fn(porting_fn: Any) -> Any:
    """Pass a callable through; otherwise convert to a tuple of port names."""
    if callable(porting_fn):
        return porting_fn
    if isinstance(porting_fn, str):
        raise TypeError(
            f"porting_fn={porting_fn!r} is a bare string; use [{porting_fn!r}] "
            f"for a single port."
        )
    return tuple(porting_fn)


class Site(eqx.Module):
    """The placement of a `Factor` at one position in a DFG."""

    name: str = eqx.field(static=True)
    factor: AbstractFactor
    parents: tuple[str, ...] = eqx.field(
        static=True, converter=_convert_parents
    )  # parent addresses, in order
    porting_fn: (
        Callable[[Sequence[PyTree[Array]]], Mapping[str, PyTree[Array]]]
        | tuple[str, ...]
    ) = eqx.field(
        static=True, converter=_convert_porting_fn
    )  # routes parent outputs into the factor's inputs
    param_key: str | int | None = eqx.field(static=True)  # address into the params tree
    info_key: str | None = eqx.field(static=True)  # key into the DFG's info entries
    site_info: Any = eqx.field(static=True)  # per-site metadata for the factor


Site.__init__.__doc__ = """Construct a `Site`.

**Arguments:**

- `name`: This site's address in the DFG namespace. Unique among sites and
    disjoint from the input-port names.
- `factor`: The `Factor` placed here.
- `parents`: Addresses (input ports or site names) feeding this factor;
    `porting_fn` maps them onto its named input ports.
- `porting_fn`: How parents route into the factor's input dict: a tuple of
    port names (1:1 with `parents`), or a callable `parents -> input dict`
    for non-trivial routing.
- `param_key`: Address of this site's parameters within the DFG's `params`.
- `info_key`: Like `param_key`, for runtime info; `None` passes `info=None`.
- `site_info`: per-site metadata passed to the factor.
"""


DFGParams: TypeAlias = PyTree[Array]


class DFGInfo(eqx.Module):
    """DFG-level runtime info.

    Separate from per-site info, which lives in `entries` under each site's
    `info_key`; this configures how the DAG itself runs.

    **Arguments:**

    - `expose_site_outputs`: When `True` (and `aux` is requested), prepend a
        name-keyed dict of every site's main output to the aux return.
    - `entries`: Per-`info_key` mapping, scattered to sites by
        `distribute_info`. A child-`DFG` site's entry is itself a `DFGInfo`.
    """

    expose_site_outputs: bool = eqx.field(static=True)
    entries: Mapping[str, PyTree[Array]] = eqx.field(default_factory=dict)


class AbstractDFG(AbstractFactor):
    """A `Factor` built as a DAG of placed factors.

    Holds the structure and concretises `sample` / `sample_with_references`
    as an eager topological walk.

    The DFG owns a flat namespace of addresses, one per input port and one
    per `Site`, and the two sets must be disjoint. Every `parents` entry and
    `output_name` is such an address. A DFG is itself a `Factor`, so it can
    nest as the `factor` of a `Site`.
    """

    sites: eqx.AbstractVar[tuple[Site, ...]]
    output_name: eqx.AbstractVar[str]
    topological_order: eqx.AbstractVar[tuple[int, ...]]
    sites_by_name: eqx.AbstractVar[Mapping[str, int]]

    @staticmethod
    def _derive(
        sites: tuple[Site, ...],
        input_ports: Mapping[str, PortSpec],
        output_name: str,
    ) -> tuple[dict[str, PortSpec], dict[str, int], tuple[int, ...], PortSpec]:
        """Validate the DAG and derive the fields a final class must set."""
        input_ports = dict(input_ports)
        sites_by_name = AbstractDFG._validate(sites, input_ports, output_name)
        topological_order = AbstractDFG._topo_sort(sites, sites_by_name)
        if output_name in input_ports:
            output_spec = input_ports[output_name]
        else:
            output_spec = sites[sites_by_name[output_name]].factor.output_spec
        return input_ports, sites_by_name, topological_order, output_spec

    @staticmethod
    def _validate(
        sites: tuple[Site, ...],
        input_ports: Mapping[str, PortSpec],
        output_name: str,
    ) -> dict[str, int]:
        site_names = [s.name for s in sites]
        seen: set[str] = set()
        duplicates: set[str] = set()
        for n in site_names:
            if n in seen:
                duplicates.add(n)
            seen.add(n)
        if duplicates:
            raise ValueError(f"Duplicate site names: {sorted(duplicates)}")

        # Disjointness with input port names.
        overlap = set(site_names) & set(input_ports.keys())
        if overlap:
            raise ValueError(
                f"Site names overlap with input port names: {sorted(overlap)}"
            )

        sites_by_name = {s.name: i for i, s in enumerate(sites)}

        valid_names = set(input_ports) | set(sites_by_name)
        if output_name not in valid_names:
            raise ValueError(
                f"output_name {output_name!r} is not a DFG address; input ports: "
                f"{tuple(input_ports)}, site names: {tuple(sites_by_name)}."
            )

        for site in sites:
            unknown_parents = tuple(p for p in site.parents if p not in valid_names)
            if unknown_parents:
                raise ValueError(
                    f"Site {site.name!r} has unknown parent(s) "
                    f"{unknown_parents}; input ports: {tuple(input_ports)}, "
                    f"site names: {tuple(sites_by_name)}."
                )

            if callable(site.porting_fn):
                continue

            port_names = tuple(site.porting_fn)
            if len(port_names) != len(site.parents):
                raise ValueError(
                    f"Site {site.name!r} has {len(site.parents)} parent(s) but "
                    f"tuple porting_fn has {len(port_names)} port(s)."
                )

            seen_ports = set()
            duplicate_ports = []
            for port in port_names:
                if port in seen_ports and port not in duplicate_ports:
                    duplicate_ports.append(port)
                seen_ports.add(port)
            if duplicate_ports:
                raise ValueError(
                    f"Site {site.name!r} tuple porting_fn contains duplicate "
                    f"port(s): {duplicate_ports}."
                )

            actual_ports = set(port_names)
            expected_ports = set(site.factor.input_ports)
            if actual_ports != expected_ports:
                missing = expected_ports - actual_ports
                extra = actual_ports - expected_ports
                raise ValueError(
                    f"Site {site.name!r} tuple porting_fn must match factor "
                    f"input ports; missing {missing}, extra {extra}."
                )

        return sites_by_name

    @staticmethod
    def _topo_sort(
        sites: tuple[Site, ...],
        sites_by_name: Mapping[str, int],
    ) -> tuple[int, ...]:
        """Kahn's algorithm"""
        in_degree = [0] * len(sites)
        adj = [[] for _ in range(len(sites))]
        for j, site in enumerate(sites):
            for parent_name in site.parents:
                if parent_name in sites_by_name:
                    i = sites_by_name[parent_name]
                    in_degree[j] += 1
                    adj[i].append(j)

        queue = deque(i for i in range(len(sites)) if in_degree[i] == 0)
        topo = []
        while queue:
            i = queue.popleft()
            topo.append(i)
            for j in adj[i]:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    queue.append(j)

        if len(topo) != len(sites):
            remaining = [s.name for k, s in enumerate(sites) if in_degree[k] > 0]
            raise ValueError(
                f"Cycle detected in DAG. Sites still with incoming edges: "
                f"{remaining}"
            )
        return tuple(topo)

    @staticmethod
    def _init_params(sites: tuple[Site, ...], key: Key[Array, ""]) -> DFGParams:
        """Initialise parameters once per distinct `param_key` across `sites`."""
        seen = {}
        for site in sites:
            if site.param_key is not None and site.param_key not in seen:
                seen[site.param_key] = site.factor

        if not seen:
            return {}

        pkeys = list(seen.keys())
        rng_keys = jax.random.split(key, len(pkeys))
        return {
            pkey: seen[pkey].init_params(rng_keys[i]) for i, pkey in enumerate(pkeys)
        }

    def sample(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: DFGParams,
        info: DFGInfo | None = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> PyTree[Array] | tuple[PyTree[Array], tuple[PyTree[Array], ...]]:
        """Run the DAG once and return the `output_name` value.

        Samples each site in topological order, routing parent outputs
        through its `porting_fn`.
        """
        values, auxes = self._run(
            key, inputs, params, info, n_references=None, with_aux=return_aux
        )
        output = values[self.output_name]
        if return_aux:
            return output, auxes
        return output

    def sample_with_references(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: DFGParams,
        info: DFGInfo | None = None,
        site_info: Any = None,
        n_references: int = 1,
    ) -> tuple[PyTree[Array], tuple[PyTree[Array], ...]]:
        """Like `sample`, but draws each site via its factor's
        `sample_with_references`, so every site yields `n_references + 1`
        samples. Returns `(output, per_site_auxes)`.
        """
        values, auxes = self._run(
            key, inputs, params, info, n_references=n_references, with_aux=True
        )
        return values[self.output_name], auxes

    def _run(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: DFGParams,
        info: DFGInfo | None,
        n_references: int | None,
        with_aux: bool,
    ) -> tuple[dict[str, PyTree[Array]], tuple[PyTree[Array], ...]]:
        """Shared eager topological walk for `sample` / `sample_with_references`."""
        per_site_params = self.distribute_params(params)
        per_site_info = self.distribute_info(info)
        if len(self.sites) > 0:
            keys = jax.random.split(key, len(self.sites))
        else:
            keys = jnp.empty((0, 2), dtype=jnp.uint32)

        values = dict(inputs)
        auxes = [None] * len(self.sites)

        for site_idx in self.topological_order:
            site = self.sites[site_idx]
            parent_outputs = [values[a] for a in site.parents]
            if callable(site.porting_fn):
                site_inputs = site.porting_fn(parent_outputs)
            else:
                site_inputs = dict(zip(site.porting_fn, parent_outputs))
            k = keys[site_idx]
            par = per_site_params[site_idx]
            inf = per_site_info[site_idx]
            si = site.site_info

            if n_references is None:
                if with_aux:
                    output, aux = site.factor.sample(
                        k, site_inputs, par, inf, si, return_aux=True
                    )
                else:
                    output = site.factor.sample(
                        k, site_inputs, par, inf, si, return_aux=False
                    )
                    aux = None
            else:
                output, aux = site.factor.sample_with_references(
                    k,
                    site_inputs,
                    par,
                    inf,
                    si,
                    n_references=n_references,
                )

            values[site.name] = output
            auxes[site_idx] = aux

        aux_out = tuple(auxes)
        if with_aux and info is not None and info.expose_site_outputs:
            site_outputs = {s.name: values[s.name] for s in self.sites}
            if n_references is not None:
                n_samples = n_references + 1
                site_outputs = jax.tree.map(
                    lambda x: jnp.broadcast_to(
                        jnp.expand_dims(x, 0), (n_samples,) + x.shape
                    ),
                    site_outputs,
                )
            aux_out = (site_outputs, aux_out)

        return values, aux_out

    def distribute_params(self, params: DFGParams) -> tuple[ParamsTree, ...]:
        """Scatter the shared params mapping into a per-site tuple."""
        return tuple(
            params[site.param_key] if site.param_key is not None else None
            for site in self.sites
        )

    def gather_param_grads(
        self,
        params: DFGParams,
        site_grads: Mapping[int, ParamsTree],
    ) -> DFGParams:
        """Gather per-site parameter gradients into the shared params dict."""

        def dot_fn(p: DFGParams):
            per_site = self.distribute_params(p)
            total = 0.0
            for idx, g in site_grads.items():
                total = total + sum(
                    jnp.vdot(a, b)
                    for a, b in zip(jax.tree.leaves(g), jax.tree.leaves(per_site[idx]))
                )
            return total

        return eqx.filter_grad(dot_fn)(params)

    def distribute_info(self, info: DFGInfo | None) -> tuple[InfoTree, ...]:
        """Scatter `info.entries` into a per-site tuple."""
        if info is None:
            return tuple(None for _ in self.sites)
        return tuple(
            info.entries[site.info_key] if site.info_key is not None else None
            for site in self.sites
        )


class DFG(AbstractDFG):
    """A concrete, eagerly-walked DAG of factors."""

    sites: tuple[Site, ...]
    input_ports: Mapping[str, PortSpec] = eqx.field(static=True)
    output_spec: PortSpec = eqx.field(static=True)
    output_name: str = eqx.field(static=True)
    topological_order: tuple[int, ...] = eqx.field(static=True)
    sites_by_name: Mapping[str, int] = eqx.field(static=True)

    def __init__(
        self,
        sites: tuple[Site, ...],
        input_ports: Mapping[str, PortSpec],
        output_name: str,
    ) -> None:
        input_ports, sites_by_name, topological_order, output_spec = self._derive(
            sites, input_ports, output_name
        )
        self.sites = sites
        self.input_ports = input_ports
        self.output_name = output_name
        self.topological_order = topological_order
        self.sites_by_name = sites_by_name
        self.output_spec = output_spec

    def init_params(self, key: Key[Array, ""]) -> DFGParams:
        """Initialise parameters once per distinct `param_key`."""
        return self._init_params(self.sites, key)
