r"""Compile ``PAsymSwap`` into ``thrml`` Ising kernels two ways and compare them.

This script rebuilds the 5x5 biased-random-walk demo of arXiv 2608.01615
(Thermalizers), Section IV.1, Eqs. (30)-(34), Figs. 5-6, and compiles the
per-edge ``PAsymSwap`` gate (Eq. 33) into ``thrml`` Ising kernels in two forms:

* **output form**: the paper's kernel class, ``n_in = n_out = 2, n_h = 1``; the
  EBM samples both output spins, so at any finite coupling cap the kernel has
  full support and can leak occupation out of the one-particle sector;
* **decision form**: same inputs, same hidden spin, same cap, but the single
  output spin is the *branch* (hop / no hop); the two-site move is then the
  deterministic branch table that ``PAsymSwap.branches[1:]`` applies in torx,
  so occupation number is conserved by construction.

Everything the paper states is followed: 5x5 periodic torus, one particle at
(0, 0), site logits ``a_i = 2 sin[2pi((2x+y)/5 + 0.2)] + 0.75 cos[2pi((x-2y)/5
- 0.4)]``, ``gamma = 2``, ``dt = 0.05``, hop ``p_ij = gamma sigma(a_j - a_i) dt``,
six colour classes, ``M = 10`` macro steps (60 layers), 4096 chains, ``K = 30``
block-Gibbs sweeps per layer via ``thrml.sample_states``, half-L1 occupancy
error against the CTMC ``exp(QT) p(0)``. Each kernel is trained by exact
enumeration of its conditional under the uniform measure over the four inputs
to minimise ``E_x KL(P(.|x) || P~(.|x))`` (Adam, 3000 steps, every coupling and
field held in ``[-J_max, J_max]`` by a tanh reparametrisation).

Two things the paper does not state are swept rather than chosen: the cap
``J_max`` (``h_max = J_max``) and the placement of the five spins on a bipartite
lattice (``full``: inputs couple to outputs and hidden spin; ``bip``: inputs on
the hidden spin's colour, coupling to the outputs only).

Requires ``thrml >= 0.1.4`` (``pip install thrml``), which is not in the
``examples`` extra. CPU only; one cap with all five variants takes about half a
minute on a laptop, a minute with ``--context-matching``, the full sweep a few.

Usage::

    python examples/scripts/pasymswap_compile_forms.py            # J_max = 1.5
    python examples/scripts/pasymswap_compile_forms.py --sweep    # all caps
    python examples/scripts/pasymswap_compile_forms.py --context-matching

Reference results (2026-09-08, CPU, seeds as in this file)
-----------------------------------------------------------

Table 1: cap sweep. Entries are median per-gate TV / final total mass / final
half-L1 occupancy error. The paper's row is compiled-only; after model-context
matching the paper reports half-L1 0.30 and after REINFORCE on top 0.08, both at
a cap and placement the paper does not state.

===============  =====================  =====================  =====================
J_max            output ``full``        output ``bip``         decision ``dec_full``
===============  =====================  =====================  =====================
0.5              0.376 / 12.50 / 5.75   0.443 / 12.51 / 5.75   0.073 / 1.000 / 0.65
1.0              0.154 / 12.33 / 5.67   0.217 / 12.57 / 5.78   0.0009 / 1.000 / 0.072
1.5              0.058 / 10.78 / 4.89   0.095 / 12.23 / 5.61   0.0000 / 1.000 / 0.034
2.0              0.022 / 6.72 / 2.86    0.039 / 9.90 / 4.45    0.0000 / 1.000 / 0.021
3.0              0.003 / 2.12 / 1.10    0.006 / 3.34 / 1.45    0.0000 / 1.000 / 0.025
4.0              0.0006 / 1.23 / 0.79   0.0011 / 1.47 / 0.85   0.0000 / 1.000 / 0.027
6.0              0.0001 / 7.74 / 3.44   0.0001 / 1.05 / 0.72   0.0000 / 1.000 / 0.056
10.0             0.004 / 3.21 / 1.53    0.0001 / 2.04 / 1.03   0.0008 / 1.000 / 0.17
paper (2608.01615 Fig. 5)  0.096 / 12.2 / 5.64  (cap, placement not stated)
===============  =====================  =====================  =====================

``dec_bip`` at 1.5: 0.0000 / 1.000 / 0.020; ``dec_nh0`` (no hidden spin) at 1.5:
0.005 / 1.000 / 0.16. At ``J_max >= 3`` the output-form kernel no longer
thermalises in 30 sweeps (thrml-vs-enumeration TV 0.2-0.49), so the output
form has a cap window bounded below by leakage and above by mixing. The floor
of the diagnostic is 0.016 (the analytic Trotter circuit vs the CTMC) plus
0.018 (4096-chain noise), so decision-form values of 0.02-0.03 are not
resolvable from zero at this chain count.

Table 2: model-context matching (Section III.2), both forms, ``J_max = 1.5``,
``full`` placement, every gate instance (edge x macro step, 500 instances)
re-optimised under the empirical input marginal it receives from the thrml
rollout; 1500 Adam steps per round, three rounds. Entries are final mass /
half-L1 / median TV under the model measure.

=====  ===========================  ===========================
round  output form                  decision form
=====  ===========================  ===========================
0      10.78 / 4.89 / 0.058         1.000 / 0.034 / 0.0000
1      7.86 / 3.43 / 0.050          1.000 / 0.021 / 0.0000
2      5.32 / 2.16 / 0.036          1.000 / 0.029 / 0.0000
3      3.67 / 1.33 / 0.025          1.000 / 0.022 / 0.0000
=====  ===========================  ===========================

Context matching moves capacity to the inputs the rollout visits and shrinks
the output form's leak round by round without making it zero (the kernel's
support is unchanged); for the decision form there is nothing to match. The
decision form's half-L1 of 0.02-0.03 sits below the paper's post-training
0.30 / 0.08 at our cap 1.5 / ``bip`` placement; the paper states neither cap
nor placement, so their post-training numbers may sit at a different cap.
"""

import argparse
import json
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm

try:
    from thrml import Block, sample_states, SamplingSchedule, SpinNode
    from thrml.models.ising import IsingEBM, IsingSamplingProgram
except ImportError:  # pragma: no cover
    sys.exit(
        "This example needs thrml (https://github.com/extropic-ai/thrml): "
        "pip install thrml"
    )

jax.config.update("jax_enable_x64", False)

# ---------------------------------------------------------------------------
# 1. The paper's program: lattice, field, rates, gates, colour classes, CTMC
# ---------------------------------------------------------------------------
L, GAMMA, DT, M, K_SWEEPS, N_CHAINS = 5, 2.0, 0.05, 10, 30, 4096
NSITES = L * L
NUM_GATES = 50  # undirected edges of the 5x5 torus, one PAsymSwap each


def idx(x, y):
    return (x % L) + L * (y % L)


XY = [(i % L, i // L) for i in range(NSITES)]
A_FIELD = np.array(
    [
        2 * np.sin(2 * np.pi * ((2 * x + y) / L + 0.2))
        + 0.75 * np.cos(2 * np.pi * ((x - 2 * y) / L - 0.4))
        for x, y in XY
    ]
)


def sigma(u):
    return 1.0 / (1.0 + np.exp(-u))


def classes():
    """Six colour classes as in Fig. 6 (H3 / V3 hold the periodic edges)."""
    colour = {0: 0, 1: 1, 2: 0, 3: 1, 4: 2}
    H = [[], [], []]
    V = [[], [], []]
    for y in range(L):
        for x in range(L):
            H[colour[x]].append((idx(x, y), idx(x + 1, y)))
    for x in range(L):
        for y in range(L):
            V[colour[y]].append((idx(x, y), idx(x, y + 1)))
    return H + V


CLASSES = classes()
GATES = [g for c in CLASSES for g in c]
assert len(GATES) == NUM_GATES
assert all(len({v for g in c for v in g}) == 2 * len(c) for c in CLASSES)


def hop(i, j):
    """p_ij of Eq. (34): the hop i -> j."""
    return GAMMA * sigma(A_FIELD[j] - A_FIELD[i]) * DT


def target_gate(i, j):
    """Eq. (33); columns are inputs (z_i z_j) in order 00, 01, 10, 11."""
    pij, pji = hop(i, j), hop(j, i)
    P = np.eye(4)
    P[1, 1], P[2, 1] = 1 - pji, pji  # input 01: z_j = 1 hops to i with p_ji
    P[2, 2], P[1, 2] = 1 - pij, pij  # input 10: z_i = 1 hops to j with p_ij
    return P


TARGETS = np.stack([target_gate(*g) for g in GATES])  # (50, 4, 4)

# CTMC of Eq. (30)
Q = np.zeros((NSITES, NSITES))
for i, j in GATES:
    Q[j, i] += GAMMA * sigma(A_FIELD[j] - A_FIELD[i])
    Q[i, j] += GAMMA * sigma(A_FIELD[i] - A_FIELD[j])
np.fill_diagonal(Q, -Q.sum(0))
p0 = np.zeros(NSITES)
p0[idx(0, 0)] = 1.0
M_CTMC = expm(Q * M * DT) @ p0


def trotter_exact():
    """The analytic Trotter circuit of Eqs. (32)-(33) applied exactly."""
    q = p0.copy()
    for _ in range(M):
        for c in CLASSES:
            qn = q.copy()
            for i, j in c:
                pij, pji = hop(i, j), hop(j, i)
                qn[i] = (1 - pij) * q[i] + pji * q[j]
                qn[j] = (1 - pji) * q[j] + pij * q[i]
            q = qn
    return q


M_TROTTER = trotter_exact()


def half_l1(m):
    return 0.5 * np.abs(m - M_CTMC).sum()


# ---------------------------------------------------------------------------
# 2. Kernels by exact enumeration
# ---------------------------------------------------------------------------
SPIN4 = jnp.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=jnp.float32)
SPIN2 = jnp.array([-1.0, 1.0])


def logpsi_out(params, x, topo):
    """log sum_w exp(-E) for the output-form kernel, over y in SPIN4 order."""
    hy, A, C, B, hw = (params[k] for k in ("hy", "A", "C", "B", "hw"))
    if topo == "bip":
        C = 0.0 * C
    y = SPIN4
    field = hw + C @ x + y @ B
    return y @ hy + (x @ A) @ y.T + jnp.log(2 * jnp.cosh(field))


def cond_out(params, topo):
    lp = jax.vmap(lambda x: logpsi_out(params, x, topo))(SPIN4)  # (4 in, 4 out)
    return jax.nn.softmax(lp, axis=1)


def logpsi_dec(params, x, topo):
    """Decision form: single output spin d, optional hidden spin w."""
    hd, A, C, B, hw = (params[k] for k in ("hd", "A", "C", "B", "hw"))
    d = SPIN2
    base = hd * d + (A @ x) * d
    if topo == "dec_nh0":
        return base
    if topo == "dec_bip":
        C = 0.0 * C
    return base + jnp.log(2 * jnp.cosh(hw + C @ x + B * d))


def cond_dec(params, topo):
    lp = jax.vmap(lambda x: logpsi_dec(params, x, topo))(SPIN4)  # (4 in, 2)
    pd = jax.nn.softmax(lp, axis=1)[:, 1]  # P(d = +1 | x)
    # induced output conditional under the branch table (swap iff d = +1 and
    # the pair is 01 or 10); this is PAsymSwap's matrix with (p, q) = pd[2], pd[1]
    rows = [
        jnp.array([1.0, 0.0, 0.0, 0.0]),
        jnp.array([0.0, 1 - pd[1], pd[1], 0.0]),
        jnp.array([0.0, pd[2], 1 - pd[2], 0.0]),
        jnp.array([0.0, 0.0, 0.0, 1.0]),
    ]
    return jnp.stack(rows)


def init_params(kind, n):
    k = jax.random.split(jax.random.PRNGKey(1), 6)
    s = 0.3
    if kind == "out":
        return {
            "hy": s * jax.random.normal(k[0], (n, 2)),
            "A": s * jax.random.normal(k[1], (n, 2, 2)),
            "C": s * jax.random.normal(k[2], (n, 2)),
            "B": s * jax.random.normal(k[3], (n, 2)),
            "hw": s * jax.random.normal(k[4], (n,)),
        }
    return {
        "hd": s * jax.random.normal(k[0], (n,)),
        "A": s * jax.random.normal(k[1], (n, 2)),
        "C": s * jax.random.normal(k[2], (n, 2)),
        "B": s * jax.random.normal(k[3], (n,)),
        "hw": s * jax.random.normal(k[4], (n,)),
    }


def capped(u, cap):
    return jax.tree_util.tree_map(lambda t: cap * jnp.tanh(t), u)


def make_cond(kind, topo):
    if kind == "out":
        return lambda p: cond_out(p, topo)
    return lambda p: cond_dec(p, topo)


def train(kind, topo, targets, measure, cap, steps=3000, lr=0.05, init=None):
    """Minimise sum_x mu(x) KL(P(.|x) || P~(.|x)) per gate with Adam.

    ``targets`` is (n, 4, 4) column-stochastic, ``measure`` is (n, 4) over
    inputs. All couplings and fields stay inside [-cap, cap] via tanh.
    """
    n = targets.shape[0]
    cond = make_cond(kind, topo)
    T = jnp.asarray(np.transpose(targets, (0, 2, 1)))  # (n, input, output)
    mu = jnp.asarray(measure)

    def loss_one(u, T1, mu1):
        Pm = cond(capped(u, cap))
        safe_T = jnp.where(T1 > 0, T1, 1.0)
        kl = jnp.sum(
            jnp.where(T1 > 0, T1 * (jnp.log(safe_T) - jnp.log(Pm + 1e-30)), 0.0),
            axis=1,
        )
        return jnp.sum(mu1 * kl)

    def loss(U):
        return jnp.sum(jax.vmap(loss_one)(U, T, mu))

    U = init if init is not None else init_params(kind, n)
    m = jax.tree_util.tree_map(jnp.zeros_like, U)
    v = jax.tree_util.tree_map(jnp.zeros_like, U)
    b1, b2 = 0.9, 0.999

    @jax.jit
    def step(U, m, v, t):
        g = jax.grad(loss)(U)
        m = jax.tree_util.tree_map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, m, g)
        v = jax.tree_util.tree_map(lambda v_, g_: b2 * v_ + (1 - b2) * g_ * g_, v, g)
        mh = jax.tree_util.tree_map(lambda m_: m_ / (1 - b1**t), m)
        vh = jax.tree_util.tree_map(lambda v_: v_ / (1 - b2**t), v)
        U = jax.tree_util.tree_map(
            lambda u_, mh_, vh_: u_ - lr * mh_ / (jnp.sqrt(vh_) + 1e-8), U, mh, vh
        )
        return U, m, v

    for t in range(1, steps + 1):
        U, m, v = step(U, m, v, t)
    P = capped(U, cap)
    conds = jax.vmap(cond)(P)  # (n, input, output)
    tv_in = 0.5 * jnp.abs(conds - T).sum(-1)  # (n, input)
    return U, P, np.asarray(conds), np.asarray(tv_in), float(loss(U))


# ---------------------------------------------------------------------------
# 3. Rollouts through thrml's block-Gibbs sampler, one IsingEBM per colour class
# ---------------------------------------------------------------------------
_CACHE = {}


def class_sampler(kind, topo, n, K):
    """Build (once) the jitted sampler for a colour class of ``n`` gates."""
    key = (kind, topo, n, K)
    if key in _CACHE:
        return _CACHE[key]
    X = [SpinNode() for _ in range(2 * n)]
    W = [SpinNode() for _ in range(n)]
    Y = [SpinNode() for _ in range(2 * n if kind == "out" else n)]
    nodes, edges = [], []
    for g in range(n):
        x1, x2, w = X[2 * g], X[2 * g + 1], W[g]
        if kind == "out":
            y1, y2 = Y[2 * g], Y[2 * g + 1]
            nodes += [x1, x2, w, y1, y2]
            edges += [(x1, y1), (x1, y2), (x2, y1), (x2, y2), (w, y1), (w, y2)]
            if topo == "full":
                edges += [(x1, w), (x2, w)]
        else:
            d = Y[g]
            nodes += [x1, x2, w, d]
            edges += [(x1, d), (x2, d)]
            if topo != "dec_nh0":
                edges += [(w, d)]
                if topo == "dec_full":
                    edges += [(x1, w), (x2, w)]
    Xb, Wb, Yb = Block(X), Block(W), Block(Y)
    sched = SamplingSchedule(n_warmup=K - 1, n_samples=1, steps_per_sample=1)

    @jax.jit
    def run(weights, biases, keys, xstate, y0, w0):
        ebm = IsingEBM(nodes, edges, biases, weights, jnp.array(1.0))
        prog = IsingSamplingProgram(ebm, [Yb, Wb], [Xb])

        def one(k, x, y, w):
            return sample_states(k, prog, sched, [y, w], [x], [Yb])[0][0]

        return jax.vmap(one)(keys, xstate, y0, w0)

    _CACHE[key] = run
    return run


def class_arrays(kind, topo, P, gate_ids):
    """Weights and biases in the order ``class_sampler`` laid out its graph."""
    biases, weights = [], []
    for gi in gate_ids:
        if kind == "out":
            biases += [0.0, 0.0, P["hw"][gi], P["hy"][gi, 0], P["hy"][gi, 1]]
            weights += [
                P["A"][gi, 0, 0],
                P["A"][gi, 0, 1],
                P["A"][gi, 1, 0],
                P["A"][gi, 1, 1],
                P["B"][gi, 0],
                P["B"][gi, 1],
            ]
            if topo == "full":
                weights += [P["C"][gi, 0], P["C"][gi, 1]]
        else:
            hw = P["hw"][gi] if topo != "dec_nh0" else 0.0
            biases += [0.0, 0.0, hw, P["hd"][gi]]
            weights += [P["A"][gi, 0], P["A"][gi, 1]]
            if topo != "dec_nh0":
                weights += [P["B"][gi]]
                if topo == "dec_full":
                    weights += [P["C"][gi, 0], P["C"][gi, 1]]
    return (
        jnp.asarray(np.array(weights, dtype=np.float32)),
        jnp.asarray(np.array(biases, dtype=np.float32)),
    )


def apply_decision(xin, d):
    """The branch table: swap the pair iff ``d`` and the pair is 01 or 10."""
    y = xin.copy()
    swap = d & (xin[..., 0] != xin[..., 1])
    y[swap] = y[swap][:, ::-1]
    return y


def rollout(
    kind, topo, P_by_instance, K=K_SWEEPS, n_chains=N_CHAINS, seed=0
) -> dict[str, Any]:
    """Roll ``M`` macro steps of the compiled circuit through thrml.

    ``P_by_instance`` is a list over macro steps of per-gate parameter pytrees
    (indexed by gate id 0..49) or one pytree used for every step. Returns the
    mass per macro step, the final marginal, its half-L1 error against the
    CTMC, the chain-to-chain mass standard deviation and the empirical input
    distribution every gate instance received.
    """
    key = jax.random.PRNGKey(seed)
    z = np.zeros((n_chains, NSITES), dtype=bool)
    z[:, idx(0, 0)] = True
    masses = [1.0]
    input_hist = np.zeros((M, NUM_GATES, 4))
    gate_offset = np.cumsum([0] + [len(c) for c in CLASSES])
    for m in range(M):
        P = P_by_instance[m] if isinstance(P_by_instance, list) else P_by_instance
        for ci, c in enumerate(CLASSES):
            gids = list(range(gate_offset[ci], gate_offset[ci + 1]))
            n = len(c)
            ii = np.array([g[0] for g in c])
            jj = np.array([g[1] for g in c])
            xin = np.stack([z[:, ii], z[:, jj]], -1)  # (chains, n, 2) bool
            xcode = 2 * xin[..., 0].astype(int) + xin[..., 1].astype(int)
            for g in range(n):
                counts = np.bincount(xcode[:, g], minlength=4)
                input_hist[m, gids[g]] += counts / n_chains
            run = class_sampler(kind, topo, n, K)
            wts, bs = class_arrays(kind, topo, P, gids)
            n_y = 2 * n if kind == "out" else n
            key, k1, k2 = jax.random.split(key, 3)
            keys = jax.random.split(k1, n_chains)
            xstate = jnp.asarray(xin.reshape(n_chains, 2 * n))
            y0 = jax.random.bernoulli(k2, 0.5, (n_chains, n_y))
            w0 = jax.random.bernoulli(jax.random.fold_in(k2, 1), 0.5, (n_chains, n))
            ys = np.asarray(run(wts, bs, keys, xstate, y0, w0))
            if kind == "out":
                y = ys.reshape(n_chains, n, 2)
            else:
                y = apply_decision(xin, ys.reshape(n_chains, n))
            z[:, ii] = y[..., 0]
            z[:, jj] = y[..., 1]
        masses.append(float(z.sum(1).mean()))
    m_final = z.mean(0)
    return dict(
        mass=masses,
        m_final=m_final,
        half_l1=float(half_l1(m_final)),
        input_hist=input_hist,
        mass_std=float(z.sum(1).std()),
    )


# ---------------------------------------------------------------------------
# 4. Validation of the thrml kernel against exact enumeration
# ---------------------------------------------------------------------------
def validate_thrml(kind, topo, P, K=K_SWEEPS, n=20000, gate=0):
    """Worst case over the four inputs of the TV between thrml's K-sweep output
    law and the enumerated conditional (Monte-Carlo floor about 0.005)."""
    P_gate = jax.tree_util.tree_map(lambda t: t[gate], P)
    cond = np.asarray(make_cond(kind, topo)(P_gate))
    run = class_sampler(kind, topo, 1, K)
    wts, bs = class_arrays(kind, topo, P, [gate])
    n_y = 2 if kind == "out" else 1
    worst = 0.0
    for xi in range(4):
        xin = np.array([[xi // 2, xi % 2]], dtype=bool).repeat(n, 0)
        keys = jax.random.split(jax.random.PRNGKey(7 + xi), n)
        y0 = jax.random.bernoulli(jax.random.PRNGKey(99), 0.5, (n, n_y))
        w0 = jax.random.bernoulli(jax.random.PRNGKey(98), 0.5, (n, 1))
        ys = np.asarray(run(wts, bs, keys, jnp.asarray(xin), y0, w0))
        if kind == "out":
            code = 2 * ys[:, 0].astype(int) + ys[:, 1].astype(int)
        else:
            y = apply_decision(xin[:, None, :], ys)[:, 0]
            code = 2 * y[:, 0].astype(int) + y[:, 1].astype(int)
        emp = np.bincount(code, minlength=4) / n
        worst = max(worst, 0.5 * np.abs(emp - cond[xi]).sum())
    return float(worst)


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
VARIANTS = [
    ("out", "full"),
    ("out", "bip"),
    ("dec", "dec_full"),
    ("dec", "dec_bip"),
    ("dec", "dec_nh0"),
]
SWEEP_CAPS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0]
PAPER = dict(tv=0.096, mass=12.2, half_l1=5.64)  # 2608.01615 Fig. 5, compiled-only


def context_matching(kind, topo, U0, cap, rounds=3, steps=1500):
    """Section III.2: re-optimise every gate instance under its empirical input
    marginal from the thrml rollout of the current compiled circuit."""
    U_inst = jax.tree_util.tree_map(
        lambda t: jnp.tile(t, (M,) + (1,) * (t.ndim - 1)), U0
    )

    def split(P_inst):
        return [
            jax.tree_util.tree_map(
                lambda t: t[m * NUM_GATES : (m + 1) * NUM_GATES], P_inst
            )
            for m in range(M)
        ]

    r = rollout(kind, topo, split(capped(U_inst, cap)))
    hist: list[dict[str, Any]] = [
        dict(round=0, mass=r["mass"][-1], half_l1=r["half_l1"])
    ]
    print(
        f"  {kind}/{topo} round 0: mass {r['mass'][-1]:.3f}  half-L1 {r['half_l1']:.3f}"
    )
    for rd in range(1, rounds + 1):
        mu = r["input_hist"].reshape(M * NUM_GATES, 4)
        T_inst = np.tile(TARGETS, (M, 1, 1))
        U_inst, P_inst, _, tv_in, _ = train(
            kind, topo, T_inst, mu, cap, steps=steps, init=U_inst
        )
        r = rollout(kind, topo, split(P_inst), seed=rd)
        tv_model = float(np.median((tv_in * mu).sum(1)))
        hist.append(
            dict(round=rd, mass=r["mass"][-1], half_l1=r["half_l1"], tv=tv_model)
        )
        print(
            f"  {kind}/{topo} round {rd}: mass {r['mass'][-1]:.3f}  "
            f"half-L1 {r['half_l1']:.3f}  median TV (model measure) {tv_model:.4f}"
        )
    return hist


def main(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--caps",
        type=float,
        nargs="+",
        default=[1.5],
        help="coupling caps J_max to run (default 1.5)",
    )
    parser.add_argument(
        "--sweep", action="store_true", help=f"run the full cap sweep {SWEEP_CAPS}"
    )
    parser.add_argument(
        "--context-matching",
        action="store_true",
        help="run three rounds of model-context matching at the first cap",
    )
    parser.add_argument("--out", default=None, help="write results to this JSON file")
    args = parser.parse_args(argv)
    caps = SWEEP_CAPS if args.sweep else args.caps

    t0 = time.time()
    uniform = np.full((NUM_GATES, 4), 0.25)
    print(
        f"CTMC marginal sum {M_CTMC.sum():.6f}; analytic Trotter circuit (no "
        f"compile) half-L1 vs CTMC = {half_l1(M_TROTTER):.4f}"
    )
    hops = [hop(i, j) for i, j in GATES] + [hop(j, i) for i, j in GATES]
    print(f"hop probabilities p_ij in [{min(hops):.4f}, {max(hops):.4f}]")
    print(
        f"paper (2608.01615 Fig. 5), compiled-only: TV {PAPER['tv']}, "
        f"mass {PAPER['mass']}, half-L1 {PAPER['half_l1']}"
    )
    print()
    hdr = (
        f"{'form':5s} {'topo':9s} {'J_max':>5s} {'med TV':>7s} {'mass':>7s} "
        f"{'mass sd':>7s} {'half-L1':>7s} {'thrml-enum':>10s}"
    )
    print(hdr)
    print("-" * len(hdr))
    results: dict[str, Any] = dict(caps=caps, rows=[], paper=PAPER)
    trained = {}
    for cap in caps:
        for kind, topo in VARIANTS:
            U, P, _, tv_in, _ = train(kind, topo, TARGETS, uniform, cap)
            trained[(cap, kind, topo)] = U
            tv = float(np.median(tv_in.mean(1)))
            val = validate_thrml(kind, topo, P)
            r = rollout(kind, topo, P)
            row = dict(
                kind=kind,
                topo=topo,
                cap=cap,
                med_tv=tv,
                mass=r["mass"],
                mass_std=r["mass_std"],
                half_l1=r["half_l1"],
                thrml_vs_enum_tv=val,
            )
            results["rows"].append(row)
            print(
                f"{kind:5s} {topo:9s} {cap:5.1f} {tv:7.4f} {r['mass'][-1]:7.3f} "
                f"{r['mass_std']:7.3f} {r['half_l1']:7.3f} {val:10.4f}",
                flush=True,
            )
    print(f"\n[{time.time() - t0:.0f}s] caps done")

    if args.context_matching:
        cap = caps[0]
        print(f"\nModel-context matching at J_max = {cap}, 'full' placement, 3 rounds")
        results["context_matching"] = {}
        for kind, topo in [("out", "full"), ("dec", "dec_full")]:
            hist = context_matching(kind, topo, trained[(cap, kind, topo)], cap)
            results["context_matching"][f"{kind}/{topo}"] = hist
        print(f"\n[{time.time() - t0:.0f}s] total")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1, default=float)


if __name__ == "__main__":
    main()
