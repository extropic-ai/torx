<div align="center">
  <img src="docs_site/assets/brand/logo.svg" alt="Torx Logo" width="200" style="margin-bottom: 10px;">
</div>

<h1 align='center'>Torx</h1>
<h2 align='center'>Probabilistic circuits in JAX.</h2>

Torx is a [JAX](https://github.com/google/jax)-based library for PSCs and DFGs with all the benefits of JAX (native GPU/TPU acceleration, differentiability, vectorization).

## Installation

Requires Python 3.11+.

```bash
pip install extro-torx
```

or

```bash
git clone https://github.com/extropic-ai/torx.git
cd torx
pip install -e ".[testing,examples]"
```

## Documentation

Available at [docs.torx.ai](https://docs.torx.ai/en/latest/).


## Quick example

Build and simulate a simple probabilistic circuit:

```python
import jax
import jax.numpy as jnp
from torx import psc

gates = [
    psc.PNOT(0),
    psc.PCNOT([0, 1]),
]

circuit = psc.DiscretePCircuit(gates)
thetas = circuit.init_params(jax.random.key(0))
sim = psc.StateVectorSimulator()
compiled = sim.build_circuit(circuit, thetas)

state = jnp.array([1.0, 0.0, 0.0, 0.0])

density = sim.density(compiled, state)
print(f"Final distribution: {density}")
```

## Citation

If you found this library useful in academic research, please cite:

```bibtex
@misc{verdon2026frameworkstochasticdifferentiableprogramming,
  title         = {A Framework for Stochastic Differentiable Programming},
  author        = {Guillaume Verdon and Leo Tyrpak and Owen Lockwood and Seth Morton and Alexander Neagoe and Anton Sugolov and Ian MacCormack and Mirko Amico},
  year          = {2026},
  eprint        = {2608.01612},
  archivePrefix = {arXiv},
  primaryClass  = {cs.ET},
  url           = {https://arxiv.org/abs/2608.01612},
}
```

## See also: other libraries in the JAX ecosystem

[thrml](https://github.com/extropic-ai/thrml): probabilistic graphical models and block Gibbs sampling in JAX.

[Awesome JAX](https://github.com/lockwo/awesome-jax): a longer list of other JAX projects.
