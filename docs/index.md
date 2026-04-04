<div align="center" class="torx-logo">
  <img src="_static/logo/logo.svg" alt="Torx Logo" width="200" style="margin-bottom: 20px;">
</div>

<h1 align='center' class="torx-title">Torx</h1>

Probabilistic Circuits in JAX.

## Installation

Requires Python 3.10+.

```bash
pip install torx
```

For development:

```bash
git clone https://github.com/extropic-ai/torx.git
cd torx
pip install -e ".[testing,examples]"
```

## Quick example

Build and simulate a simple probabilistic circuit:

```python
import jax.numpy as jnp
import torx

# Create gates
gates = [
    torx.PNOT(jnp.array(0.0), 0),
    torx.PCNOT(jnp.array(0.0), [0, 1]),
]

# Build circuit
circuit = torx.DiscretePCircuit(gates)

# Create simulator and compile circuit
sim = torx.StateVectorSimulator()
compiled = sim.build_circuit(circuit)

# Initial state: |00)
state = jnp.array([1.0, 0, 0, 0])

# Get final distribution
density = sim.density(compiled, state)
print(f"Final distribution: {density}")
```

## Citation

If you found this library useful in academic research, please cite:

```bibtex
% Citation coming soon
```

## See also: other libraries in the JAX ecosystem

[Awesome JAX](https://github.com/lockwo/awesome-jax): a longer list of other JAX projects.
