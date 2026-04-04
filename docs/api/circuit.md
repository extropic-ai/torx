# Circuit

A circuit is an ordered sequence of probabilistic gates applied to an initial state. Use [`DiscretePCircuit`][torx.DiscretePCircuit] for purely discrete (binary/pdit) gates, or [`HybridPCircuit`][torx.HybridPCircuit] for circuits mixing discrete and continuous gates.

::: torx.AbstractPCircuit
    options:
        members: false
        inherited_members: false

---

::: torx.DiscretePCircuit
    options:
        members:
            - __init__

---

::: torx.HybridPCircuit
    options:
        members:
            - __init__
