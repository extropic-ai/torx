# Simulators

??? abstract "Abstract base classes"

    ::: torx.AbstractSimulator
        options:
            members:
                - expval
                - expval_all
                - build_circuit


    ::: torx.AbstractCompiledPCircuit
        options:
            members:
                - from_pcircuit
                - to_pcircuit

---

::: torx.StateVectorSimulator
    options:
        members:
            - density
            - expval
            - expval_all

---

::: torx.SampleSimulator
    options:
        members:
            - __init__
            - sample
            - expval
            - expval_all

!!! note "Conditional sample gates"

    `SampleSimulator` supports conditional sample gates such as
    [`torx.PConditionalBernoulli`][] for forward sampling and expectation
    estimates. Gradient methods for these gates are not implemented yet and
    raise `NotImplementedError`.

---

::: torx.CompiledStateVectorPCircuit
    options:
        members:
            - from_pcircuit
            - to_pcircuit

---

::: torx.CompiledSamplePCircuit
    options:
        members:
            - from_pcircuit
            - to_pcircuit

---


::: torx.HybridState
    options:
        members: false

---

::: torx.HybridSampleSimulator
    options:
        members:
            - sample
            - expval
            - expval_all
