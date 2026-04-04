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
