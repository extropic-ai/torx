# Gates

??? abstract "Abstract base classes"

    All gates implement the following interface specified by [`torx.AbstractPGate`][].

    ::: torx.AbstractPGate
        options:
            members: false
            inherited_members: false

    Discrete gates (those with matrix representations) implement [`torx.AbstractDiscreteGate`][].

    ::: torx.AbstractDiscreteGate
        options:
            members:
                - get_matrix

    K-branch gates (those with K possible outcomes) implement the [`torx.AbstractKBranchGate`][] interface.

    ::: torx.AbstractKBranchGate
        options:
            members:
                - num_branches
                - probs
                - branches

    ::: torx.AbstractSingleBinaryPGate
        options:
            members:
                - __init__

    ::: torx.AbstractMultiBinaryPGate
        options:
            members:
                - __init__

    ::: torx.AbstractSinglePditGate
        options:
            members:
                - __init__

    ::: torx.AbstractMultiPditGate
        options:
            members:
                - __init__

---

::: torx.PNOT
    options:
        members:
            - __init__

---

::: torx.PCNOT
    options:
        members:
            - __init__

---

::: torx.PSWAP
    options:
        members:
            - __init__

---

::: torx.PJUMP
    options:
        members:
            - __init__

---

::: torx.PMultiCNOT
    options:
        members:
            - __init__

---

::: torx.PDEMUX
    options:
        members:
            - __init__

---

::: torx.POR
    options:
        members:
            - __init__

---

::: torx.PReset
    options:
        members:
            - __init__

---

::: torx.PCopy
    options:
        members:
            - __init__

---

::: torx.PCSWAP
    options:
        members:
            - __init__

---

::: torx.PISING
    options:
        members:
            - get_generator
            - get_matrix

## Higher-Dimensional Gates

The following gates support k-dimensional systems (k > 2) and enable mixed-dimension circuits.

::: torx.PditShift
    options:
        members:
            - __init__

---

::: torx.PditSWAP
    options:
        members:
            - __init__

---

::: torx.PditCycle
    options:
        members:
            - __init__

---

## Hybrid Gates

The following gates support hybrid discrete-continuous systems. They operate on continuous
state variables and may be controlled by discrete sites.


??? abstract "Abstract base classes"

    ::: torx.AbstractHybridGate
        options:
            members:
                - sample

    ::: torx.AbstractContinuousGate
        options:
            members: false

    ::: torx.AbstractControlledContinuousGate
        options:
            members: false

---

::: torx.GaussianNoiseGate
    options:
        members: false
        inherited_members: false

---

::: torx.AffineGaussianGate
    options:
        members: false
        inherited_members: false

---

::: torx.MixtureGaussianGate
    options:
        members: false
        inherited_members: false

---

::: torx.JumpDiffusionGate
    options:
        members: false
        inherited_members: false

