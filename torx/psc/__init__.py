"""Probabilistic Circuits in JAX."""

import importlib.metadata

from ._circuit import (
    AbstractPCircuit as AbstractPCircuit,
    DiscretePCircuit as DiscretePCircuit,
    HybridPCircuit as HybridPCircuit,
    HybridState as HybridState,
)
from .gates import (
    AbstractAffineGaussianGate as AbstractAffineGaussianGate,
    AbstractContinuousGate as AbstractContinuousGate,
    AbstractControlledContinuousGate as AbstractControlledContinuousGate,
    AbstractDiscreteGate as AbstractDiscreteGate,
    AbstractGeneratorGate as AbstractGeneratorGate,
    AbstractHybridGate as AbstractHybridGate,
    AbstractKBranchGate as AbstractKBranchGate,
    AbstractMultiBinaryPGate as AbstractMultiBinaryPGate,
    AbstractMultiPditGate as AbstractMultiPditGate,
    AbstractPGate as AbstractPGate,
    AbstractSingleBinaryPGate as AbstractSingleBinaryPGate,
    AbstractSinglePditGate as AbstractSinglePditGate,
    AffineGaussianGate as AffineGaussianGate,
    Diffuse as Diffuse,
    Displace as Displace,
    GaussianNoiseGate as GaussianNoiseGate,
    HybridSites as HybridSites,
    JumpDiffusionGate as JumpDiffusionGate,
    Mix as Mix,
    MixtureGaussianGate as MixtureGaussianGate,
    PCNOT as PCNOT,
    PCopy as PCopy,
    PCSWAP as PCSWAP,
    PDEMUX as PDEMUX,
    PditCycle as PditCycle,
    PditShift as PditShift,
    PditSWAP as PditSWAP,
    PISING as PISING,
    PJUMP as PJUMP,
    PMultiCNOT as PMultiCNOT,
    PNOT as PNOT,
    POR as POR,
    PReset as PReset,
    PSWAP as PSWAP,
    Scale as Scale,
)
from .simulation import (
    AbstractCompiledPCircuit as AbstractCompiledPCircuit,
    AbstractSimulator as AbstractSimulator,
    AffineGaussianSimulator as AffineGaussianSimulator,
    BranchingSimulator as BranchingSimulator,
    CompiledAffineGaussianPCircuit as CompiledAffineGaussianPCircuit,
    CompiledBranchingPCircuit as CompiledBranchingPCircuit,
    CompiledStateVectorPCircuit as CompiledStateVectorPCircuit,
    GaussianMoments as GaussianMoments,
    sample_circuit as sample_circuit,
    StateVectorSimulator as StateVectorSimulator,
)
from .visualization import draw_circuit as draw_circuit, TextDrawing as TextDrawing

__version__ = importlib.metadata.version("extro-torx")
