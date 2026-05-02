"""Probabilistic Circuits in JAX."""

import importlib.metadata

from ._circuit import (
    AbstractPCircuit as AbstractPCircuit,
    DiscretePCircuit as DiscretePCircuit,
    HybridPCircuit as HybridPCircuit,
)
from .gates import (
    AbstractContinuousGate as AbstractContinuousGate,
    AbstractControlledContinuousGate as AbstractControlledContinuousGate,
    AbstractDiscreteGate as AbstractDiscreteGate,
    AbstractHybridGate as AbstractHybridGate,
    AbstractKBranchGate as AbstractKBranchGate,
    AbstractMultiBinaryPGate as AbstractMultiBinaryPGate,
    AbstractMultiPditGate as AbstractMultiPditGate,
    AbstractPGate as AbstractPGate,
    AbstractSingleBinaryPGate as AbstractSingleBinaryPGate,
    AbstractSinglePditGate as AbstractSinglePditGate,
    AffineGaussianGate as AffineGaussianGate,
    GaussianNoiseGate as GaussianNoiseGate,
    HybridSites as HybridSites,
    JumpDiffusionGate as JumpDiffusionGate,
    MixtureGaussianGate as MixtureGaussianGate,
    PCNOT as PCNOT,
    PConditionalBernoulliLayer as PConditionalBernoulliLayer,
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
)
from .simulation import (
    AbstractCompiledPCircuit as AbstractCompiledPCircuit,
    AbstractSimulator as AbstractSimulator,
    CompiledHybridPCircuit as CompiledHybridPCircuit,
    CompiledSamplePCircuit as CompiledSamplePCircuit,
    CompiledStateVectorPCircuit as CompiledStateVectorPCircuit,
    HybridSampleSimulator as HybridSampleSimulator,
    HybridState as HybridState,
    SampleSimulator as SampleSimulator,
    StateVectorSimulator as StateVectorSimulator,
)
from .visualization import draw_circuit as draw_circuit, TextDrawing as TextDrawing

__version__ = importlib.metadata.version("torx")
