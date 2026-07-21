"""Probabilistic Circuits in JAX."""

import importlib.metadata

from ._circuit import (
    AbstractPCircuit as AbstractPCircuit,
    DiscretePCircuit as DiscretePCircuit,
    HybridPCircuit as HybridPCircuit,
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
    AbstractSampler as AbstractSampler,
    AbstractSimulator as AbstractSimulator,
    AffineGaussianSimulator as AffineGaussianSimulator,
    CompiledAffineGaussianPCircuit as CompiledAffineGaussianPCircuit,
    CompiledHybridPCircuit as CompiledHybridPCircuit,
    CompiledSamplePCircuit as CompiledSamplePCircuit,
    CompiledStateVectorPCircuit as CompiledStateVectorPCircuit,
    GaussianMoments as GaussianMoments,
    HybridSampleSimulator as HybridSampleSimulator,
    HybridState as HybridState,
    JaxPRNGSampler as JaxPRNGSampler,
    SampleSimulator as SampleSimulator,
    StateVectorSimulator as StateVectorSimulator,
)
from .visualization import draw_circuit as draw_circuit, TextDrawing as TextDrawing

__version__ = importlib.metadata.version("extro-torx")
