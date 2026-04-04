"""Custom type aliases for torx."""

from jaxtyping import Array, Float, Int

StateVector = Float[Array, "dimensions"]
BitString = Int[Array, "pbits"]
