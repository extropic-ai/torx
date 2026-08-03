"""Custom type aliases for torx."""

from typing import TypeAlias

from jaxtyping import Array, Float, Int

StateVector: TypeAlias = Float[Array, " dimensions"]
BitString: TypeAlias = Int[Array, " pbits"]
