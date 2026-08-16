"""Pure helper functions with no engine-specific logic and no side effects.

Every function here is deterministic, depends only on its arguments, and is
named for the general operation it performs rather than for lighting or
gradients, so any of the other eight engines can import it unchanged. Nothing
in this module reads or writes the filesystem, mutates global state, or holds
instance state, and nothing imports from the rest of this package.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


def compute_gradient_magnitude(array: np.ndarray) -> np.ndarray:
    """Elementwise magnitude of the two-dimensional numerical gradient.

    Args:
        array: Two-dimensional float array, at least 2 elements on each axis.

    Returns:
        Array of the same shape holding sqrt(d/d(axis0)^2 + d/d(axis1)^2) at
        every position.

    Raises:
        ValueError: If the array is not 2-D or is too small on either axis for
            numpy.gradient to compute a numerical derivative.
    """
    data = np.asarray(array, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"expected a 2-D array, received shape {data.shape}")

    first_axis_gradient, second_axis_gradient = np.gradient(data)
    return np.sqrt(first_axis_gradient ** 2 + second_axis_gradient ** 2)


def safe_ratio(numerator: float, denominator: float, floor: float) -> float:
    """Divide two scalars, flooring the denominator to avoid a zero divide.

    Args:
        numerator: Value to divide.
        denominator: Value to divide by; floored to at least floor.
        floor: Smallest denominator magnitude used, strictly positive.

    Returns:
        numerator / max(denominator, floor).

    Raises:
        ValueError: If floor is not strictly positive.
    """
    if floor <= 0.0:
        raise ValueError(f"floor must be strictly positive, received {floor}")
    return float(numerator) / max(float(denominator), floor)


def clip_to_unit_interval(value: float) -> float:
    """Constrain a scalar to [0, 1], mapping non-finite input to 0.0.

    Args:
        value: Any real number.

    Returns:
        The value bounded into [0, 1]; 0.0 if it is NaN or infinite.
    """
    if not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def describe_array_shape(array: Optional[np.ndarray]) -> str:
    """Render an array's shape as a readable string for trace logging.

    Args:
        array: Array to describe, or None.

    Returns:
        Shape string such as "512x512" or "none" when array is None.
    """
    if array is None:
        return "none"
    return "x".join(str(dimension) for dimension in np.asarray(array).shape)


def build_computation_step(step_number: int,
                           name: str,
                           description: str,
                           input_shape: str,
                           output_shape: str,
                           key_values: dict[str, Any]) -> dict:
    """Assemble one entry of the computation trace.

    Args:
        step_number: 1-based position in the pipeline.
        name: Short label, e.g. "Gradient magnitude".
        description: Full sentence describing what was done and why.
        input_shape: Human-readable shape of the input.
        output_shape: Human-readable shape of the output.
        key_values: Named scalars a reader would want to see.

    Returns:
        Dictionary with the fixed trace schema.
    """
    return {
        "step": step_number,
        "name": name,
        "description": description,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "key_values": key_values,
    }


def compose_confidence_penalties(penalties: Sequence[float]) -> float:
    """Combine independent confidence multipliers into a single weight.

    Multiplicative rather than additive so that several mild degradations
    compound, and any single disqualifying factor of 0.0 dominates.

    Args:
        penalties: Multipliers, each expected in [0, 1].

    Returns:
        Product of all multipliers, bounded into [0, 1]; 1.0 when empty.
    """
    weight = 1.0
    for penalty in penalties:
        weight *= penalty
    return clip_to_unit_interval(weight)
