"""Pure helper functions with no engine-specific logic and no side effects.

Every function here is generic enough for reuse by any other engine in this
system, per Rule 7 - none of them are named after PRNU, cameras, or
forgery-specific concepts.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def clip_to_unit_interval(value: float) -> float:
    """Clamp a float into [0.0, 1.0].

    Args:
        value: Any real number.

    Returns:
        value clipped into [0.0, 1.0].
    """
    return float(np.clip(value, 0.0, 1.0))


def compose_confidence_penalties(penalties: list) -> float:
    """Combine independent [0, 1] confidence multipliers into one weight.

    Args:
        penalties: List of per-check confidence multipliers.

    Returns:
        Product of all penalties, or 1.0 if the list is empty.
    """
    weight = 1.0
    for penalty in penalties:
        weight *= float(penalty)
    return clip_to_unit_interval(weight)


def build_computation_step(step: int,
                           name: str,
                           description: str,
                           input_shape: str = "",
                           output_shape: str = "",
                           key_values: Optional[dict] = None) -> dict:
    """Assemble one computation_steps log entry in the fixed report shape.

    Args:
        step: 1-indexed position of this step in the pipeline.
        name: Short step name.
        description: Human-readable explanation for the report generator.
        input_shape: Text description of the step's input shape.
        output_shape: Text description of the step's output shape.
        key_values: Numeric/summary values worth surfacing in the report.

    Returns:
        Dictionary matching the computation_steps contract.
    """
    return {
        "step": step,
        "name": name,
        "description": description,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "key_values": key_values or {},
    }


def grid_neighbourhood_median(grid: np.ndarray, window: int) -> np.ndarray:
    """Median of each cell's surrounding window x window neighbourhood.

    Args:
        grid: 2-D array of per-cell scalar statistics.
        window: Odd neighbourhood side length (e.g. 3 for a 3x3 window).

    Returns:
        2-D array, same shape as grid, holding each cell's local median
        (excluding the cell itself).
    """
    half = window // 2
    height, width = grid.shape
    medians = np.empty_like(grid, dtype=np.float64)
    for row in range(height):
        for col in range(width):
            row_start, row_end = max(0, row - half), min(height, row + half + 1)
            col_start, col_end = max(0, col - half), min(width, col + half + 1)
            neighbourhood = grid[row_start:row_end, col_start:col_end].copy()
            exclude_mask = np.ones(neighbourhood.shape, dtype=bool)
            exclude_mask[row - row_start, col - col_start] = False
            values = neighbourhood[exclude_mask]
            medians[row, col] = float(np.median(values)) if values.size \
                else float(grid[row, col])
    return medians


def top_k_fraction_mean(values: np.ndarray, fraction: float) -> float:
    """Mean of the top `fraction` largest values in an array.

    Args:
        values: 1-D or N-D array of scalars.
        fraction: Fraction in (0, 1] of the largest values to average.

    Returns:
        Mean of the top fraction, or 0.0 for an empty array.
    """
    flat = np.asarray(values, dtype=np.float64).ravel()
    if flat.size == 0:
        return 0.0
    count = max(1, int(np.ceil(flat.size * fraction)))
    top_values = np.sort(flat)[-count:]
    return float(np.mean(top_values))
