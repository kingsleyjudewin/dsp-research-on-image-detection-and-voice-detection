"""Pure helper functions with no engine-specific logic and no side effects.

Every function here is generic enough for reuse by any other engine in this
system, per Rule 7 - none of them are named after wavelets, blocks, or
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


def robust_sigma_from_mad(values: np.ndarray, mad_constant: float) -> float:
    """Estimate a Gaussian standard deviation from the median absolute value.

    sigma = median(|values|) / mad_constant.

    Args:
        values: Array of coefficients assumed zero-mean and approximately
            Gaussian/Laplacian distributed.
        mad_constant: Distribution-specific normalising constant.

    Returns:
        Non-negative sigma estimate.
    """
    return float(np.median(np.abs(values))) / mad_constant


def binomial_coefficient(n: int, k: int) -> float:
    """Compute C(n, k), returning 0 for out-of-range k.

    Args:
        n: Non-negative integer.
        k: Integer, may be negative or exceed n.

    Returns:
        C(n, k) as a float, 0.0 when k < 0 or k > n.
    """
    if k < 0 or k > n:
        return 0.0
    numerator = 1.0
    denominator = 1.0
    for i in range(k):
        numerator *= (n - i)
        denominator *= (i + 1)
    return numerator / denominator


def solve_weighted_log_linear_fit(x_values: np.ndarray,
                                  y_values: np.ndarray,
                                  weights: np.ndarray) -> tuple:
    """Closed-form weighted least squares for y = intercept - slope * x.

    Solves the standard 2x2 normal-equation system for a weighted linear
    regression of y on x, matching the min_w sum(w*(y-(intercept-slope*x))^2)
    formulation.

    Args:
        x_values: Independent variable samples.
        y_values: Dependent variable samples.
        weights: Non-negative weight per sample.

    Returns:
        Tuple of (intercept, slope). Returns (0.0, 0.0) if the system is
        singular (e.g. fewer than 2 distinct weighted points).
    """
    sum_w = float(np.sum(weights))
    sum_wx = float(np.sum(weights * x_values))
    sum_wxx = float(np.sum(weights * x_values * x_values))
    sum_wy = float(np.sum(weights * y_values))
    sum_wxy = float(np.sum(weights * x_values * y_values))

    matrix = np.array([[sum_w, -sum_wx], [-sum_wx, sum_wxx]])
    vector = np.array([sum_wy, -sum_wxy])
    try:
        intercept, slope = np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        return 0.0, 0.0
    return float(intercept), float(slope)


def pixel_coordinate_grids(size: int) -> tuple:
    """Build row/column index grids for a square array, for moment sums.

    Args:
        size: Side length of the square grid.

    Returns:
        Tuple of (x_grid, y_grid), each size x size float64 arrays holding
        the column index and row index respectively at each position.
    """
    y_grid, x_grid = np.meshgrid(np.arange(size, dtype=np.float64),
                                 np.arange(size, dtype=np.float64),
                                 indexing="ij")
    return x_grid, y_grid


def generate_neighbour_offsets(max_offset: int, count: int) -> list:
    """Deterministically select `count` non-zero integer lattice offsets.

    Scans (dx, dy) in row-major order over [-max_offset, max_offset]^2,
    skipping (0, 0), and returns the first `count` found. Deterministic and
    reproducible given no other selection rule is specified.

    Args:
        max_offset: Maximum absolute value of dx and dy.
        count: Number of offsets to return.

    Returns:
        List of (dx, dy) integer tuples, length min(count, available).
    """
    offsets = []
    for dx in range(-max_offset, max_offset + 1):
        for dy in range(-max_offset, max_offset + 1):
            if dx == 0 and dy == 0:
                continue
            offsets.append((dx, dy))
            if len(offsets) >= count:
                return offsets
    return offsets
