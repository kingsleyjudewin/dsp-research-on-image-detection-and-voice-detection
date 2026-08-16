"""Pure helper functions with no engine-specific logic and no side effects.

Every function here is generic enough for reuse by any other engine in this
system, per Rule 7 - none of them are named after JPEG, quantization, or
forgery-specific concepts.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.fft import dctn
from scipy.ndimage import minimum_filter1d, uniform_filter1d


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


def split_into_tiles(image: np.ndarray, tile_size: int) -> np.ndarray:
    """Split a 2-D array into non-overlapping square tiles.

    Args:
        image: 2-D array whose dimensions are exact multiples of tile_size.
        tile_size: Side length of each tile.

    Returns:
        Array of shape (n_tiles, tile_size, tile_size), in row-major tile order.

    Raises:
        ValueError: If either dimension is not a multiple of tile_size.
    """
    height, width = image.shape
    if height % tile_size or width % tile_size:
        raise ValueError(f"image {height}x{width} is not an exact multiple of "
                         f"tile size {tile_size}")
    tiled = image.reshape(height // tile_size, tile_size,
                          width // tile_size, tile_size)
    # Bring the two tile-index axes adjacent so the flatten below yields
    # whole tiles in row-major order rather than interleaved rows.
    return np.swapaxes(tiled, 1, 2).reshape(-1, tile_size, tile_size)


def tilewise_dct_two_dimensional(tiles: np.ndarray) -> np.ndarray:
    """Apply an orthonormal type-II 2-D DCT to each tile independently.

    Args:
        tiles: Array of shape (n_tiles, size, size).

    Returns:
        Array of the same shape holding each tile's 2-D DCT coefficients.
    """
    return dctn(tiles, type=2, norm="ortho", axes=(1, 2))


def integer_bin_histogram(values: np.ndarray) -> tuple:
    """Histogram integer-valued data into unit-width bins, one bin per value.

    Args:
        values: Array of values that are already effectively integer-valued.

    Returns:
        Tuple of (bin_centres, counts); both empty when values is empty.
    """
    if values.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    rounded = np.rint(values).astype(np.int64)
    lowest, highest = int(rounded.min()), int(rounded.max())
    centres = np.arange(lowest, highest + 1, dtype=np.int64)
    counts = np.bincount(rounded - lowest,
                         minlength=centres.size).astype(np.float64)
    return centres, counts


def unit_norm(values: np.ndarray, floor: float) -> np.ndarray:
    """Scale a vector to unit L2 length, leaving a near-zero vector unchanged.

    Args:
        values: 1-D array.
        floor: Norm below which the vector is treated as degenerate.

    Returns:
        The vector scaled to unit L2 norm, or unchanged if its norm is below
        the floor.
    """
    norm = float(np.linalg.norm(values))
    if norm < floor:
        return values
    return values / norm


def trailing_minimum(values: np.ndarray, window_length: int) -> np.ndarray:
    """Minimum over a trailing window ending at (and including) each position.

    For position f the window covers values[f - window_length : f + 1],
    clamped at the array start.

    Args:
        values: 1-D array.
        window_length: Number of preceding positions included alongside f.

    Returns:
        Array of the same shape holding each position's trailing minimum.
    """
    if values.size == 0:
        return values
    # scipy's origin shifts the window; origin = n//2 with size = n+1 places
    # it exactly on [f-n, f]. Verified against a direct slice-minimum for
    # window lengths 2 through 6.
    return minimum_filter1d(values, size=window_length + 1,
                            origin=window_length // 2, mode="nearest")


def moving_average(values: np.ndarray, window_length: int) -> np.ndarray:
    """Smooth a 1-D array with a centred uniform (box) filter.

    Args:
        values: 1-D array.
        window_length: Filter length in samples.

    Returns:
        Smoothed array of the same shape.
    """
    if values.size == 0:
        return values
    return uniform_filter1d(values, size=window_length, mode="nearest")
