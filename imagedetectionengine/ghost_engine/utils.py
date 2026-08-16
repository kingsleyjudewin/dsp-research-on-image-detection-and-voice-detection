"""Pure helper functions with no engine-specific logic and no side effects.

Every function here is generic enough for reuse by any other engine in this
system, per Rule 7 - none of them are named after ghosts, resampling, or
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


def centre_crop(image: np.ndarray, window: int) -> np.ndarray:
    """Crop the centred square window of a 2-D array.

    Args:
        image: 2-D array.
        window: Desired side length; larger than the array returns it whole.

    Returns:
        The centred sub-array, or the input when it is already smaller.
    """
    height, width = image.shape[:2]
    if height <= window and width <= window:
        return image
    side = min(window, height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return image[top:top + side, left:left + side]


def min_max_normalise_along_axis(stack: np.ndarray, axis: int) -> np.ndarray:
    """Rescale each position to [0, 1] using extrema taken along one axis.

    Args:
        stack: Array holding one slice per sweep step along `axis`.
        axis: Axis over which the minimum and maximum are taken.

    Returns:
        Array of the same shape, each position rescaled independently.
        Positions whose range is zero become 0.
    """
    lowest = np.min(stack, axis=axis, keepdims=True)
    highest = np.max(stack, axis=axis, keepdims=True)
    span = highest - lowest
    safe_span = np.where(span > 0.0, span, 1.0)
    normalised = (stack - lowest) / safe_span
    return np.where(span > 0.0, normalised, 0.0)


def radial_frequency_grid(shape: tuple) -> np.ndarray:
    """Distance of every DFT bin from the centred DC term, normalised to 1.

    Assumes the spectrum has already been fftshift-ed so DC sits at the
    centre.

    Args:
        shape: (height, width) of the spectrum.

    Returns:
        Float array of the same shape, 0 at the centre rising to about 1 at
        the corners.
    """
    height, width = shape
    rows = np.arange(height) - height / 2.0
    columns = np.arange(width) - width / 2.0
    radius = np.sqrt(rows[:, None] ** 2 + columns[None, :] ** 2)
    half_diagonal = np.sqrt((height / 2.0) ** 2 + (width / 2.0) ** 2)
    return radius / half_diagonal


def two_class_split_by_threshold(values: np.ndarray,
                                 threshold: float) -> np.ndarray:
    """Label values below a threshold as class 0 and the rest as class 1.

    Args:
        values: Array of scalars.
        threshold: Split point.

    Returns:
        Integer array of the same shape holding 0/1 labels.
    """
    return (values >= threshold).astype(np.int64)


def class_mean_and_standard_deviation(values: np.ndarray) -> tuple:
    """Return the mean and population standard deviation of an array.

    Args:
        values: 1-D array of samples.

    Returns:
        Tuple of (mean, standard deviation); (0.0, 0.0) when empty.
    """
    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))
