"""Pure helper functions with no engine-specific logic and no side effects.

Every function here is deterministic, depends only on its arguments, and is
named for the general operation it performs rather than for Benford's Law, so
any of the other eight engines can import it unchanged. Nothing in this module
reads or writes the filesystem, mutates global state, or holds instance state.
"""

from __future__ import annotations

import io
from functools import lru_cache
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image


def generate_zigzag_indices(block_size: int) -> tuple[tuple[int, int], ...]:
    """Return (row, column) coordinates of a block in JPEG zig-zag scan order.

    Computed from the anti-diagonal traversal rule rather than stored as a
    literal table, so no magic numbers are introduced and any block size works.
    Index 0 is always the DC coefficient at (0, 0).

    Args:
        block_size: Edge length of the square transform block, e.g. 8.

    Returns:
        Tuple of (row, column) pairs, length block_size**2, in scan order.

    Raises:
        ValueError: If block_size is not a positive integer.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, received {block_size}")

    coordinates: list[tuple[int, int]] = []
    for diagonal_index in range(2 * block_size - 1):
        # Collect every cell lying on this anti-diagonal (row + column == index).
        diagonal_cells = [
            (row, diagonal_index - row)
            for row in range(block_size)
            if 0 <= diagonal_index - row < block_size
        ]
        # Even-numbered anti-diagonals are traversed bottom-left to top-right,
        # odd-numbered ones top-right to bottom-left. That alternation is what
        # produces the characteristic zig-zag.
        if diagonal_index % 2 == 0:
            diagonal_cells.reverse()
        coordinates.extend(diagonal_cells)
    return tuple(coordinates)


def extract_first_significant_digit(values: np.ndarray, base: int) -> np.ndarray:
    """Extract the leading significant digit of each value in the given base.

    Implements d = floor(|v| / base**floor(log_base |v|)) elementwise. Zero has
    no leading digit, so zero inputs map to the sentinel 0, which lies outside
    the valid digit range {1, ..., base-1} and is therefore unambiguous. Results
    are returned unclamped so the caller can audit floating-point edge cases.

    Args:
        values: Numeric array of any shape.
        base: Digit base, must be at least 2.

    Returns:
        Integer array, same shape as values, holding leading digits with 0
        marking "input was zero".

    Raises:
        ValueError: If base is less than 2.
    """
    if base < 2:
        raise ValueError(f"base must be at least 2, received {base}")

    magnitudes = np.abs(np.asarray(values, dtype=np.float64))
    nonzero_mask = magnitudes > 0.0

    digits = np.zeros(magnitudes.shape, dtype=np.int64)
    if not np.any(nonzero_mask):
        return digits

    # log_base(x) computed as log(x)/log(base); evaluated only where x > 0 so no
    # divide-by-zero or log(0) warning can be raised.
    surviving = magnitudes[nonzero_mask]
    exponents = np.floor(np.log(surviving) / np.log(float(base)))
    digits[nonzero_mask] = np.floor(surviving / np.power(float(base), exponents))
    return digits


def compute_probability_mass_function(digits: np.ndarray, base: int) -> np.ndarray:
    """Build the normalised histogram of digits over the support {1..base-1}.

    Args:
        digits: Integer array of leading digits. Entries outside {1..base-1}
            (notably the 0 sentinel) are ignored.
        base: Digit base defining the support width.

    Returns:
        Float array of length base-1, summing to 1.0, where element i holds the
        observed relative frequency of digit i+1. Returns all zeros when no
        digit lies in the valid support.
    """
    support_width = base - 1
    flattened = np.asarray(digits, dtype=np.int64).ravel()

    in_support = flattened[(flattened >= 1) & (flattened <= support_width)]
    if in_support.size == 0:
        return np.zeros(support_width, dtype=np.float64)

    # minlength covers digits 0..base-1; slice off index 0, which the support
    # excludes by definition.
    counts = np.bincount(in_support, minlength=base)[1:base]
    return counts.astype(np.float64) / float(in_support.size)


def _encode_probe_and_read_luminance_table(quality_factor: int,
                                           block_size: int) -> list:
    """Encode a blank grayscale probe and read back its quantization table.

    Args:
        quality_factor: JPEG quality in 1..100.
        block_size: Edge length of the probe image.

    Returns:
        Flat list of quantization steps in raster order.

    Raises:
        OSError: If Pillow fails to encode or decode the probe.
    """
    # A single-channel probe is used deliberately: it carries only a luminance
    # table, so there is no chrominance table to disambiguate, and it avoids
    # naming a channel count here. Verified to return byte-identical tables to
    # a 3-channel probe at quality 50, 80, 95 and 100.
    probe = Image.fromarray(np.zeros((block_size, block_size), dtype=np.uint8))
    buffer = io.BytesIO()
    probe.save(buffer, format="JPEG", quality=int(quality_factor))
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return list(decoded.quantization[0])


@lru_cache(maxsize=None)
def load_standard_jpeg_quantization_table(quality_factor: int,
                                          block_size: int) -> tuple:
    """Return the standard IJG luminance quantization table at a quality factor.

    The table is obtained by encoding a blank image with Pillow and reading back
    the quantization table the encoder chose, rather than hardcoding the IJG
    matrix and its quality-scaling rule. This keeps the standard tables in the
    library that owns them and introduces no unsourced numeric literals.

    Pillow returns tables in raster (row-major) order, verified empirically
    against the published IJG luminance table whose first row is
    16, 11, 10, 16, 24, 40, 51, 61.

    Memoised because the sweep requests the same few tables repeatedly; the
    function remains referentially transparent.

    Args:
        quality_factor: JPEG quality in 1..100.
        block_size: Transform block edge length, e.g. 8.

    Returns:
        Nested tuple of shape (block_size, block_size) of steps. A tuple rather
        than an ndarray so the result is hashable and cacheable.

    Raises:
        ValueError: If the encoder returns a table of unexpected length.
        OSError: If Pillow fails to encode the probe image.
    """
    luminance_table = _encode_probe_and_read_luminance_table(quality_factor,
                                                             block_size)
    expected_length = block_size * block_size
    if len(luminance_table) != expected_length:
        raise ValueError(
            f"expected a {expected_length}-entry quantization table at quality "
            f"{quality_factor}, received {len(luminance_table)}"
        )

    table = np.asarray(luminance_table, dtype=np.float64).reshape(block_size,
                                                                 block_size)
    return tuple(tuple(row) for row in table.tolist())


def compute_saturated_pixel_fraction(channel: np.ndarray,
                                     minimum_level: int,
                                     maximum_level: int) -> float:
    """Fraction of pixels pinned at either end of the intensity range.

    Args:
        channel: Single-channel image array.
        minimum_level: Intensity treated as fully dark, e.g. 0.
        maximum_level: Intensity treated as fully bright, e.g. 255.

    Returns:
        Fraction in [0, 1]; 0.0 for an empty array.
    """
    flattened = np.asarray(channel).ravel()
    if flattened.size == 0:
        return 0.0

    pinned = np.count_nonzero(
        (flattened <= minimum_level) | (flattened >= maximum_level)
    )
    return float(pinned) / float(flattened.size)


def safe_normalise_distribution(values: np.ndarray,
                                probability_floor: float) -> np.ndarray:
    """Clip a vector to a strictly positive floor and rescale it to sum to 1.

    Guarantees a valid probability vector for logarithmic divergences, where a
    zero entry in the denominator would otherwise produce infinity.

    Args:
        values: Non-negative vector.
        probability_floor: Smallest permitted entry, strictly positive.

    Returns:
        Float vector of the same length, strictly positive, summing to 1.0.

    Raises:
        ValueError: If probability_floor is not strictly positive.
    """
    if probability_floor <= 0.0:
        raise ValueError("probability_floor must be strictly positive")

    clipped = np.clip(np.asarray(values, dtype=np.float64), probability_floor, None)
    return clipped / clipped.sum()


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
        name: Short label, e.g. "DCT extraction".
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
