"""Pure helper functions with no engine-specific logic and no side effects.

Every function here is deterministic, depends only on its arguments, and is
named for the general operation it performs rather than for CFA or demosaicing,
so any of the other eight engines can import it unchanged. Nothing in this
module reads or writes the filesystem, mutates global state, or holds instance
state, and nothing imports from the rest of this package.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from scipy.special import gammaln, logsumexp

# Structural constants owned by this module rather than by constants.py. Rule 7
# requires utils.py to import nothing from the rest of the package, so the two
# definitional values it needs are named here instead of being carried in as
# bare literals. Neither is a forensic parameter: one is the base of the common
# logarithm, the other the maximum value of an 8-bit sample.
DECIMAL_LOG_BASE: float = 10.0
EIGHT_BIT_MAXIMUM: float = 255.0


def build_gaussian_window(half_width: int, standard_deviation: float) -> np.ndarray:
    """Build a square, unnormalised Gaussian window.

    Args:
        half_width: K, giving a (2K+1) x (2K+1) window.
        standard_deviation: Sigma of the Gaussian, in pixels.

    Returns:
        Float array of shape (2*half_width+1, 2*half_width+1), peak 1.0 at the
        centre.

    Raises:
        ValueError: If half_width is negative or standard_deviation is not
            strictly positive.
    """
    if half_width < 0:
        raise ValueError(f"half_width must be non-negative, received {half_width}")
    if standard_deviation <= 0.0:
        raise ValueError("standard_deviation must be strictly positive, "
                         f"received {standard_deviation}")

    span = np.arange(-half_width, half_width + 1, dtype=np.float64)
    rows, columns = np.meshgrid(span, span, indexing="ij")
    squared_radius = rows ** 2 + columns ** 2
    return np.exp(-squared_radius / (2.0 * standard_deviation ** 2))


def build_checkerboard_parity_mask(shape: tuple, parity: int) -> np.ndarray:
    """Mark the lattice sites whose (row + column) matches a given parity.

    A quincunx / checkerboard lattice is exactly the set of positions where
    (row + column) is even or odd, so this expresses both the acquired and the
    interpolated sublattice of a Bayer green plane, and any other two-phase
    interleaved sampling grid.

    Args:
        shape: (rows, columns) of the mask to build.
        parity: 0 to mark even (row + column), 1 to mark odd.

    Returns:
        Boolean array of the requested shape.

    Raises:
        ValueError: If parity is not 0 or 1.
    """
    if parity not in (0, 1):
        raise ValueError(f"parity must be 0 or 1, received {parity}")

    rows, columns = np.indices(shape)
    return ((rows + columns) % 2) == parity


def normalise_kernel(kernel: np.ndarray) -> np.ndarray:
    """Scale a kernel so its entries sum to one.

    Args:
        kernel: Array with a strictly positive sum.

    Returns:
        Float array of the same shape summing to 1.0.

    Raises:
        ValueError: If the kernel sums to zero.
    """
    array = np.asarray(kernel, dtype=np.float64)
    total = array.sum()
    if total == 0.0:
        raise ValueError("cannot normalise a kernel whose entries sum to zero")
    return array / total


def partition_into_blocks(array: np.ndarray, block_size: int) -> np.ndarray:
    """Reshape a 2-D array into a grid of non-overlapping square blocks.

    Any trailing rows or columns that do not fill a whole block are dropped, so
    the caller does not have to crop first.

    Args:
        array: Two-dimensional input.
        block_size: Edge length of each block.

    Returns:
        Array of shape (block_rows, block_columns, block_size, block_size).

    Raises:
        ValueError: If block_size is not positive or the array is not 2-D.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, received {block_size}")

    data = np.asarray(array)
    if data.ndim != 2:
        raise ValueError(f"expected a 2-D array, received shape {data.shape}")

    block_rows = data.shape[0] // block_size
    block_columns = data.shape[1] // block_size
    if block_rows == 0 or block_columns == 0:
        raise ValueError(f"array of shape {data.shape} holds no whole "
                         f"{block_size}x{block_size} block")

    trimmed = data[:block_rows * block_size, :block_columns * block_size]
    return trimmed.reshape(block_rows, block_size,
                           block_columns, block_size).transpose(0, 2, 1, 3)


def masked_block_mean(values: np.ndarray,
                      mask: np.ndarray,
                      block_size: int) -> tuple:
    """Mean of the values inside each block, counting only masked-in positions.

    Args:
        values: Two-dimensional array to average.
        mask: Boolean array of the same shape; True positions are counted.
        block_size: Edge length of each block.

    Returns:
        Tuple of (means, counts): the per-block mean over masked-in positions
        and the number of such positions. Blocks with no masked-in position
        yield a mean of 0.0 and a count of 0.

    Raises:
        ValueError: If values and mask have different shapes.
    """
    data = np.asarray(values, dtype=np.float64)
    selector = np.asarray(mask, dtype=bool)
    if data.shape != selector.shape:
        raise ValueError(f"values shape {data.shape} does not match mask shape "
                         f"{selector.shape}")

    value_blocks = partition_into_blocks(data * selector, block_size)
    count_blocks = partition_into_blocks(selector.astype(np.float64), block_size)

    totals = value_blocks.sum(axis=(2, 3))
    counts = count_blocks.sum(axis=(2, 3))
    means = np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)
    return means, counts


def aggregate_blocks(values: np.ndarray, group_size: int) -> np.ndarray:
    """Sum a block-resolution map onto a coarser grid of block groups.

    Used to cumulate additive per-block quantities such as log-likelihood
    ratios onto larger output blocks.

    Args:
        values: Two-dimensional block-resolution map.
        group_size: Number of blocks per side in each group.

    Returns:
        Two-dimensional array of group sums.

    Raises:
        ValueError: If group_size is not positive.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, received {group_size}")
    if group_size == 1:
        return np.asarray(values, dtype=np.float64)
    return partition_into_blocks(values, group_size).sum(axis=(2, 3))


def log_binomial_tail(successes: int, trials: int, probability: float) -> float:
    """Natural log of the upper binomial tail, computed without underflow.

    Evaluates log( sum_{i=successes}^{trials} C(trials,i) p^i (1-p)^(trials-i) )
    by summing the terms in log space with a log-sum-exp, which stays exact
    where the plain sum underflows to zero.

    Verified against scipy.stats.binom.logsf to within 7e-15 wherever scipy
    itself remains finite; scipy returns -inf from around 1024 trials with every
    trial a success, where this function still returns the exact value.

    Args:
        successes: Lower limit of the tail, the observed count.
        trials: Number of independent trials.
        probability: Per-trial success probability, strictly inside (0, 1).

    Returns:
        Natural logarithm of the tail probability. Returns 0.0 (log of 1) when
        successes is zero or negative, since the tail is then certain.

    Raises:
        ValueError: If trials is negative or probability is outside (0, 1).
    """
    if trials < 0:
        raise ValueError(f"trials must be non-negative, received {trials}")
    if not 0.0 < probability < 1.0:
        raise ValueError(f"probability must lie in (0, 1), received {probability}")
    if successes <= 0:
        return 0.0
    if successes > trials:
        return -np.inf

    indices = np.arange(successes, trials + 1, dtype=np.float64)
    # log C(n, i) via log-gamma, which stays exact for large n where the
    # factorials themselves overflow.
    log_coefficients = (gammaln(trials + 1.0) - gammaln(indices + 1.0)
                        - gammaln(trials - indices + 1.0))
    log_terms = (log_coefficients + indices * np.log(probability)
                 + (trials - indices) * np.log1p(-probability))
    return float(logsumexp(log_terms))


def natural_log_to_log10(value: float) -> float:
    """Convert a natural logarithm to a base-10 logarithm.

    Args:
        value: A natural logarithm, possibly -inf.

    Returns:
        The same quantity in base 10, preserving -inf.
    """
    if value == -np.inf:
        return -np.inf
    return float(value) / float(np.log(DECIMAL_LOG_BASE))


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


def upscale_block_map(block_map: np.ndarray, block_size: int) -> np.ndarray:
    """Expand a block-resolution map back to pixel resolution.

    Nearest-neighbour replication, so no value is invented by interpolation and
    the block structure of the measurement stays visible to a reader.

    Args:
        block_map: Two-dimensional block-resolution array.
        block_size: Pixels per block along each axis.

    Returns:
        Array of shape (rows * block_size, columns * block_size).

    Raises:
        ValueError: If block_size is not positive.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, received {block_size}")
    data = np.asarray(block_map)
    return np.repeat(np.repeat(data, block_size, axis=0), block_size, axis=1)


def find_connected_components(mask: np.ndarray) -> list:
    """Group the True cells of a boolean map into 4-connected components.

    A small iterative flood fill rather than a library call, so the function
    carries no dependency beyond numpy and behaves identically everywhere.

    Args:
        mask: Two-dimensional boolean array.

    Returns:
        List of components, each a list of (row, column) index pairs.
    """
    selector = np.asarray(mask, dtype=bool)
    unvisited = selector.copy()
    components: list = []

    for start in zip(*np.nonzero(selector)):
        if not unvisited[start]:
            continue
        component: list = []
        pending = [start]
        unvisited[start] = False
        while pending:
            row, column = pending.pop()
            component.append((int(row), int(column)))
            for neighbour in ((row - 1, column), (row + 1, column),
                              (row, column - 1), (row, column + 1)):
                if (0 <= neighbour[0] < selector.shape[0]
                        and 0 <= neighbour[1] < selector.shape[1]
                        and unvisited[neighbour]):
                    unvisited[neighbour] = False
                    pending.append(neighbour)
        components.append(component)

    return components


def apply_diverging_colormap(values: np.ndarray,
                             low_colour: Sequence[int],
                             mid_colour: Sequence[int],
                             high_colour: Sequence[int]) -> np.ndarray:
    """Map a [0, 1] scalar field onto a three-stop diverging colour ramp.

    Args:
        values: Two-dimensional array, expected in [0, 1] and clipped if not.
        low_colour: Three channel values for value 0.0.
        mid_colour: Three channel values for value 0.5.
        high_colour: Three channel values for value 1.0.

    Returns:
        uint8 array of shape (rows, columns, 3).
    """
    scalar = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    low = np.asarray(low_colour, dtype=np.float64)
    mid = np.asarray(mid_colour, dtype=np.float64)
    high = np.asarray(high_colour, dtype=np.float64)

    # Two linear segments meeting at 0.5, so the neutral colour marks the
    # decision point rather than an arbitrary place on a continuous ramp.
    lower_weight = np.clip(scalar * 2.0, 0.0, 1.0)[..., np.newaxis]
    upper_weight = np.clip((scalar - 0.5) * 2.0, 0.0, 1.0)[..., np.newaxis]

    lower_segment = low * (1.0 - lower_weight) + mid * lower_weight
    blended = lower_segment * (1.0 - upper_weight) + high * upper_weight
    return np.clip(blended, 0.0, EIGHT_BIT_MAXIMUM).astype(np.uint8)


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
        name: Short label, e.g. "Prediction error".
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
