"""Pure helper functions with no engine-specific logic and no side effects.

Every function here is deterministic, depends only on its arguments, and is
named for the general operation it performs rather than for vanishing points or
forgery detection, so any of the other eight engines can import it unchanged.
Nothing in this module reads or writes the filesystem, mutates global state, or
holds instance state, and nothing imports from the rest of this package.

The projective-geometry helpers work in homogeneous coordinates throughout, per
the SKILL file's Implementation Note that this is what lets a vanishing point
genuinely at infinity be represented without special-casing.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

# Structural constants owned by this module rather than by constants.py. Rule 7
# requires utils.py to import nothing from the rest of the package, so the
# definitional values it needs are named here instead of appearing as bare
# literals. None of them is a forensic parameter.
HOMOGENEOUS_LENGTH: int = 3
CARTESIAN_LENGTH: int = 2
SCALE_COMPONENT_INDEX: int = 2
QUARTER_TURN_RADIANS: float = float(np.pi / 2.0)
HALF_TURN_RADIANS: float = float(np.pi)
EIGHT_BIT_LEVEL_COUNT: int = 256
# Coefficient of the determinant term in the quadratic discriminant
# b^2 - 4ac. Pure algebra, not a tunable value.
QUADRATIC_DISCRIMINANT_COEFFICIENT: float = 4.0


def line_through_points(first_point: Sequence[float],
                        second_point: Sequence[float]) -> np.ndarray:
    """Homogeneous coefficients of the line joining two Cartesian points.

    The line is the cross product of the two points in homogeneous form, which
    is the standard projective construction. The result is scaled so its first
    two entries form a unit vector, making the dot product with a homogeneous
    point equal to the signed perpendicular distance in pixels.

    Args:
        first_point: (x, y) of one point on the line.
        second_point: (x, y) of another point on the line.

    Returns:
        Length-3 float array of line coefficients.

    Raises:
        ValueError: If the two points coincide, leaving the line undefined.
    """
    start = np.array([first_point[0], first_point[1], 1.0], dtype=np.float64)
    end = np.array([second_point[0], second_point[1], 1.0], dtype=np.float64)

    line = np.cross(start, end)
    normal_magnitude = float(np.hypot(line[0], line[1]))
    if normal_magnitude == 0.0:
        raise ValueError(f"cannot build a line through coincident points "
                         f"{tuple(first_point)} and {tuple(second_point)}")
    return line / normal_magnitude


def intersect_lines(first_line: np.ndarray,
                    second_line: np.ndarray) -> np.ndarray:
    """Homogeneous intersection point of two homogeneous lines.

    Returns the cross product without dividing through by the scale component,
    so two exactly parallel lines yield a valid point at infinity rather than a
    division by zero.

    Args:
        first_line: Length-3 homogeneous line coefficients.
        second_line: Length-3 homogeneous line coefficients.

    Returns:
        Length-3 homogeneous point. Its third entry is zero when the lines are
        parallel.
    """
    return np.cross(np.asarray(first_line, dtype=np.float64),
                    np.asarray(second_line, dtype=np.float64))


def homogeneous_to_cartesian(point: np.ndarray,
                             infinity_tolerance: float) -> Optional[np.ndarray]:
    """Convert a homogeneous point to Cartesian coordinates, if it is finite.

    Args:
        point: Length-3 homogeneous point.
        infinity_tolerance: Smallest scale component treated as finite.

    Returns:
        Length-2 array of (x, y), or None when the point lies at infinity.

    Raises:
        ValueError: If the point does not have three components.
    """
    vector = np.asarray(point, dtype=np.float64)
    if vector.size != HOMOGENEOUS_LENGTH:
        raise ValueError(f"expected a length-{HOMOGENEOUS_LENGTH} homogeneous "
                         f"point, received size {vector.size}")

    scale = float(vector[SCALE_COMPONENT_INDEX])
    if abs(scale) < infinity_tolerance:
        return None
    return vector[:CARTESIAN_LENGTH] / scale


def acute_angle_between_directions(first_direction: np.ndarray,
                                   second_direction: np.ndarray) -> float:
    """Acute angle between two undirected directions, folded into [0, pi/2].

    Lines have no orientation, so a direction and its negation describe the same
    line; taking the absolute value of the dot product before the arccosine is
    what folds the result into the acute range.

    Args:
        first_direction: Length-2 direction vector, need not be unit length.
        second_direction: Length-2 direction vector.

    Returns:
        Angle in radians within [0, pi/2]; 0.0 if either vector is degenerate.
    """
    first = np.asarray(first_direction, dtype=np.float64)
    second = np.asarray(second_direction, dtype=np.float64)

    magnitudes = float(np.linalg.norm(first) * np.linalg.norm(second))
    if magnitudes == 0.0:
        return 0.0

    cosine = float(np.clip(abs(float(np.dot(first, second))) / magnitudes,
                           -1.0, 1.0))
    return float(np.arccos(cosine))


def signed_direction_angle(from_point: Sequence[float],
                           to_point: Sequence[float]) -> float:
    """Direction of the segment from one point to another, in radians.

    Args:
        from_point: (x, y) of the segment start.
        to_point: (x, y) of the segment end.

    Returns:
        Angle in (-pi, pi] measured from the positive x axis.
    """
    delta_x = float(to_point[0]) - float(from_point[0])
    delta_y = float(to_point[1]) - float(from_point[1])
    return float(np.arctan2(delta_y, delta_x))


def wrap_angle_difference(first_angle: float, second_angle: float) -> float:
    """Smallest absolute difference between two angles, accounting for wrap.

    Without wrapping, directions of +179 and -179 degrees would appear 358
    degrees apart rather than 2.

    Args:
        first_angle: Angle in radians.
        second_angle: Angle in radians.

    Returns:
        Absolute difference in radians within [0, pi].
    """
    difference = float(first_angle) - float(second_angle)
    wrapped = (difference + HALF_TURN_RADIANS) % (2.0 * HALF_TURN_RADIANS)
    return float(abs(wrapped - HALF_TURN_RADIANS))


def point_to_line_distance(point: Sequence[float],
                           line: np.ndarray) -> float:
    """Perpendicular distance in pixels from a Cartesian point to a line.

    Assumes the line was normalised so its first two coefficients form a unit
    vector, as line_through_points guarantees.

    Args:
        point: (x, y) of the point.
        line: Length-3 homogeneous line coefficients, unit-normalised.

    Returns:
        Non-negative distance in pixels.
    """
    homogeneous_point = np.array([point[0], point[1], 1.0], dtype=np.float64)
    return float(abs(np.dot(np.asarray(line, dtype=np.float64),
                            homogeneous_point)))


def smallest_eigenvector(symmetric_matrix: np.ndarray) -> np.ndarray:
    """Unit eigenvector of the smallest eigenvalue of a symmetric matrix.

    This is the least-squares solution of a homogeneous system: for a scatter
    matrix built as the sum of outer products of constraint vectors, it is the
    vector most nearly orthogonal to all of them.

    Uses eigh rather than eig because the input is symmetric by construction,
    which guarantees real eigenvalues and an orthonormal basis.

    Args:
        symmetric_matrix: Square symmetric array.

    Returns:
        Unit-norm eigenvector belonging to the smallest eigenvalue.

    Raises:
        numpy.linalg.LinAlgError: If the decomposition does not converge.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(
        np.asarray(symmetric_matrix, dtype=np.float64))
    return np.asarray(eigenvectors[:, int(np.argmin(eigenvalues))],
                      dtype=np.float64)


def fit_line_least_squares(points: np.ndarray) -> np.ndarray:
    """Total-least-squares line through a set of Cartesian points.

    Minimises perpendicular rather than vertical distance, so the fit does not
    degrade for near-vertical point sets.

    Args:
        points: Array of shape (n, 2) holding (x, y) coordinates.

    Returns:
        Length-3 homogeneous line coefficients, unit-normalised.

    Raises:
        ValueError: If fewer than two points are supplied.
    """
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.shape[0] < CARTESIAN_LENGTH:
        raise ValueError(f"need at least {CARTESIAN_LENGTH} points to fit a "
                         f"line, received {coordinates.shape[0]}")

    centroid = coordinates.mean(axis=0)
    centred = coordinates - centroid
    # The line's normal is the direction of least variance, i.e. the smallest
    # eigenvector of the centred scatter matrix.
    normal = smallest_eigenvector(centred.T @ centred)
    offset = -float(np.dot(normal, centroid))
    return np.array([normal[0], normal[1], offset], dtype=np.float64)


def mean_perpendicular_distance(points: np.ndarray,
                                line: np.ndarray) -> float:
    """Average perpendicular distance from a set of points to a line.

    Args:
        points: Array of shape (n, 2) holding (x, y) coordinates.
        line: Length-3 homogeneous line coefficients, unit-normalised.

    Returns:
        Non-negative mean distance in pixels; 0.0 for an empty point set.
    """
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.shape[0] == 0:
        return 0.0

    homogeneous = np.hstack([coordinates,
                             np.ones((coordinates.shape[0], 1),
                                     dtype=np.float64)])
    return float(np.mean(np.abs(homogeneous @ np.asarray(line,
                                                         dtype=np.float64))))


def smallest_eigenvalue_of_two_by_two(matrix: np.ndarray) -> float:
    """Smaller eigenvalue of a 2x2 symmetric matrix, in closed form.

    Avoids a full decomposition inside an optimiser's residual function, where
    it would be evaluated thousands of times.

    Args:
        matrix: 2x2 symmetric array.

    Returns:
        The smaller eigenvalue, clipped at zero so floating-point noise on a
        positive-semidefinite input cannot produce a negative result.
    """
    trace = float(matrix[0, 0] + matrix[1, 1])
    determinant = float(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0])
    discriminant = max(trace * trace
                   - QUADRATIC_DISCRIMINANT_COEFFICIENT * determinant,
                   0.0)
    return max(0.5 * (trace - float(np.sqrt(discriminant))), 0.0)


def compute_colour_histogram(pixels: np.ndarray,
                             bins_per_channel: int) -> np.ndarray:
    """Normalised joint colour histogram of a set of pixels.

    Args:
        pixels: Array of shape (n, channels) of 8-bit intensities.
        bins_per_channel: Number of bins along each channel axis.

    Returns:
        Flat float array of length bins_per_channel ** channels, summing to 1.0.
        Returns a uniform histogram when no pixels are supplied.

    Raises:
        ValueError: If bins_per_channel is not positive.
    """
    if bins_per_channel <= 0:
        raise ValueError(f"bins_per_channel must be positive, received "
                         f"{bins_per_channel}")

    samples = np.asarray(pixels, dtype=np.float64)
    channel_count = samples.shape[1] if samples.ndim > 1 else 1
    bin_count = bins_per_channel ** channel_count
    if samples.size == 0:
        return np.full(bin_count, 1.0 / bin_count, dtype=np.float64)

    # Quantise each channel, then fold the per-channel bin indices into one
    # flat index by treating them as digits in base bins_per_channel.
    scaled = np.floor(samples * bins_per_channel / EIGHT_BIT_LEVEL_COUNT)
    quantised = np.clip(scaled, 0, bins_per_channel - 1).astype(np.int64)
    weights = bins_per_channel ** np.arange(channel_count, dtype=np.int64)
    flat_indices = quantised @ weights

    counts = np.bincount(flat_indices, minlength=bin_count).astype(np.float64)
    return counts / counts.sum()


def chi_square_distance(first_histogram: np.ndarray,
                        second_histogram: np.ndarray,
                        epsilon: float) -> float:
    """Chi-square distance between two normalised histograms.

    Args:
        first_histogram: Non-negative vector summing to 1.
        second_histogram: Non-negative vector of the same length.
        epsilon: Floor added to the denominator to avoid dividing by zero.

    Returns:
        Non-negative distance; 0.0 for identical histograms.

    Raises:
        ValueError: If the histograms have different lengths.
    """
    first = np.asarray(first_histogram, dtype=np.float64)
    second = np.asarray(second_histogram, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(f"histogram shapes {first.shape} and {second.shape} "
                         f"do not match")

    difference = first - second
    return float(0.5 * np.sum(difference * difference / (first + second + epsilon)))


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
        Shape string such as "512x512x3" or "none" when array is None.
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
        name: Short label, e.g. "Vanishing point estimation".
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
