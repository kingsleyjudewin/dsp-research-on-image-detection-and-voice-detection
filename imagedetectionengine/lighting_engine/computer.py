"""Core mathematical computation for the lighting / illumination engine.

One class, implementing the ONLY technique in this SKILL file with an actual
formula and no ML dependency: the Sobel-gradient magnitude heuristic (Pipeline
A, Rao, Ghanekar, Chitnis, Dawkhar & Mishra 2025). Pipeline B (Spherical-
Harmonics photometric consistency) has no implementable formula anywhere in the
SKILL - every step is tagged "(not specified in the corpus)". Pipeline C is
excluded under the no-ML constraint. See constants.KNOWN_UNIMPLEMENTED_MODULES.

Every formula carries a SKILL VERIFICATION block naming the equation, its
variables, its source paper and the range its output should occupy.

READ THE MODULE DOCSTRING IN constants.py before trusting this engine's output:
the SKILL file itself states this is the most weakly evidenced module in the
system, and the class below computes a generic edge-strength statistic, not
validated lighting-consistency evidence.
"""

from __future__ import annotations

import logging

import numpy as np

from . import constants
from .contracts import GradientMagnitudeResult
from .utils import compute_gradient_magnitude, safe_ratio

logger = logging.getLogger(__name__)


class GradientMagnitudeComputer:
    """Computes the Sobel-gradient magnitude heuristic of SKILL Pipeline A."""

    def compute(self, grayscale: np.ndarray) -> GradientMagnitudeResult:
        """Run Pipeline A end to end on a prepared grayscale image.

        Args:
            grayscale: Float64 single-plane image, at least 2x2.

        Returns:
            GradientMagnitudeResult holding the magnitude map and the derived
            scale-invariant ratio.

        Raises:
            ValueError: If grayscale is not 2-D or is smaller than the minimum
                size numpy.gradient itself requires.
        """
        magnitude = self._gradient_magnitude(grayscale)
        # ENHANCEMENT 3: statistics come from the interior only; see _interior.
        measured = self._interior(magnitude)
        max_gradient = self._maximum(measured)
        median_gradient = self._median(measured)

        is_degenerate = (median_gradient
                         < constants.MINIMUM_MEDIAN_GRADIENT_FOR_RATIO)
        # ENHANCEMENT 1: a degenerate image publishes no score at all rather
        # than max_grad over the 1e-9 floor, measured at 9.7e10 on renders
        # whose median gradient is exactly zero.
        ratio = (constants.DEGENERATE_IMAGE_RAW_SCORE if is_degenerate
                 else safe_ratio(max_gradient, median_gradient,
                                 constants.MEDIAN_GRADIENT_FLOOR))

        return GradientMagnitudeResult(
            gradient_magnitude=magnitude,
            max_gradient=max_gradient,
            median_gradient=median_gradient,
            ratio=ratio,
            is_degenerate=is_degenerate,
        )

    @staticmethod
    def _interior(gradient_magnitude: np.ndarray) -> np.ndarray:
        """Drop the boundary rows and columns numpy.gradient computes one-sidedly.

        ENHANCEMENT 3. numpy.gradient uses a centred difference in the interior
        but a one-sided one on the first and last row and column, so boundary
        positions come from a different estimator and are not comparable with
        the rest. Measured, "fake .jpeg" put its maximum at row 0 with 172 tied
        pixels; excluding the border moved its ratio by -15.50%. The FULL map is
        still returned by compute(), so the evidence image is unchanged.

        Args:
            gradient_magnitude: Per-pixel gradient magnitude map.

        Returns:
            The interior of the map, or the map unchanged when it is too small
            to have an interior.
        """
        width = constants.GRADIENT_BORDER_EXCLUSION_WIDTH
        minimum = constants.MINIMUM_SIDE_FOR_BORDER_EXCLUSION
        if (gradient_magnitude.shape[0] < minimum
                or gradient_magnitude.shape[1] < minimum):
            return gradient_magnitude
        return gradient_magnitude[width:-width, width:-width]

    @staticmethod
    def _gradient_magnitude(grayscale: np.ndarray) -> np.ndarray:
        """Per-pixel gradient magnitude, transcribed from the source MATLAB.

        Args:
            grayscale: Float64 single-plane image.

        Returns:
            Float64 array of the same shape holding sqrt(Gx^2 + Gy^2).
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: [Gx, Gy] = gradient(double(gray_img));
        #          gradient_mag = sqrt(Gx.^2 + Gy.^2);
        # Variables: gray_img = the grayscale image; Gx, Gy = the numerical
        #            gradient along each image axis; gradient_mag = the
        #            per-pixel gradient magnitude.
        # Source: Rao, Ghanekar, Chitnis, Dawkhar & Mishra 2025 (CISCON) -
        #            SKILL Pipeline A step 2, "the paper's own MATLAB code,
        #            transcribed exactly".
        # Expected range: non-negative, in grey-level units per pixel.
        # Note: MATLAB's gradient(A) returns (FX, FY) = (horizontal,
        #            vertical); numpy.gradient(A) returns
        #            (d/d(axis0), d/d(axis1)) = (vertical, horizontal) - the
        #            opposite order. Verified in constants.py that this cannot
        #            affect the result: sqrt(a^2+b^2) is symmetric in its two
        #            arguments, so no axis swap is needed or performed.
        # ────────────────────────────────────────────────────
        return compute_gradient_magnitude(grayscale)

    @staticmethod
    def _maximum(gradient_magnitude: np.ndarray) -> float:
        """The scalar max_grad of the source's MATLAB code.

        Args:
            gradient_magnitude: Per-pixel gradient magnitude map.

        Returns:
            The maximum value in the map.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: max_grad = max(gradient_mag(:));
        # Variables: gradient_mag = the per-pixel gradient magnitude map;
        #            max_grad = its single largest value.
        # Source: Rao, Ghanekar, Chitnis, Dawkhar & Mishra 2025 - SKILL
        #            Pipeline A step 2.
        # Expected range: non-negative, "unbounded, image- and
        #            content-dependent - no normalization or calibration
        #            given" (SKILL "Output" -> Pipeline A).
        # ────────────────────────────────────────────────────
        return float(np.max(gradient_magnitude))

    @staticmethod
    def _median(gradient_magnitude: np.ndarray) -> float:
        """Median gradient magnitude, the normalising divisor.

        Args:
            gradient_magnitude: Per-pixel gradient magnitude map.

        Returns:
            The median value in the map.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: (no formula printed in the SKILL for this step)
        # Variables: the median of gradient_mag, used as the divisor of
        #            max_grad.
        # Source: SKILL "Output" -> Pipeline A: "if used, normalize per-image
        #            (e.g. divide by the image's own median gradient
        #            magnitude) rather than using an absolute cutoff, since
        #            the source gives no calibration at all." This is an
        #            engineering recommendation named in the SKILL, not a
        #            corpus-validated formula - there is no equation number
        #            for it because none is given.
        # Expected range: non-negative. Zero only for a perfectly flat image,
        #            handled explicitly as a degenerate case rather than
        #            divided through.
        # ────────────────────────────────────────────────────
        return float(np.median(gradient_magnitude))
