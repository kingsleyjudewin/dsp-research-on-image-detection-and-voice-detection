"""Pre- and post-computation reliability gates for the noise-pattern engine.

Two SKILL-documented failure modes are checked numerically: saturated pixels
(Eq. 19's attenuation vanishing near 255) and flat/textureless blocks (Eq.
19-20's texture dependence, generalised the same way this system's other
engines gate against degenerate content). One more - denoising/smoothing
attacks, "the single largest documented failure mode across this whole
module" - is not numerically checkable from ImageMetadata alone, so it is a
standing caveat, EXCEPT when Pipeline D's triage classifier is available
(calibration-gated), in which case its "Gaussian blur"/denoised label
directly informs a confidence penalty, per the SKILL's own instruction to
"down-weight the PRNU modules' confidence here".
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from . import constants
from .contracts import ConditionReport, ImageMetadata, NoiseTriageResult
from .utils import compose_confidence_penalties

logger = logging.getLogger(__name__)

CheckResult = tuple  # (passed: bool, confidence_penalty: float, note: str)

STANDING_LIMITATIONS_NOTE = (
    "Standing SKILL caveats: PRNU-based detection is fundamentally blind to "
    "content-preserving edits that don't disturb the sensor-noise layer "
    "(e.g. recoloring a stain); the single largest documented failure mode "
    "is a denoising/smoothing attack on the tampered region, which removes "
    "the very signal being measured and is not detectable from metadata "
    "alone unless Pipeline D's triage classifier was run."
)


class ConditionChecker:
    """Decides whether the noise engine may run, and at what confidence."""

    def check(self, metadata: ImageMetadata, image: Optional[np.ndarray],
             block_size: int) -> ConditionReport:
        """Run the pre-computation conditions against the input.

        Args:
            metadata: Container and resolution facts.
            image: BGR uint8 or grayscale array, or None.
            block_size: Resolution-scaled block side length in pixels.

        Returns:
            ConditionReport aggregating every check.
        """
        skip_reason = self._premise_failure_reason(image, block_size)
        if skip_reason:
            return ConditionReport(is_reliable=False,
                                   confidence_weight=constants.ZERO_CONFIDENCE,
                                   reliability_note=skip_reason, skip_engine=True)
        return ConditionReport(is_reliable=True,
                               confidence_weight=constants.FULL_CONFIDENCE,
                               reliability_note=STANDING_LIMITATIONS_NOTE,
                               skip_engine=False)

    def _premise_failure_reason(self, image: Optional[np.ndarray],
                                block_size: int) -> Optional[str]:
        """Explain why the engine cannot run at all, if it cannot.

        Args:
            image: BGR uint8 or grayscale array, or None.
            block_size: Resolution-scaled block side length in pixels.

        Returns:
            Explanation string, or None when the engine may proceed.
        """
        if image is None:
            return "Engine skipped: no image array was supplied."
        array = np.asarray(image)
        if array.ndim not in (constants.GRAYSCALE_IMAGE_DIMENSION_COUNT,
                              constants.COLOUR_IMAGE_DIMENSION_COUNT):
            return (f"Engine skipped: expected a 2-D grayscale or 3-D colour "
                    f"image, received an array of {array.ndim} dimensions.")
        if array.shape[0] < block_size or array.shape[1] < block_size:
            return (f"Engine skipped: image is {array.shape[0]}x"
                    f"{array.shape[1]}, smaller than the resolution-scaled "
                    f"{block_size}x{block_size} block size Pipeline A "
                    f"requires.")
        return None

    @staticmethod
    def assess_saturation(intensity_values: np.ndarray) -> CheckResult:
        """Check the fraction of near-saturated pixels in the image.

        Args:
            intensity_values: Grayscale intensity array.

        Returns:
            CheckResult reflecting the saturated-pixel fraction.
        """
        saturated_fraction = float(np.mean(
            intensity_values >= constants.SATURATION_INTENSITY_FLOOR))
        if saturated_fraction < constants.MAXIMUM_SATURATED_PIXEL_FRACTION:
            return True, constants.FULL_CONFIDENCE, ""
        penalty = compose_confidence_penalties([1.0 - saturated_fraction])
        return (False, penalty,
                f"{saturated_fraction:.1%} of pixels are at or above "
                f"{constants.SATURATION_INTENSITY_FLOOR}; the multiplicative "
                f"PRNU term vanishes near sensor saturation (Eq. 19).")

    @staticmethod
    def assess_flatness(block_variances: list) -> CheckResult:
        """Check the fraction of flat/textureless residual blocks.

        Args:
            block_variances: Residual-pixel variance per tiled block.

        Returns:
            CheckResult reflecting the flat-block fraction.
        """
        if not block_variances:
            return False, constants.ZERO_CONFIDENCE, "No blocks to assess."
        flat = sum(1 for value in block_variances
                  if value < constants.BLOCK_VARIANCE_DEGENERACY_FLOOR)
        fraction = flat / len(block_variances)
        if fraction < constants.MAXIMUM_FLAT_BLOCK_FRACTION:
            return True, constants.FULL_CONFIDENCE, ""
        return (False, constants.ZERO_CONFIDENCE,
                f"{flat}/{len(block_variances)} blocks have residual "
                f"variance below {constants.BLOCK_VARIANCE_DEGENERACY_FLOOR}; "
                f"local noise-level comparison is unreliable on flat content.")

    @staticmethod
    def assess_triage_denoising_penalty(triage: NoiseTriageResult) -> CheckResult:
        """Apply the SKILL-instructed confidence penalty for suspected denoising.

        SKILL, Pipeline D output: "use only as a categorical gate (e.g. 'this
        region was likely denoised - down-weight the PRNU modules'
        confidence here')". No numeric penalty is given; see
        constants.NOISE_TRIAGE_DENOISED_CONFIDENCE_PENALTY.

        Args:
            triage: Output of NoiseTriageClassifier.classify, or a
                not-run result when calibration data was unavailable.

        Returns:
            CheckResult; always passes (this never blocks the engine), but
            carries a confidence penalty when denoising is suspected.
        """
        if not triage.ran or triage.label != "Gaussian blur":
            return True, constants.FULL_CONFIDENCE, ""
        return (True, constants.NOISE_TRIAGE_DENOISED_CONFIDENCE_PENALTY,
                f"Pipeline D triage suspects denoising/blur (z_score="
                f"{triage.z_score:.4f}); PRNU-based confidence down-weighted "
                f"per the SKILL's own instruction.")
