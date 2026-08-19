"""Pre- and post-computation reliability gates for the ghost engine.

This module's dominant caveats are not per-image measurables but structural
properties of the techniques themselves, and two of them are unusually
strong:

  * Pipeline B is DIRECTIONAL. It "works only when the spliced region's
    original quality q0 is lower than the surrounding re-save quality q1
    (q1 > q0); does not resolve the reverse case." q0 is exactly what is
    unknown, so this cannot be tested per image.
  * Pipeline A is ACTIVELY, QUANTITATIVELY DEFEATED by three counter-forensic
    techniques published in its own source paper. The subtlest finding there
    deserves repeating in full, because it is worse than plain non-detection:
    of the images that defeated detection via geometric distortion, "not a
    single 'detection success' also found the correct synthetic
    transformation map" - meaning an alarm raised on an attacked image is
    firing on a spurious match, not a genuine one.

Both are carried as standing caveats, and the consequence is stated once
here and repeated in every note: a LOW score from this engine is NOT
evidence of authenticity.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from . import constants
from .contracts import ConditionReport, GhostResult, ImageMetadata
from .utils import compose_confidence_penalties

logger = logging.getLogger(__name__)

CheckResult = tuple  # (passed: bool, confidence_penalty: float, note: str)

STANDING_LIMITATIONS_NOTE = (
    "Standing SKILL caveats - a LOW score from this engine is NOT evidence "
    "of authenticity. The ghost detector is fundamentally directional: it "
    "resolves a splice only when the pasted region's original quality was "
    "LOWER than the quality the composite was re-saved at, and it cannot "
    "resolve the reverse case at all. Its sensitivity depends strongly on "
    f"the quality gap, exceeding 95% only once that gap exceeds "
    f"{constants.RELIABLE_QUALITY_FACTOR_GAP}, and fading to nothing as the "
    "gap approaches zero. Neither quantity is computable from the image "
    "alone, since the pasted region's original quality is precisely what is "
    "unknown. The resampling detector is additionally defeated outright by "
    "three counter-forensic techniques published in its own source paper "
    "(5x5 median filtering, edge-modulated geometric distortion, and their "
    "dual-path combination), and by ordinary sinc/spline interpolation with "
    "no attack at all."
)


NULL_GHOST_RESULT_NOTE = (
    "No ghost candidate survived validation across any of the {total} swept "
    "combinations - no region behaved like a separately-quantized paste. Read "
    "this as an absence of detectable ghost evidence, NOT as evidence of "
    "authenticity: the method is blind by construction to a paste whose "
    "original quality exceeded the re-save quality, and faint when the two "
    "are close. Ground truth confirms the ambiguity is structural rather than "
    "incidental - an authentic single-compression image and a genuine splice "
    "of higher-quality content into a lower-quality host both return exactly "
    f"this null result, so its confidence is weighted at "
    f"{constants.NULL_GHOST_RESULT_CONFIDENCE}."
)


class ConditionChecker:
    """Decides whether the ghost engine may run, and at what confidence."""

    def check(self, metadata: ImageMetadata,
              image: Optional[np.ndarray]) -> ConditionReport:
        """Run the pre-computation conditions against the input.

        Args:
            metadata: Container and resolution facts.
            image: BGR uint8 or grayscale array, or None.

        Returns:
            ConditionReport aggregating every pre-computation check.
        """
        skip_reason = self._premise_failure_reason(image)
        if skip_reason:
            return ConditionReport(is_reliable=False,
                                   confidence_weight=constants.ZERO_CONFIDENCE,
                                   reliability_note=skip_reason, skip_engine=True)
        return ConditionReport(is_reliable=True,
                               confidence_weight=constants.FULL_CONFIDENCE,
                               reliability_note=STANDING_LIMITATIONS_NOTE,
                               skip_engine=False)

    @staticmethod
    def _premise_failure_reason(image: Optional[np.ndarray]) -> Optional[str]:
        """Explain why the engine cannot run at all, if it cannot.

        Args:
            image: BGR uint8 or grayscale array, or None.

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

        minimum = constants.MINIMUM_IMAGE_DIMENSION
        if array.shape[0] < minimum or array.shape[1] < minimum:
            return (f"Engine skipped: image is {array.shape[0]}x"
                    f"{array.shape[1]}, smaller than the {minimum}x{minimum} "
                    f"smoothing window the ghost difference-energy statistic "
                    f"is defined over.")
        return None

    @staticmethod
    def assess_jpeg_history(metadata: ImageMetadata) -> CheckResult:
        """Check that the ghost detector has a JPEG history to exploit.

        SKILL: "JPEG ghost detection requires the dubious image to already be
        JPEG-derived (has 8x8 block-DCT structure to exploit)" and "Ghost
        detector is not meaningful on images with no JPEG compression history
        at all."

        Args:
            metadata: Container facts, whose format field carries the
                container the orchestrator decoded.

        Returns:
            CheckResult; failing marks the engine inapplicable to this input.
        """
        container = (metadata.format or "").strip().upper()
        if container.startswith("JPG") or container.startswith("JPEG"):
            return True, constants.FULL_CONFIDENCE, ""
        return (False, constants.ZERO_CONFIDENCE,
                f"Container format is reported as '{metadata.format}', not "
                f"JPEG. The ghost detector exploits 8x8 block-DCT "
                f"quantization structure and the SKILL states it is 'not "
                f"meaningful on images with no JPEG compression history at "
                f"all', so no ghost statistic is interpretable here.")

    @staticmethod
    def assess_resampling_applicability(metadata: ImageMetadata) -> CheckResult:
        """Judge whether Pipeline A's output is worth reporting at all.

        SKILL: the resampling detector "fails outright after even moderate
        JPEG compression - periodic blocking artifacts interfere with the
        periodic resampling residual", stated both as an assumption and as an
        empirical finding ("virtually all resampling detectors fail after
        moderate JPEG compression"). No numeric boundary for "moderate" is
        given, so this only annotates the auxiliary result; it never gates
        the engine, because Pipeline A does not score.

        Args:
            metadata: Container and compression facts.

        Returns:
            CheckResult; always passes, carrying an explanatory note.
        """
        container = (metadata.format or "").strip().upper()
        is_jpeg = container.startswith("JPG") or container.startswith("JPEG")
        light = (metadata.estimated_compression_level
                 >= constants.RESAMPLING_MAXIMUM_COMPRESSION_LEVEL)
        if not is_jpeg or light:
            return True, constants.FULL_CONFIDENCE, ""
        return (True, constants.FULL_CONFIDENCE,
                f"Resampling periodicity (auxiliary) was computed on a JPEG "
                f"input at estimated quality "
                f"{metadata.estimated_compression_level:.0f}; the SKILL "
                f"states this detector 'fails outright after even moderate "
                f"JPEG compression', so its rho should be read as "
                f"uninformative here rather than as an absence of "
                f"resampling.")

    @staticmethod
    def assess_segmentation_health(result: GhostResult) -> CheckResult:
        """Check that the ghost sweep actually had combinations to evaluate.

        Combinations rejected by the ghost-candidate validity test are NOT a
        failure - rejecting them is the whole point, and on a clean image
        every combination is expected to be rejected. That outcome is a
        genuine negative result (no ghost found) rather than a broken
        computation, so it must not be reported as unreliability; treating it
        as such would make the engine abstain on exactly the images it
        handles best. Only an empty sweep is a real failure.

        Args:
            result: Pipeline B's result.

        Returns:
            CheckResult; fails only when nothing was evaluated at all.
        """
        total = result.combinations_evaluated
        if total == 0:
            return False, constants.ZERO_CONFIDENCE, (
                "No (q2, dx, dy) combination was evaluated, so no ghost "
                "statistic exists.")

        if result.degenerate_segmentation_count < total:
            return True, constants.FULL_CONFIDENCE, ""
        # ENHANCEMENT 3: a null result still votes, but not at full weight -
        # an authentic image and a reverse-direction splice produce the same
        # null. See constants.NULL_GHOST_RESULT_CONFIDENCE.
        return (True, constants.NULL_GHOST_RESULT_CONFIDENCE,
                NULL_GHOST_RESULT_NOTE.format(total=total))

    @staticmethod
    def assess_sweep_coverage(result: GhostResult) -> CheckResult:
        """Down-weight confidence when the sweep was pruned below the full search.

        SKILL Implementation Notes explicitly sanction pruning the
        6400-combination sweep, but a pruned sweep can still miss the true
        (q2, dx, dy) triple, so the confidence reflects how much of the
        paper's search space was actually covered.

        Args:
            result: Pipeline B's result.

        Returns:
            CheckResult; always passes, carrying a coverage-proportional
            confidence weight.
        """
        coverage = (result.combinations_evaluated
                    / constants.FULL_SEARCH_COMBINATION_COUNT)
        if coverage >= constants.FULL_CONFIDENCE:
            return True, constants.FULL_CONFIDENCE, ""
        return (True, compose_confidence_penalties([coverage ** 0.5]),
                f"Swept {result.combinations_evaluated} of the paper's "
                f"{constants.FULL_SEARCH_COMBINATION_COUNT} "
                f"(q2, dx, dy) combinations "
                f"({result.quality_factors_swept} quality factors x "
                f"{result.grid_shifts_swept} grid shifts). Pruning is "
                f"sanctioned by the SKILL for cost, but a pruned sweep can "
                f"miss the true combination, so confidence is scaled by "
                f"coverage.")
