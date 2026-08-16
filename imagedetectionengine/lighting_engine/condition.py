"""Pre-computation input gate for the lighting / illumination engine.

Unlike the other engines in this system, the SKILL file gives NO quantified
reliable/unreliable operating envelope for this module: "cannot be meaningfully
characterized for this module from the current corpus - neither implementable
technique here ... has a stated, quantified operating envelope specific to
lighting consistency". There is therefore no JPEG-quality gate, no resampling
gate, no resolution-adequacy gate to check here - inventing one would use a
value not present in the SKILL file.

What IS checked is purely structural: can numpy.gradient run on this input at
all, and is the input degenerate. Everything else this module has to say about
its own trustworthiness is the standing, unconditional confidence ceiling
applied in scorer.py and engine.py, not a per-image condition.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from . import constants
from .contracts import ConditionReport, GradientMagnitudeResult, ImageMetadata
from .utils import compose_confidence_penalties

logger = logging.getLogger(__name__)

# A check reports (passed, confidence_penalty, note). passed=False marks the
# result unreliable; penalty is a multiplier folded into the final weight.
CheckResult = tuple[bool, float, str]


class ConditionChecker:
    """Decides whether the lighting engine may run, and at what confidence."""

    def check(self,
              metadata: ImageMetadata,
              image: Optional[np.ndarray] = None) -> ConditionReport:
        """Run the structural pre-computation conditions against the input.

        Args:
            metadata: Container and resolution facts. Compression level is
                deliberately not consulted; see the module docstring.
            image: BGR uint8 array. When None, the engine cannot run at all.

        Returns:
            ConditionReport aggregating every check.
        """
        skip_reason = self._premise_failure_reason(image)
        if skip_reason:
            return ConditionReport(
                is_reliable=False,
                confidence_weight=constants.ZERO_CONFIDENCE,
                reliability_note=skip_reason,
                skip_engine=True,
            )

        results = self._run_all_checks(metadata)
        notes = [note for _, _, note in results if note]
        penalties = [penalty for _, penalty, _ in results]
        is_reliable = all(passed for passed, _, _ in results)

        return ConditionReport(
            is_reliable=is_reliable,
            confidence_weight=compose_confidence_penalties(penalties),
            reliability_note=" ".join(notes) if notes
            else "Input is structurally valid.",
            skip_engine=False,
        )

    @staticmethod
    def _premise_failure_reason(image: Optional[np.ndarray]) -> Optional[str]:
        """Explain why the engine cannot run at all, if it cannot.

        Args:
            image: BGR uint8 array, or None.

        Returns:
            Explanation string, or None when the engine may proceed.
        """
        if image is None:
            return ("Engine skipped: no image array was supplied, so no "
                    "gradient could be computed.")

        array = np.asarray(image)
        if array.ndim not in (constants.GRAYSCALE_IMAGE_DIMENSION_COUNT,
                              constants.COLOUR_IMAGE_DIMENSION_COUNT):
            return (f"Engine skipped: expected a 2-D grayscale or 3-D colour "
                    f"image, received an array of {array.ndim} dimensions.")

        minimum = constants.MINIMUM_GRADIENT_AXIS_LENGTH
        if array.shape[0] < minimum or array.shape[1] < minimum:
            return (f"Engine skipped: image is {array.shape[0]}x"
                    f"{array.shape[1]}, below the {minimum}x{minimum} minimum "
                    f"numpy.gradient itself requires on each axis.")
        return None

    def _run_all_checks(self, metadata: ImageMetadata) -> list[CheckResult]:
        """Evaluate every documented pre-computation condition.

        Args:
            metadata: Container and resolution facts.

        Returns:
            One CheckResult per condition evaluated. Currently empty: the
            SKILL documents no quantified condition to check before the
            gradient itself has been computed. Kept as a method, rather than
            inlined, so a future SKILL revision with an actual operating
            envelope has an obvious place to add checks.
        """
        return []

    @staticmethod
    def assess_gradient_field(result: GradientMagnitudeResult) -> CheckResult:
        """Judge whether the computed gradient field was degenerate.

        SKILL gives no formal degeneracy criterion; this is the structural
        floor below which the ratio itself becomes a division of noise by
        near-zero noise rather than a meaningful statistic.

        Args:
            result: Output of the mathematical core.

        Returns:
            CheckResult reflecting whether the image had enough texture.
        """
        if not result.is_degenerate:
            return True, constants.FULL_CONFIDENCE, ""

        return (False, constants.ZERO_CONFIDENCE,
                f"Median gradient magnitude is {result.median_gradient:.4f}, "
                f"below the "
                f"{constants.MINIMUM_MEDIAN_GRADIENT_FOR_RATIO} floor treated "
                f"as usable texture; the image is flat or near-flat, so "
                f"max_grad / median is not a meaningful ratio here.")
