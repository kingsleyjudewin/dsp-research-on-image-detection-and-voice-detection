"""Pre- and post-computation reliability gates for the JPEG-artifact engine.

Several of this SKILL's unreliability conditions are *fundamental blind
spots*, not per-image measurables - the integer quantization-step-ratio case
("no periodic double-quantization artifact is introduced at all"), the
QF_second < QF_first regime, and the non-aligned/equal-QF cases the no-ML
path forfeits by excluding Pipelines C and D. None can be tested from the
image alone, because each depends on the *primary* compression history that
is precisely what is unknown.

They are therefore carried as standing caveats on every result. The most
important consequence, stated here once and repeated in every note this
module emits: a LOW score from this engine is NOT evidence of authenticity.
A forged image can produce exactly zero signal here by construction.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from . import constants
from .contracts import (ConditionReport, DoubleCompressionResult,
                        ImageMetadata, JpegHistoryResult, QualityFactorResult)
from .utils import compose_confidence_penalties

logger = logging.getLogger(__name__)

CheckResult = tuple  # (passed: bool, confidence_penalty: float, note: str)

STANDING_LIMITATIONS_NOTE = (
    "Standing SKILL caveats - a LOW score from this engine is NOT evidence "
    "of authenticity. Double-quantization periodicity is undetectable in "
    "principle when the secondary/primary quantization-step ratio is an "
    "integer (the SKILL calls this a fundamental, unavoidable blind spot of "
    "every DQ-histogram method, with detection accuracy near zero), and is "
    "substantially degraded when the image was re-saved at LOWER quality "
    "than its original. Content spliced from a never-compressed source and "
    "never itself re-encoded is undetectable by this module family "
    "entirely. COVERAGE GAP: excluding the SKILL's [ML] Pipelines C and D "
    "leaves this engine blind to non-aligned (block-grid-shifted) double "
    "JPEG compression and to the QF1=QF2 case, which the SKILL states are "
    "handled by no training-free technique in its corpus."
)


class ConditionChecker:
    """Decides whether the JPEG engine may run, and at what confidence."""

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

        block = constants.DCT_BLOCK_SIZE
        if array.shape[0] < block or array.shape[1] < block:
            return (f"Engine skipped: image is {array.shape[0]}x"
                    f"{array.shape[1]}, smaller than a single {block}x{block} "
                    f"DCT block.")
        return None

    @staticmethod
    def assess_block_count(unsaturated_blocks: int,
                           total_blocks: int) -> CheckResult:
        """Check that enough unsaturated blocks survive for stable statistics.

        SKILL: statistics are restricted to unsaturated blocks, and Pipeline
        B needs a populated per-frequency histogram across many blocks. No
        minimum count is given in the corpus for B.

        Args:
            unsaturated_blocks: Blocks kept after the saturation filter.
            total_blocks: Blocks before filtering.

        Returns:
            CheckResult reflecting whether the block count suffices.
        """
        if unsaturated_blocks >= constants.MINIMUM_BLOCKS_FOR_HISTOGRAM:
            return True, constants.FULL_CONFIDENCE, ""
        return (False, constants.ZERO_CONFIDENCE,
                f"Only {unsaturated_blocks} of {total_blocks} 8x8 blocks are "
                f"unsaturated, below the "
                f"{constants.MINIMUM_BLOCKS_FOR_HISTOGRAM} needed for a "
                f"populated per-frequency DCT histogram; Pipeline B's "
                f"periodicity statistics cannot be estimated.")

    @staticmethod
    def assess_jpeg_history(history: JpegHistoryResult) -> CheckResult:
        """Apply Pipeline A.1's gate: was this image ever JPEG-compressed?

        SKILL: "run A.1 first as a cheap global gate (is this image
        JPEG-derived at all?)". The whole module exploits 8x8 block-DCT
        quantization structure, so an image with no JPEG history offers
        nothing for Pipeline B to measure.

        Args:
            history: Pipeline A.1's result.

        Returns:
            CheckResult; failing marks the engine inapplicable to this input.
        """
        if history.is_jpeg_derived:
            return True, constants.FULL_CONFIDENCE, ""
        return (False, constants.ZERO_CONFIDENCE,
                f"JPEG-history feature s={history.history_feature:.4f} does "
                f"not exceed the t={history.threshold} decision threshold, so "
                f"this image shows no evidence of ever having been JPEG "
                f"compressed. Every technique in this module exploits 8x8 "
                f"block-DCT quantization structure, so no double-compression "
                f"statistic is meaningful here.")

    @staticmethod
    def assess_quality_factor(quality: QualityFactorResult) -> CheckResult:
        """Penalise the low-quality-factor regime the SKILL flags as unreliable.

        SKILL: "Very low quality factor (QF~50): high-frequency coefficients
        quantize to zero, destroying the statistics both Luo's and Mahdian's
        methods rely on" - Luo's own table shows quantization-step accuracy
        falling to 41-63% at QF=50 versus 91%+ at QF=95.

        Args:
            quality: Pipeline A.3's result.

        Returns:
            CheckResult carrying a confidence penalty in the low-QF regime.
        """
        if not quality.sweep_ran:
            return True, constants.FULL_CONFIDENCE, ""
        if quality.quality_factor > constants.LOW_QUALITY_FACTOR_FLOOR:
            return True, constants.FULL_CONFIDENCE, ""
        return (True, constants.ZERO_CONFIDENCE,
                f"Estimated quality factor is {quality.quality_factor}, at or "
                f"below the QF={constants.LOW_QUALITY_FACTOR_FLOOR} point the "
                f"SKILL flags as destroying these statistics: high-frequency "
                f"coefficients quantize to zero, and the source paper's own "
                f"quantization-step accuracy falls to 41-63% here.")

    @staticmethod
    def assess_usable_frequencies(result: DoubleCompressionResult) -> CheckResult:
        """Check that enough of the 10 Pipeline-B frequencies survived exclusion.

        SKILL Implementation Notes instruct excluding a frequency whose
        coefficients are overwhelmingly zero "rather than letting a
        degenerate near-empty histogram silently corrupt the estimate". If
        too few survive, the aggregate is not meaningful.

        Args:
            result: Pipeline B's result.

        Returns:
            CheckResult reflecting the surviving-frequency count.
        """
        total = len(constants.DOUBLE_QUANTIZATION_FREQUENCIES)
        minimum = int(np.ceil(total * constants.MINIMUM_USABLE_FREQUENCY_FRACTION))
        if result.usable_frequency_count >= minimum:
            return True, constants.FULL_CONFIDENCE, ""
        return (False, constants.ZERO_CONFIDENCE,
                f"Only {result.usable_frequency_count} of {total} selected "
                f"low-frequency DCT positions carry usable statistics (the "
                f"rest are almost entirely zero-quantized), below the "
                f"{minimum} required; the aggregate periodicity score would "
                f"rest on too few frequencies to be meaningful.")

    @staticmethod
    def assess_table_detection_margin(quality: QualityFactorResult) -> CheckResult:
        """Down-weight confidence when no candidate table matches cleanly.

        SKILL: a poor maximum pixel-match rate "suggests an inconsistent/
        mixed compression history, itself weak tampering evidence (this
        specific use is an engineering extrapolation, not validated in the
        corpus)". Because it is explicitly unvalidated it only moves
        confidence, never raw_score.

        Args:
            quality: Pipeline A.3's result.

        Returns:
            CheckResult; always passes, but scales confidence by the winning
            candidate's own exact-pixel-match rate.
        """
        if not quality.sweep_ran:
            return True, constants.FULL_CONFIDENCE, ""
        if quality.pixel_match_ratio >= constants.FULL_CONFIDENCE:
            return True, constants.FULL_CONFIDENCE, ""
        return (True, compose_confidence_penalties([quality.pixel_match_ratio]),
                f"Best candidate quantization table reproduces only "
                f"{quality.pixel_match_ratio:.1%} of pixels exactly "
                f"(runner-up {quality.runner_up_match_ratio:.1%}); a clean "
                f"single compression history normally reproduces near 100%. "
                f"Treated as a confidence discount only - the SKILL marks "
                f"this reading an unvalidated engineering extrapolation.")
