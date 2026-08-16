"""Pre-computation input gate for the Benford engine.

Every condition the SKILL file documents under "Unreliable / inapplicable when"
and "Reliable when" is checked here, before any transform runs. When the
engine's premise fails outright the report sets skip_engine, and the engine
returns a null vote without touching the pixels.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from . import constants
from .contracts import ConditionReport, ImageMetadata
from .utils import (compose_confidence_penalties,
                    compute_saturated_pixel_fraction)

logger = logging.getLogger(__name__)

# A check reports (passed, confidence_penalty, note). passed=False marks the
# result unreliable; penalty is a multiplier folded into the final weight.
CheckResult = tuple[bool, float, str]


class ConditionChecker:
    """Decides whether the Benford engine may run, and at what confidence."""

    def check(self,
              metadata: ImageMetadata,
              image: Optional[np.ndarray] = None) -> ConditionReport:
        """Run every documented unreliability condition against the input.

        Args:
            metadata: Container, resolution and compression facts.
            image: BGR uint8 array. When None, pixel-level checks are skipped
                and only metadata conditions are evaluated.

        Returns:
            ConditionReport aggregating every check.
        """
        results = self._run_all_checks(metadata, image)

        notes = [note for _, _, note in results if note]
        penalties = [penalty for _, penalty, _ in results]
        is_reliable = all(passed for passed, _, _ in results)

        # The premise fails only when there is no block-transform history at
        # all; every other condition degrades confidence but still yields a
        # usable measurement.
        skip_engine = self._premise_fails(metadata)
        if skip_engine:
            notes.append(
                "Engine skipped: the SKILL file requires the image to have "
                "passed through JPEG block-transform quantization at least "
                "once, which is what imposes the Benford-fitting structure "
                "this engine measures."
            )

        return ConditionReport(
            is_reliable=is_reliable and not skip_engine,
            confidence_weight=(constants.ZERO_CONFIDENCE if skip_engine
                               else compose_confidence_penalties(penalties)),
            reliability_note=" ".join(notes) if notes
            else "All documented reliability conditions satisfied.",
            skip_engine=skip_engine,
        )

    def _run_all_checks(self,
                        metadata: ImageMetadata,
                        image: Optional[np.ndarray]) -> list[CheckResult]:
        """Evaluate every individual condition and collect the verdicts.

        Args:
            metadata: Container, resolution and compression facts.
            image: BGR uint8 array, or None to skip pixel-level checks.

        Returns:
            One CheckResult per condition evaluated.
        """
        results: list[CheckResult] = [
            self._check_compression_history(metadata),
            self._check_quality_factor(metadata),
            self._check_wavelet_codec(metadata),
            self._check_resolution(metadata),
            self._check_resampling(metadata),
        ]
        if image is not None:
            results.append(self._check_saturation(image))
        return results

    def _premise_fails(self, metadata: ImageMetadata) -> bool:
        """Report whether the image lacks any block-transform history.

        Args:
            metadata: Container and compression facts.

        Returns:
            True when the engine cannot meaningfully run at all.
        """
        container_is_jpeg = self._normalise_format(metadata) in \
            constants.JPEG_FORMAT_NAMES
        # A re-saved JPEG may arrive in a lossless container, so a positive
        # quality-factor estimate rescues a non-JPEG container.
        compression_evidence = (metadata.estimated_compression_level >
                                constants.MINIMUM_JPEG_QUALITY_FACTOR)
        return not container_is_jpeg and not compression_evidence

    def _check_compression_history(self, metadata: ImageMetadata) -> CheckResult:
        """Verify the image carries a JPEG block-transform history.

        SKILL "Unreliable / inapplicable when": "Never-JPEG-compressed (pure
        RAW/PNG pipeline) images - no paper validates Benford features on
        uncompressed-domain coefficients directly".

        Args:
            metadata: Container and compression facts.

        Returns:
            CheckResult for this condition.
        """
        if self._normalise_format(metadata) in constants.JPEG_FORMAT_NAMES:
            return True, constants.FULL_CONFIDENCE, ""

        if metadata.estimated_compression_level > constants.MINIMUM_JPEG_QUALITY_FACTOR:
            return (False,
                    constants.CONFIDENCE_PENALTY_UNKNOWN_COMPRESSION_HISTORY,
                    f"Container format is {metadata.format}, not JPEG; relying "
                    f"on an estimated quality factor of "
                    f"{metadata.estimated_compression_level:.0f} as indirect "
                    f"evidence of prior block-transform quantization. No paper "
                    f"in the corpus validates Benford features on "
                    f"uncompressed-domain coefficients.")

        return (False,
                constants.ZERO_CONFIDENCE,
                f"Container format is {metadata.format} with no evidence of "
                f"prior JPEG compression.")

    def _check_quality_factor(self, metadata: ImageMetadata) -> CheckResult:
        """Check the estimated quality factor against the documented floors.

        SKILL "Unreliable / inapplicable when": "Very low quality factor
        (QF~50 or below) where high-frequency coefficients quantize to zero,
        removing their leading digit entirely." SKILL "Reliable when": "quality
        factor >= ~80 for the strongest chi-square separation".

        Args:
            metadata: Compression facts.

        Returns:
            CheckResult for this condition.
        """
        quality_factor = metadata.estimated_compression_level

        if quality_factor <= constants.MINIMUM_RELIABLE_QUALITY_FACTOR:
            return (False,
                    constants.ZERO_CONFIDENCE,
                    f"Estimated quality factor {quality_factor:.0f} is at or "
                    f"below {constants.MINIMUM_RELIABLE_QUALITY_FACTOR:.0f}, "
                    f"where high-frequency coefficients quantize to zero and "
                    f"lose their leading digit entirely.")

        if quality_factor < constants.STRONG_SEPARATION_QUALITY_FACTOR:
            return (True,
                    constants.CONFIDENCE_PENALTY_WEAK_QUALITY_FACTOR,
                    f"Estimated quality factor {quality_factor:.0f} is below "
                    f"the {constants.STRONG_SEPARATION_QUALITY_FACTOR:.0f} "
                    f"needed for strongest separation; detection still works "
                    f"here but with reduced deviation.")

        return True, constants.FULL_CONFIDENCE, ""

    def _check_wavelet_codec(self, metadata: ImageMetadata) -> CheckResult:
        """Flag wavelet-codec input, where the DCT-domain method is off-domain.

        SKILL "Unreliable / inapplicable when": for JPEG2000 double-compression
        Singh 2015 Table II reports deviation ratios of 0 to 2.2204e-16, i.e.
        "the DWT-domain double-compression detector essentially does not work".
        This engine implements the DCT path only.

        Args:
            metadata: Container facts.

        Returns:
            CheckResult for this condition.
        """
        if self._normalise_format(metadata) not in constants.WAVELET_CODEC_FORMAT_NAMES:
            return True, constants.FULL_CONFIDENCE, ""

        return (False,
                constants.CONFIDENCE_PENALTY_WAVELET_CODEC,
                f"Container format {metadata.format} is a wavelet codec. This "
                f"engine implements the DCT-domain pipeline; the corpus "
                f"records the DWT-domain equivalent as a negative result "
                f"(deviation ratio ~0), so the measurement is off-domain.")

    def _check_resolution(self, metadata: ImageMetadata) -> CheckResult:
        """Verify the image yields at least as many blocks as the validated set.

        SKILL "Bit depth / resolution": "Bonettini et al.'s corpus is 256x256",
        which at an 8x8 grid is 1024 blocks per frequency.

        Args:
            metadata: Resolution facts.

        Returns:
            CheckResult for this condition.
        """
        try:
            height, width = int(metadata.resolution[0]), int(metadata.resolution[1])
        except (TypeError, ValueError, IndexError):
            return (False,
                    constants.CONFIDENCE_PENALTY_BELOW_VALIDATED_RESOLUTION,
                    f"Resolution metadata {metadata.resolution!r} is "
                    f"unreadable; block-count adequacy could not be verified.")

        available_blocks = ((height // constants.DCT_BLOCK_SIZE) *
                            (width // constants.DCT_BLOCK_SIZE))
        if available_blocks >= constants.MINIMUM_ANALYSIS_BLOCK_COUNT:
            return True, constants.FULL_CONFIDENCE, ""

        return (False,
                constants.CONFIDENCE_PENALTY_BELOW_VALIDATED_RESOLUTION,
                f"Image yields {available_blocks} blocks of "
                f"{constants.DCT_BLOCK_SIZE}x{constants.DCT_BLOCK_SIZE}, below "
                f"the {constants.MINIMUM_ANALYSIS_BLOCK_COUNT} available in the "
                f"smallest validated corpus image "
                f"({constants.MINIMUM_VALIDATED_IMAGE_DIMENSION} square); the "
                f"digit histogram will be noisier than any published result.")

    def _check_resampling(self, metadata: ImageMetadata) -> CheckResult:
        """Warn when prior resizing may have masked a genuine deviation.

        SKILL "Documented failure cases": Wang et al. 2009 show a compensation
        operation (histogram equalization, rescaling) can restore Benford
        conformance without undoing the tampering, raising false-negative risk.

        Args:
            metadata: Resampling facts.

        Returns:
            CheckResult for this condition.
        """
        if not metadata.is_resized:
            return True, constants.FULL_CONFIDENCE, ""

        return (True,
                constants.CONFIDENCE_PENALTY_RESAMPLED_INPUT,
                "Image is reported as resized. Rescaling is a documented "
                "compensation operation that can restore Benford conformance "
                "without undoing tampering, so a low score here carries "
                "elevated false-negative risk.")

    def _check_saturation(self, image: np.ndarray) -> CheckResult:
        """Reject near-saturated or degenerate (constant) input.

        SKILL "Reliable when" requires the image be "not near-saturated/
        degenerate".

        Args:
            image: BGR uint8 array.

        Returns:
            CheckResult for this condition.
        """
        try:
            channel = np.asarray(image)[..., constants.ANALYSIS_CHANNEL_INDEX]
        except (IndexError, TypeError) as error:
            logger.warning("saturation check could not index the analysis "
                           "channel: %s", error)
            return (False, constants.ZERO_CONFIDENCE,
                    "Analysis channel could not be read for saturation check.")

        if float(np.var(channel.astype(np.float64))) < constants.MINIMUM_CHANNEL_VARIANCE:
            return (False, constants.ZERO_CONFIDENCE,
                    "Analysis channel is constant (degenerate); its transform "
                    "carries no digit statistic.")

        saturated_fraction = compute_saturated_pixel_fraction(
            channel,
            constants.MINIMUM_INTENSITY_LEVEL,
            constants.SATURATION_INTENSITY_LEVEL,
        )
        if saturated_fraction > constants.MAX_ACCEPTABLE_SATURATION_FRACTION:
            return (False, constants.ZERO_CONFIDENCE,
                    f"{saturated_fraction:.1%} of pixels are pinned at an "
                    f"intensity extreme, exceeding the "
                    f"{constants.MAX_ACCEPTABLE_SATURATION_FRACTION:.0%} limit; "
                    f"the image is near-saturated.")

        return True, constants.FULL_CONFIDENCE, ""

    def assess_sample_quality(self,
                              zero_coefficient_rate: float,
                              evaluated_configuration_count: int) -> CheckResult:
        """Judge the surviving sample after the transform has run.

        SKILL "Implementation notes": "track what fraction of coefficients were
        excluded as a data-quality signal, since a very high zero-rate (heavy
        quantization) will make the pmf estimate noisy/unreliable regardless."

        Args:
            zero_coefficient_rate: Mean fraction of coefficients discarded.
            evaluated_configuration_count: Grid cells that produced a score.

        Returns:
            CheckResult reflecting post-computation data quality.
        """
        if evaluated_configuration_count == 0:
            return (False, constants.ZERO_CONFIDENCE,
                    "No sweep configuration retained enough non-zero "
                    "coefficients to estimate a digit histogram.")

        if zero_coefficient_rate > constants.MAX_ACCEPTABLE_ZERO_COEFFICIENT_RATE:
            return (False, constants.ZERO_CONFIDENCE,
                    f"{zero_coefficient_rate:.1%} of quantized coefficients "
                    f"were zero and had to be excluded, exceeding the "
                    f"{constants.MAX_ACCEPTABLE_ZERO_COEFFICIENT_RATE:.0%} "
                    f"limit; the surviving digit histogram is not a meaningful "
                    f"population.")

        return True, constants.FULL_CONFIDENCE, ""

    @staticmethod
    def _normalise_format(metadata: ImageMetadata) -> str:
        """Upper-case the container format for case-insensitive comparison.

        Args:
            metadata: Container facts.

        Returns:
            Upper-case format string, empty when unset.
        """
        return str(metadata.format or "").strip().upper()
