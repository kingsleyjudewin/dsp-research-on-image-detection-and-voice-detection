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
    def assess_sweep_applicability(excluded_count: int,
                                   scored_count: int,
                                   residual_lattice_count: int,
                                   native_quality_factor) -> CheckResult:
        """Report how much of the corpus sweep applied to this image at all.

        ENHANCEMENT 1, forced by diagnostic testing. The SKILL's sweep
        {80, 85, 90, 95, 100} comes from Bonettini et al., whose corpus is
        UNCOMPRESSED, so every one of those factors is that image's FIRST
        quantization. On an image already saved at a lower quality factor, all
        of them re-quantize more finely than its own encoder did, which cannot
        add information and measurably distorts the digit histogram. Those cells
        are excluded; this reports how thin what remains is.

        Args:
            excluded_count: Cells excluded as finer than the image's own.
            scored_count: Cells that survived and were scored.
            residual_lattice_count: Scored cells still carrying a lattice.
            native_quality_factor: The image's own quality factor, or None.

        Returns:
            CheckResult; fails only when nothing survived.
        """
        total = excluded_count + scored_count
        if scored_count == 0:
            return (False, constants.ZERO_CONFIDENCE,
                    f"All {total} sweep configurations quantized this image "
                    f"more finely than its own encoder did (estimated quality "
                    f"factor {native_quality_factor}), so none of them can "
                    f"report anything but the engine's own arithmetic. No "
                    f"measurement is returned.")

        return (True, constants.FULL_CONFIDENCE,
                ConditionChecker._describe_sweep_coverage(
                    excluded_count, scored_count, residual_lattice_count,
                    native_quality_factor))

    @staticmethod
    def _describe_sweep_coverage(excluded_count: int,
                                 scored_count: int,
                                 residual_lattice_count: int,
                                 native_quality_factor) -> str:
        """Word the sweep-coverage and residual-lattice findings for the report.

        Args:
            excluded_count: Cells excluded as finer than the image's own.
            scored_count: Cells that survived and were scored.
            residual_lattice_count: Scored cells still carrying a lattice.
            native_quality_factor: The image's own quality factor, or None.

        Returns:
            Explanatory text, empty when there is nothing to report.
        """
        total = excluded_count + scored_count
        notes = []
        if excluded_count:
            notes.append(
                f"{excluded_count} of {total} sweep configurations were "
                f"excluded because the standard table applied there is finer "
                f"than this image's own quantization (estimated quality factor "
                f"{native_quality_factor}); re-quantizing more finely than the "
                f"original encoder cannot recover discarded information and "
                f"distorts the leading-digit histogram. The score rests on the "
                f"{scored_count} configurations at or below the image's own "
                f"quality factor.")
        if residual_lattice_count:
            notes.append(
                f"{residual_lattice_count} scored configuration(s) still show a "
                f"lattice in the coefficients. At or below the image's own "
                f"quality factor this engine cannot have created it, so it is "
                f"evidence of an EARLIER compression - the double-compression "
                f"signature Singh et al. describe.")
        return " ".join(notes)

    @staticmethod
    def assess_published_regime(representative_chi_square: float) -> CheckResult:
        """Check the image sits inside the regime the corpus actually measured.

        ENHANCEMENT 4, forced by diagnostic testing, and deliberately a
        CONFIDENCE PENALTY rather than a reliability gate.

        First built as a hard gate on the published range, then WITHDRAWN: on
        an unmanipulated single-compressed image the statistic read 0.10212 at
        QF90 and 1.24708 at QF50 against a published 0.0112-0.0126 at both. The
        gap varies with quality factor, so no borrowed threshold is defensible.
        Only the order of magnitude stays honest, so it earns a confidence
        penalty rather than a refusal to vote. See constants.py.

        Args:
            representative_chi_square: Chi-square measured nearest the image's
                own quality factor.

        Returns:
            CheckResult; always passes, carrying a penalty when far outside.
        """
        if representative_chi_square <= constants.CHI_SQUARE_VALIDITY_CEILING:
            return True, constants.FULL_CONFIDENCE, ""

        multiple = (representative_chi_square /
                    constants.CHI_SQUARE_VALIDITY_CEILING)
        return (True,
                constants.CONFIDENCE_PENALTY_OUTSIDE_PUBLISHED_REGIME,
                f"Chi-square against the classical Benford curve measured "
                f"{representative_chi_square:.4f}, which is {multiple:.1f}x the "
                f"largest value any paper in this corpus reports for ANY image "
                f"class (altered images reach "
                f"{constants.MOIN_ALTERED_CHI_SQUARE_MAXIMUM:.4f}; unaltered "
                f"images sit at "
                f"{constants.MOIN_UNALTERED_CHI_SQUARE_MINIMUM:.4f}-"
                f"{constants.MOIN_UNALTERED_CHI_SQUARE_MAXIMUM:.4f}). This "
                f"image's coefficient population is unlike the material the "
                f"corpus characterises, so confidence is reduced. Note this "
                f"engine does not reproduce the published chi-square scale even "
                f"on unmanipulated images, so the gap is NOT itself evidence "
                f"of tampering.")

    @staticmethod
    def _normalise_format(metadata: ImageMetadata) -> str:
        """Upper-case the container format for case-insensitive comparison.

        Args:
            metadata: Container facts.

        Returns:
            Upper-case format string, empty when unset.
        """
        return str(metadata.format or "").strip().upper()
