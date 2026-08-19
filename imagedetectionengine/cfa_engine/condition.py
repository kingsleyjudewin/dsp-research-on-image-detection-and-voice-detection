"""Pre-computation input gate for the CFA / demosaicing engine.

Every condition the SKILL file documents under "Reliable when" and "Unreliable /
inapplicable when" is checked here, before any transform runs. When the engine's
premise fails outright the report sets skip_engine, and the engine returns a null
vote without touching the pixels.

Two conditions can only be judged after the mathematical core has run - whether
enough textured blocks survived, and whether any global CFA signature exists at
all (Eq. 13's requirement that mu1 > 0). Those are exposed as separate methods
that the engine calls at Stage 4.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from . import constants
from .contracts import ConditionReport, GaussianMixtureFit, ImageMetadata
from .utils import compose_confidence_penalties, compute_saturated_pixel_fraction

logger = logging.getLogger(__name__)

# A check reports (passed, confidence_penalty, note). passed=False marks the
# result unreliable; penalty is a multiplier folded into the final weight.
CheckResult = tuple[bool, float, str]


class ConditionChecker:
    """Decides whether the CFA engine may run, and at what confidence."""

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

        skip_reason = self._premise_failure_reason(metadata, image)
        if skip_reason:
            notes.append(skip_reason)

        return ConditionReport(
            is_reliable=is_reliable and skip_reason is None,
            confidence_weight=(constants.ZERO_CONFIDENCE if skip_reason
                               else compose_confidence_penalties(penalties)),
            reliability_note=" ".join(notes) if notes
            else "All documented reliability conditions satisfied.",
            skip_engine=skip_reason is not None,
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
            self._check_compression_quality(metadata),
            self._check_resampling(metadata),
            self._check_resolution(metadata),
        ]
        if image is not None:
            results.append(self._check_colour_input(image))
            results.append(self._check_saturation(image))
        return results

    def _premise_failure_reason(self,
                                metadata: ImageMetadata,
                                image: Optional[np.ndarray]) -> Optional[str]:
        """Explain why the engine cannot meaningfully run, if it cannot.

        Args:
            metadata: Container and compression facts.
            image: BGR uint8 array, or None.

        Returns:
            Explanation string, or None when the engine may proceed.
        """
        quality = metadata.estimated_compression_level
        if (quality > constants.NO_COMPRESSION_QUALITY_FACTOR
                and quality <= constants.UNUSABLE_JPEG_QUALITY_FACTOR):
            return (f"Engine skipped: estimated JPEG quality factor "
                    f"{quality:.0f} is at or below "
                    f"{constants.UNUSABLE_JPEG_QUALITY_FACTOR:.0f}, where "
                    f"Ferrara et al. state their algorithm 'is unable to "
                    f"discriminate between the presence and absence of CFA "
                    f"artifacts'. Compression has erased the demosaicing "
                    f"correlation this engine measures.")

        if image is None:
            return None
        return self._colour_premise_failure(image)

    @staticmethod
    def _colour_premise_failure(image: np.ndarray) -> Optional[str]:
        """Explain why the pixel data cannot support CFA analysis, if it cannot.

        Args:
            image: Candidate input array.

        Returns:
            Explanation string, or None when the pixel data is usable.
        """
        array = np.asarray(image)
        if (array.ndim != constants.EXPECTED_IMAGE_DIMENSION_COUNT
                or array.shape[-1] != constants.EXPECTED_CHANNEL_COUNT):
            return ("Engine skipped: the image is not a three-channel colour "
                    "array. A colour filter array leaves its trace across the "
                    "colour planes, so a grayscale or single-plane input "
                    "carries no CFA signature to measure.")

        green = array[..., constants.ANALYSIS_CHANNEL_INDEX].astype(np.float64)
        if float(np.var(green)) < constants.MINIMUM_CHANNEL_VARIANCE:
            return ("Engine skipped: the green channel is constant to within "
                    "less than one grey level. A degenerate plane has no "
                    "prediction error, so the acquired-versus-interpolated "
                    "variance ratio of Eq. 11 is undefined.")

        red = array[..., constants.RED_CHANNEL_INDEX].astype(np.float64)
        blue = array[..., constants.BLUE_CHANNEL_INDEX].astype(np.float64)
        if (float(np.var(red - green)) < constants.MINIMUM_INTER_CHANNEL_VARIANCE
                and float(np.var(blue - green))
                < constants.MINIMUM_INTER_CHANNEL_VARIANCE):
            return ("Engine skipped: the three colour planes are identical to "
                    "within less than one grey level, so this is monochrome "
                    "content in a colour container. A colour filter array "
                    "leaves its trace across the planes, so there is no "
                    "demosaicing correlation here to measure and no CFA phase "
                    "to verify.")
        return None

    @staticmethod
    def _check_compression_quality(metadata: ImageMetadata) -> CheckResult:
        """Check the estimated JPEG quality against the documented thresholds.

        SKILL "Reliable when": "image is uncompressed or JPEG quality >= ~95%
        (Ferrara)". SKILL "Unreliable when": "JPEG compression below ~90-95%
        quality - the dominant, universally-reported failure mode", with
        Ferrara's AUC collapsing "from 0.9975 (uncompressed) toward chance as
        quality drops to 85%" and Bammey's detection rate falling from 100% at
        QF100 to 67% at QF90.

        Args:
            metadata: Compression facts.

        Returns:
            CheckResult for this condition.
        """
        quality = metadata.estimated_compression_level

        if quality <= constants.NO_COMPRESSION_QUALITY_FACTOR:
            return (True, constants.FULL_CONFIDENCE,
                    "No JPEG compression reported, which is the uncompressed "
                    "condition all quantitative results in the corpus are "
                    "measured under.")

        if quality >= constants.RELIABLE_JPEG_QUALITY_FACTOR:
            return True, constants.FULL_CONFIDENCE, ""

        if quality >= constants.DEGRADED_JPEG_QUALITY_FACTOR:
            return (False, constants.CONFIDENCE_PENALTY_DEGRADED_QUALITY,
                    f"Estimated JPEG quality {quality:.0f} is below the "
                    f"{constants.RELIABLE_JPEG_QUALITY_FACTOR:.0f} the corpus "
                    f"requires. Bammey et al. report detection falling to 67% "
                    f"at this quality, and note the loss of small-forgery "
                    f"detection specifically.")

        return (False, constants.CONFIDENCE_PENALTY_SEVERELY_COMPRESSED,
                f"Estimated JPEG quality {quality:.0f} is below "
                f"{constants.DEGRADED_JPEG_QUALITY_FACTOR:.0f}. Ferrara's AUC "
                f"is collapsing toward chance through this range and Jeon's "
                f"phase-estimation accuracy drops to 19.74-34.78%.")

    @staticmethod
    def _check_resampling(metadata: ImageMetadata) -> CheckResult:
        """Flag prior resizing, which shifts or destroys the CFA phase.

        SKILL "Unreliable / inapplicable when": "Aggressive resizing/rescaling/
        re-demosaicing after tampering shifts or destroys the periodic CFA phase
        entirely (general limitation, not separately quantified in the corpus)".

        Args:
            metadata: Resampling facts.

        Returns:
            CheckResult for this condition.
        """
        if not metadata.is_resized:
            return True, constants.FULL_CONFIDENCE, ""

        return (False, constants.CONFIDENCE_PENALTY_RESAMPLED_INPUT,
                "Image is reported as resized. Resampling shifts or destroys "
                "the periodic CFA phase across the whole image, so an elevated "
                "score here may reflect the resize rather than any tampering. "
                "The post-computation check on Eq. 13 will report whether any "
                "CFA signature survived at all.")

    @staticmethod
    def _check_resolution(metadata: ImageMetadata) -> CheckResult:
        """Check the image is large enough for the corpus's block geometry.

        SKILL Pipeline C step 7 reports accuracies only for block sizes from 32
        upward, so below that the phase estimator has no validated behaviour.

        Args:
            metadata: Resolution facts.

        Returns:
            CheckResult for this condition.
        """
        try:
            height = int(metadata.resolution[0])
            width = int(metadata.resolution[1])
        except (TypeError, ValueError, IndexError):
            return (False, constants.CONFIDENCE_PENALTY_PHASE_UNVERIFIED,
                    f"Resolution metadata {metadata.resolution!r} is "
                    f"unreadable; block-geometry adequacy could not be checked.")

        if min(height, width) >= constants.MINIMUM_PHASE_ESTIMATION_BLOCK_SIZE:
            return True, constants.FULL_CONFIDENCE, ""

        return (False, constants.CONFIDENCE_PENALTY_PHASE_UNVERIFIED,
                f"Image is {height}x{width}, smaller than the "
                f"{constants.MINIMUM_PHASE_ESTIMATION_BLOCK_SIZE}-pixel "
                f"minimum block size Jeon et al. report any accuracy for, so "
                f"the CFA phase cannot be verified by SVD.")

    @staticmethod
    def _check_colour_input(image: np.ndarray) -> CheckResult:
        """Confirm the input carries the three planes CFA analysis needs.

        Args:
            image: Candidate input array.

        Returns:
            CheckResult for this condition.
        """
        array = np.asarray(image)
        if (array.ndim == constants.EXPECTED_IMAGE_DIMENSION_COUNT
                and array.shape[-1] == constants.EXPECTED_CHANNEL_COUNT):
            return True, constants.FULL_CONFIDENCE, ""

        return (False, constants.ZERO_CONFIDENCE,
                f"Input has shape {array.shape}, not a three-channel colour "
                f"image.")

    @staticmethod
    def _check_saturation(image: np.ndarray) -> CheckResult:
        """Reject near-saturated input, where prediction error is uninformative.

        SKILL "Unreliable / inapplicable when": "Flat/uniform and saturated
        regions - near-zero prediction error regardless of CFA presence".

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
                    "Analysis channel could not be read for the saturation "
                    "check.")

        saturated_fraction = compute_saturated_pixel_fraction(
            channel,
            constants.MINIMUM_INTENSITY_LEVEL,
            constants.SATURATION_INTENSITY_LEVEL,
        )
        if saturated_fraction <= constants.MAX_ACCEPTABLE_SATURATION_FRACTION:
            return True, constants.FULL_CONFIDENCE, ""

        return (False, constants.ZERO_CONFIDENCE,
                f"{saturated_fraction:.1%} of green-channel pixels are pinned "
                f"at an intensity extreme, exceeding the "
                f"{constants.MAX_ACCEPTABLE_SATURATION_FRACTION:.0%} limit. "
                f"Clipped pixels carry near-zero prediction error regardless of "
                f"whether a CFA signature is present.")

    @staticmethod
    def assess_texture_adequacy(excluded_block_fraction: float) -> CheckResult:
        """Judge how much of the image was too flat or too edgy to measure.

        Args:
            excluded_block_fraction: Fraction of blocks the preprocessor
                dropped as flat or edge-dominated.

        Returns:
            CheckResult reflecting post-preprocessing data quality.
        """
        if excluded_block_fraction <= constants.MAX_ACCEPTABLE_EXCLUDED_BLOCK_FRACTION:
            return True, constants.FULL_CONFIDENCE, ""

        return (False, constants.CONFIDENCE_PENALTY_LOW_TEXTURE,
                f"{excluded_block_fraction:.1%} of blocks were excluded as "
                f"almost flat or edge-dominated, exceeding the "
                f"{constants.MAX_ACCEPTABLE_EXCLUDED_BLOCK_FRACTION:.0%} limit. "
                f"Ferrara et al. state the method 'is less effective in the "
                f"presence of either almost flat areas or sharp edges', so the "
                f"surviving blocks are not a representative sample of the "
                f"scene.")

    @staticmethod
    def assess_sample_quality(valid_block_count: int) -> CheckResult:
        """Judge whether enough blocks survived to fit a two-component mixture.

        Args:
            valid_block_count: Blocks that contributed to the mixture fit.

        Returns:
            CheckResult reflecting the population size behind the fit.
        """
        if valid_block_count == 0:
            return (False, constants.ZERO_CONFIDENCE,
                    "No block carried enough texture to contribute to the "
                    "statistic, so no measurement was possible.")

        if valid_block_count >= constants.MINIMUM_MIXTURE_SAMPLE_COUNT:
            return True, constants.FULL_CONFIDENCE, ""

        return (True, constants.CONFIDENCE_PENALTY_SMALL_SAMPLE,
                f"Only {valid_block_count} blocks contributed to the Gaussian "
                f"mixture, below the {constants.MINIMUM_MIXTURE_SAMPLE_COUNT} "
                f"of the corpus's 256x256 working block size. The fit is "
                f"reported but is not a population estimate.")

    @staticmethod
    def assess_global_cfa_presence(mixture: GaussianMixtureFit) -> CheckResult:
        """Check that some CFA signature exists in the image at all.

        Ferrara Eq. 13 states that under M1 the feature is distributed with
        mu1 > 0. A fitted mean at or below zero means no block anywhere showed
        the acquired-versus-interpolated variance imbalance, so there is no
        authentic population for tampered blocks to stand out against. That
        happens on non-Bayer sensors, on globally resampled images, and on
        heavily recompressed ones - all documented inapplicability conditions.

        Args:
            mixture: The fitted Gaussian mixture.

        Returns:
            CheckResult reflecting whether the engine's premise held.
        """
        # ENHANCEMENT 2: the test is on the SEPARATION of the two components,
        # not merely the sign of mu1. mu2 is fixed at 0 by Eq. 14, so
        # mu1/sigma1 is the standardised distance between them. Testing showed
        # the sign test passes an image that never went through a colour filter
        # array at all (mu1 = 0.0020, scored 0.6733 as reliable). Measurements
        # behind the threshold are recorded in constants.
        separation = ConditionChecker._mixture_separation(mixture)
        if (mixture.authentic_mean > constants.MIXTURE_TAMPERED_MEAN
                and separation >= constants.MINIMUM_MIXTURE_SEPARATION_RATIO):
            return True, constants.FULL_CONFIDENCE, ""

        return (False, constants.ZERO_CONFIDENCE,
                ConditionChecker._describe_absent_cfa(mixture, separation))

    @staticmethod
    def _describe_absent_cfa(mixture: GaussianMixtureFit,
                             separation: float) -> str:
        """Word the "no usable CFA signature" finding for the report.

        Args:
            mixture: The fitted Gaussian mixture.
            separation: Standardised distance between its components.

        Returns:
            Human-readable explanation.
        """
        deviation = float(np.sqrt(max(mixture.authentic_variance, 0.0)))
        return (f"No usable CFA signature was found anywhere in the image. The "
                f"fitted authentic-component mean is "
                f"{mixture.authentic_mean:.4f} against a component standard "
                f"deviation of {deviation:.4f}, a separation of "
                f"{separation:.3f} where Eq. 13 requires a positive mean and "
                f"this engine requires at least "
                f"{constants.MINIMUM_MIXTURE_SEPARATION_RATIO:.1f} standard "
                f"deviations before the two hypotheses can be told apart. "
                f"Genuine Bayer imagery measures 1.9 to 24.9 here; imagery "
                f"that never passed a colour filter array measures 0.006 to "
                f"0.013. The image was either not captured through a Bayer CFA "
                f"- which includes synthetic and AI-generated imagery - or was "
                f"resampled or recompressed hard enough to erase the "
                f"demosaicing correlation. A high score here would be noise.")

    @staticmethod
    def _mixture_separation(mixture: GaussianMixtureFit) -> float:
        """Standardised distance between the two mixture components.

        Eq. 14 fixes the tampered component's mean at zero, so this is simply
        mu1 / sigma1.

        Args:
            mixture: The fitted Gaussian mixture.

        Returns:
            The ratio, or 0.0 when the component variance is degenerate.
        """
        variance = float(max(mixture.authentic_variance, 0.0))
        if variance <= 0.0:
            return 0.0
        return float(mixture.authentic_mean) / float(np.sqrt(variance))

    @staticmethod
    def assess_mixture_convergence(mixture: GaussianMixtureFit) -> CheckResult:
        """Penalise a mixture fit that never converged.

        ENHANCEMENT 4. The SKILL's Pipeline A step 6 sets convergence at a
        log-likelihood increase below 1e-3 or 500 iterations, whichever comes
        first. Hitting the cap means the second condition stopped the fit, not
        the first. Two of the six diagnostic images (campic.jpeg, campic2.jpeg)
        ran the full 500 iterations with converged=False and their posteriors
        were used at full weight.

        Args:
            mixture: The fitted Gaussian mixture.

        Returns:
            CheckResult; always passes, carrying a penalty when unconverged.
        """
        if getattr(mixture, "converged", True):
            return True, constants.FULL_CONFIDENCE, ""

        return (True, constants.CONFIDENCE_PENALTY_EM_NOT_CONVERGED,
                f"The Gaussian-mixture fit reached its "
                f"{constants.EM_MAXIMUM_ITERATIONS}-iteration cap without the "
                f"log-likelihood settling below "
                f"{constants.EM_LOG_LIKELIHOOD_TOLERANCE}. The parameters used "
                f"for every per-block posterior are therefore the last iterate "
                f"rather than a converged estimate.")
