"""Pre-computation input gate for the perspective / geometry engine.

Every condition the SKILL file documents under "Reliable when" and "Unreliable /
inapplicable when" is checked here. Unusually for this system, several of them
can only be judged AFTER the vanishing point has been estimated - camera tilt is
tested through the position of the vanishing line, and the single-dominant-VP
assumption through the RANSAC inlier fraction - so those are exposed as separate
methods the engine calls at Stage 4.

The SKILL is emphatic that the vanishing-point confidence is a hard gate, not a
soft weight: "This confidence must gate whether the module proceeds at all ... a
low-confidence VP estimate should cause the module to abstain rather than emit a
possibly-spurious height-ratio score." assess_vanishing_point_confidence is that
gate.

Note that compression is deliberately NOT penalised here. The SKILL records
robustness to downsampling and low-quality JPEG recompression as this method's
distinguishing strength, "a regime where most trace-based methods ... fail".
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from . import constants
from .contracts import (ConditionReport, HeightRatioAnalysis, ImageMetadata,
                        VanishingPointEstimate)
from .utils import compose_confidence_penalties

logger = logging.getLogger(__name__)

# A check reports (passed, confidence_penalty, note). passed=False marks the
# result unreliable; penalty is a multiplier folded into the final weight.
CheckResult = tuple[bool, float, str]


class ConditionChecker:
    """Decides whether the geometry engine may run, and at what confidence."""

    def check(self,
              metadata: ImageMetadata,
              image: Optional[np.ndarray] = None) -> ConditionReport:
        """Run every pre-computation condition against the input.

        Args:
            metadata: Container, resolution and compression facts.
            image: BGR uint8 array. When None, only metadata conditions run.

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
        """Evaluate every pre-computation condition and collect the verdicts.

        Args:
            metadata: Container, resolution and compression facts.
            image: BGR uint8 array, or None to skip pixel-level checks.

        Returns:
            One CheckResult per condition evaluated.
        """
        return [
            self._check_compression(metadata),
            self._check_resampling(metadata),
            self._check_resolution(metadata),
        ]

    @staticmethod
    def _premise_failure_reason(metadata: ImageMetadata,
                                image: Optional[np.ndarray]) -> Optional[str]:
        """Explain why the engine cannot run at all, if it cannot.

        Args:
            metadata: Container and resolution facts.
            image: BGR uint8 array, or None.

        Returns:
            Explanation string, or None when the engine may proceed.
        """
        if image is None:
            return ("Engine skipped: no image array was supplied, so no "
                    "geometric structure could be extracted.")

        array = np.asarray(image)
        if array.ndim not in (constants.GRAYSCALE_IMAGE_DIMENSION_COUNT,
                              constants.COLOUR_IMAGE_DIMENSION_COUNT):
            return (f"Engine skipped: expected a 2-D grayscale or 3-D colour "
                    f"image, received an array of {array.ndim} dimensions.")

        if min(array.shape[0], array.shape[1]) < \
                constants.MINIMUM_IMAGE_SIDE_PIXELS:
            return (f"Engine skipped: the image is "
                    f"{array.shape[0]}x{array.shape[1]}, below the "
                    f"{constants.MINIMUM_IMAGE_SIDE_PIXELS}-pixel minimum side. "
                    f"Neither the Hough vote threshold nor the minimum segment "
                    f"length can be met at this size, so no vanishing point "
                    f"could be estimated.")
        return None

    @staticmethod
    def _check_compression(metadata: ImageMetadata) -> CheckResult:
        """Record the compression level without penalising it.

        SKILL "Key findings": Yao et al.'s method is "explicitly validated as
        robust to down-sampling and low-quality JPEG recompression, a regime
        where most trace-based methods (resampling detectors, JPEG-ghost
        detectors) fail - this module is complementary to, not redundant with,
        the DSP-artifact-based modules elsewhere in this engine specifically
        because it survives exactly the post-processing that defeats them."

        Args:
            metadata: Compression facts.

        Returns:
            CheckResult that never fails and never penalises.
        """
        quality = metadata.estimated_compression_level
        if quality <= constants.NO_COMPRESSION_QUALITY_FACTOR:
            return True, constants.FULL_CONFIDENCE, ""

        if quality < constants.NOTABLE_JPEG_QUALITY_FACTOR:
            return (True, constants.FULL_CONFIDENCE,
                    f"Estimated JPEG quality is {quality:.0f}, which would "
                    f"disable most trace-based engines. This engine is "
                    f"deliberately not penalised for it: Yao et al. validate "
                    f"the perspective constraint as robust to low-quality "
                    f"recompression, which is what makes this vote "
                    f"complementary rather than redundant.")
        return True, constants.FULL_CONFIDENCE, ""

    @staticmethod
    def _check_resampling(metadata: ImageMetadata) -> CheckResult:
        """Note prior resizing, which this method largely survives.

        Downsampling is a documented strength, so this is a light penalty
        covering the residual risk that resampling blurred the edges the Hough
        transform depends on, not a challenge to the geometry itself.

        Args:
            metadata: Resampling facts.

        Returns:
            CheckResult for this condition.
        """
        if not metadata.is_resized:
            return True, constants.FULL_CONFIDENCE, ""

        return (True, constants.CONFIDENCE_PENALTY_RESAMPLED_INPUT,
                "Image is reported as resized. Yao et al. validate this method "
                "as robust to down-sampling, so the geometric constraint still "
                "holds; the mild penalty reflects only that resampling softens "
                "the edges the line detector relies on.")

    @staticmethod
    def _check_resolution(metadata: ImageMetadata) -> CheckResult:
        """Check the declared resolution is large enough to analyse.

        Args:
            metadata: Resolution facts.

        Returns:
            CheckResult for this condition.
        """
        try:
            height = int(metadata.resolution[0])
            width = int(metadata.resolution[1])
        except (TypeError, ValueError, IndexError):
            return (True, constants.FULL_CONFIDENCE,
                    f"Resolution metadata {metadata.resolution!r} is "
                    f"unreadable; the image array itself was measured instead.")

        if min(height, width) >= constants.MINIMUM_IMAGE_SIDE_PIXELS:
            return True, constants.FULL_CONFIDENCE, ""
        return (False, constants.ZERO_CONFIDENCE,
                f"Declared resolution {height}x{width} is below the "
                f"{constants.MINIMUM_IMAGE_SIDE_PIXELS}-pixel minimum side.")

    @staticmethod
    def assess_vanishing_point_confidence(
            estimate: VanishingPointEstimate) -> CheckResult:
        """Apply the SKILL's hard gate on vanishing-point confidence.

        SKILL "Output": "a low-confidence VP estimate should cause the module to
        abstain rather than emit a possibly-spurious height-ratio score", with
        "RANSAC inlier count/fraction for A4, or line-fit residual for A1" named
        as the indicators. A low inlier fraction is also exactly the
        multiple-vanishing-point (Manhattan-world) condition the SKILL rules
        inapplicable, since competing vanishing points split the consensus.

        Args:
            estimate: The vanishing-point estimate under test.

        Returns:
            CheckResult; a failure here means the engine must abstain.
        """
        if estimate.homogeneous_point is None:
            return (False, constants.ZERO_CONFIDENCE,
                    f"No vanishing point could be estimated. {estimate.note}")

        if estimate.is_at_infinity or estimate.vanishing_line_row is None:
            return (False, constants.ZERO_CONFIDENCE,
                    "The estimated vanishing point lies at infinity, so the "
                    "supporting lines are exactly parallel in the image and fix "
                    "no vanishing-line row. Eq. 7 has no v0 to work with.")

        if estimate.inlier_fraction < \
                constants.MINIMUM_VANISHING_POINT_INLIER_FRACTION:
            return (False, constants.ZERO_CONFIDENCE,
                    f"Only {estimate.inlier_count} of "
                    f"{estimate.total_line_count} lines "
                    f"({estimate.inlier_fraction:.0%}) agree on a single "
                    f"vanishing point, below the "
                    f"{constants.MINIMUM_VANISHING_POINT_INLIER_FRACTION:.0%} "
                    f"required. Both source methods assume a single dominant "
                    f"vanishing point; a split consensus indicates a "
                    f"Manhattan-world scene, which the corpus rules "
                    f"inapplicable.")
        return ConditionChecker._check_residual(estimate)

    @staticmethod
    def _check_residual(estimate: VanishingPointEstimate) -> CheckResult:
        """Reject an estimate whose supporting lines converge too loosely.

        Args:
            estimate: The vanishing-point estimate under test.

        Returns:
            CheckResult; a failure means the engine must abstain.
        """
        if estimate.line_fit_residual_pixels > \
                constants.MAXIMUM_LINE_FIT_RESIDUAL_PIXELS:
            return (False, constants.ZERO_CONFIDENCE,
                    f"Line-fit residual is "
                    f"{estimate.line_fit_residual_pixels:.1f} pixels, above the "
                    f"{constants.MAXIMUM_LINE_FIT_RESIDUAL_PIXELS:.0f}-pixel "
                    f"limit; the detected edges do not converge tightly enough "
                    f"to trust the vanishing point.")
        return ConditionChecker._assess_line_evidence(estimate)

    @staticmethod
    def _assess_line_evidence(estimate: VanishingPointEstimate) -> CheckResult:
        """Weight a confident estimate by how much evidence stands behind it.

        Args:
            estimate: A vanishing-point estimate that passed the hard gates.

        Returns:
            CheckResult that passes, possibly with a penalty.
        """
        penalties: list = []
        notes: list = []

        if estimate.total_line_count < constants.MINIMUM_LINES_FOR_CONFIDENT_ESTIMATE:
            penalties.append(constants.CONFIDENCE_PENALTY_SPARSE_LINE_EVIDENCE)
            notes.append(
                f"Only {estimate.total_line_count} usable lines were available, "
                f"below the {constants.MINIMUM_LINES_FOR_CONFIDENT_ESTIMATE} at "
                f"which the estimate is treated as well supported.")

        if estimate.method == "A4_recurrence":
            penalties.append(constants.CONFIDENCE_PENALTY_RECURRENCE_FALLBACK)
            notes.append(
                "The vanishing point came from the recurrence fallback rather "
                "than from explicit parallel lines, which the corpus names as "
                "the primary route when lines are available.")

        return (True, compose_confidence_penalties(penalties),
                " ".join(notes))

    @staticmethod
    def assess_vanishing_line_position(vanishing_line_row: Optional[float],
                                       image_height: int) -> CheckResult:
        """Test camera tilt through the position of the vanishing line.

        SKILL "Unreliable / inapplicable when" -> significant camera tilt/roll:
        Yao et al.'s Eq. 4 "assumes negligible tilt", and a tilt-compensated
        version needs a second vertical vanishing point the paper states is
        unreliable to estimate. The paper's own experiments give the operational
        criterion used here: Eq. 4 "still holds approximately as long as the
        tilt is small enough that the vanishing line is inside the image".

        Args:
            vanishing_line_row: Yao's v0, or None when undetermined.
            image_height: Height of the analysed image in pixels.

        Returns:
            CheckResult for this condition.
        """
        if vanishing_line_row is None:
            return (False, constants.ZERO_CONFIDENCE,
                    "No vanishing-line row was determined, so camera tilt "
                    "could not be assessed.")

        if 0.0 <= vanishing_line_row <= float(image_height - 1):
            return True, constants.FULL_CONFIDENCE, ""

        return (False, constants.ZERO_CONFIDENCE,
                f"The estimated vanishing line sits at row "
                f"{vanishing_line_row:.0f}, outside the image's 0 to "
                f"{image_height - 1} range. Yao et al. state their simplified "
                f"projection holds only 'as long as the tilt is small enough "
                f"that the vanishing line is inside the image', so the camera "
                f"is tilted or rolled beyond what this method supports.")

    @staticmethod
    def assess_object_pairs(analysis: HeightRatioAnalysis) -> CheckResult:
        """Judge whether enough object pairs were measurable to vote.

        SKILL "Input requirements" requires "at least one pair of
        same-reference-plane objects is identifiable", and SKILL B step 5
        recommends averaging "several measurements of beta ... to improve
        accuracy - the paper does not give a specific minimum count".

        Args:
            analysis: The height-ratio analysis to judge.

        Returns:
            CheckResult reflecting the evidence behind the score.
        """
        if analysis.evaluated_pair_count < constants.MINIMUM_OBJECT_PAIR_COUNT:
            return (False, constants.ZERO_CONFIDENCE,
                    f"No object pair passed the ground-plane sanity check "
                    f"({analysis.rejected_pair_count} candidate pairs were "
                    f"rejected as too short, floating above the vanishing line, "
                    f"or too dissimilar to compare). The corpus requires at "
                    f"least one identifiable pair of objects on a common "
                    f"reference plane.")

        if analysis.evaluated_pair_count < constants.MINIMUM_PAIRS_FOR_CORROBORATION:
            return (True, constants.CONFIDENCE_PENALTY_UNCORROBORATED,
                    f"Only {analysis.evaluated_pair_count} object pair(s) were "
                    f"measurable, so the corpus's recommendation to average "
                    f"several measurements of the height ratio could not be "
                    f"honoured.")

        return True, constants.FULL_CONFIDENCE, ""

    @staticmethod
    def assess_vanishing_line_precision(
            analysis: HeightRatioAnalysis,
            estimate: VanishingPointEstimate) -> CheckResult:
        """Check the vanishing line is precise enough to support the verdict.

        Eq. 7 differences image rows against v0, so its output moves with any
        error in v0, and how fast depends on the pair. Measured here, a pair at
        depths 12 m and 28 m tolerated 6.5 pixels of horizon error before
        flipping verdict; one at 6 m and 60 m tolerated 1.4. This compares that
        tolerance against the SKILL's own named A1 confidence indicator, the
        line-fit residual - a proxy for how well the lines converge rather than
        a direct v0 error, so it is used conservatively. Without this check the
        engine emits confident false positives.

        Args:
            analysis: The height-ratio analysis to judge.
            estimate: The vanishing-point estimate the analysis rests on.

        Returns:
            CheckResult reflecting whether v0 is precise enough.
        """
        deciding = min(analysis.measurements,
                       key=lambda item: item.consistency, default=None)
        if deciding is None:
            return True, constants.FULL_CONFIDENCE, ""

        tolerance = deciding.tolerable_vanishing_line_error_pixels
        residual = estimate.line_fit_residual_pixels
        if not np.isfinite(tolerance) or not np.isfinite(residual):
            return True, constants.FULL_CONFIDENCE, ""
        if residual <= tolerance:
            return True, constants.FULL_CONFIDENCE, ""

        return (False, constants.ZERO_CONFIDENCE,
                ConditionChecker._imprecise_note(deciding, tolerance, residual))

    @staticmethod
    def _imprecise_note(deciding, tolerance: float, residual: float) -> str:
        """Note explaining that v0 is too imprecise to support the verdict.

        Args:
            deciding: The least consistent HeightRatioMeasurement.
            tolerance: Vanishing-line error the verdict could absorb.
            residual: The estimate's measured line-fit residual.

        Returns:
            Explanatory text for the reliability field.
        """
        return (f"VANISHING LINE NOT PRECISE ENOUGH: the deciding object pair "
                f"would flip verdict on only {tolerance:.1f} pixels of error in "
                f"the vanishing line, but the supporting lines converge to a "
                f"residual of {residual:.1f} pixels. Its measured height ratio "
                f"moves by {deciding.ratio_sensitivity_per_pixel:.4f} per pixel "
                f"of horizon error, so this pair straddles too wide a depth "
                f"range to be decided at the precision available. The "
                f"measurement is reported for inspection but not counted.")

    @staticmethod
    def assess_expected_ratio_provenance(
            analysis: HeightRatioAnalysis,
            regions_were_supplied: bool) -> CheckResult:
        """Flag when the expected height ratio was assumed rather than known.

        SKILL B step 2 requires the expected ratio to come from "general prior
        knowledge about the object classes" or "a trusted reference pair". An
        image-only contract supplies neither, so the engine assumes the regions
        depict the same kind of object. How damaging that is depends on where
        the regions came from. With caller-supplied regions the caller has
        already judged them comparable, and the benchmark separates cleanly
        (0.0715 authentic against 1.0000 for a 55%-oversized splice). With
        automatically proposed regions the assumption is usually false, and the
        benchmark returned 1.0000 for authentic and forged alike - no
        discriminative power - so no counted vote is cast, per the SKILL's
        instruction to abstain rather than "emit a possibly-spurious
        height-ratio score".

        Args:
            analysis: The height-ratio analysis to judge.
            regions_were_supplied: Whether the caller localised the objects.

        Returns:
            CheckResult reflecting where the expected ratio came from.
        """
        if not analysis.any_ratio_assumed:
            return True, constants.FULL_CONFIDENCE, ""
        if regions_were_supplied:
            return (True, constants.CONFIDENCE_PENALTY_ASSUMED_RATIO_PRIOR,
                    ConditionChecker._assumed_ratio_note())
        return (False, constants.ZERO_CONFIDENCE,
                ConditionChecker._uncountable_ratio_note())

    @staticmethod
    def _assumed_ratio_note() -> str:
        """Note for an assumed ratio over caller-supplied regions.

        Returns:
            Explanatory text for the reliability field.
        """
        return (f"ASSUMED RATIO PRIOR: no expected height ratio was supplied, "
                f"so pairs were scored against an assumed ratio of "
                f"{constants.DEFAULT_EXPECTED_HEIGHT_RATIO:.1f}. The regions "
                f"themselves were supplied by the caller, who is taken to have "
                f"judged them comparable, so the vote stands at reduced "
                f"confidence. Supply expected ratios to remove this caveat.")

    @staticmethod
    def _uncountable_ratio_note() -> str:
        """Note for an assumed ratio over automatically proposed regions.

        Returns:
            Explanatory text for the reliability field.
        """
        return (f"NO COUNTED VOTE: no expected height ratio was supplied, so "
                f"pairs were scored against an assumed ratio of "
                f"{constants.DEFAULT_EXPECTED_HEIGHT_RATIO:.1f}, and the "
                f"regions were proposed automatically by superpixel "
                f"segmentation, so there is no basis for believing any two "
                f"depict the same class of object. Measured on this engine's "
                f"benchmark that combination scores 1.0 for authentic and "
                f"forged images alike, so the height-ratio test is reported "
                f"for inspection but not counted. Supply object regions and "
                f"expected height ratios to obtain a real vote.")
