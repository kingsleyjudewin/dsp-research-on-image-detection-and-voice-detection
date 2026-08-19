"""Entry point for the perspective / geometric consistency engine.

Orchestrates the four processing stages in order:

    Stage 1  input validation and condition checking   (condition.py)
    Stage 2  preprocessing                             (preprocessor.py)
    Stage 3  core mathematical computation             (computer.py)
    Stage 4  score extraction and output assembly      (scorer.py, visualizer.py)

Stage 3 runs the vanishing-point estimator first and the height-ratio test
second, because Eq. 7 cannot be evaluated without v0. Between them sits the
SKILL's hard confidence gate: "a low-confidence VP estimate should cause the
module to abstain rather than emit a possibly-spurious height-ratio score". When
that gate fires the engine returns a null vote even though it has already done
most of its work, which is the intended behaviour.

The engine never raises into its caller. Every failure path returns a fully
populated EngineOutput with is_reliable=False and a reliability_note naming what
went wrong, so the fusion layer can always account for this engine's vote.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from . import constants
from .computer import HeightRatioAnalyser, VanishingPointEstimator
from .condition import ConditionChecker
from .contracts import (CalibrationSettings, ConditionReport, EngineInput,
                        EngineOutput, FlaggedRegion, HeightRatioAnalysis,
                        PreparedScene, VanishingPointEstimate)
from .preprocessor import GeometryPreprocessor
from .scorer import GeometryScorer
from .utils import (build_computation_step, clip_to_unit_interval,
                    compose_confidence_penalties, describe_array_shape)
from .visualizer import GeometryVisualizer

logger = logging.getLogger(__name__)


class GeometryEngine:
    """Detects forgery from objects that break the scene's perspective."""

    def __init__(self,
                 calibration: Optional[CalibrationSettings] = None) -> None:
        """Build the engine and its collaborators.

        Args:
            calibration: Optional calibration state. Its supplied_regions and
                supplied_expected_ratios fields are how a caller replaces the
                automatic SLIC proposal and the assumed height-ratio prior,
                which is the single most effective way to strengthen this
                engine's vote.
        """
        self.calibration = calibration or CalibrationSettings()
        self.condition_checker = ConditionChecker()
        self.preprocessor = GeometryPreprocessor(
            self.calibration.supplied_regions)
        self.vanishing_point_estimator = VanishingPointEstimator()
        self.height_ratio_analyser = HeightRatioAnalyser()
        self.scorer = GeometryScorer(self.calibration)
        self.visualizer = GeometryVisualizer()

    def analyse(self, engine_input: EngineInput) -> EngineOutput:
        """Analyse one image and return this engine's forensic vote.

        Args:
            engine_input: Image plus metadata.

        Returns:
            EngineOutput, always populated, never raising.
        """
        started_at = time.perf_counter()
        try:
            return self._run_pipeline(engine_input, started_at)
        except Exception as error:  # noqa: BLE001 - boundary must not leak
            logger.exception("geometry engine failed during analysis")
            return self._build_failure_output(
                started_at,
                f"{type(error).__name__}: {error}",
                self._describe_input(engine_input),
            )

    def _run_pipeline(self,
                      engine_input: EngineInput,
                      started_at: float) -> EngineOutput:
        """Execute the four stages in order.

        Args:
            engine_input: Image plus metadata.
            started_at: perf_counter value taken at entry.

        Returns:
            EngineOutput for a completed or deliberately abstained analysis.
        """
        image = engine_input.image
        metadata = engine_input.metadata

        # Stage 1 - condition checking runs before any pixel is transformed.
        report = self.condition_checker.check(metadata, image)
        steps = [self._describe_condition_check(image, metadata, report)]
        if report.skip_engine:
            return self._build_skip_output(started_at, report, steps)

        # Stage 2.
        scene = self.preprocessor.prepare(image)
        steps.append(self._describe_preprocessing(image, scene))

        # Stage 3a - vanishing point, then the SKILL's hard confidence gate.
        estimate = self.vanishing_point_estimator.estimate(scene)
        steps.append(self._describe_vanishing_point(scene, estimate))

        gates = self._run_geometry_gates(estimate, scene)
        if any(not passed for passed, _, _ in gates):
            return self._build_abstain_output(started_at, report, gates, steps)

        # Stage 3b and Stage 4.
        return self._complete_analysis(started_at, report, scene, estimate,
                                       metadata, steps)

    def _run_geometry_gates(self,
                            estimate: VanishingPointEstimate,
                            scene: PreparedScene) -> list:
        """Apply the two gates that can only be judged after estimating the VP.

        Args:
            estimate: The vanishing-point estimate.
            scene: The prepared scene, for the image height.

        Returns:
            List of CheckResult tuples.
        """
        return [
            self.condition_checker.assess_vanishing_point_confidence(estimate),
            self.condition_checker.assess_vanishing_line_position(
                estimate.vanishing_line_row, int(scene.grayscale.shape[0])),
        ]

    def _complete_analysis(self,
                           started_at: float,
                           report: ConditionReport,
                           scene: PreparedScene,
                           estimate: VanishingPointEstimate,
                           metadata,
                           steps: list[dict]) -> EngineOutput:
        """Run the height-ratio test and assemble the final output.

        Args:
            started_at: perf_counter value taken at entry.
            report: Verdict of the pre-computation gate.
            scene: The prepared scene.
            estimate: The accepted vanishing-point estimate.
            metadata: Image metadata, selecting the calibration bucket.
            steps: Trace accumulated so far, appended to in place.

        Returns:
            Fully populated EngineOutput.
        """
        analysis = self.height_ratio_analyser.analyse(
            estimate.vanishing_line_row, scene.regions,
            self.calibration.supplied_expected_ratios)
        steps.append(self._describe_height_ratio(scene, estimate, analysis))

        return self._build_success_output(started_at, report, scene, estimate,
                                          analysis, metadata, steps)

    @staticmethod
    def _describe_condition_check(image: np.ndarray,
                                  metadata,
                                  report: ConditionReport) -> dict:
        """Record the condition-checking stage in the computation trace.

        Args:
            image: Original image array.
            metadata: Image metadata that was evaluated.
            report: Verdict of the gate.

        Returns:
            One computation-step dictionary.
        """
        return build_computation_step(
            step_number=constants.COMPUTATION_STEP_CONDITION_CHECK,
            name="Condition check",
            description=("Evaluated the pre-computation reliability conditions. "
                         "Unlike the trace-based engines in this system, JPEG "
                         "compression carries no penalty here: the corpus "
                         "validates the perspective constraint as robust to "
                         "down-sampling and low-quality recompression, which is "
                         "what makes this vote complementary to theirs. Tilt "
                         "and vanishing-point confidence cannot be judged yet "
                         "and are gated after estimation."),
            input_shape=describe_array_shape(image),
            output_shape="condition report",
            key_values={
                "is_reliable": report.is_reliable,
                "skip_engine": report.skip_engine,
                "confidence_weight": round(report.confidence_weight, constants.TRACE_DECIMAL_PLACES),
                "estimated_jpeg_quality": metadata.estimated_compression_level,
                "is_resized": metadata.is_resized,
                "container_format": metadata.format,
            },
        )

    def _describe_preprocessing(self,
                                image: np.ndarray,
                                scene: PreparedScene) -> dict:
        """Record the preprocessing stage in the computation trace.

        Args:
            image: Original image array.
            scene: Output of Stage 2.

        Returns:
            One computation-step dictionary.
        """
        supplied = bool(self.calibration.supplied_regions)
        return build_computation_step(
            step_number=constants.COMPUTATION_STEP_PREPROCESSING,
            name="Line, keypoint and region extraction",
            description=(
                f"Extracted straight edges by Canny plus probabilistic Hough "
                f"transform for the parallel-line module, "
                f"{constants.SIFT_DESCRIPTOR_LENGTH}-dimensional SIFT "
                f"descriptors for the recurrence module, and candidate object "
                f"regions "
                f"{'supplied by the caller' if supplied else 'by SLIC superpixel proposal'}."),
            input_shape=describe_array_shape(image),
            output_shape=describe_array_shape(scene.grayscale),
            key_values={
                "line_segment_count": len(scene.line_segments),
                "sift_keypoint_count": int(scene.keypoint_positions.shape[0]),
                "candidate_region_count": len(scene.regions),
                "region_source": "supplied" if supplied else "slic",
                "minimum_line_pair_angle_degrees":
                    constants.MINIMUM_LINE_PAIR_ANGLE_DEGREES,
            },
        )

    @staticmethod
    def _describe_vanishing_point(scene: PreparedScene,
                                  estimate: VanishingPointEstimate) -> dict:
        """Record the vanishing-point estimation stage in the trace.

        Args:
            scene: Output of Stage 2.
            estimate: The vanishing-point estimate.

        Returns:
            One computation-step dictionary.
        """
        return build_computation_step(
            step_number=constants.COMPUTATION_STEP_VANISHING_POINT,
            name="Vanishing point and vanishing line",
            description=(
                "Estimated the reference plane's vanishing point, preferring "
                "Yao et al.'s explicit-line route (Hough segments, 5-degree "
                "minimum-pair-angle filter, homogeneous least-squares "
                "intersection, Levenberg-Marquardt refinement) and falling back "
                "to R-VPD's recurrence route (single-linkage visual words over "
                "SIFT descriptors, geometric-consistency scoring, implicit line "
                "fitting, weighted RANSAC). Because the camera is assumed "
                "level, the vanishing line is the horizontal line through that "
                "point, giving v0. " + estimate.note),
            input_shape=f"{len(scene.line_segments)} line segments, "
                        f"{int(scene.keypoint_positions.shape[0])} keypoints",
            output_shape="vanishing point and horizon row",
            key_values={
                "method": estimate.method,
                "vanishing_line_row": (None if estimate.vanishing_line_row is None
                                       else round(estimate.vanishing_line_row, constants.TRACE_COARSE_DECIMAL_PLACES)),
                "inlier_count": estimate.inlier_count,
                "total_line_count": estimate.total_line_count,
                "inlier_fraction": round(estimate.inlier_fraction, constants.TRACE_DECIMAL_PLACES),
                "line_fit_residual_pixels":
                    round(estimate.line_fit_residual_pixels, 3)
                    if np.isfinite(estimate.line_fit_residual_pixels) else None,
                "is_at_infinity": estimate.is_at_infinity,
            },
        )

    @staticmethod
    def _describe_height_ratio(scene: PreparedScene,
                               estimate: VanishingPointEstimate,
                               analysis: HeightRatioAnalysis) -> dict:
        """Record the height-ratio consistency test in the trace.

        Args:
            scene: Output of Stage 2.
            estimate: The accepted vanishing-point estimate.
            analysis: The height-ratio analysis.

        Returns:
            One computation-step dictionary.
        """
        worst = min(analysis.measurements,
                    key=lambda item: item.consistency, default=None)
        return build_computation_step(
            step_number=constants.COMPUTATION_STEP_HEIGHT_RATIO,
            name="Height-ratio consistency (Eq. 7-8)",
            description=(
                f"For each admissible object pair, recovered the ratio of their "
                f"real-world heights from image coordinates and the vanishing "
                f"line at row {estimate.vanishing_line_row:.1f} using Eq. 7, "
                f"then scored its divergence from the expected ratio through "
                f"Eq. 8's Gaussian consistency measure with sigma = "
                f"{constants.RATIO_SIGMA_FRACTION_OF_EXPECTED} x alpha. Pairs "
                f"whose bases sat at or above the vanishing line, or which were "
                f"too short to measure, were rejected before scoring, per the "
                f"corpus's degenerate-pair caution."),
            input_shape=f"{len(scene.regions)} candidate regions",
            output_shape=f"{analysis.evaluated_pair_count} measured pairs",
            key_values=GeometryEngine._height_ratio_key_values(analysis, worst),
        )

    @staticmethod
    def _height_ratio_key_values(analysis: HeightRatioAnalysis, worst) -> dict:
        """Collect the headline numbers from the height-ratio test.

        Args:
            analysis: The height-ratio analysis.
            worst: The least consistent measurement, or None.

        Returns:
            Dictionary of named scalars for the trace.
        """
        return {
                "evaluated_pair_count": analysis.evaluated_pair_count,
                "rejected_pair_count": analysis.rejected_pair_count,
                "minimum_consistency": round(analysis.minimum_consistency, constants.TRACE_DECIMAL_PLACES),
                # ENHANCEMENT 1: the statistic that actually drives raw_score.
                **GeometryEngine._corroboration_key_values(analysis),
                "mean_consistency": round(analysis.mean_consistency, constants.TRACE_DECIMAL_PLACES),
                "mean_measured_ratio": round(analysis.mean_measured_ratio, constants.TRACE_DECIMAL_PLACES),
                "worst_pair": (None if worst is None else
                               [worst.first_region_id, worst.second_region_id]),
                "worst_pair_measured_ratio": (None if worst is None else
                                              round(worst.measured_ratio, constants.TRACE_DECIMAL_PLACES)),
                "worst_pair_expected_ratio": (None if worst is None else
                                              round(worst.expected_ratio, constants.TRACE_DECIMAL_PLACES)),
                "decision_threshold":
                    constants.CONSISTENCY_DECISION_THRESHOLD,
                "expected_ratio_was_assumed": analysis.any_ratio_assumed,
                # Eq. 7 is markedly more sensitive to v0 for pairs straddling
                # very different depths. Reporting how much vanishing-line error
                # the deciding pair could absorb before flipping verdict tells a
                # reader how much weight the finding can carry.
                "worst_pair_ratio_shift_per_pixel_of_v0_error":
                    (None if worst is None else
                     round(worst.ratio_sensitivity_per_pixel, constants.TRACE_FINE_DECIMAL_PLACES)),
                "worst_pair_tolerable_v0_error_pixels":
                    (None if worst is None or
                     not np.isfinite(worst.tolerable_vanishing_line_error_pixels)
                     else round(worst.tolerable_vanishing_line_error_pixels,
                                constants.TRACE_COARSE_DECIMAL_PLACES)),
        }

    def _build_success_output(self,
                              started_at: float,
                              report: ConditionReport,
                              scene: PreparedScene,
                              estimate: VanishingPointEstimate,
                              analysis: HeightRatioAnalysis,
                              metadata,
                              steps: list[dict]) -> EngineOutput:
        """Score the analysis and assemble the final output.

        Args:
            started_at: perf_counter value taken at entry.
            report: Verdict of the pre-computation gate.
            scene: The prepared scene.
            estimate: The accepted vanishing-point estimate.
            analysis: The height-ratio analysis.
            metadata: Image metadata, selecting the calibration bucket.
            steps: Trace accumulated so far, appended to in place.

        Returns:
            Fully populated EngineOutput.
        """
        raw_score, probability, confidence, is_reliable, note = \
            self._score_analysis(report, estimate, analysis, metadata, steps)
        return EngineOutput(
            engine_name=constants.ENGINE_NAME,
            raw_score=float(raw_score),
            # Contract: probability is defined only when the vote is reliable.
            probability=(clip_to_unit_interval(probability) if is_reliable
                         else None),
            confidence=confidence,
            is_reliable=is_reliable,
            reliability_note=note,
            evidence_map=self._render_evidence(scene, estimate, analysis),
            flagged_regions=self._collect_flagged_regions(scene, analysis),
            computation_steps=steps,
            processing_time_ms=self._elapsed_milliseconds(started_at),
            skill_version=constants.SKILL_VERSION,
        )

    def _score_analysis(self,
                        report: ConditionReport,
                        estimate: VanishingPointEstimate,
                        analysis: HeightRatioAnalysis,
                        metadata,
                        steps: list[dict]) -> tuple:
        """Reduce the analysis to a score, probability and confidence.

        Args:
            report: Verdict of the pre-computation gate.
            estimate: The accepted vanishing-point estimate.
            analysis: The height-ratio analysis.
            metadata: Image metadata, selecting the calibration bucket.
            steps: Trace to append the calibration record to.

        Returns:
            Tuple of (raw_score, probability, confidence, is_reliable, note).
        """
        checks = self._post_computation_checks(estimate, analysis)
        raw_score = self.scorer.reduce_analysis_to_scalar(analysis)
        probability, route, is_calibrated, calibration_note = \
            self.scorer.to_probability(raw_score,
                                       metadata.estimated_compression_level)

        confidence = self._compose_confidence(report, checks, is_calibrated)
        steps.append(self._describe_calibration(raw_score, probability, route,
                                                is_calibrated, calibration_note,
                                                confidence))

        notes = [report.reliability_note]
        notes.extend(note for _, _, note in checks if note)
        notes.append(calibration_note)
        return (raw_score, probability, confidence,
                report.is_reliable and all(passed for passed, _, _ in checks),
                " ".join(note for note in notes if note))

    def _post_computation_checks(self,
                                 estimate: VanishingPointEstimate,
                                 analysis: HeightRatioAnalysis) -> list:
        """Run the conditions judgeable only once the analysis exists.

        Args:
            estimate: The accepted vanishing-point estimate.
            analysis: The height-ratio analysis.

        Returns:
            List of CheckResult tuples.
        """
        return [
            self.condition_checker.assess_vanishing_point_confidence(estimate),
            self.condition_checker.assess_object_pairs(analysis),
            self.condition_checker.assess_vanishing_line_precision(analysis,
                                                                   estimate),
            self.condition_checker.assess_expected_ratio_provenance(
                analysis, bool(self.calibration.supplied_regions)),
        ]

    @staticmethod
    def _corroboration_key_values(analysis: HeightRatioAnalysis) -> dict:
        """Trace fields for ENHANCEMENT 1's corroborated statistic.

        Args:
            analysis: The height-ratio analysis.

        Returns:
            Dictionary of named scalars.
        """
        return {
            "corroborated_consistency":
                round(analysis.corroborated_consistency,
                      constants.TRACE_DECIMAL_PLACES),
            "worst_object_id": analysis.worst_object_id,
            "worst_object_partner_count": analysis.worst_object_partner_count,
        }

    @staticmethod
    def _compose_confidence(report: ConditionReport,
                            checks: list,
                            is_calibrated: bool) -> float:
        """Fold every confidence penalty into one fusion weight.

        Args:
            report: Verdict of the pre-computation gate.
            checks: Post-computation check results.
            is_calibrated: Whether measured calibration data backed the result.

        Returns:
            Composed weight in [0, 1].
        """
        penalties = [report.confidence_weight]
        penalties.extend(penalty for _, penalty, _ in checks)
        if not is_calibrated:
            # The paper's own calibration rests on eight example images, so it
            # is treated as a seed rather than a measured mapping.
            penalties.append(constants.CONFIDENCE_PENALTY_PAPER_CALIBRATION)
        return compose_confidence_penalties(penalties)

    @staticmethod
    def _describe_calibration(raw_score: float,
                              probability: float,
                              route: str,
                              is_calibrated: bool,
                              calibration_note: str,
                              confidence: float) -> dict:
        """Record the reduction and calibration stage in the trace.

        Args:
            raw_score: 1 - C_min, published as raw_score.
            probability: Calibrated probability.
            route: Calibration route that was used.
            is_calibrated: Whether measured calibration data backed the result.
            calibration_note: Explanation from the scorer.
            confidence: Final composed confidence weight.

        Returns:
            One computation-step dictionary.
        """
        return build_computation_step(
            step_number=constants.COMPUTATION_STEP_CALIBRATION,
            name="Score reduction and calibration",
            description=(f"Reduced the per-pair consistency scores to a single "
                         f"tampering score as 1 - C_min, the corpus's own "
                         f"recommended reduction, then mapped it to a "
                         f"probability by the '{route}' route. "
                         f"{calibration_note}"),
            input_shape="per-pair consistency scores",
            output_shape="scalar probability",
            key_values={"raw_score": round(raw_score, constants.TRACE_SCORE_DECIMAL_PLACES),
                        "probability": round(probability, constants.TRACE_SCORE_DECIMAL_PLACES),
                        "calibration_route": route,
                        "is_calibrated": is_calibrated,
                        "paper_threshold":
                            constants.CONSISTENCY_DECISION_THRESHOLD,
                        "confidence": round(confidence, constants.TRACE_DECIMAL_PLACES)},
        )

    @staticmethod
    def _collect_flagged_regions(scene: PreparedScene,
                                 analysis: HeightRatioAnalysis) -> Optional[list]:
        """Report the regions whose pair consistency fell below the threshold.

        Args:
            scene: The prepared scene, for region geometry.
            analysis: The height-ratio analysis.

        Returns:
            List of FlaggedRegion, or None when every pair was consistent.
        """
        by_identifier = {region.identifier: region for region in scene.regions}
        flagged: dict = {}

        for measurement in analysis.measurements:
            if measurement.is_consistent:
                continue
            score = 1.0 - measurement.consistency
            for identifier, partner in ((measurement.first_region_id,
                                         measurement.second_region_id),
                                        (measurement.second_region_id,
                                         measurement.first_region_id)):
                if identifier in by_identifier and \
                        score > flagged.get(identifier, (0.0, 0))[0]:
                    flagged[identifier] = (score, partner)

        regions = [GeometryEngine._build_flagged_region(by_identifier[key],
                                                        value[0], value[1])
                   for key, value in flagged.items()]
        regions.sort(key=lambda item: item.tampering_score, reverse=True)
        return regions or None

    @staticmethod
    def _build_flagged_region(region,
                              tampering_score: float,
                              paired_with: int) -> FlaggedRegion:
        """Convert one inconsistent region into a reportable record.

        Args:
            region: The ObjectRegion that was flagged.
            tampering_score: 1 - C for its worst pair.
            paired_with: Identifier of the region it clashed with.

        Returns:
            FlaggedRegion in pixel coordinates.
        """
        return FlaggedRegion(
            top_left=(int(region.top_row), int(region.left_column)),
            height=int(region.bottom_row - region.top_row),
            width=int(region.right_column - region.left_column),
            tampering_score=float(tampering_score),
            paired_with=int(paired_with),
        )

    def _render_evidence(self,
                         scene: PreparedScene,
                         estimate: VanishingPointEstimate,
                         analysis: HeightRatioAnalysis) -> Optional[np.ndarray]:
        """Render the annotated geometry overlay.

        Args:
            scene: The prepared scene.
            estimate: The accepted vanishing-point estimate.
            analysis: The height-ratio analysis.

        Returns:
            BGR evidence image, or None when rendering failed.
        """
        try:
            return self.visualizer.render_evidence_map(
                scene.colour_image.astype(np.uint8), estimate, scene.regions,
                analysis)
        except (ValueError, IndexError, TypeError) as error:
            # Visualisation must never invalidate a completed measurement.
            logger.warning("evidence map rendering failed: %s", error)
            return None

    def _build_abstain_output(self,
                              started_at: float,
                              report: ConditionReport,
                              gates: list,
                              steps: list[dict]) -> EngineOutput:
        """Assemble the null vote returned when the geometry gates fire.

        SKILL "Output" requires exactly this behaviour: a low-confidence
        vanishing point "should cause the module to abstain rather than emit a
        possibly-spurious height-ratio score".

        Args:
            started_at: perf_counter value taken at entry.
            report: Verdict of the pre-computation gate.
            gates: The post-estimation check results that fired.
            steps: Trace accumulated so far.

        Returns:
            EngineOutput carrying no measurement.
        """
        reasons = [note for passed, _, note in gates if not passed and note]
        note = " ".join([report.reliability_note] + reasons)
        logger.info("geometry engine abstained: %s", " ".join(reasons))
        return self._null_output(started_at, note, steps)

    def _build_skip_output(self,
                           started_at: float,
                           report: ConditionReport,
                           steps: list[dict]) -> EngineOutput:
        """Assemble the null vote returned when the engine's premise fails.

        Args:
            started_at: perf_counter value taken at entry.
            report: Verdict of the pre-computation gate.
            steps: Trace accumulated so far.

        Returns:
            EngineOutput carrying no measurement.
        """
        logger.info("geometry engine skipped: %s", report.reliability_note)
        return self._null_output(started_at, report.reliability_note, steps)

    def _null_output(self,
                     started_at: float,
                     note: str,
                     steps: list[dict]) -> EngineOutput:
        """Build an EngineOutput carrying no measurement.

        Args:
            started_at: perf_counter value taken at entry.
            note: Explanation for the report.
            steps: Trace accumulated so far.

        Returns:
            EngineOutput with probability None and zero confidence.
        """
        return EngineOutput(
            engine_name=constants.ENGINE_NAME,
            raw_score=0.0,
            probability=None,
            confidence=constants.ZERO_CONFIDENCE,
            is_reliable=False,
            reliability_note=note,
            evidence_map=None,
            flagged_regions=None,
            computation_steps=steps,
            processing_time_ms=self._elapsed_milliseconds(started_at),
            skill_version=constants.SKILL_VERSION,
        )

    def _build_failure_output(self,
                              started_at: float,
                              error_description: str,
                              input_description: str) -> EngineOutput:
        """Assemble the output for an unexpected internal failure.

        Args:
            started_at: perf_counter value taken at entry.
            error_description: Exception type and message.
            input_description: What was passed in, for diagnosis.

        Returns:
            EngineOutput describing the failure.
        """
        step = build_computation_step(
            step_number=constants.COMPUTATION_STEP_FAILURE,
            name="Failure",
            description=("The engine raised an unexpected exception and "
                         "returned a null vote so the fusion layer can proceed "
                         "without it."),
            input_shape=input_description,
            output_shape="none",
            key_values={"error": error_description},
        )
        return self._null_output(
            started_at,
            f"Geometry engine failed and returned no measurement. "
            f"Error: {error_description}. Input: {input_description}.",
            [step])

    @staticmethod
    def _describe_input(engine_input: Optional[EngineInput]) -> str:
        """Summarise the input for a failure note.

        Args:
            engine_input: The input under analysis, possibly malformed.

        Returns:
            Readable one-line description.
        """
        try:
            return (f"image {describe_array_shape(engine_input.image)}, "
                    f"format {engine_input.metadata.format}, "
                    f"quality {engine_input.metadata.estimated_compression_level}")
        except AttributeError:
            return f"unreadable EngineInput ({type(engine_input).__name__})"

    @staticmethod
    def _elapsed_milliseconds(started_at: float) -> float:
        """Wall-clock milliseconds since the given start time.

        Args:
            started_at: perf_counter value taken at entry.

        Returns:
            Elapsed duration in milliseconds.
        """
        return (time.perf_counter() - started_at) * constants.MILLISECONDS_PER_SECOND
