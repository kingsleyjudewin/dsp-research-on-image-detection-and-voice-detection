"""Entry point for the CFA / demosaicing artifact forgery-detection engine.

Orchestrates the four processing stages in order:

    Stage 1  input validation and condition checking   (condition.py)
    Stage 2  preprocessing                             (preprocessor.py)
    Stage 3  core mathematical computation             (computer.py)
    Stage 4  score extraction and output assembly      (scorer.py, visualizer.py)

Stage 3 runs three of the four pipelines the SKILL documents: Pipeline C first
to fix the CFA phase, then Pipeline A as the primary measurement, then Pipeline
B as a confirmatory layer with false-alarm guarantees. Pipeline D (lateral
chromatic aberration) is deliberately not folded into this vote - the SKILL
describes it as "a signal from lens optics rather than sensor demosaicing"
and "fusable as an independent detector", so combining it here would give the
fusion layer one correlated vote where it should receive two independent ones.

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
from .computer import (CfaLikelihoodComputer, CfaPhaseEstimator,
                       GridConsistencyComputer)
from .condition import ConditionChecker
from .contracts import (CalibrationSettings, CfaComputation, ConditionReport,
                        EngineInput, EngineOutput, FlaggedRegion, PreparedImage)
from .preprocessor import CfaPreprocessor
from .scorer import CfaScorer
from .utils import (build_computation_step, clip_to_unit_interval,
                    compose_confidence_penalties, describe_array_shape,
                    find_connected_components)
from .visualizer import CfaVisualizer

logger = logging.getLogger(__name__)


class CfaEngine:
    """Detects forgery from breaks in the camera's demosaicing correlation."""

    def __init__(self,
                 calibration: Optional[CalibrationSettings] = None,
                 feature_block_size: Optional[int] = None,
                 enable_grid_consistency_layer: bool = True) -> None:
        """Build the engine and its collaborators.

        Args:
            calibration: Optional calibration state. Without it the engine
                still runs, but reports its probability as provisional.
            feature_block_size: Override for Ferrara's block size B. Defaults
                to the corpus-recommended direct block size.
            enable_grid_consistency_layer: Whether to run Bammey's confirmatory
                a contrario layer. It roughly triples runtime, so it can be
                switched off for bulk triage.
        """
        self.condition_checker = ConditionChecker()
        self.phase_estimator = CfaPhaseEstimator()
        self.computer = CfaLikelihoodComputer(feature_block_size)
        self.preprocessor = CfaPreprocessor(self.computer.feature_block_size)
        self.grid_computer = (GridConsistencyComputer()
                              if enable_grid_consistency_layer else None)
        self.scorer = CfaScorer(calibration)
        self.visualizer = CfaVisualizer()

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
            logger.exception("CFA engine failed during analysis")
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
            EngineOutput for a completed or deliberately skipped analysis.
        """
        image = engine_input.image
        metadata = engine_input.metadata

        # Stage 1 - condition checking runs before any pixel is transformed.
        report = self.condition_checker.check(metadata, image)
        steps = [self._describe_condition_check(image, metadata, report)]
        if report.skip_engine:
            return self._build_skip_output(started_at, report, steps)

        # Stage 2.
        prepared = self.preprocessor.prepare(image)
        steps.append(self._describe_preprocessing(image, prepared))

        # Stage 3.
        computation = self._compute(prepared, steps)

        # Stage 4.
        return self._build_success_output(started_at, report, prepared,
                                          computation, metadata, steps)

    def _compute(self, prepared: PreparedImage, steps: list[dict]) -> CfaComputation:
        """Run Pipelines C, A and B in that order, tracing each.

        Args:
            prepared: Output of Stage 2.
            steps: Trace to append to, in place.

        Returns:
            CfaComputation carrying every intermediate map.
        """
        phase = self.phase_estimator.estimate(prepared.colour_image)
        steps.append(self._describe_phase_estimation(prepared, phase))

        computation = self.computer.compute(prepared, phase)
        steps.extend(self._describe_likelihood_pipeline(prepared, computation))

        if self.grid_computer is not None:
            computation.grid_consistency = self.grid_computer.analyse(
                prepared.colour_image)
            steps.append(self._describe_grid_consistency(prepared, computation))
        return computation

    @staticmethod
    def _describe_condition_check(image: np.ndarray,
                                  metadata,
                                  report: ConditionReport) -> dict:
        """Record the condition-checking stage in the computation trace.

        Args:
            image: Original BGR image.
            metadata: Image metadata that was evaluated.
            report: Verdict of the gate.

        Returns:
            One computation-step dictionary.
        """
        return build_computation_step(
            step_number=constants.COMPUTATION_STEP_CONDITION_CHECK,
            name="Condition check",
            description=("Evaluated every unreliability condition documented in "
                         "the SKILL file - JPEG quality against Ferrara's 95% "
                         "reliability floor and 85% discrimination floor, prior "
                         "resampling, colour-plane availability, saturation and "
                         "resolution - before any computation ran."),
            input_shape=describe_array_shape(image),
            output_shape="condition report",
            key_values={
                "is_reliable": report.is_reliable,
                "skip_engine": report.skip_engine,
                "confidence_weight": round(report.confidence_weight, 4),
                "estimated_jpeg_quality": metadata.estimated_compression_level,
                "is_resized": metadata.is_resized,
                "container_format": metadata.format,
            },
        )

    def _describe_preprocessing(self,
                                image: np.ndarray,
                                prepared: PreparedImage) -> dict:
        """Record the preprocessing stage in the computation trace.

        Args:
            image: Original BGR image.
            prepared: Output of Stage 2.

        Returns:
            One computation-step dictionary.
        """
        return build_computation_step(
            step_number=constants.COMPUTATION_STEP_PREPROCESSING,
            name="Channel extraction and texture screening",
            description=(
                f"Extracted the {constants.ANALYSIS_CHANNEL_NAME} channel, the "
                f"only channel Ferrara et al. operate on, cropped the image to "
                f"a whole Bayer grid from the bottom and right so the CFA phase "
                f"is preserved, and screened out blocks that are almost flat or "
                f"edge-dominated - the two scene conditions Ferrara et al. name "
                f"as defeating the method."),
            input_shape=describe_array_shape(image),
            output_shape=describe_array_shape(prepared.green_channel),
            key_values={
                "analysis_channel": constants.ANALYSIS_CHANNEL_NAME,
                "feature_block_size": self.computer.feature_block_size,
                "excluded_block_fraction":
                    round(prepared.excluded_block_fraction, 4),
                "flat_variance_threshold":
                    constants.FLAT_BLOCK_VARIANCE_THRESHOLD,
                "sharp_edge_gradient_threshold":
                    constants.SHARP_EDGE_GRADIENT_THRESHOLD,
            },
        )

    @staticmethod
    def _describe_phase_estimation(prepared: PreparedImage, phase) -> dict:
        """Record Pipeline C in the computation trace.

        Args:
            prepared: Output of Stage 2.
            phase: The CFA phase estimate.

        Returns:
            One computation-step dictionary.
        """
        return build_computation_step(
            step_number=constants.COMPUTATION_STEP_PHASE_ESTIMATION,
            name="CFA phase estimation (Pipeline C)",
            description=(
                "Determined which of the four Bayer configurations the image "
                "was captured on, using the SVD colour-difference estimator of "
                "Jeon et al. 2017 rather than trusting EXIF, which may be "
                "stripped or wrong. This fixes which lattice holds acquired "
                "green samples; identifying it with the wrong parity would "
                "invert the sign of every feature downstream. " + phase.note),
            input_shape=describe_array_shape(prepared.colour_image),
            output_shape="CFA configuration label",
            key_values={
                "configuration": phase.configuration_name,
                "green_acquired_parity": phase.green_acquired_parity,
                "estimation_block_size": phase.block_size,
                "was_verified_by_svd": phase.was_estimated,
                "diagonal_scores": [round(score, 3)
                                    for score in phase.diagonal_scores],
            },
        )

    def _describe_likelihood_pipeline(self,
                                      prepared: PreparedImage,
                                      computation: CfaComputation) -> list[dict]:
        """Record the two mathematical stages of Pipeline A in the trace.

        Args:
            prepared: Output of Stage 2.
            computation: Result of the mathematical core.

        Returns:
            Two computation-step dictionaries.
        """
        return [
            self._describe_feature_stage(prepared, computation),
            self._describe_posterior_stage(computation),
        ]

    @staticmethod
    def _describe_feature_stage(prepared: PreparedImage,
                                computation: CfaComputation) -> dict:
        """Record the prediction-error and mixture-fitting stage.

        Args:
            prepared: Output of Stage 2.
            computation: Result of the mathematical core.

        Returns:
            One computation-step dictionary.
        """
        mixture = computation.mixture
        return build_computation_step(
            step_number=constants.COMPUTATION_STEP_PREDICTION_ERROR,
            name="Prediction error, local variance and mixture fit",
            description=(
                "Predicted every green pixel from its four nearest neighbours "
                "with the fixed bilinear kernel (Eq. 9), measured the "
                "lattice-masked locally-weighted variance of the residual "
                "(Eq. 10), and reduced each block to the log ratio of the "
                "geometric-mean variance at acquired versus interpolated "
                "positions (Eq. 11-12). A two-component Gaussian mixture was "
                "then fitted by expectation-maximization, with the tampered "
                "component's mean held at zero exactly as Eq. 14 specifies."),
            input_shape=describe_array_shape(prepared.green_channel),
            output_shape=describe_array_shape(computation.feature_map),
            key_values={
                "authentic_component_mean": round(mixture.authentic_mean, 4),
                "authentic_component_variance":
                    round(mixture.authentic_variance, 4),
                "tampered_component_variance":
                    round(mixture.tampered_variance, 4),
                "em_mixing_weight": round(mixture.mixing_weight, 4),
                "em_iterations": mixture.iterations,
                "em_converged": mixture.converged,
                "valid_block_count": computation.valid_block_count,
                "eq13_requires_positive_mean": True,
            },
        )

    @staticmethod
    def _describe_posterior_stage(computation: CfaComputation) -> dict:
        """Record the Bayesian posterior and map-filtering stage.

        Args:
            computation: Result of the mathematical core.

        Returns:
            One computation-step dictionary.
        """
        tampered_blocks = int(np.count_nonzero(
            (computation.tampering_map >
             constants.TAMPERED_BLOCK_PROBABILITY_THRESHOLD)
            & computation.block_validity_mask))
        return build_computation_step(
            step_number=constants.COMPUTATION_STEP_POSTERIOR_MAP,
            name="Posterior likelihood map (Pipeline A)",
            description=(
                f"Converted each block's feature into the posterior probability "
                f"of authenticity by Bayes' rule with the equal priors Eq. 15-16 "
                f"prescribe, then applied the "
                f"{constants.MAP_FILTER_SIZE}x{constants.MAP_FILTER_SIZE} "
                f"{constants.MAP_FILTER_RULE} filter of step 8, which the "
                f"corpus reports outperforms a mean filter. The map published "
                f"here is the tampering score 1 - Pr(M1|L), so 0 means "
                f"confidently authentic and 1 confidently tampered."),
            input_shape=describe_array_shape(computation.feature_map),
            output_shape=describe_array_shape(computation.tampering_map),
            key_values={
                "output_block_size": computation.output_block_size,
                "blocks_above_tampering_threshold": tampered_blocks,
                "tampering_threshold":
                    constants.TAMPERED_BLOCK_PROBABILITY_THRESHOLD,
                "map_filter": f"{constants.MAP_FILTER_SIZE}x"
                              f"{constants.MAP_FILTER_SIZE} "
                              f"{constants.MAP_FILTER_RULE}",
                "mean_tampering_probability":
                    round(float(np.mean(computation.tampering_map)), 4),
            },
        )

    @staticmethod
    def _describe_grid_consistency(prepared: PreparedImage,
                                   computation: CfaComputation) -> dict:
        """Record Pipeline B in the computation trace.

        Args:
            prepared: Output of Stage 2.
            computation: Result of the mathematical core.

        Returns:
            One computation-step dictionary.
        """
        grid = computation.grid_consistency
        return build_computation_step(
            step_number=constants.COMPUTATION_STEP_GRID_CONSISTENCY,
            name="Grid-position consistency (Pipeline B)",
            description=(
                "Re-mosaiced the image under each of the four candidate CFA "
                "grid positions, estimated the eight inter-channel demosaicing "
                "filters by least squares for each, and had every block vote "
                "for the position with the smallest reconstruction residual. "
                "The votes were then tested against an a contrario null model "
                "in which blocks vote at random with probability 1/4, giving a "
                "Number of False Alarms - a bound Pipeline A does not provide. "
                + grid.note),
            input_shape=describe_array_shape(prepared.colour_image),
            output_shape=f"{grid.window_count} windows tested",
            key_values={
                "dominant_grid_position": grid.dominant_position_index,
                "dominant_log10_nfa": round(grid.dominant_log10_nfa, 2),
                "disagreeing_windows": len(grid.forged_windows),
                "windows_tested": grid.window_count,
                "vote_block_size": constants.GRID_VOTE_BLOCK_SIZE,
                "nfa_detection_threshold_log10":
                    constants.NFA_DETECTION_LOG10_THRESHOLD,
                "is_conclusive": grid.is_conclusive,
            },
        )

    def _build_success_output(self,
                              started_at: float,
                              report: ConditionReport,
                              prepared: PreparedImage,
                              computation: CfaComputation,
                              metadata,
                              steps: list[dict]) -> EngineOutput:
        """Score the computation and assemble the final output.

        Args:
            started_at: perf_counter value taken at entry.
            report: Verdict of the pre-computation gate.
            prepared: Output of Stage 2.
            computation: Result of the mathematical core.
            metadata: Image metadata, selecting the calibration bucket.
            steps: Trace accumulated so far, appended to in place.

        Returns:
            Fully populated EngineOutput.
        """
        raw_score, probability, confidence, is_reliable, note = \
            self._score_computation(report, prepared, computation, metadata,
                                    steps)
        return EngineOutput(
            engine_name=constants.ENGINE_NAME,
            raw_score=float(raw_score),
            # Contract: probability is defined only when the vote is reliable.
            probability=(clip_to_unit_interval(probability) if is_reliable
                         else None),
            confidence=confidence,
            is_reliable=is_reliable,
            reliability_note=note,
            evidence_map=self._render_evidence(computation),
            flagged_regions=self._collect_flagged_regions(computation),
            computation_steps=steps,
            processing_time_ms=self._elapsed_milliseconds(started_at),
            skill_version=constants.SKILL_VERSION,
        )

    def _score_computation(self,
                           report: ConditionReport,
                           prepared: PreparedImage,
                           computation: CfaComputation,
                           metadata,
                           steps: list[dict]) -> tuple:
        """Reduce the computation to a score, probability and confidence.

        Args:
            report: Verdict of the pre-computation gate.
            prepared: Output of Stage 2.
            computation: Result of the mathematical core.
            metadata: Image metadata, selecting the calibration bucket.
            steps: Trace to append the calibration record to.

        Returns:
            Tuple of (raw_score, probability, confidence, is_reliable, note).
        """
        checks = self._run_post_computation_checks(prepared, computation)
        raw_score, rule = self.scorer.reduce_map_to_scalar(
            computation.tampering_map, computation.block_validity_mask)
        probability, route, is_calibrated, calibration_note = \
            self.scorer.to_probability(raw_score,
                                       metadata.estimated_compression_level)

        confidence = self._compose_confidence(report, checks, computation,
                                              is_calibrated)
        steps.append(self._describe_calibration(raw_score, rule, probability,
                                                route, is_calibrated,
                                                calibration_note, confidence))

        notes = [report.reliability_note] + [note for _, _, note in checks if note]
        notes.append(calibration_note)
        is_reliable = report.is_reliable and all(passed for passed, _, _ in checks)
        return (raw_score, probability, confidence, is_reliable,
                " ".join(note for note in notes if note))

    def _run_post_computation_checks(self,
                                     prepared: PreparedImage,
                                     computation: CfaComputation) -> list:
        """Run the conditions that can only be judged after computing.

        Args:
            prepared: Output of Stage 2.
            computation: Result of the mathematical core.

        Returns:
            List of CheckResult tuples.
        """
        return [
            self.condition_checker.assess_texture_adequacy(
                prepared.excluded_block_fraction),
            self.condition_checker.assess_sample_quality(
                computation.valid_block_count),
            self.condition_checker.assess_global_cfa_presence(
                computation.mixture),
        ]

    @staticmethod
    def _compose_confidence(report: ConditionReport,
                            checks: list,
                            computation: CfaComputation,
                            is_calibrated: bool) -> float:
        """Fold every confidence penalty into one fusion weight.

        Args:
            report: Verdict of the pre-computation gate.
            checks: Post-computation check results.
            computation: Result of the mathematical core.
            is_calibrated: Whether measured calibration data backed the result.

        Returns:
            Composed weight in [0, 1].
        """
        penalties = [report.confidence_weight]
        penalties.extend(penalty for _, penalty, _ in checks)
        if not is_calibrated:
            # An uncalibrated probability expresses ordering, not likelihood,
            # so the fusion layer is told to trust it less.
            penalties.append(constants.CONFIDENCE_PENALTY_UNCALIBRATED)
        if not computation.phase.was_estimated:
            penalties.append(constants.CONFIDENCE_PENALTY_PHASE_UNVERIFIED)
        if (computation.grid_consistency is not None
                and not computation.grid_consistency.is_conclusive):
            penalties.append(constants.CONFIDENCE_PENALTY_GRID_LAYER_INCONCLUSIVE)
        return compose_confidence_penalties(penalties)

    @staticmethod
    def _describe_calibration(raw_score: float,
                              reduction_rule: str,
                              probability: float,
                              route: str,
                              is_calibrated: bool,
                              calibration_note: str,
                              confidence: float) -> dict:
        """Record the reduction and calibration stage in the trace.

        Args:
            raw_score: Reduced scalar published as raw_score.
            reduction_rule: Which of the SKILL's three reductions was applied.
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
            name="Map reduction and calibration",
            description=(f"Reduced the per-block tampering map to a scalar with "
                         f"the '{reduction_rule}' rule, one of the three the "
                         f"SKILL file names, then mapped it onto a probability "
                         f"via the '{route}' route. {calibration_note}"),
            input_shape="per-block tampering map",
            output_shape="scalar probability",
            key_values={"raw_score": round(raw_score, 6),
                        "reduction_rule": reduction_rule,
                        "reduction_percentile":
                            constants.MAP_REDUCTION_PERCENTILE,
                        "probability": round(probability, 6),
                        "calibration_route": route,
                        "is_calibrated": is_calibrated,
                        "confidence": round(confidence, 4)},
        )

    def _collect_flagged_regions(self,
                                 computation: CfaComputation) -> Optional[list]:
        """Group blocks above the tampering threshold into suspect regions.

        Args:
            computation: Result of the mathematical core.

        Returns:
            List of FlaggedRegion, or None when nothing was flagged. Unlike the
            global-only engines this one localises, so this field is populated
            whenever the map has suspect blocks.
        """
        suspect = ((computation.tampering_map >
                    constants.TAMPERED_BLOCK_PROBABILITY_THRESHOLD)
                   & computation.block_validity_mask)
        if not np.any(suspect):
            return None

        regions = [
            self._build_region(component, computation)
            for component in find_connected_components(suspect)
            if len(component) >= constants.MINIMUM_FLAGGED_REGION_BLOCK_COUNT
        ]
        if not regions:
            return None
        regions.sort(key=lambda region: region.mean_tampering_probability,
                     reverse=True)
        return regions

    @staticmethod
    def _build_region(component: list,
                      computation: CfaComputation) -> FlaggedRegion:
        """Turn one connected component of flagged blocks into a pixel region.

        Args:
            component: List of (block row, block column) index pairs.
            computation: Result of the mathematical core.

        Returns:
            FlaggedRegion in pixel coordinates of the cropped image.
        """
        rows = [row for row, _ in component]
        columns = [column for _, column in component]
        block_size = computation.output_block_size
        probabilities = [float(computation.tampering_map[row, column])
                         for row, column in component]

        return FlaggedRegion(
            top_left=(min(rows) * block_size, min(columns) * block_size),
            height=(max(rows) - min(rows) + 1) * block_size,
            width=(max(columns) - min(columns) + 1) * block_size,
            mean_tampering_probability=float(np.mean(probabilities)),
            block_count=len(component),
        )

    def _render_evidence(self,
                         computation: CfaComputation) -> Optional[np.ndarray]:
        """Render the per-block posterior as a heatmap.

        Args:
            computation: Result of the mathematical core.

        Returns:
            BGR evidence image, or None when rendering failed.
        """
        try:
            return self.visualizer.render_evidence_map(
                computation.tampering_map,
                computation.block_validity_mask,
                computation.output_block_size)
        except (ValueError, IndexError) as error:
            # Visualisation must never invalidate a completed measurement.
            logger.warning("evidence map rendering failed: %s", error)
            return None

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
        logger.info("CFA engine skipped: %s", report.reliability_note)
        return EngineOutput(
            engine_name=constants.ENGINE_NAME,
            raw_score=0.0,
            probability=None,
            confidence=constants.ZERO_CONFIDENCE,
            is_reliable=False,
            reliability_note=report.reliability_note,
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
        return EngineOutput(
            engine_name=constants.ENGINE_NAME,
            raw_score=0.0,
            probability=None,
            confidence=constants.ZERO_CONFIDENCE,
            is_reliable=False,
            reliability_note=(f"CFA engine failed and returned no measurement. "
                              f"Error: {error_description}. "
                              f"Input: {input_description}."),
            evidence_map=None,
            flagged_regions=None,
            computation_steps=[build_computation_step(
                step_number=constants.COMPUTATION_STEP_FAILURE,
                name="Failure",
                description=("The engine raised an unexpected exception and "
                             "returned a null vote so the fusion layer can "
                             "proceed without it."),
                input_shape=input_description,
                output_shape="none",
                key_values={"error": error_description},
            )],
            processing_time_ms=self._elapsed_milliseconds(started_at),
            skill_version=constants.SKILL_VERSION,
        )

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
