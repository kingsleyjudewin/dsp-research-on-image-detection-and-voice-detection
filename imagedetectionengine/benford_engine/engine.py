"""Entry point for the Benford's Law forgery-detection engine.

Orchestrates the four processing stages in order:

    Stage 1  input validation and condition checking   (condition.py)
    Stage 2  preprocessing                             (preprocessor.py)
    Stage 3  core mathematical computation             (computer.py)
    Stage 4  score extraction and output assembly      (scorer.py, visualizer.py)

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
from .computer import BenfordComputer
from .condition import ConditionChecker
from .contracts import (BenfordComputation, CalibrationSettings, ConditionReport,
                        EngineInput, EngineOutput)
from .preprocessor import BenfordPreprocessor
from .scorer import BenfordScorer
from .utils import (build_computation_step, clip_to_unit_interval,
                    compose_confidence_penalties, describe_array_shape)
from .visualizer import BenfordVisualizer

logger = logging.getLogger(__name__)


class BenfordEngine:
    """Detects forgery by measuring departure from Benford's Law in DCT digits."""

    def __init__(self,
                 calibration: Optional[CalibrationSettings] = None,
                 computer: Optional[BenfordComputer] = None) -> None:
        """Build the engine and its collaborators.

        Args:
            calibration: Optional calibration state. Without it the engine
                still runs, but reports its probability as provisional.
            computer: Optional pre-configured computer, for sweeping a
                non-default grid. Defaults to the corpus-recommended grid.
        """
        self.condition_checker = ConditionChecker()
        self.preprocessor = BenfordPreprocessor()
        self.computer = computer or BenfordComputer()
        self.scorer = BenfordScorer(calibration)
        self.visualizer = BenfordVisualizer()

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
            logger.exception("Benford engine failed during analysis")
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

        # Stages 2 and 3.
        blocks = self.preprocessor.prepare(image)
        steps.append(self._describe_preprocessing(image, blocks))

        computation = self.computer.compute(
            blocks, estimated_quality_factor=metadata.estimated_compression_level
        )
        steps.extend(self._describe_computation(blocks, computation))

        # Stage 4.
        return self._build_success_output(started_at, report, computation,
                                          metadata.estimated_compression_level,
                                          steps)

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
                         "the SKILL file against the image's container format, "
                         "estimated quality factor, resolution, resampling flag "
                         "and saturation before any computation ran."),
            input_shape=describe_array_shape(image),
            output_shape="condition report",
            key_values={
                "is_reliable": report.is_reliable,
                "skip_engine": report.skip_engine,
                "confidence_weight": round(report.confidence_weight, 4),
                "estimated_quality_factor": metadata.estimated_compression_level,
                "container_format": metadata.format,
            },
        )

    def _describe_preprocessing(self,
                                image: np.ndarray,
                                blocks: np.ndarray) -> dict:
        """Record the preprocessing stage in the computation trace.

        Args:
            image: Original BGR image.
            blocks: Prepared block array.

        Returns:
            One computation-step dictionary.
        """
        return build_computation_step(
            step_number=constants.COMPUTATION_STEP_PREPROCESSING,
            name="Channel selection and blocking",
            description=(f"Extracted the {constants.ANALYSIS_CHANNEL_NAME} "
                         f"channel, the only channel choice stated explicitly in "
                         f"the corpus (Moin & Islam 2017), then cropped to a "
                         f"whole block grid and partitioned into non-overlapping "
                         f"{constants.DCT_BLOCK_SIZE}x{constants.DCT_BLOCK_SIZE} "
                         f"blocks."),
            input_shape=describe_array_shape(image),
            output_shape=describe_array_shape(blocks),
            key_values={
                "analysis_channel": constants.ANALYSIS_CHANNEL_NAME,
                "block_size": constants.DCT_BLOCK_SIZE,
                "block_count": int(blocks.shape[0]),
            },
        )

    def _describe_computation(self,
                              blocks: np.ndarray,
                              computation: BenfordComputation) -> list[dict]:
        """Record the mathematical stages in the computation trace.

        Args:
            blocks: Prepared block array.
            computation: Result of the mathematical core.

        Returns:
            Computation-step dictionaries for the transform and the sweep.
        """
        return [
            build_computation_step(
                step_number=constants.COMPUTATION_STEP_BLOCK_DCT,
                name="Block DCT",
                description=("Applied the JPEG-normalized 2D DCT-II of Thai et "
                             "al. 2012 Eq. 8 to every block, reproducing "
                             "libjpeg-compatible coefficient magnitudes."),
                input_shape=describe_array_shape(blocks),
                output_shape=describe_array_shape(blocks),
                key_values={"dct_type": constants.DCT_TYPE,
                            "normalisation": constants.DCT_NORMALISATION},
            ),
            self._describe_sweep(blocks, computation),
        ]

    def _describe_sweep(self,
                        blocks: np.ndarray,
                        computation: BenfordComputation) -> dict:
        """Record the divergence sweep in the computation trace.

        Args:
            blocks: Prepared block array.
            computation: Result of the mathematical core.

        Returns:
            One computation-step dictionary.
        """
        grid_description = (
            f"bases {list(self.computer.digit_bases)}, zig-zag frequencies "
            f"{list(self.computer.zigzag_frequency_indices)} (DC excluded), "
            f"quality factors {list(self.computer.quality_factor_sweep)}"
        )
        return build_computation_step(
            step_number=constants.COMPUTATION_STEP_DIVERGENCE_SWEEP,
            name="Digit-divergence sweep",
            description=(
                f"For each cell of the sweep grid ({grid_description}), "
                f"quantized the coefficients with the standard JPEG table, "
                f"extracted leading digits, built the empirical pmf, "
                f"least-squares fitted the generalized Benford curve "
                f"p(d)=beta*log_b(1+1/(gamma+d^delta)), and measured the "
                f"unaveraged symmetric KL divergence of Bonettini et al. 2021 "
                f"Eq. 6. Zero-valued coefficients have no leading digit and "
                f"were excluded as the SKILL file instructs."
            ),
            input_shape=describe_array_shape(blocks),
            output_shape=f"{len(computation.samples)} divergence samples",
            key_values=self._sweep_key_values(computation),
        )

    @staticmethod
    def _sweep_key_values(computation: BenfordComputation) -> dict:
        """Collect the headline numbers from the sweep for the trace.

        Includes Moin et al.'s published unaltered chi-square range alongside
        the measured value, so the report generator can state the comparison
        without needing the SKILL file itself.

        Args:
            computation: Result of the mathematical core.

        Returns:
            Dictionary of named scalars.
        """
        return {
            "evaluated_configurations": len(computation.samples),
            "skipped_configurations": computation.skipped_configuration_count,
            "mean_divergence": round(computation.mean_divergence, 6),
            "max_divergence": round(computation.max_divergence, 6),
            "zero_coefficient_rate":
                round(computation.overall_zero_coefficient_rate, 4),
            "representative_chi_square":
                round(computation.representative_chi_square, 6),
            "moin_unaltered_chi_square_reference":
                [constants.MOIN_UNALTERED_CHI_SQUARE_MINIMUM,
                 constants.MOIN_UNALTERED_CHI_SQUARE_MAXIMUM],
            "digit_clamp_events": computation.digit_clamp_event_count,
        }

    def _build_success_output(self,
                              started_at: float,
                              report: ConditionReport,
                              computation: BenfordComputation,
                              estimated_quality_factor: float,
                              steps: list[dict]) -> EngineOutput:
        """Score the computation and assemble the final output.

        Args:
            started_at: perf_counter value taken at entry.
            report: Verdict of the pre-computation gate.
            computation: Result of the mathematical core.
            estimated_quality_factor: Selects the calibration bucket.
            steps: Trace accumulated so far, appended to in place.

        Returns:
            Fully populated EngineOutput.
        """
        raw_score, probability, confidence, is_reliable, note = \
            self._score_computation(report, computation,
                                    estimated_quality_factor, steps)
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
            # This engine is global-only; the corpus records an explicit
            # negative result for localising tampering with digit statistics.
            flagged_regions=None,
            computation_steps=steps,
            processing_time_ms=self._elapsed_milliseconds(started_at),
            skill_version=constants.SKILL_VERSION,
        )

    def _score_computation(self,
                           report: ConditionReport,
                           computation: BenfordComputation,
                           estimated_quality_factor: float,
                           steps: list[dict]) -> tuple:
        """Reduce the computation to a score, probability and confidence.

        Args:
            report: Verdict of the pre-computation gate.
            computation: Result of the mathematical core.
            estimated_quality_factor: Selects the calibration bucket.
            steps: Trace to append the calibration record to.

        Returns:
            Tuple of (raw_score, probability, confidence, is_reliable, note).
        """
        sample_ok, sample_penalty, sample_note = \
            self.condition_checker.assess_sample_quality(
                computation.overall_zero_coefficient_rate,
                len(computation.samples))
        raw_score = self._select_raw_score(computation)
        probability, route, is_calibrated, calibration_note = \
            self.scorer.to_probability(raw_score, estimated_quality_factor)
        confidence = self._compose_confidence(report.confidence_weight,
                                              sample_penalty, is_calibrated)
        steps.append(self._describe_calibration(
            computation, raw_score, probability, route, is_calibrated,
            calibration_note, confidence))

        notes = [report.reliability_note, sample_note, calibration_note]
        return (raw_score, probability, confidence,
                report.is_reliable and sample_ok,
                " ".join(n for n in notes if n))

    @staticmethod
    def _compose_confidence(condition_weight: float,
                            sample_penalty: float,
                            is_calibrated: bool) -> float:
        """Fold every confidence penalty into one fusion weight.

        Args:
            condition_weight: Weight from the pre-computation gate.
            sample_penalty: Penalty from post-transform data quality.
            is_calibrated: Whether measured calibration data backed the result.

        Returns:
            Composed weight in [0, 1].
        """
        penalties = [condition_weight, sample_penalty]
        if not is_calibrated:
            # An uncalibrated probability expresses ordering, not likelihood,
            # so the fusion layer is told to trust it less.
            penalties.append(constants.CONFIDENCE_PENALTY_UNCALIBRATED)
        return compose_confidence_penalties(penalties)

    @staticmethod
    def _describe_calibration(computation: BenfordComputation,
                              raw_score: float,
                              probability: float,
                              route: str,
                              is_calibrated: bool,
                              calibration_note: str,
                              confidence: float) -> dict:
        """Record the calibration stage in the computation trace.

        Args:
            computation: Result of the mathematical core.
            raw_score: Aggregated divergence published as raw_score.
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
            name="Calibration",
            description=(f"Mapped the raw divergence onto a probability using "
                         f"the '{route}' route. {calibration_note}"),
            input_shape="scalar raw score",
            output_shape="scalar probability",
            key_values={"raw_score": round(raw_score, 6),
                        "aggregation_rule": constants.DIVERGENCE_AGGREGATION_RULE,
                        "mean_divergence": round(computation.mean_divergence, 6),
                        "max_divergence": round(computation.max_divergence, 6),
                        "probability": round(probability, 6),
                        "calibration_route": route,
                        "is_calibrated": is_calibrated,
                        "confidence": round(confidence, 4)},
        )

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
        logger.info("Benford engine skipped: %s", report.reliability_note)
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
            reliability_note=(f"Benford engine failed and returned no "
                              f"measurement. Error: {error_description}. "
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
    def _select_raw_score(computation: BenfordComputation) -> float:
        """Reduce the divergence grid to the single scalar the SKILL prescribes.

        SKILL family A step 7 sanctions either aggregation - "threshold a single
        scalar - e.g. the mean or max divergence across the swept (b,n,Delta)
        grid". The choice between them lives in
        constants.DIVERGENCE_AGGREGATION_RULE, where the measurement that
        settled it is recorded.

        Args:
            computation: Result of the mathematical core.

        Returns:
            The aggregated divergence to publish as raw_score.

        Raises:
            ValueError: If the configured rule is not recognised.
        """
        rule = constants.DIVERGENCE_AGGREGATION_RULE
        if rule == "max":
            return float(computation.max_divergence)
        if rule == "mean":
            return float(computation.mean_divergence)
        raise ValueError(f"unknown divergence aggregation rule {rule!r}")

    def _render_evidence(self,
                         computation: BenfordComputation) -> Optional[np.ndarray]:
        """Render evidence from the most deviant grid sample.

        The worst-conforming configuration is plotted because it is the one
        carrying the finding; plotting the mean would hide it.

        Args:
            computation: Result of the mathematical core.

        Returns:
            BGR evidence image, or None when nothing was measured.
        """
        if not computation.samples:
            return None

        worst = max(computation.samples, key=lambda sample: sample.divergence)
        try:
            return self.visualizer.render_evidence_map(worst.empirical_pmf,
                                                       worst.fitted_pmf)
        except (ValueError, IndexError) as error:
            # Visualisation must never invalidate a completed measurement.
            logger.warning("evidence map rendering failed: %s", error)
            return None

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
