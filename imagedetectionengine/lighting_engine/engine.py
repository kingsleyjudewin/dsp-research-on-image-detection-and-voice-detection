"""Entry point for the lighting / illumination inconsistency engine.

Orchestrates the four processing stages in order:

    Stage 1  input validation and condition checking   (condition.py)
    Stage 2  preprocessing                             (preprocessor.py)
    Stage 3  core mathematical computation              (computer.py)
    Stage 4  score extraction and output assembly       (scorer.py, visualizer.py)

READ constants.py's module docstring before wiring this engine's vote into a
fusion layer at ordinary weight. The SKILL file's own Corpus Gap section
states this module has "the thinnest evidentiary base of all nine" detectors
in the system, and its only implementable, non-ML technique is a two-sentence
unvalidated heuristic with an internally inconsistent decision rule. This
engine therefore applies constants.MAXIMUM_CONFIDENCE_CEILING unconditionally,
on top of every other confidence factor - this is a standing instruction from
the SKILL file, not a per-image condition.

The engine never raises into its caller. Every failure path returns a fully
populated EngineOutput with is_reliable=False and a reliability_note naming
what went wrong, so the fusion layer can always account for this engine's vote.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from . import constants
from .computer import GradientMagnitudeComputer
from .condition import ConditionChecker
from .contracts import (CalibrationSettings, ConditionReport, EngineInput,
                        EngineOutput, GradientMagnitudeResult, PreparedImage)
from .preprocessor import LightingPreprocessor
from .scorer import LightingScorer
from .utils import (build_computation_step, clip_to_unit_interval,
                    compose_confidence_penalties, describe_array_shape)
from .visualizer import LightingVisualizer

logger = logging.getLogger(__name__)


class LightingEngine:
    """Computes a weak, unvalidated gradient-strength auxiliary signal.

    Not a validated lighting-consistency detector - see the module docstring.
    """

    def __init__(self,
                 calibration: Optional[CalibrationSettings] = None) -> None:
        """Build the engine and its collaborators.

        Args:
            calibration: Optional calibration state. Without it the engine
                still runs, but reports its probability as provisional.
        """
        self.condition_checker = ConditionChecker()
        self.preprocessor = LightingPreprocessor()
        self.computer = GradientMagnitudeComputer()
        self.scorer = LightingScorer(calibration)
        self.visualizer = LightingVisualizer()

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
            logger.exception("lighting engine failed during analysis")
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
        steps = [self._describe_condition_check(image, report)]
        if report.skip_engine:
            return self._build_skip_output(started_at, report, steps)

        # Stage 2 and 3.
        prepared = self.preprocessor.prepare(image)
        steps.append(self._describe_preprocessing(image, prepared))

        result = self.computer.compute(prepared.grayscale)
        steps.append(self._describe_gradient_computation(prepared, result))

        # Stage 4.
        return self._build_success_output(started_at, report, result, steps)

    @staticmethod
    def _describe_condition_check(image: np.ndarray,
                                  report: ConditionReport) -> dict:
        """Record the condition-checking stage in the computation trace.

        Args:
            image: Original image array.
            report: Verdict of the gate.

        Returns:
            One computation-step dictionary.
        """
        return build_computation_step(
            step_number=constants.COMPUTATION_STEP_CONDITION_CHECK,
            name="Condition check",
            description=("Verified the input is structurally usable for "
                         "numpy.gradient. The SKILL file states no quantified "
                         "reliable/unreliable operating envelope exists for "
                         "this module, so no quality-factor or resampling "
                         "gate is applied here, unlike the other engines in "
                         "this system."),
            input_shape=describe_array_shape(image),
            output_shape="condition report",
            key_values={
                "is_reliable": report.is_reliable,
                "skip_engine": report.skip_engine,
                "confidence_weight": round(report.confidence_weight, constants.TRACE_DECIMAL_PLACES),
                "maximum_confidence_ceiling":
                    constants.MAXIMUM_CONFIDENCE_CEILING,
            },
        )

    @staticmethod
    def _describe_preprocessing(image: np.ndarray,
                                prepared: PreparedImage) -> dict:
        """Record the preprocessing stage in the computation trace.

        Args:
            image: Original image array.
            prepared: Output of Stage 2.

        Returns:
            One computation-step dictionary.
        """
        return build_computation_step(
            step_number=constants.COMPUTATION_STEP_PREPROCESSING,
            name="Grayscale conversion",
            description=("Converted to grayscale, matching the SKILL's "
                         "'double(gray_img) per the source's own MATLAB "
                         "code'. The exact RGB weighting is not printed in "
                         "the SKILL; ITU-R BT.601 luma weights were used, "
                         "verified to match MATLAB's own rgb2gray default."),
            input_shape=describe_array_shape(image),
            output_shape=describe_array_shape(prepared.grayscale),
            key_values={
                "luma_weight_red": constants.LUMA_WEIGHT_RED,
                "luma_weight_green": constants.LUMA_WEIGHT_GREEN,
                "luma_weight_blue": constants.LUMA_WEIGHT_BLUE,
            },
        )

    @staticmethod
    def _describe_gradient_computation(
            prepared: PreparedImage,
            result: GradientMagnitudeResult) -> dict:
        """Record the gradient-magnitude stage in the computation trace.

        Args:
            prepared: Output of Stage 2.
            result: Output of the mathematical core.

        Returns:
            One computation-step dictionary.
        """
        return build_computation_step(
            step_number=constants.COMPUTATION_STEP_GRADIENT_COMPUTATION,
            name="Gradient magnitude (SKILL Pipeline A)",
            description=(
                "Computed the numerical image gradient via numpy.gradient "
                "(the direct equivalent of MATLAB's gradient(), per the "
                "SKILL's own Implementation Notes), took its per-pixel "
                "magnitude sqrt(Gx^2+Gy^2), and formed a scale-invariant "
                "ratio of the maximum to the median gradient magnitude - the "
                "normalization the SKILL's Output section recommends in "
                "place of an absolute cutoff, since none is given. The "
                "paper's stated decision rule also mentions detecting "
                "'multiple light directions', which is NOT implemented: the "
                "SKILL itself flags this as an unresolved internal "
                "inconsistency, since a single scalar magnitude carries no "
                "directional information."),
            input_shape=describe_array_shape(prepared.grayscale),
            output_shape=describe_array_shape(result.gradient_magnitude),
            key_values={
                "max_gradient": round(result.max_gradient, constants.TRACE_DECIMAL_PLACES),
                "median_gradient": round(result.median_gradient, constants.TRACE_DECIMAL_PLACES),
                "ratio": round(result.ratio, constants.TRACE_DECIMAL_PLACES),
                "is_degenerate": result.is_degenerate,
            },
        )

    def _build_success_output(self,
                              started_at: float,
                              report: ConditionReport,
                              result: GradientMagnitudeResult,
                              steps: list[dict]) -> EngineOutput:
        """Score the computation and assemble the final output.

        Args:
            started_at: perf_counter value taken at entry.
            report: Verdict of the pre-computation gate.
            result: Output of the mathematical core.
            steps: Trace accumulated so far, appended to in place.

        Returns:
            Fully populated EngineOutput.
        """
        raw_score, probability, confidence, is_reliable, note = \
            self._score_computation(report, result, steps)
        return EngineOutput(
            engine_name=constants.ENGINE_NAME,
            raw_score=float(raw_score),
            # Contract: probability is defined only when the vote is reliable.
            probability=(clip_to_unit_interval(probability) if is_reliable
                         else None),
            confidence=confidence,
            is_reliable=is_reliable,
            reliability_note=note,
            evidence_map=self._render_evidence(result),
            # The SKILL gives no threshold for turning the gradient map into
            # discrete suspect regions; this engine is a global scalar signal.
            flagged_regions=None,
            computation_steps=steps,
            processing_time_ms=self._elapsed_milliseconds(started_at),
            skill_version=constants.SKILL_VERSION,
        )

    def _score_computation(self,
                           report: ConditionReport,
                           result: GradientMagnitudeResult,
                           steps: list[dict]) -> tuple:
        """Reduce the computation to a score, probability and confidence.

        Args:
            report: Verdict of the pre-computation gate.
            result: Output of the mathematical core.
            steps: Trace to append the calibration record to.

        Returns:
            Tuple of (raw_score, probability, confidence, is_reliable, note).
        """
        field_ok, field_penalty, field_note = \
            self.condition_checker.assess_gradient_field(result)
        probability, route, is_calibrated, calibration_note = \
            self.scorer.to_probability(result.ratio)
        confidence = self._compose_confidence(report.confidence_weight,
                                              field_penalty, is_calibrated)
        steps.append(self._describe_calibration(result, probability, route,
                                                is_calibrated,
                                                calibration_note, confidence))

        notes = [report.reliability_note, field_note, calibration_note]
        return (result.ratio, probability, confidence,
                report.is_reliable and field_ok,
                " ".join(note for note in notes if note))

    @staticmethod
    def _compose_confidence(condition_weight: float,
                            field_penalty: float,
                            is_calibrated: bool) -> float:
        """Fold every confidence penalty into one fusion weight, then cap it.

        Args:
            condition_weight: Weight from the pre-computation gate.
            field_penalty: Penalty from the degenerate-image check.
            is_calibrated: Whether measured calibration data backed the result.

        Returns:
            Composed weight in [0, constants.MAXIMUM_CONFIDENCE_CEILING].
        """
        penalties = [condition_weight, field_penalty]
        if not is_calibrated:
            penalties.append(constants.CONFIDENCE_PENALTY_UNCALIBRATED)
        composed = compose_confidence_penalties(penalties)
        # The SKILL's Corpus Gap instruction applies regardless of every other
        # factor above: this module never earns more than a near-zero vote.
        return min(composed, constants.MAXIMUM_CONFIDENCE_CEILING)

    @staticmethod
    def _describe_calibration(result: GradientMagnitudeResult,
                              probability: float,
                              route: str,
                              is_calibrated: bool,
                              calibration_note: str,
                              confidence: float) -> dict:
        """Record the calibration stage in the computation trace.

        Args:
            result: Output of the mathematical core.
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
            description=(f"Mapped the gradient ratio onto a probability using "
                         f"the '{route}' route, then capped confidence at "
                         f"{constants.MAXIMUM_CONFIDENCE_CEILING} regardless "
                         f"of route, per the SKILL's own Corpus Gap "
                         f"instruction. {calibration_note}"),
            input_shape="scalar ratio",
            output_shape="scalar probability",
            key_values={"raw_score": round(result.ratio, constants.TRACE_SCORE_DECIMAL_PLACES),
                        "probability": round(probability, constants.TRACE_SCORE_DECIMAL_PLACES),
                        "calibration_route": route,
                        "is_calibrated": is_calibrated,
                        "confidence": round(confidence, constants.TRACE_DECIMAL_PLACES)},
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
        logger.info("lighting engine skipped: %s", report.reliability_note)
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
            reliability_note=(f"Lighting engine failed and returned no "
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

    def _render_evidence(self,
                         result: GradientMagnitudeResult) -> Optional[np.ndarray]:
        """Render the gradient-magnitude map as evidence.

        Args:
            result: Output of the mathematical core.

        Returns:
            BGR evidence image, or None when rendering failed.
        """
        try:
            return self.visualizer.render_evidence_map(
                result.gradient_magnitude)
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
                    f"format {engine_input.metadata.format}")
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
