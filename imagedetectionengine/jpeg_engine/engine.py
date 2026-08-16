"""Entry point orchestrating the JPEG-compression-artifact engine.

Stage 1: condition.py pre-checks. Stage 2: preprocessor.py (luminance, block
grid, 8x8 block DCT over unsaturated blocks). Stage 3: computer.py (A.1, A.2,
A.3, B). Stage 4: scorer.py + visualizer.py + this file's output assembly.

Pipeline roles follow the SKILL's own explicit assignment (see the SCOPE
DECISION note in constants.py): A.1 gates, A.2 and A.3 condition, and only
Pipeline B drives raw_score. Pipelines C and D are [ML - excluded]; the
coverage that forfeits - non-aligned double JPEG and the QF1=QF2 case -
is surfaced in every reliability_note rather than being absorbed silently.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from . import constants
from .computer import (DoubleCompressionDetector, JpegHistoryIdentifier,
                       QualityFactorDetector, QuantizationStepEstimator)
from .condition import STANDING_LIMITATIONS_NOTE, ConditionChecker
from .contracts import (CalibrationSettings, EngineInput, EngineOutput,
                        QualityFactorResult)
from .preprocessor import JpegPreprocessor
from .scorer import JpegScorer
from .utils import build_computation_step, compose_confidence_penalties
from .visualizer import JpegVisualizer

logger = logging.getLogger(__name__)

# Report-facing text for computation_steps. Each entry's SCORED/NOT SCORED
# tag reflects the SKILL's own role assignment for that pipeline; the report
# generator reads these verbatim, so the distinction has to survive into the
# forensic write-up rather than living only in this file's comments.
STEP_NAMES = {
    "blocks": "Block DCT extraction",
    "history": "Pipeline A.1: JPEG-history gate (NOT SCORED)",
    "quality": "Pipeline A.3: quality-factor recovery (CONDITIONING + CONFIDENCE)",
    "steps": "Pipeline A.2: per-frequency quantization steps (CONDITIONING)",
    "periodicity": "Pipeline B: double-quantization periodicity (SCORE-DRIVING)",
    "scoring": "Scoring",
}

STEP_DESCRIPTIONS = {
    "blocks": ("Converted to luminance, cropped to the 8x8 block grid, and "
               "applied a level-shifted 8x8 DCT to every unsaturated block "
               "(saturated blocks are excluded so truncation error can be "
               "neglected, per the SKILL)."),
    "history": ("Computed the ratio of AC-coefficient mass in R1=(-1,+1) to "
                "that in R2=(-2,-1)u(+1,+2). Used only to decide whether "
                "this module applies at all - being JPEG is not evidence of "
                "forgery."),
    "quality": ("Recompressed at every candidate quality factor and took the "
                "one reproducing the most pixels exactly. Feeds the low-QF "
                "reliability gate, the confidence discount, and the "
                "quantization steps Pipeline B needs - never raw_score."),
    "steps": ("Estimated the quantization step at each AC DCT frequency from "
              "the histogram of rounded coefficients. The SKILL calls this a "
              "nuisance parameter, not a tampering score; it is reported for "
              "the record and used only as a fallback when the "
              "quality-factor sweep is disabled."),
    "periodicity": ("Divided each selected low-frequency coefficient by its "
                    "quantization step to recover the integer domain the "
                    "double-quantization model is written over, histogrammed "
                    "it, took the normalised FFT magnitude, removed the "
                    "decaying trend by trailing-minimum subtraction, and "
                    "measured the surviving peak prominences."),
}


class JpegEngine:
    """Orchestrates condition-checking, preprocessing, computation, and scoring."""

    def __init__(self, calibration: Optional[CalibrationSettings] = None) -> None:
        """Bind stateless collaborators and optional one-time calibration.

        Args:
            calibration: Reference scores, per-image-size history threshold,
                Eq. 5 window length, and the A.3 sweep toggle.
        """
        self.condition_checker = ConditionChecker()
        self.preprocessor = JpegPreprocessor()
        self.history_identifier = JpegHistoryIdentifier()
        self.step_estimator = QuantizationStepEstimator()
        self.quality_detector = QualityFactorDetector()
        self.double_compression_detector = DoubleCompressionDetector()
        self.scorer = JpegScorer(calibration)
        self.visualizer = JpegVisualizer()
        self.calibration = calibration or CalibrationSettings()

    def analyse(self, engine_input: EngineInput) -> EngineOutput:
        """Run the full JPEG-artifact pipeline on one image.

        Args:
            engine_input: Image and metadata supplied by the orchestrator.

        Returns:
            EngineOutput. Never raises - all failure paths return a full
            EngineOutput with is_reliable=False and a descriptive note.
        """
        start_time = time.perf_counter()
        try:
            return self._analyse_unguarded(engine_input, start_time)
        except (ValueError, RuntimeError, OSError,
                np.linalg.LinAlgError) as error:
            logger.exception("jpeg engine failed")
            return self._failure_output(str(error), start_time)

    def _analyse_unguarded(self, engine_input: EngineInput,
                           start_time: float) -> EngineOutput:
        """The real pipeline body, wrapped by analyse()'s error handling.

        Args:
            engine_input: Image and metadata supplied by the orchestrator.
            start_time: perf_counter() timestamp from analyse().

        Returns:
            EngineOutput.
        """
        condition = self.condition_checker.check(engine_input.metadata,
                                                  engine_input.image)
        if condition.skip_engine:
            return self._null_output(condition.reliability_note, start_time)

        prepared = self.preprocessor.prepare(engine_input.image,
                                             engine_input.metadata)
        blocks = self.preprocessor.extract_block_dct(prepared.luminance)

        block_check = self.condition_checker.assess_block_count(
            blocks.unsaturated_block_count, blocks.total_block_count)
        if not block_check[0]:
            return self._null_output(
                f"{block_check[2]} {STANDING_LIMITATIONS_NOTE}", start_time)

        return self._run_pipelines(prepared, blocks, condition, block_check,
                                   start_time)

    def _run_pipelines(self, prepared, blocks, condition, block_check,
                       start_time) -> EngineOutput:
        """Run A.1's gate, then the conditioning and scoring pipelines.

        Args:
            prepared: PreparedImage holding the cropped luminance.
            blocks: BlockDctResult for the unsaturated blocks.
            condition: Pre-computation ConditionReport.
            block_check: Result of assess_block_count.
            start_time: perf_counter() timestamp from analyse().

        Returns:
            EngineOutput.
        """
        history = self.history_identifier.compute_history_feature(
            blocks.coefficients, self._history_threshold())
        history_check = self.condition_checker.assess_jpeg_history(history)
        if not history_check[0]:
            return self._null_output(
                f"{history_check[2]} {STANDING_LIMITATIONS_NOTE}", start_time)

        quality = self._run_quality_sweep(prepared.luminance)
        steps = self.step_estimator.compute(blocks.coefficients)
        double_compression = self.double_compression_detector.compute(
            blocks.coefficients, self._trend_window_length(),
            self._quantization_table(quality, steps))

        checks = {
            "block_count": block_check,
            "jpeg_history": history_check,
            "quality_factor": self.condition_checker.assess_quality_factor(quality),
            "usable_frequencies":
                self.condition_checker.assess_usable_frequencies(double_compression),
            "table_margin":
                self.condition_checker.assess_table_detection_margin(quality),
        }
        return self._assemble(blocks, history, quality, steps,
                              double_compression, condition, checks, start_time)

    def _assemble(self, blocks, history, quality, steps, double_compression,
                  condition, checks, start_time) -> EngineOutput:
        """Score Pipeline B's result and build the final EngineOutput.

        Args:
            blocks: BlockDctResult for the unsaturated blocks.
            history: Pipeline A.1's result.
            quality: Pipeline A.3's result.
            steps: Pipeline A.2's result.
            double_compression: Pipeline B's result.
            condition: Pre-computation ConditionReport.
            checks: Dict of post-computation CheckResult tuples.
            start_time: perf_counter() timestamp from analyse().

        Returns:
            EngineOutput.
        """
        scoring = self._score(double_compression.aggregate_score, condition, checks)
        evidence_map = self.visualizer.render_spectra(double_compression.spectra)
        steps_log = self._build_computation_steps(
            blocks, history, quality, steps, double_compression, scoring)
        note = self._compose_reliability_note(checks, scoring)

        elapsed_ms = ((time.perf_counter() - start_time)
                      * constants.MILLISECONDS_PER_SECOND)
        return EngineOutput(
            engine_name=constants.ENGINE_NAME, raw_score=scoring["raw_score"],
            probability=scoring["probability"], confidence=scoring["confidence"],
            is_reliable=scoring["is_reliable"], reliability_note=note,
            evidence_map=evidence_map,
            # Pipeline A/B do not localise; see visualizer.py's module note.
            flagged_regions=None, computation_steps=steps_log,
            processing_time_ms=elapsed_ms, skill_version=constants.SKILL_VERSION)

    def _history_threshold(self) -> float:
        """Resolve the Pipeline A.1 decision threshold t.

        Returns:
            The orchestrator's per-image-size threshold when supplied,
            otherwise the midpoint of the SKILL's reported range.
        """
        if self.calibration.history_threshold is not None:
            return float(self.calibration.history_threshold)
        return constants.DEFAULT_HISTORY_THRESHOLD

    def _trend_window_length(self) -> int:
        """Resolve the Eq. 5 trailing-window length n.

        Returns:
            The orchestrator's calibrated n when supplied, otherwise the
            engineering default (the SKILL gives no corpus value).
        """
        if self.calibration.trend_removal_window_length is not None:
            return int(self.calibration.trend_removal_window_length)
        return constants.TREND_REMOVAL_WINDOW_LENGTH

    def _quantization_table(self, quality: QualityFactorResult,
                            steps) -> np.ndarray:
        """Resolve the per-frequency steps Pipeline B needs to recover Eq. 4's
        integer coefficient domain.

        A.3's recovered table is preferred: the paper benchmarks table
        detection at 94-99% and it was verified end-to-end here, whereas
        A.2's Eq. 13 argmax is unreliable at exactly the low frequencies
        Pipeline B uses. A.2 is the fallback only when the caller disabled
        the sweep; its zero entries then exclude their frequency.

        Args:
            quality: Pipeline A.3's result.
            steps: Pipeline A.2's result.

        Returns:
            8x8 integer array of quantization steps.
        """
        if quality.sweep_ran:
            return self.quality_detector.encoder_quantization_table(
                quality.quality_factor)
        return steps.steps

    def _run_quality_sweep(self, luminance: np.ndarray) -> QualityFactorResult:
        """Run Pipeline A.3, or report cleanly that it was disabled.

        The sweep costs 100 JPEG encode/decode round trips, so the
        orchestrator can switch it off for throughput; its outputs feed
        confidence and the low-QF gate, never raw_score.

        Args:
            luminance: Float64 luminance array of the image under test.

        Returns:
            QualityFactorResult, with sweep_ran=False when disabled.
        """
        if not self.calibration.run_quality_factor_sweep:
            return QualityFactorResult(
                quality_factor=0, pixel_match_ratio=0.0,
                runner_up_match_ratio=0.0, sweep_ran=False,
                note="Quality-factor sweep disabled by the caller; the "
                     "low-quality-factor reliability gate could not be "
                     "evaluated for this image.")
        return self.quality_detector.detect(luminance)

    def _score(self, aggregate_score, condition, checks) -> dict:
        """Turn Pipeline B's aggregate into the full scoring bundle.

        Args:
            aggregate_score: Pipeline B's mean strongest peak prominence.
            condition: Pre-computation ConditionReport.
            checks: Dict of post-computation CheckResult tuples.

        Returns:
            Dict with raw_score, probability, confidence, is_reliable,
            route, is_calibrated, and score_note.
        """
        probability, route, is_calibrated, score_note = self.scorer.to_probability(
            aggregate_score)
        confidence = compose_confidence_penalties(
            [condition.confidence_weight]
            + [check[1] for check in checks.values()])
        is_reliable = condition.is_reliable and all(
            check[0] for check in checks.values())
        return {
            "raw_score": float(aggregate_score),
            "probability": probability if is_reliable else None,
            "confidence": confidence if is_reliable else constants.ZERO_CONFIDENCE,
            "is_reliable": is_reliable, "route": route,
            "is_calibrated": is_calibrated, "score_note": score_note,
        }

    def _build_computation_steps(self, blocks, history, quality, steps,
                                 double_compression, scoring) -> list:
        """Assemble the computation_steps log for the report generator.

        Args:
            blocks: BlockDctResult for the unsaturated blocks.
            history: Pipeline A.1's result.
            quality: Pipeline A.3's result.
            steps: Pipeline A.2's result.
            double_compression: Pipeline B's result.
            scoring: Dict returned by _score.

        Returns:
            List of computation-step dicts.
        """
        specs = [
            (STEP_NAMES["blocks"], STEP_DESCRIPTIONS["blocks"],
             self._block_key_values(blocks)),
            (STEP_NAMES["history"], STEP_DESCRIPTIONS["history"],
             self._history_key_values(history)),
            (STEP_NAMES["quality"], STEP_DESCRIPTIONS["quality"],
             self._quality_key_values(quality)),
            (STEP_NAMES["steps"], STEP_DESCRIPTIONS["steps"],
             {"usable_frequency_count": steps.usable_frequency_count}),
            (STEP_NAMES["periodicity"], STEP_DESCRIPTIONS["periodicity"],
             self._double_compression_key_values(double_compression)),
            (STEP_NAMES["scoring"],
             f"Aggregate peak prominence converted to a probability via the "
             f"{scoring['route']} route.",
             {"is_calibrated": scoring["is_calibrated"],
              "raw_score": round(scoring["raw_score"],
                                constants.TRACE_DECIMAL_PLACES)}),
        ]
        return [build_computation_step(index, name, description, key_values=values)
               for index, (name, description, values) in enumerate(specs, start=1)]

    @staticmethod
    def _block_key_values(blocks) -> dict:
        """Report values for the block-DCT computation_steps entry."""
        return {"total_blocks": blocks.total_block_count,
               "unsaturated_blocks": blocks.unsaturated_block_count,
               "blocks_per_row": blocks.blocks_per_row,
               "blocks_per_column": blocks.blocks_per_column}

    @staticmethod
    def _history_key_values(history) -> dict:
        """Report values for Pipeline A.1's computation_steps entry."""
        return {"history_feature_s": round(history.history_feature,
                                          constants.TRACE_DECIMAL_PLACES),
               "threshold": history.threshold,
               "is_jpeg_derived": history.is_jpeg_derived,
               "region_one_count": history.region_one_count,
               "region_two_count": history.region_two_count}

    @staticmethod
    def _quality_key_values(quality) -> dict:
        """Report values for Pipeline A.3's computation_steps entry."""
        if not quality.sweep_ran:
            return {"sweep_ran": False, "note": quality.note}
        return {"sweep_ran": True,
               "estimated_quality_factor": quality.quality_factor,
               "pixel_match_ratio": round(quality.pixel_match_ratio,
                                         constants.TRACE_DECIMAL_PLACES),
               "runner_up_match_ratio": round(quality.runner_up_match_ratio,
                                             constants.TRACE_DECIMAL_PLACES)}

    @staticmethod
    def _double_compression_key_values(result) -> dict:
        """Report values for Pipeline B's computation_steps entry."""
        return {"usable_frequencies": result.usable_frequency_count,
               "total_frequencies": len(constants.DOUBLE_QUANTIZATION_FREQUENCIES),
               "peak_bearing_frequencies": result.peak_bearing_frequency_count,
               "aggregate_score": round(result.aggregate_score,
                                       constants.TRACE_DECIMAL_PLACES),
               "per_frequency_strongest_prominence": [
                   round(entry.strongest_prominence,
                        constants.TRACE_DECIMAL_PLACES)
                   for entry in result.spectra]}

    @staticmethod
    def _compose_reliability_note(checks, scoring) -> str:
        """Combine every note fragment into one reliability_note string.

        Args:
            checks: Dict of post-computation CheckResult tuples.
            scoring: Dict returned by _score.

        Returns:
            Combined reliability_note string.
        """
        fragments = [STANDING_LIMITATIONS_NOTE]
        fragments.extend(check[2] for check in checks.values() if check[2])
        fragments.append(scoring["score_note"])
        return " ".join(fragments)

    @staticmethod
    def _null_output(note: str, start_time: float) -> EngineOutput:
        """Build the EngineOutput for a skipped or ungradeable run.

        Args:
            note: Explanation for why the engine did not produce a score.
            start_time: perf_counter() timestamp from analyse().

        Returns:
            EngineOutput with is_reliable=False and no probability.
        """
        elapsed_ms = ((time.perf_counter() - start_time)
                      * constants.MILLISECONDS_PER_SECOND)
        return EngineOutput(
            engine_name=constants.ENGINE_NAME, raw_score=0.0, probability=None,
            confidence=constants.ZERO_CONFIDENCE, is_reliable=False,
            reliability_note=note, evidence_map=None, flagged_regions=None,
            computation_steps=[], processing_time_ms=elapsed_ms,
            skill_version=constants.SKILL_VERSION)

    @staticmethod
    def _failure_output(error_message: str, start_time: float) -> EngineOutput:
        """Build the EngineOutput for an unexpected computation failure.

        Args:
            error_message: str(exception) describing what failed.
            start_time: perf_counter() timestamp from analyse().

        Returns:
            EngineOutput with is_reliable=False and the error in the note.
        """
        elapsed_ms = ((time.perf_counter() - start_time)
                      * constants.MILLISECONDS_PER_SECOND)
        return EngineOutput(
            engine_name=constants.ENGINE_NAME, raw_score=0.0, probability=None,
            confidence=constants.ZERO_CONFIDENCE, is_reliable=False,
            reliability_note=f"Engine failed during computation: {error_message}",
            evidence_map=None, flagged_regions=None, computation_steps=[],
            processing_time_ms=elapsed_ms, skill_version=constants.SKILL_VERSION)
