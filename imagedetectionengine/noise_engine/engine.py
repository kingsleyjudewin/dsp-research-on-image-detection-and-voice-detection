"""Entry point orchestrating the noise-pattern forgery-detection engine.

Stage 1: condition.py pre-checks. Stage 2: preprocessor.py (grayscale,
wavelet residual extraction, block tiling). Stage 3: computer.py (all four
pipelines). Stage 4: scorer.py + visualizer.py + this file's output assembly.

Only Pipeline A (blind local noise-level inconsistency) drives raw_score/
probability - the SKILL explicitly names it PRIMARY for this engine's
single-image, no-reference contract. Pipeline C always runs as auxiliary
evidence. Pipelines B and D only run when the orchestrator supplies the
extra calibration data they need (same-camera reference images / a
training-set baseline magnitude, respectively) - neither is required for the
engine to produce a score.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from . import constants
from .computer import (BlindSpectralAnalyser, LocalNoiseInconsistencyComputer,
                       NoiseTriageClassifier, ReferencePRNUEstimator)
from .condition import STANDING_LIMITATIONS_NOTE, ConditionChecker
from .contracts import (CalibrationSettings, EngineInput, EngineOutput,
                        NoiseTriageResult, ReferencePRNUResult)
from .preprocessor import NoisePreprocessor
from .scorer import NoiseScorer
from .utils import build_computation_step, compose_confidence_penalties
from .visualizer import NoiseVisualizer

logger = logging.getLogger(__name__)


class NoiseEngine:
    """Orchestrates condition-checking, preprocessing, computation, and scoring."""

    def __init__(self, calibration: Optional[CalibrationSettings] = None) -> None:
        """Bind stateless collaborators and optional one-time calibration.

        Args:
            calibration: Reference scores, same-camera reference images,
                and/or the Pipeline D training-set baseline magnitude.
        """
        self.condition_checker = ConditionChecker()
        self.preprocessor = NoisePreprocessor()
        self.local_inconsistency_computer = LocalNoiseInconsistencyComputer()
        self.spectral_analyser = BlindSpectralAnalyser()
        self.reference_prnu_estimator = ReferencePRNUEstimator()
        self.triage_classifier = NoiseTriageClassifier()
        self.scorer = NoiseScorer(calibration)
        self.visualizer = NoiseVisualizer()
        self.calibration = calibration or CalibrationSettings()

    def analyse(self, engine_input: EngineInput) -> EngineOutput:
        """Run the full noise-pattern pipeline on one image.

        Args:
            engine_input: Image and metadata supplied by the orchestrator.

        Returns:
            EngineOutput. Never raises - all failure paths return a full
            EngineOutput with is_reliable=False and a descriptive note.
        """
        start_time = time.perf_counter()
        try:
            return self._analyse_unguarded(engine_input, start_time)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            logger.exception("noise engine failed")
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
        image = engine_input.image
        block_size = self._estimate_block_size(image)
        condition = self.condition_checker.check(engine_input.metadata, image,
                                                  block_size)
        if condition.skip_engine:
            return self._null_output(condition.reliability_note, start_time)

        prepared = self.preprocessor.prepare(image, engine_input.metadata)
        residual = self.preprocessor.extract_residual(prepared.grayscale)
        blocks = self.preprocessor.tile_blocks(residual, prepared.grayscale,
                                               block_size)
        if not blocks:
            return self._null_output(
                "Engine skipped: no blocks could be tiled at the "
                "resolution-scaled block size.", start_time)

        return self._score_and_assemble(prepared.grayscale, residual, blocks,
                                        block_size, condition, start_time)

    def _estimate_block_size(self, image: Optional[np.ndarray]) -> int:
        """Estimate the resolution-scaled block size before full preprocessing.

        Args:
            image: BGR uint8 or grayscale array, or None.

        Returns:
            Block side length in pixels; a structural default if image is None.
        """
        if image is None:
            return constants.MINIMUM_BLOCK_SIZE_PIXELS
        array = np.asarray(image)
        if array.ndim < constants.GRAYSCALE_IMAGE_DIMENSION_COUNT:
            return constants.MINIMUM_BLOCK_SIZE_PIXELS
        return self.preprocessor.resolution_scaled_block_size(
            array.shape[0], array.shape[1])

    def _score_and_assemble(self, grayscale, residual, blocks, block_size,
                            condition, start_time) -> EngineOutput:
        """Run all four pipelines and assemble the final EngineOutput.

        Args:
            grayscale: Float64 grayscale image.
            residual: Float64 noise-residual array W.
            blocks: Tiled NoiseBlock list.
            block_size: Block side length in pixels.
            condition: Pre-computation ConditionReport.
            start_time: perf_counter() timestamp from analyse().

        Returns:
            EngineOutput.
        """
        local_result = self.local_inconsistency_computer.compute(blocks, block_size)
        spectral_result = self.spectral_analyser.compute(residual)
        triage_result = self._run_pipeline_d(grayscale)
        prnu_result = self._run_pipeline_b(residual, grayscale, blocks)

        checks = self._run_post_computation_checks(blocks, triage_result)
        scoring = self._score(local_result.aggregate_scalar, condition, checks)

        evidence_map = self.visualizer.render_heatmap(local_result.heatmap, block_size)
        flagged_regions = self._build_flagged_regions(local_result.flagged_blocks,
                                                       block_size)
        steps = self._build_computation_steps(
            local_result, spectral_result, prnu_result, triage_result,
            scoring["route"], scoring["is_calibrated"])
        note = self._compose_reliability_note(condition, checks, prnu_result, scoring)

        elapsed_ms = (time.perf_counter() - start_time) * constants.MILLISECONDS_PER_SECOND
        return EngineOutput(
            engine_name=constants.ENGINE_NAME, raw_score=scoring["raw_score"],
            probability=scoring["probability"], confidence=scoring["confidence"],
            is_reliable=scoring["is_reliable"], reliability_note=note,
            evidence_map=evidence_map, flagged_regions=flagged_regions,
            computation_steps=steps, processing_time_ms=elapsed_ms,
            skill_version=constants.SKILL_VERSION)

    def _run_pipeline_d(self, grayscale) -> NoiseTriageResult:
        """Run the calibration-gated noise-type triage classifier.

        Args:
            grayscale: Float64 grayscale image.

        Returns:
            NoiseTriageResult, or a not-run result if no baseline was supplied.
        """
        baseline = self.calibration.noise_triage_reference_mean_magnitude
        if baseline is None:
            return NoiseTriageResult(
                ran=False, note="No training-set baseline magnitude "
                                "supplied; Pipeline D skipped.")
        spectrum = self.triage_classifier.fft_magnitude_spectrum(grayscale)
        return self.triage_classifier.classify(spectrum, baseline)

    def _run_pipeline_b(self, residual, grayscale, blocks) -> ReferencePRNUResult:
        """Run the calibration-gated reference-based PRNU estimator.

        Args:
            residual: Test image's noise-residual array W.
            grayscale: Test image's grayscale intensity array.
            blocks: Tiled NoiseBlock list, used to size the test block.

        Returns:
            ReferencePRNUResult, or a not-run result if no reference images
            were supplied.
        """
        reference_images = self.calibration.same_camera_reference_images
        if not reference_images:
            return ReferencePRNUResult(
                ran=False, note="No same-camera reference images supplied; "
                                "Pipeline B skipped.")
        reference_residuals, reference_intensities = [], []
        for reference_image in reference_images:
            prepared = self.preprocessor.prepare(reference_image, metadata=None)
            reference_residuals.append(self.preprocessor.extract_residual(
                prepared.grayscale))
            reference_intensities.append(prepared.grayscale)
        return self.reference_prnu_estimator.compute(
            reference_residuals, reference_intensities, residual, grayscale)

    def _run_post_computation_checks(self, blocks, triage_result) -> dict:
        """Run all post-computation condition checks.

        Args:
            blocks: Tiled NoiseBlock list.
            triage_result: Output of _run_pipeline_d.

        Returns:
            Dict of CheckResult tuples keyed by check name.
        """
        intensity_values = np.concatenate(
            [block.intensity_pixels.ravel() for block in blocks])
        residual_variances = [float(np.var(block.residual_pixels))
                              for block in blocks]
        return {
            "saturation": self.condition_checker.assess_saturation(intensity_values),
            "flatness": self.condition_checker.assess_flatness(residual_variances),
            "triage": self.condition_checker.assess_triage_denoising_penalty(
                triage_result),
        }

    def _score(self, aggregate_scalar, condition, checks) -> dict:
        """Turn Pipeline A's aggregate scalar into the full scoring bundle.

        Args:
            aggregate_scalar: Pipeline A's top-k% mean anomaly scalar.
            condition: Pre-computation ConditionReport.
            checks: Dict of post-computation CheckResult tuples.

        Returns:
            Dict with raw_score, probability, confidence, is_reliable,
            route, is_calibrated, and score_note.
        """
        probability, route, is_calibrated, score_note = self.scorer.to_probability(
            aggregate_scalar)
        confidence = compose_confidence_penalties(
            [condition.confidence_weight] + [check[1] for check in checks.values()])
        # ENHANCEMENT 4: the saturation check's pass/fail result used to be
        # computed and then discarded - only its confidence penalty was read.
        # Measured on fake .jpeg: 65% of pixels at or above 250, the check
        # returned failed, and the engine still published FAKE at probability
        # 0.9991, on an image where the SKILL states the multiplicative PRNU
        # term vanishes outright (Eq. 19's attenuation).
        # ENHANCEMENT 5: an uncalibrated run no longer votes; see
        # constants.ABSTAIN_WHEN_UNCALIBRATED.
        may_vote = is_calibrated or not constants.ABSTAIN_WHEN_UNCALIBRATED
        is_reliable = (condition.is_reliable and checks["flatness"][0]
                       and checks["saturation"][0] and may_vote)
        return {
            "raw_score": float(aggregate_scalar),
            "probability": probability if is_reliable else None,
            "confidence": confidence if is_reliable else constants.ZERO_CONFIDENCE,
            "is_reliable": is_reliable, "route": route,
            "is_calibrated": is_calibrated, "score_note": score_note,
        }

    def _build_flagged_regions(self, flagged_blocks, block_size) -> Optional[list]:
        """Convert flagged blocks into the flagged_regions contract.

        Args:
            flagged_blocks: List of NoiseBlock objects Pipeline A flagged.
            block_size: Block side length in pixels.

        Returns:
            List of region dicts, or None if no blocks were flagged.
        """
        if not flagged_blocks:
            return None
        return [{"pixel_row": block.pixel_row, "pixel_col": block.pixel_col,
                "block_size": block_size} for block in flagged_blocks]

    def _build_computation_steps(self, local_result, spectral_result,
                                 prnu_result, triage_result, route,
                                 is_calibrated) -> list:
        """Assemble the computation_steps log for the report generator.

        Args:
            local_result: Pipeline A's LocalInconsistencyResult.
            spectral_result: Pipeline C's SpectralAnalysisResult.
            prnu_result: Pipeline B's ReferencePRNUResult.
            triage_result: Pipeline D's NoiseTriageResult.
            route: Scoring route name from the scorer.
            is_calibrated: Whether the empirical-CDF route was used.

        Returns:
            List of computation-step dicts.
        """
        specs = [
            ("Pipeline A: blind local noise-level inconsistency (PRIMARY, SCORED)",
             "Wavelet residual extraction, block-wise variance, comparison "
             "against local neighbourhood median.",
             self._pipeline_a_key_values(local_result)),
            ("Pipeline C: blind cell-based PRNU spectral analysis (AUXILIARY, NOT SCORED)",
             "Cell-wise DFT-magnitude-histogram peak features, aggregated "
             "via S_mean/S_rms.",
             self._pipeline_c_key_values(spectral_result)),
            ("Pipeline B: reference-based PRNU (AUXILIARY, NOT SCORED)",
             "ML PRNU estimation + ZM/Wiener preprocessing; only the "
             "camera-agnostic pieces are implemented (see "
             "KNOWN_UNIMPLEMENTED_MODULES).",
             self._pipeline_b_key_values(prnu_result)),
            ("Pipeline D: FFT-spectrum noise-type triage (AUXILIARY, CONFIDENCE-ONLY)",
             "Categorical gate only, per the SKILL's own instruction - "
             "never contributes to raw_score.",
             self._pipeline_d_key_values(triage_result)),
            ("Scoring", f"Aggregate anomaly scalar converted to probability "
             f"via the {route} route.", {"is_calibrated": is_calibrated}),
        ]
        return [build_computation_step(index, name, description, key_values=values)
               for index, (name, description, values) in enumerate(specs, start=1)]

    def _pipeline_a_key_values(self, local_result) -> dict:
        """Report values for Pipeline A's computation_steps entry."""
        return {"total_blocks": local_result.total_blocks,
               "flagged_block_count": local_result.flagged_block_count,
               "aggregate_scalar": round(local_result.aggregate_scalar,
                                        constants.TRACE_DECIMAL_PLACES),
               "legacy_top_k_scalar": round(local_result.legacy_top_k_scalar,
                                           constants.TRACE_DECIMAL_PLACES)}

    def _pipeline_c_key_values(self, spectral_result) -> dict:
        """Report values for Pipeline C's computation_steps entry."""
        return {"grid_cells": spectral_result.grid_cells,
               "S_mean_P_val": round(spectral_result.s_mean_p_val,
                                    constants.TRACE_DECIMAL_PLACES),
               "S_rms_P_val": round(spectral_result.s_rms_p_val,
                                   constants.TRACE_DECIMAL_PLACES)}

    def _pipeline_b_key_values(self, prnu_result) -> dict:
        """Report values for Pipeline B's computation_steps entry."""
        if not prnu_result.ran:
            return {"ran": False, "note": prnu_result.note}
        return {"ran": True,
               "mean_block_correlation": round(prnu_result.mean_block_correlation,
                                              constants.TRACE_DECIMAL_PLACES),
               "reference_image_count": prnu_result.reference_image_count}

    def _pipeline_d_key_values(self, triage_result) -> dict:
        """Report values for Pipeline D's computation_steps entry."""
        if not triage_result.ran:
            return {"ran": False, "note": triage_result.note}
        return {"ran": True, "label": triage_result.label,
               "z_score": round(triage_result.z_score,
                               constants.TRACE_DECIMAL_PLACES)}

    def _compose_reliability_note(self, condition, checks, prnu_result,
                                  scoring) -> str:
        """Combine every note fragment into one reliability_note string.

        Args:
            condition: Pre-computation ConditionReport.
            checks: Dict of post-computation CheckResult tuples.
            prnu_result: Pipeline B's ReferencePRNUResult.
            scoring: Dict returned by _score.

        Returns:
            Combined reliability_note string.
        """
        fragments = [STANDING_LIMITATIONS_NOTE]
        for check in checks.values():
            if check[2]:
                fragments.append(check[2])
        if prnu_result.note:
            fragments.append(f"Pipeline B: {prnu_result.note}")
        fragments.append(scoring["score_note"])
        return " ".join(fragments)

    def _null_output(self, note: str, start_time: float) -> EngineOutput:
        """Build the EngineOutput for a skipped or ungradeable run.

        Args:
            note: Explanation for why the engine did not produce a score.
            start_time: perf_counter() timestamp from analyse().

        Returns:
            EngineOutput with is_reliable=False and no probability.
        """
        elapsed_ms = (time.perf_counter() - start_time) * constants.MILLISECONDS_PER_SECOND
        return EngineOutput(
            engine_name=constants.ENGINE_NAME, raw_score=0.0, probability=None,
            confidence=constants.ZERO_CONFIDENCE, is_reliable=False,
            reliability_note=note, evidence_map=None, flagged_regions=None,
            computation_steps=[], processing_time_ms=elapsed_ms,
            skill_version=constants.SKILL_VERSION)

    def _failure_output(self, error_message: str, start_time: float) -> EngineOutput:
        """Build the EngineOutput for an unexpected computation failure.

        Args:
            error_message: str(exception) describing what failed.
            start_time: perf_counter() timestamp from analyse().

        Returns:
            EngineOutput with is_reliable=False and the error in the note.
        """
        elapsed_ms = (time.perf_counter() - start_time) * constants.MILLISECONDS_PER_SECOND
        return EngineOutput(
            engine_name=constants.ENGINE_NAME, raw_score=0.0, probability=None,
            confidence=constants.ZERO_CONFIDENCE, is_reliable=False,
            reliability_note=f"Engine failed during computation: {error_message}",
            evidence_map=None, flagged_regions=None, computation_steps=[],
            processing_time_ms=elapsed_ms, skill_version=constants.SKILL_VERSION)
