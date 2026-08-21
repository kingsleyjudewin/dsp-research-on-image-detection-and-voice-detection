"""Entry point orchestrating the wavelet-domain forgery-detection engine.

Stage 1: condition.py pre-checks. Stage 2: preprocessor.py (grayscale, Haar
LL extraction, block tiling). Stage 3: computer.py (all three pipelines).
Stage 4: scorer.py + visualizer.py + this file's output assembly.

Only Pipeline C (copy-move detection) drives raw_score/probability - see
constants.py's SCOPE DECISION note. Pipelines A and B run in full and are
reported as auxiliary, non-scoring evidence in computation_steps and
reliability_note, per their own SKILL-stated caveats.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from . import constants
from .computer import CompressionHistoryEstimator, CopyMoveDetector, NoiseResidualExtractor
from .condition import STANDING_LIMITATIONS_NOTE, ConditionChecker
from .contracts import CalibrationSettings, EngineInput, EngineOutput
from .preprocessor import WaveletPreprocessor
from .scorer import WaveletScorer
from .utils import build_computation_step, compose_confidence_penalties
from .visualizer import WaveletVisualizer

logger = logging.getLogger(__name__)


class WaveletEngine:
    """Orchestrates condition-checking, preprocessing, computation, and scoring."""

    def __init__(self,
                calibration: Optional[CalibrationSettings] = None,
                block_size: int = constants.DEFAULT_BLOCK_SIZE) -> None:
        """Bind stateless collaborators and optional one-time calibration.

        Args:
            calibration: Reference scores and/or sigmoid overrides.
            block_size: R, the block side length used by Pipeline C.
        """
        self.condition_checker = ConditionChecker()
        self.preprocessor = WaveletPreprocessor()
        self.noise_extractor = NoiseResidualExtractor()
        self.compression_estimator = CompressionHistoryEstimator()
        self.copy_move_detector = CopyMoveDetector()
        self.scorer = WaveletScorer(calibration)
        self.visualizer = WaveletVisualizer()
        self.block_size = block_size

    def analyse(self, engine_input: EngineInput) -> EngineOutput:
        """Run the full wavelet-domain pipeline on one image.

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
            logger.exception("wavelet engine failed")
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
        # ENHANCEMENT 6: cap the analysed resolution before anything else.
        image, scale = self.preprocessor.limit_resolution(engine_input.image)
        self._downscale_note = self._describe_downscale(engine_input.image, image,
                                                         scale)
        condition = self.condition_checker.check(engine_input.metadata, image,
                                                  self.block_size)
        if condition.skip_engine:
            return self._null_output(condition.reliability_note, start_time)

        prepared = self.preprocessor.prepare(image, engine_input.metadata)
        ll_subband = self.preprocessor.extract_ll_subband(prepared.grayscale)
        blocks = self.preprocessor.tile_blocks(ll_subband, self.block_size)

        block_check = self.condition_checker.assess_block_count_sufficiency(len(blocks))
        if not block_check[0]:
            return self._null_output(block_check[2], start_time)

        return self._score_and_assemble(prepared.grayscale, ll_subband, blocks,
                                        condition, block_check, start_time)

    @staticmethod
    def _describe_downscale(original, analysed, scale: float) -> str:
        """Describe the analysed-resolution cap for the reliability note.

        Args:
            original: The caller's own image array.
            analysed: The array actually analysed.
            scale: Linear scale factor applied, 1.0 when none was.

        Returns:
            Note fragment, or an empty string when nothing was downscaled.
        """
        if scale >= 1.0:
            return ""
        floor_pixels = int(round(2 * constants.DEFAULT_BLOCK_SIZE / scale))
        return (f"ENHANCEMENT 6: analysed at {analysed.shape[1]}x"
                f"{analysed.shape[0]} instead of {original.shape[1]}x"
                f"{original.shape[0]} (scale {scale:.3f}); at full resolution "
                f"the stride-1 candidate search is intractable. This sets a "
                f"floor on detectable forgery size - a duplicated region "
                f"smaller than roughly {floor_pixels} px across in the "
                f"original will not be resolved.")

    def _score_and_assemble(self, grayscale, ll_subband, blocks,
                            condition, block_check, start_time) -> EngineOutput:
        """Run all three pipelines and assemble the final EngineOutput.

        Args:
            grayscale: Float64 grayscale image.
            ll_subband: Haar-DWT LL subband.
            blocks: Tiled Block list.
            condition: Pre-computation ConditionReport.
            block_check: Result of assess_block_count_sufficiency.
            start_time: perf_counter() timestamp from analyse().

        Returns:
            EngineOutput.
        """
        moment_check = self._assess_moments(blocks)
        copy_move = self.copy_move_detector.compute(blocks, ll_subband.shape,
                                                     self.block_size)
        noise_result, compression_result, auxiliary_errors = (
            self._run_auxiliary_pipelines(grayscale))

        scoring = self._score(copy_move, condition, block_check, moment_check)
        evidence_map = self.visualizer.render_duplicate_overlay(
            ll_subband, copy_move.duplicate_map)
        flagged_regions = self._build_flagged_regions(copy_move.confirmed_pairs)
        steps = self._build_computation_steps(
            blocks, copy_move, noise_result, compression_result,
            scoring["route"], scoring["is_calibrated"])
        note = self._compose_reliability_note(
            condition, moment_check, compression_result, scoring["score_note"],
            auxiliary_errors, copy_move)

        elapsed_ms = (time.perf_counter() - start_time) * constants.MILLISECONDS_PER_SECOND
        return EngineOutput(
            engine_name=constants.ENGINE_NAME, raw_score=scoring["raw_score"],
            probability=scoring["probability"], confidence=scoring["confidence"],
            is_reliable=scoring["is_reliable"], reliability_note=note,
            evidence_map=evidence_map, flagged_regions=flagged_regions,
            computation_steps=steps, processing_time_ms=elapsed_ms,
            skill_version=constants.SKILL_VERSION)

    def _run_auxiliary_pipelines(self, grayscale) -> tuple:
        """Run both non-scoring pipelines, surviving a failure in either.

        ENHANCEMENT 2: Pipelines A and B are explicitly non-scoring, so a
        failure inside either must not be able to void Pipeline C's result.
        Measured before this change: 3 of the 6 corpus images raised out of
        Pipeline A's swt2 call, and analyse()'s top-level except turned that
        into a whole-engine failure that discarded a completed, correct
        Pipeline C result - 485s, 496s and 209s of work respectively.

        Args:
            grayscale: Float64 grayscale image.

        Returns:
            Tuple of (Pipeline A result or None, Pipeline B result or None,
            list of error notes for whichever failed).
        """
        noise_result, noise_error = self._run_auxiliary(
            self.noise_extractor.compute, grayscale)
        compression_result, compression_error = self._run_auxiliary(
            self.compression_estimator.compute, grayscale)
        return noise_result, compression_result, [
            note for note in (noise_error, compression_error) if note]

    @staticmethod
    def _run_auxiliary(computation, grayscale) -> tuple:
        """Run a non-scoring pipeline, capturing rather than propagating failure.

        Args:
            computation: The auxiliary pipeline's compute callable.
            grayscale: Float64 grayscale image to pass to it.

        Returns:
            Tuple of (result or None, error note or empty string).
        """
        try:
            return computation(grayscale), ""
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            logger.warning("auxiliary pipeline failed: %s", error)
            return None, (f"{computation.__self__.__class__.__name__} "
                          f"(auxiliary, non-scoring) failed and was skipped: "
                          f"{error}. raw_score is unaffected.")

    def _score(self, copy_move, condition, block_check, moment_check) -> dict:
        """Turn the raw fraction-flagged scalar into the full scoring bundle.

        Args:
            copy_move: Pipeline C's CopyMoveResult.
            condition: Pre-computation ConditionReport.
            block_check: Result of assess_block_count_sufficiency.
            moment_check: Combined result of _assess_moments.

        Returns:
            Dict with raw_score, probability, confidence, is_reliable,
            route, is_calibrated, and score_note.
        """
        probability, route, is_calibrated, score_note = self.scorer.to_probability(
            copy_move.fraction_flagged)
        confidence = compose_confidence_penalties(
            [condition.confidence_weight, block_check[1], moment_check[1]])
        # ENHANCEMENT 5: an overflowed candidate search produced no usable
        # match set, so the zero it returns is an absence of evidence, not
        # evidence of absence.
        is_reliable = (condition.is_reliable and moment_check[0]
                       and not copy_move.search_overflowed)
        return {
            "raw_score": float(copy_move.fraction_flagged),
            "probability": probability if is_reliable else None,
            "confidence": confidence if is_reliable else constants.ZERO_CONFIDENCE,
            "is_reliable": is_reliable, "route": route,
            "is_calibrated": is_calibrated, "score_note": score_note,
        }

    def _assess_moments(self, blocks) -> tuple:
        """Compute mu_00 and pixel variance per block, run both degeneracy checks.

        Args:
            blocks: Tiled Block list.

        Returns:
            Combined CheckResult (passed, confidence_penalty, note) - the
            more restrictive verdict of assess_moment_degeneracy and
            assess_texture_degeneracy.
        """
        mu00_values = [float(np.sum(block.pixels)) for block in blocks]
        variances = [float(np.var(block.pixels)) for block in blocks]
        mass_check = self.condition_checker.assess_moment_degeneracy(mu00_values)
        texture_check = self.condition_checker.assess_texture_degeneracy(variances)
        passed = mass_check[0] and texture_check[0]
        penalty = compose_confidence_penalties([mass_check[1], texture_check[1]])
        note = " ".join(fragment for fragment in (mass_check[2], texture_check[2])
                        if fragment)
        return passed, penalty, note

    def _build_flagged_regions(self, confirmed_pairs) -> Optional[list]:
        """Convert confirmed duplicate pairs into the flagged_regions contract.

        Args:
            confirmed_pairs: List of DuplicatePair objects.

        Returns:
            List of region dicts, or None if no pairs were confirmed.
        """
        if not confirmed_pairs:
            return None
        return [{"block_a_row": pair.block_a_row, "block_a_col": pair.block_a_col,
                "block_b_row": pair.block_b_row, "block_b_col": pair.block_b_col,
                "block_size": self.block_size, "similarity": pair.similarity}
               for pair in confirmed_pairs]

    def _build_computation_steps(self, blocks, copy_move, noise_result,
                                 compression_result, route, is_calibrated) -> list:
        """Assemble the computation_steps log for the report generator.

        Args:
            blocks: Tiled Block list.
            copy_move: CopyMoveResult, the score-driving Pipeline C output.
            noise_result: NoiseResidualResult, auxiliary Pipeline A output.
            compression_result: CompressionHistoryResult, auxiliary Pipeline B output.
            route: Scoring route name from the scorer.
            is_calibrated: Whether the empirical-CDF route was used.

        Returns:
            List of computation-step dicts.
        """
        specs = [
            ("LL subband + block tiling",
             "Single-level Haar DWT, kept the LL subband, tiled into "
             "overlapping blocks.",
             {"total_blocks": len(blocks), "block_size": self.block_size}),
            ("Pipeline C: copy-move detection (SCORE-DRIVING)",
             "Blur-invariant moment features, PCA reduction, similarity "
             "search, and 16-neighbour consistency confirmation.",
             self._pipeline_c_key_values(copy_move)),
            ("Pipeline A: noise-residual extraction (AUXILIARY, NOT SCORED)",
             "Feeds the separate noise-analysis engine per the SKILL; does "
             "not affect this engine's raw_score.",
             self._pipeline_a_key_values(noise_result)),
            ("Pipeline B: compression-history fit (AUXILIARY, LOW-TRUST, NOT SCORED)",
             "SKILL: this pipeline is documented as 100%-defeatable by a "
             "knowledgeable adversary; reported as evidence only.",
             self._pipeline_b_key_values(compression_result)),
            ("Scoring", f"Fraction-flagged scalar converted to probability "
             f"via the {route} route.", {"is_calibrated": is_calibrated}),
        ]
        return [build_computation_step(index, name, description, key_values=values)
               for index, (name, description, values) in enumerate(specs, start=1)]

    def _pipeline_c_key_values(self, copy_move) -> dict:
        """Report values for Pipeline C's computation_steps entry."""
        return {"flagged_block_count": copy_move.flagged_block_count,
               "total_blocks": copy_move.total_blocks,
               "fraction_flagged": round(copy_move.fraction_flagged,
                                        constants.TRACE_DECIMAL_PLACES),
               "confirmed_pairs": len(copy_move.confirmed_pairs)}

    def _pipeline_a_key_values(self, noise_result) -> dict:
        """Report values for Pipeline A's computation_steps entry."""
        if noise_result is None:
            return {"ran": False, "note": "auxiliary pipeline failed; skipped"}
        return {"noise_sigma": round(noise_result.sigma_estimate,
                                    constants.TRACE_DECIMAL_PLACES),
               "threshold_method": noise_result.threshold_method,
               "threshold_mode": noise_result.threshold_mode}

    def _pipeline_b_key_values(self, compression_result) -> dict:
        """Report values for Pipeline B's computation_steps entry."""
        if compression_result is None:
            return {"ran": False, "note": "auxiliary pipeline failed; skipped"}
        return {"lambda_hat": round(compression_result.lambda_hat,
                                   constants.TRACE_DECIMAL_PLACES),
               "converged": compression_result.converged,
               "iterations_run": compression_result.iterations_run}

    def _compose_reliability_note(self, condition, moment_check,
                                  compression_result, score_note,
                                  auxiliary_errors, scoring_overflow) -> str:
        """Combine every note fragment into one reliability_note string.

        Args:
            condition: Pre-computation ConditionReport.
            moment_check: Result of assess_moment_degeneracy.
            compression_result: CompressionHistoryResult, for the low-trust caveat.
            score_note: Note from the scorer's chosen route.
            auxiliary_errors: Notes for any non-scoring pipeline that failed.
            scoring_overflow: Pipeline C's CopyMoveResult, for its overflow flag.

        Returns:
            Combined reliability_note string.
        """
        fragments = [STANDING_LIMITATIONS_NOTE]
        if getattr(self, "_downscale_note", ""):
            fragments.append(self._downscale_note)
        if moment_check[2]:
            fragments.append(moment_check[2])
        if compression_result is not None:
            fragments.append(
                f"Pipeline B auxiliary evidence: lambda_hat="
                f"{compression_result.lambda_hat:.4f} "
                f"(converged={compression_result.converged}) - not used in "
                f"raw_score; treat as low-trust per the SKILL's own caveat.")
        if getattr(scoring_overflow, "search_overflowed", False):
            fragments.append(
                f"Pipeline C's candidate search exceeded "
                f"{constants.MAXIMUM_CANDIDATE_PAIRS} pairs and was abandoned; "
                f"raw_score 0.0 here means NO SEARCH RAN, not 'no duplicates "
                f"found'.")
        fragments.extend(auxiliary_errors)
        fragments.append(score_note)
        return " ".join(fragments)

    def _null_output(self, note: str, start_time: float) -> EngineOutput:
        """Build the EngineOutput for a skipped or ungradeable run.

        Args:
            note: Explanation for why the engine did not produce a score.
            start_time: perf_counter() timestamp from analyse().

        Returns:
            EngineOutput with is_reliable=False and no probability.
        """
        return EngineOutput(
            engine_name=constants.ENGINE_NAME, raw_score=0.0, probability=None,
            confidence=constants.ZERO_CONFIDENCE, is_reliable=False,
            reliability_note=note, evidence_map=None, flagged_regions=None,
            computation_steps=[],
            processing_time_ms=(time.perf_counter() - start_time)
            * constants.MILLISECONDS_PER_SECOND,
            skill_version=constants.SKILL_VERSION)

    def _failure_output(self, error_message: str, start_time: float) -> EngineOutput:
        """Build the EngineOutput for an unexpected computation failure.

        Args:
            error_message: str(exception) describing what failed.
            start_time: perf_counter() timestamp from analyse().

        Returns:
            EngineOutput with is_reliable=False and the error in the note.
        """
        return EngineOutput(
            engine_name=constants.ENGINE_NAME, raw_score=0.0, probability=None,
            confidence=constants.ZERO_CONFIDENCE, is_reliable=False,
            reliability_note=f"Engine failed during computation: {error_message}",
            evidence_map=None, flagged_regions=None, computation_steps=[],
            processing_time_ms=(time.perf_counter() - start_time)
            * constants.MILLISECONDS_PER_SECOND,
            skill_version=constants.SKILL_VERSION)
