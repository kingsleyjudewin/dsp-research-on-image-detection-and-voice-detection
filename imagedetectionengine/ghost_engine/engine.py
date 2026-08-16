"""Entry point orchestrating the Fourier-domain / JPEG-ghost engine.

Stage 1: condition.py pre-checks. Stage 2: preprocessor.py (RGB view, plus
the CFA-suppressed centre window for Pipeline A). Stage 3: computer.py
(Pipeline B's ghost sweep, Pipeline A's EM p-map). Stage 4: scorer.py +
visualizer.py + this file's output assembly.

Only Pipeline B drives raw_score; Pipeline A is reported but cannot be
scored because the corpus gives no numeric rho_T. Pipeline C is
[ML - excluded]. See the SCOPE DECISION note in constants.py.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from . import constants
from .computer import GhostDetector, ResamplingDetector
from .condition import STANDING_LIMITATIONS_NOTE, ConditionChecker
from .contracts import (CalibrationSettings, EngineInput, EngineOutput,
                        ResamplingResult)
from .preprocessor import GhostPreprocessor
from .scorer import GhostScorer
from .utils import build_computation_step, compose_confidence_penalties
from .visualizer import GhostVisualizer

logger = logging.getLogger(__name__)

STEP_NAMES = {
    "prepare": "Image preparation",
    "ghost": "Pipeline B: automated JPEG-ghost sweep (SCORE-DRIVING)",
    "resampling": "Pipeline A: resampling periodicity (AUXILIARY, UNCALIBRATED)",
    "scoring": "Scoring",
}

STEP_DESCRIPTIONS = {
    "prepare": ("Built the RGB view the ghost difference-energy statistic "
                "sums over, and a CFA-suppressed centre window for the "
                "resampling detector (downsampled x2 with nearest-neighbour "
                "to strip demosaicing periodicity that would otherwise be "
                "mistaken for resampling periodicity)."),
    "ghost": ("For each grid shift, padded the image off the 8x8 DCT grid, "
              "recompressed at each candidate quality, took the smoothed "
              "RGB squared difference, normalised each pixel across the "
              "quality sweep, segmented ghost from background, and measured "
              "the two classes' separability by Bhattacharyya distance. The "
              "most separable combination gives D_max and the tamper mask."),
    "resampling": ("Ran the two-state EM predictor to convergence to build "
                   "the per-pixel p-map, then correlated its radially "
                   "high-passed spectrum against the bank of synthetic "
                   "transformation maps. Reported only: the corpus gives no "
                   "numeric threshold for this statistic."),
}


class GhostEngine:
    """Orchestrates condition-checking, preprocessing, computation, and scoring."""

    def __init__(self, calibration: Optional[CalibrationSettings] = None) -> None:
        """Bind stateless collaborators and optional one-time calibration.

        Args:
            calibration: Reference scores, sweep-pruning controls, and the
                Pipeline A toggle.
        """
        self.condition_checker = ConditionChecker()
        self.preprocessor = GhostPreprocessor()
        self.ghost_detector = GhostDetector(self.preprocessor)
        self.resampling_detector = ResamplingDetector()
        self.scorer = GhostScorer(calibration)
        self.visualizer = GhostVisualizer()
        self.calibration = calibration or CalibrationSettings()

    def analyse(self, engine_input: EngineInput) -> EngineOutput:
        """Run the full ghost pipeline on one image.

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
            logger.exception("ghost engine failed")
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
        metadata = engine_input.metadata
        condition = self.condition_checker.check(metadata, engine_input.image)
        if condition.skip_engine:
            return self._null_output(condition.reliability_note, start_time)

        history_check = self.condition_checker.assess_jpeg_history(metadata)
        if not history_check[0]:
            return self._null_output(
                f"{history_check[2]} {STANDING_LIMITATIONS_NOTE}", start_time)

        prepared = self.preprocessor.prepare(engine_input.image, metadata)
        ghost = self.ghost_detector.compute(
            prepared.colour_rgb, self._quality_factors(), self._grid_shifts())
        resampling = self._run_resampling(prepared)
        return self._assemble(prepared, ghost, resampling, condition,
                              history_check, metadata, start_time)

    def _assemble(self, prepared, ghost, resampling, condition, history_check,
                  metadata, start_time) -> EngineOutput:
        """Score the ghost result and build the final EngineOutput.

        Args:
            prepared: PreparedImage holding both views.
            ghost: Pipeline B's GhostResult.
            resampling: Pipeline A's ResamplingResult.
            condition: Pre-computation ConditionReport.
            history_check: Result of assess_jpeg_history.
            metadata: Container and compression facts.
            start_time: perf_counter() timestamp from analyse().

        Returns:
            EngineOutput.
        """
        checks = {
            "jpeg_history": history_check,
            "segmentation": self.condition_checker.assess_segmentation_health(ghost),
            "sweep_coverage": self.condition_checker.assess_sweep_coverage(ghost),
            "resampling": self.condition_checker.assess_resampling_applicability(
                metadata),
        }
        scoring = self._score(ghost, condition, checks)
        evidence_map, regions = self._build_localisation(prepared, ghost,
                                                          scoring["is_reliable"])
        steps = self._build_computation_steps(prepared, ghost, resampling, scoring)
        note = self._compose_reliability_note(checks, resampling, scoring)

        elapsed_ms = ((time.perf_counter() - start_time)
                      * constants.MILLISECONDS_PER_SECOND)
        return EngineOutput(
            engine_name=constants.ENGINE_NAME, raw_score=scoring["raw_score"],
            probability=scoring["probability"], confidence=scoring["confidence"],
            is_reliable=scoring["is_reliable"], reliability_note=note,
            evidence_map=evidence_map, flagged_regions=regions,
            computation_steps=steps, processing_time_ms=elapsed_ms,
            skill_version=constants.SKILL_VERSION)

    def _quality_factors(self) -> list:
        """Enumerate the candidate quality factors to sweep.

        Returns:
            List of q2 values; step 1 gives the paper's full 1..100 sweep.
        """
        step = (self.calibration.quality_factor_step
                if self.calibration.quality_factor_step is not None
                else constants.DEFAULT_QUALITY_FACTOR_STEP)
        return list(range(constants.QUALITY_FACTOR_MINIMUM,
                          constants.QUALITY_FACTOR_MAXIMUM + 1, max(step, 1)))

    def _grid_shifts(self) -> list:
        """Enumerate the (dx, dy) grid shifts to sweep.

        Returns:
            List of (dx, dy) pairs; step 1 gives the paper's full 64 shifts.
        """
        step = (self.calibration.grid_shift_step
                if self.calibration.grid_shift_step is not None
                else constants.DEFAULT_GRID_SHIFT_STEP)
        offsets = list(range(0, constants.GRID_SHIFT_MAXIMUM + 1, max(step, 1)))
        return [(shift_x, shift_y) for shift_y in offsets for shift_x in offsets]

    def _run_resampling(self, prepared) -> ResamplingResult:
        """Run Pipeline A, or report cleanly why it did not run.

        Args:
            prepared: PreparedImage holding the resampling window.

        Returns:
            ResamplingResult, with ran=False when skipped or impossible.
        """
        if not self.calibration.run_resampling_detector:
            return ResamplingResult(
                ran=False, note="Resampling detector disabled by the caller.")
        if prepared.resampling_window is None:
            return ResamplingResult(ran=False,
                                    note=prepared.resampling_window_note)
        step = (self.calibration.synthetic_map_step
                if self.calibration.synthetic_map_step is not None
                else constants.DEFAULT_SYNTHETIC_MAP_STEP)
        return self.resampling_detector.compute(prepared.resampling_window,
                                                 max(step, 1))

    def _score(self, ghost, condition, checks) -> dict:
        """Turn D_max into the full scoring bundle.

        Args:
            ghost: Pipeline B's GhostResult.
            condition: Pre-computation ConditionReport.
            checks: Dict of post-computation CheckResult tuples.

        Returns:
            Dict with raw_score, probability, confidence, is_reliable,
            route, is_calibrated, and score_note.
        """
        raw_score = self.scorer.to_raw_score(ghost.max_distance)
        probability, route, is_calibrated, score_note = self.scorer.to_probability(
            raw_score)
        confidence = compose_confidence_penalties(
            [condition.confidence_weight]
            + [check[1] for check in checks.values()])
        is_reliable = condition.is_reliable and all(
            check[0] for check in checks.values())
        return {
            "raw_score": float(raw_score),
            "probability": probability if is_reliable else None,
            "confidence": confidence if is_reliable else constants.ZERO_CONFIDENCE,
            "is_reliable": is_reliable, "route": route,
            "is_calibrated": is_calibrated, "score_note": score_note,
            "max_distance": float(ghost.max_distance),
        }

    def _build_localisation(self, prepared, ghost, is_reliable) -> tuple:
        """Render the tamper mask and its connected-component regions.

        Unlike the other engines in this system, localisation here is
        corpus-backed: the SKILL calls the segmentation mask "literally a
        segmented binary tamper mask". Regions are still only emitted when
        the run is reliable and D_max actually cleared the paper's threshold.

        Args:
            prepared: PreparedImage holding the RGB view.
            ghost: Pipeline B's GhostResult.
            is_reliable: Whether the run passed every condition check.

        Returns:
            Tuple of (evidence_map, flagged_regions).
        """
        candidate = ghost.best_candidate
        if candidate is None or candidate.mask is None:
            return None, None

        evidence_map = self.visualizer.render_mask_overlay(prepared.colour_rgb,
                                                            candidate.mask)
        cleared_threshold = (ghost.max_distance
                             > constants.BHATTACHARYYA_THRESHOLD)
        if not (is_reliable and cleared_threshold):
            return evidence_map, None
        return evidence_map, self.visualizer.mask_to_regions(candidate.mask)

    def _build_computation_steps(self, prepared, ghost, resampling,
                                 scoring) -> list:
        """Assemble the computation_steps log for the report generator.

        Args:
            prepared: PreparedImage holding both views.
            ghost: Pipeline B's GhostResult.
            resampling: Pipeline A's ResamplingResult.
            scoring: Dict returned by _score.

        Returns:
            List of computation-step dicts.
        """
        specs = [
            (STEP_NAMES["prepare"], STEP_DESCRIPTIONS["prepare"],
             {"original_shape": str(prepared.original_shape),
              "resampling_window": prepared.resampling_window_note}),
            (STEP_NAMES["ghost"], STEP_DESCRIPTIONS["ghost"],
             self._ghost_key_values(ghost)),
            (STEP_NAMES["resampling"], STEP_DESCRIPTIONS["resampling"],
             self._resampling_key_values(resampling)),
            (STEP_NAMES["scoring"],
             f"D_max normalised against the paper's threshold and mapped to "
             f"a probability via the {scoring['route']} route.",
             {"D_max": round(scoring["max_distance"],
                            constants.TRACE_DECIMAL_PLACES),
              "threshold_Th": constants.BHATTACHARYYA_THRESHOLD,
              "raw_score": round(scoring["raw_score"],
                                constants.TRACE_DECIMAL_PLACES),
              "is_calibrated": scoring["is_calibrated"]}),
        ]
        return [build_computation_step(index, name, description, key_values=values)
               for index, (name, description, values) in enumerate(specs, start=1)]

    @staticmethod
    def _ghost_key_values(ghost) -> dict:
        """Report values for Pipeline B's computation_steps entry."""
        best = ghost.best_candidate
        values = {
            "D_max": round(ghost.max_distance, constants.TRACE_DECIMAL_PLACES),
            "threshold_Th": constants.BHATTACHARYYA_THRESHOLD,
            "exceeds_threshold": bool(ghost.max_distance
                                      > constants.BHATTACHARYYA_THRESHOLD),
            "combinations_evaluated": ghost.combinations_evaluated,
            "full_search_size": constants.FULL_SEARCH_COMBINATION_COUNT,
            "degenerate_segmentations": ghost.degenerate_segmentation_count,
        }
        if best is not None:
            values["estimated_region_quality_factor"] = best.quality_factor
            values["detected_grid_shift"] = (best.shift_x, best.shift_y)
        return values

    @staticmethod
    def _resampling_key_values(resampling) -> dict:
        """Report values for Pipeline A's computation_steps entry."""
        if not resampling.ran:
            return {"ran": False, "note": resampling.note}
        return {
            "ran": True, "scored": False,
            "rho_uncalibrated": round(resampling.decision_statistic,
                                      constants.TRACE_DECIMAL_PLACES),
            "best_transform_kind": resampling.best_transform_kind,
            "best_transform_value": round(resampling.best_transform_value,
                                          constants.TRACE_DECIMAL_PLACES),
            "em_iterations": resampling.iterations_run,
            "em_converged": resampling.converged,
            "synthetic_maps_evaluated": resampling.synthetic_maps_evaluated,
            "full_search_set_size": constants.FULL_SYNTHETIC_MAP_COUNT,
        }

    @staticmethod
    def _compose_reliability_note(checks, resampling, scoring) -> str:
        """Combine every note fragment into one reliability_note string.

        Args:
            checks: Dict of post-computation CheckResult tuples.
            resampling: Pipeline A's ResamplingResult.
            scoring: Dict returned by _score.

        Returns:
            Combined reliability_note string.
        """
        fragments = [STANDING_LIMITATIONS_NOTE]
        fragments.extend(check[2] for check in checks.values() if check[2])
        if resampling.note:
            fragments.append(f"Pipeline A: {resampling.note}")
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
