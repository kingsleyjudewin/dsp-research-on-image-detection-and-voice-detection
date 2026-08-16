"""Dataclass contracts for the Fourier-domain / JPEG-ghost engine.

EngineInput / EngineOutput / ImageMetadata / ConditionReport are the fixed
shapes shared across all 9 engines in this system; the fusion layer depends
on them being exact. Do not add or remove a field on those four.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ImageMetadata:
    """Container/format facts supplied by the orchestrator, not computed here."""

    estimated_compression_level: float  # 0-100
    is_resized: bool
    color_space: str
    resolution: tuple
    format: str  # JPEG, PNG, WEBP
    has_exif: bool


@dataclass
class EngineInput:
    """What the orchestrator passes into the engine."""

    image: np.ndarray  # BGR uint8, H x W x 3
    metadata: ImageMetadata


@dataclass
class EngineOutput:
    """Fixed output shape depended on by the fusion layer. Do not modify."""

    engine_name: str
    raw_score: float
    probability: Optional[float]
    confidence: float
    is_reliable: bool
    reliability_note: str
    evidence_map: Optional[np.ndarray]
    flagged_regions: Optional[list]
    computation_steps: list
    processing_time_ms: float
    skill_version: str


@dataclass
class ConditionReport:
    """Result of the pre-computation reliability gate."""

    is_reliable: bool
    confidence_weight: float  # 0.0 to 1.0
    reliability_note: str
    skip_engine: bool  # True = return null vote, no computation runs


@dataclass
class PreparedImage:
    """The two views this engine needs of one input image."""

    colour_rgb: np.ndarray  # uint8 H x W x 3, RGB, for Pipeline B
    resampling_window: Optional[np.ndarray]  # float64, for Pipeline A
    original_shape: tuple
    resampling_window_note: str


@dataclass
class GhostCandidate:
    """One (q2, dx, dy) combination's segmentation and separability."""

    quality_factor: int
    shift_x: int
    shift_y: int
    bhattacharyya_distance: float
    mask: Optional[np.ndarray]  # binary, original image shape


@dataclass
class GhostResult:
    """Pipeline B output: the sole score-driving result."""

    max_distance: float  # D_max
    best_candidate: Optional[GhostCandidate]
    combinations_evaluated: int
    quality_factors_swept: int
    grid_shifts_swept: int
    degenerate_segmentation_count: int


@dataclass
class ResamplingResult:
    """Pipeline A output: auxiliary, cannot be scored (no rho_T in corpus)."""

    ran: bool
    probability_map: Optional[np.ndarray] = None
    decision_statistic: Optional[float] = None  # rho
    best_transform_kind: str = ""  # "scaling" / "rotation"
    best_transform_value: float = 0.0
    iterations_run: int = 0
    converged: bool = False
    synthetic_maps_evaluated: int = 0
    note: str = ""


@dataclass
class CalibrationSettings:
    """Optional orchestrator-supplied calibration, mirrors the other engines."""

    authentic_reference_scores: Optional[np.ndarray] = None
    sigmoid_slope: Optional[float] = None
    sigmoid_midpoint: Optional[float] = None
    # Pipeline B sweep control; set both to 1 for the paper's full 6400 runs.
    quality_factor_step: Optional[int] = None
    grid_shift_step: Optional[int] = None
    # Pipeline A control; it costs 692 full-size DFTs at step 1.
    run_resampling_detector: bool = True
    synthetic_map_step: Optional[int] = None
    # A locally measured rho_T, for reporting only - never used to score.
    resampling_threshold: Optional[float] = None
