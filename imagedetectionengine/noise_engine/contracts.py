"""Dataclass contracts for the noise-pattern forgery-detection engine.

EngineInput / EngineOutput / ImageMetadata / ConditionReport are the fixed
shapes shared across all 9 engines in this system; the fusion layer depends
on them being exact. Do not add or remove a field on those four.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    """Grayscale image ready for wavelet residual extraction."""

    grayscale: np.ndarray  # float64, H x W
    original_shape: tuple


@dataclass
class NoiseBlock:
    """One tile of the residual map, with its pixel-space origin."""

    residual_pixels: np.ndarray  # float64, block_size x block_size
    intensity_pixels: np.ndarray  # float64, matching original-image intensity
    row: int  # grid row index (not pixel row)
    col: int  # grid column index (not pixel column)
    pixel_row: int  # top-left pixel row
    pixel_col: int  # top-left pixel column


@dataclass
class LocalInconsistencyResult:
    """Full output of Pipeline A, the sole score-driving pipeline."""

    heatmap: np.ndarray  # [0,1]-normalised, block-grid shape
    flagged_blocks: list  # list[NoiseBlock]
    total_blocks: int
    flagged_block_count: int
    aggregate_scalar: float  # fraction of blocks flagged as deviant
    block_size: int
    legacy_top_k_scalar: float = 0.0  # pre-enhancement scalar, reported only


@dataclass
class SpectralAnalysisResult:
    """Auxiliary, non-scoring output of Pipeline C."""

    s_mean_p_val: float
    s_mean_p_pos: float
    s_mean_p_pv: float
    s_rms_p_val: float
    s_rms_p_pos: float
    s_rms_p_pv: float
    grid_cells: int


@dataclass
class ReferencePRNUResult:
    """Auxiliary, non-scoring output of Pipeline B (calibration-gated)."""

    ran: bool
    mean_block_correlation: Optional[float] = None
    reference_image_count: int = 0
    note: str = ""


@dataclass
class NoiseTriageResult:
    """Auxiliary, confidence-modulating-only output of Pipeline D."""

    ran: bool
    label: Optional[str] = None  # "Gaussian blur" / "Impulse" / "No noise"
    z_score: Optional[float] = None
    note: str = ""


@dataclass
class CalibrationSettings:
    """Optional orchestrator-supplied calibration, mirrors the other engines."""

    authentic_reference_scores: Optional[np.ndarray] = None
    sigmoid_slope: Optional[float] = None
    sigmoid_midpoint: Optional[float] = None
    same_camera_reference_images: Optional[list] = None  # Pipeline B
    noise_triage_reference_mean_magnitude: Optional[float] = None  # Pipeline D
