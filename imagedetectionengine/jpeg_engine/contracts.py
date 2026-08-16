"""Dataclass contracts for the JPEG-compression-artifact engine.

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
    """Luminance image cropped to the 8x8 block grid, ready for block DCT."""

    luminance: np.ndarray  # float64, H x W, cropped to multiples of 8
    original_shape: tuple
    cropped_shape: tuple


@dataclass
class BlockDctResult:
    """Block-DCT coefficients for the unsaturated blocks of one image."""

    coefficients: np.ndarray  # float64, (n_blocks, 8, 8)
    total_block_count: int
    unsaturated_block_count: int
    blocks_per_row: int
    blocks_per_column: int


@dataclass
class JpegHistoryResult:
    """Pipeline A.1 output: the JPEG-history gate feature s (never scored)."""

    history_feature: float  # s, Eq. 7
    threshold: float
    is_jpeg_derived: bool
    region_one_count: int
    region_two_count: int


@dataclass
class QuantizationStepResult:
    """Pipeline A.2 output: per-frequency estimated steps (conditioning only)."""

    steps: np.ndarray  # int, 8 x 8; 0 marks an unusable frequency
    usable_frequency_count: int


@dataclass
class QualityFactorResult:
    """Pipeline A.3 output: estimated quality factor (conditioning + confidence)."""

    quality_factor: int  # Q_hat_F, Eq. 25
    pixel_match_ratio: float  # R at the winning candidate, Eq. 24
    runner_up_match_ratio: float  # second-best R, for margin reporting
    sweep_ran: bool
    note: str = ""


@dataclass
class FrequencySpectrum:
    """One DCT frequency's trend-removed histogram-FFT spectrum (Pipeline B)."""

    frequency: tuple  # (u, v) position in the 8x8 grid
    ordinal: int  # 1-based i, matching the SKILL's H_1 ... H_10
    spectrum: np.ndarray  # |H_i~| after Eq. 5 trend removal
    peak_positions: np.ndarray
    peak_prominences: np.ndarray
    strongest_prominence: float
    was_excluded: bool
    zero_coefficient_fraction: float


@dataclass
class DoubleCompressionResult:
    """Pipeline B output: the sole score-driving result."""

    spectra: list  # list[FrequencySpectrum]
    aggregate_score: float  # [0, 1]
    usable_frequency_count: int
    peak_bearing_frequency_count: int


@dataclass
class CalibrationSettings:
    """Optional orchestrator-supplied calibration, mirrors the other engines."""

    authentic_reference_scores: Optional[np.ndarray] = None
    sigmoid_slope: Optional[float] = None
    sigmoid_midpoint: Optional[float] = None
    history_threshold: Optional[float] = None  # per-image-size t, Pipeline A.1
    trend_removal_window_length: Optional[int] = None  # n, Eq. 5
    run_quality_factor_sweep: bool = True  # A.3 costs 100 JPEG round-trips
