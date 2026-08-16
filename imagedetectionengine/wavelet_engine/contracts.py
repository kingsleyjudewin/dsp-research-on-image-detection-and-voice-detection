"""Dataclass contracts for the wavelet-domain forgery-detection engine.

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
    """Grayscale image ready for wavelet decomposition."""

    grayscale: np.ndarray  # float64, H x W
    original_shape: tuple


@dataclass
class Block:
    """One R x R tile of the LL subband, with its pixel-space origin."""

    pixels: np.ndarray  # float64, R x R
    row: int  # top-left row in the LL subband
    col: int  # top-left column in the LL subband


@dataclass
class BlockFeature:
    """A block's contrast-normalised, PCA-reduced blur-invariant vector."""

    block_index: int
    row: int
    col: int
    vector: np.ndarray  # 1-D, length m_0 after PCA reduction


@dataclass
class DuplicatePair:
    """One confirmed (post neighbour-consistency-check) duplicate block pair."""

    block_a_row: int
    block_a_col: int
    block_b_row: int
    block_b_col: int
    similarity: float


@dataclass
class CopyMoveResult:
    """Full output of Pipeline C, the sole score-driving pipeline."""

    duplicate_map: np.ndarray  # binary, LL-subband shape
    confirmed_pairs: list  # list[DuplicatePair]
    total_blocks: int
    flagged_block_count: int
    fraction_flagged: float
    block_size: int


@dataclass
class NoiseResidualResult:
    """Auxiliary, non-scoring output of Pipeline A."""

    residual: np.ndarray  # feeds the separate noise-analysis engine
    sigma_estimate: float
    threshold_used: float
    threshold_method: str
    threshold_mode: str


@dataclass
class CompressionHistoryResult:
    """Auxiliary, non-scoring output of Pipeline B."""

    lambda_hat: float
    log_c_hat: float
    iterations_run: int
    converged: bool
    fit_residual: float  # weighted sum-of-squares residual of the final fit


@dataclass
class CalibrationSettings:
    """Optional orchestrator-supplied calibration, mirrors the other engines."""

    authentic_reference_scores: Optional[np.ndarray] = None
    sigmoid_slope: Optional[float] = None
    sigmoid_midpoint: Optional[float] = None
