"""Input/output dataclasses shared across all nine forensic engines.

The EngineOutput shape is FIXED. The fusion layer and the report generator both
bind to it by field name, so fields must not be added, removed or renamed here
without changing every engine in the system simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class ImageMetadata:
    """Facts about the image established before any engine runs.

    Populated by the orchestrator's ingest stage (container parsing, EXIF read,
    and the JPEG-compression engine's quality-factor estimate). Every field is
    consumed by this engine's condition checker.
    """

    estimated_compression_level: float  # JPEG quality factor, 0-100
    is_resized: bool
    color_space: str
    resolution: tuple                   # (height, width)
    format: str                         # JPEG, PNG, WEBP, ...
    has_exif: bool


@dataclass
class EngineInput:
    """One image plus its metadata, as handed to an engine."""

    image: np.ndarray                   # BGR uint8, H x W x 3
    metadata: ImageMetadata


@dataclass
class EngineOutput:
    """The single result shape every engine returns, success or failure.

    Attributes:
        engine_name: Stable identifier of the producing engine.
        raw_score: Engine-native statistic, before calibration. For this engine
            it is the mean symmetric-KL divergence: 0 = perfect Benford
            conformance, larger = more suspicious, unbounded above.
        probability: Calibrated tampering probability in [0, 1], or None when
            the engine did not run or could not be calibrated.
        confidence: Self-assessed reliability weight in [0, 1] for the fusion
            layer. Independent of probability: a confident 0.1 and an unreliable
            0.9 are different statements.
        is_reliable: Whether this vote should be counted at full weight.
        reliability_note: Human-readable justification, quoting the SKILL
            condition that fired. Read directly into the forensic report.
        evidence_map: Rendered visual evidence, or None.
        flagged_regions: Localised suspect regions, or None when the engine is
            global-only (as this one is).
        computation_steps: Ordered internal trace for the report generator.
        processing_time_ms: Wall-clock duration of the whole call.
        skill_version: SKILL document revision this result was produced under.
    """

    engine_name: str
    raw_score: float
    probability: Optional[float]
    confidence: float
    is_reliable: bool
    reliability_note: str
    evidence_map: Optional[np.ndarray]
    flagged_regions: Optional[list]
    computation_steps: list[dict]
    processing_time_ms: float
    skill_version: str


@dataclass
class ConditionReport:
    """Verdict of the pre-computation input gate.

    Attributes:
        is_reliable: False when a documented unreliability condition fired.
        confidence_weight: Product of every triggered penalty, in [0, 1].
        reliability_note: Concatenated explanation of everything that fired.
        skip_engine: True when the engine's premise fails outright, meaning no
            computation should run and a null vote is returned.
    """

    is_reliable: bool
    confidence_weight: float
    reliability_note: str
    skip_engine: bool


@dataclass
class DivergenceSample:
    """One point of the (base, frequency, quality factor) sweep grid.

    Produced by the computer for every grid cell that yielded enough non-zero
    coefficients to estimate a pmf.
    """

    digit_base: int
    zigzag_frequency_index: int
    quality_factor: int
    divergence: float
    chi_square: float
    sample_count: int
    zero_coefficient_rate: float
    empirical_pmf: np.ndarray
    fitted_pmf: np.ndarray
    fit_parameters: tuple


@dataclass
class BenfordComputation:
    """Everything the mathematical core produced, before scoring.

    Attributes:
        samples: One DivergenceSample per successfully evaluated grid cell.
        mean_divergence: Grid mean - the engine's raw score.
        max_divergence: Grid maximum, reported alongside per the SKILL's
            "mean or max divergence across the swept grid" wording.
        representative_chi_square: Chi-square at the quality factor closest to
            the image's own, comparable against Moin et al.'s published values.
        overall_zero_coefficient_rate: Mean fraction of coefficients discarded.
        total_block_count: Number of 8x8 blocks analysed.
        skipped_configuration_count: Grid cells that had too few samples.
        digit_clamp_event_count: Floating-point guard activations.
    """

    samples: list[DivergenceSample] = field(default_factory=list)
    mean_divergence: float = 0.0
    max_divergence: float = 0.0
    representative_chi_square: float = 0.0
    overall_zero_coefficient_rate: float = 0.0
    total_block_count: int = 0
    skipped_configuration_count: int = 0
    digit_clamp_event_count: int = 0


@dataclass
class CalibrationSettings:
    """Optional calibration state supplied once by the orchestrator.

    Attributes:
        authentic_reference_scores: Raw scores previously measured on images
            known to be authentic, keyed by quality-factor bucket name. When a
            bucket holds enough samples the empirical-CDF route is used.
        platt_slope: Override for the fallback sigmoid slope.
        platt_midpoint: Override for the fallback sigmoid midpoint.
    """

    authentic_reference_scores: Optional[dict[str, Any]] = None
    platt_slope: Optional[float] = None
    platt_midpoint: Optional[float] = None
