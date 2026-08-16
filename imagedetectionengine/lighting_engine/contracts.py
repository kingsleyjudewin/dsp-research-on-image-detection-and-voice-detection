"""Input/output dataclasses shared across all nine forensic engines.

The EngineOutput shape is FIXED. The fusion layer and the report generator both
bind to it by field name, so fields must not be added, removed or renamed here
without changing every engine in the system simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class ImageMetadata:
    """Facts about the image established before any engine runs.

    Populated by the orchestrator's ingest stage. This engine's condition
    checker deliberately does not read estimated_compression_level: the SKILL
    file gives no quality-factor operating envelope for this module.
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
        raw_score: max_grad divided by the image's own median gradient
            magnitude - a scale-invariant ratio, bounded below at 1.0 (the
            maximum of an array can never be smaller than its median) and
            unbounded above. NOT validated lighting-consistency evidence; see
            constants.py's module docstring before trusting this number.
        probability: Calibrated tampering probability in [0, 1], or None when
            the engine did not run. Always subject to
            constants.MAXIMUM_CONFIDENCE_CEILING via the confidence field.
        confidence: Self-assessed reliability weight in [0, 1] for the fusion
            layer, hard-capped near zero for this engine regardless of route.
        is_reliable: Whether the computation completed on structurally valid,
            non-degenerate input. Independent of confidence: this engine can be
            "reliable" (the number is well-formed and reproducible) while still
            being reported at near-zero confidence (the number is not validated
            lighting evidence).
        reliability_note: Human-readable justification, quoting the SKILL
            condition that fired. Read directly into the forensic report.
        evidence_map: Rendered gradient-magnitude heatmap, or None.
        flagged_regions: Always None. The SKILL gives no threshold or
            validated procedure for turning the gradient map into discrete
            suspect regions, so this engine is a global scalar signal only.
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
        is_reliable: False when the input is structurally unusable.
        confidence_weight: Product of every triggered penalty, in [0, 1],
            already bounded by MAXIMUM_CONFIDENCE_CEILING.
        reliability_note: Concatenated explanation of everything that fired.
        skip_engine: True when the input cannot be processed at all.
    """

    is_reliable: bool
    confidence_weight: float
    reliability_note: str
    skip_engine: bool


@dataclass
class PreparedImage:
    """Output of Stage 2, ready for the mathematical core.

    Attributes:
        grayscale: Float64 single-plane image, ITU-R BT.601 luma.
        original_shape: Shape of the image before conversion, for the trace.
    """

    grayscale: np.ndarray
    original_shape: tuple


@dataclass
class GradientMagnitudeResult:
    """Everything Pipeline A produced.

    Attributes:
        gradient_magnitude: Per-pixel sqrt(Gx^2 + Gy^2) map, same shape as the
            grayscale input.
        max_gradient: The scalar max_grad of the SKILL's MATLAB code.
        median_gradient: Median of gradient_magnitude, the normalising divisor.
        ratio: max_gradient divided by median_gradient (floored), the raw
            score this engine publishes.
        is_degenerate: True when the image had too little texture (median
            gradient at or below the minimum) for the ratio to be meaningful.
    """

    gradient_magnitude: np.ndarray
    max_gradient: float
    median_gradient: float
    ratio: float
    is_degenerate: bool


@dataclass
class CalibrationSettings:
    """Optional calibration state supplied once by the orchestrator.

    Attributes:
        authentic_reference_scores: Raw ratio scores previously measured on
            images known to be authentic. When enough are supplied the
            empirical-CDF route is used in place of the provisional sigmoid.
        sigmoid_slope: Override for the fallback sigmoid slope.
        sigmoid_midpoint: Override for the fallback sigmoid midpoint.
    """

    authentic_reference_scores: Optional[Any] = None
    sigmoid_slope: Optional[float] = None
    sigmoid_midpoint: Optional[float] = None
