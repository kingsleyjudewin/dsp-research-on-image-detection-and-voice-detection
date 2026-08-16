"""Input/output dataclasses shared across all nine forensic engines.

The EngineOutput shape is FIXED. The fusion layer and the report generator both
bind to it by field name, so fields must not be added, removed or renamed here
without changing every engine in the system simultaneously.

Types below EngineOutput are internal to this engine and carry the intermediate
results of the four documented pipelines.
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
    format: str                         # JPEG, PNG, WEBP, TIFF, ...
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
            it is the reduction of Ferrara's per-block tampering map
            (1 - Pr{M1|L}) to a scalar: 0 = every block carries an intact CFA
            demosaicing signature, 1 = the signature is absent, bounded to
            [0, 1] because it derives from a posterior probability.
        probability: Calibrated tampering probability in [0, 1], or None when
            the engine did not run or could not be calibrated.
        confidence: Self-assessed reliability weight in [0, 1] for the fusion
            layer. Independent of probability: a confident 0.1 and an unreliable
            0.9 are different statements.
        is_reliable: Whether this vote should be counted at full weight.
        reliability_note: Human-readable justification, quoting the SKILL
            condition that fired. Read directly into the forensic report.
        evidence_map: Rendered BGR heatmap of the per-block posterior, or None.
        flagged_regions: Localised suspect regions. Unlike the global-only
            engines, Pipeline A is a localisation method, so this is populated
            whenever blocks fall below the tampering threshold.
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
class PreparedImage:
    """Output of Stage 2, ready for the mathematical core.

    Attributes:
        green_channel: Float64 green plane, cropped to a whole Bayer grid.
        colour_image: Float64 BGR image cropped identically, needed by
            Pipeline C's colour-difference blocks and Pipeline B's mosaics.
        texture_mask: Boolean per-pixel mask, True where the pixel belongs to a
            block with enough texture to be diagnostic. Implements Ferrara's
            "almost flat areas or sharp edges" limitation.
        excluded_block_fraction: Fraction of blocks dropped as flat or
            edge-dominated.
        original_shape: Shape of the image before cropping, for the trace.
    """

    green_channel: np.ndarray
    colour_image: np.ndarray
    texture_mask: np.ndarray
    excluded_block_fraction: float
    original_shape: tuple


@dataclass
class CfaPhaseEstimate:
    """Result of Pipeline C (Jeon et al.) - which Bayer configuration is in use.

    Attributes:
        configuration_name: One of constants.CFA_CONFIGURATION_NAMES.
        green_acquired_parity: 0 or 1 - the parity of (row + column) at which
            green is a directly sensed sample. This is what Pipeline A needs.
        block_size: The M actually used, taken from the corpus ladder.
        diagonal_scores: The V^D + V^F sum for each of the two diagonal pairs.
        was_estimated: False when the image was too small for the smallest
            block size Jeon reports, in which case the parity was inferred from
            the sign of the feature instead of verified.
        note: Human-readable account of how the phase was determined.
    """

    configuration_name: str
    green_acquired_parity: int
    block_size: int
    diagonal_scores: tuple
    was_estimated: bool
    note: str


@dataclass
class GaussianMixtureFit:
    """Parameters of the two-component mixture of Ferrara Eq. 13-14.

    Attributes:
        authentic_mean: mu1, the mean of L under M1 (CFA present). Eq. 13
            requires this to be positive; a non-positive value means no CFA
            signature was found anywhere in the image.
        authentic_variance: sigma1 squared.
        tampered_variance: sigma2 squared. The tampered mean is fixed at zero
            by Eq. 14 and is therefore not stored.
        mixing_weight: EM's converged proportion of M1 blocks. Reported for the
            trace only - Eq. 15-16 fix the posterior priors at 1/2.
        iterations: EM iterations actually run.
        converged: True when the log-likelihood tolerance was met before the
            iteration cap.
        final_log_likelihood: Log-likelihood at the returned parameters.
    """

    authentic_mean: float
    authentic_variance: float
    tampered_variance: float
    mixing_weight: float
    iterations: int
    converged: bool
    final_log_likelihood: float


@dataclass
class GridConsistencyResult:
    """Result of Pipeline B (Bammey et al.) - the a contrario confirmatory layer.

    Attributes:
        dominant_position_index: Globally most-voted grid position P0, or None
            when no position was statistically meaningful.
        dominant_log10_nfa: log10 NFA of that global vote.
        vote_map: Per-block winning grid position index.
        forged_windows: (row, column, position_index, log10_nfa) per window
            holding a significant position that disagrees with P0.
        window_count: Number of windows tested, the z of the NFA formula.
        is_conclusive: True when a globally dominant position was meaningful at
            the false-alarm budget.
        note: Human-readable account of what the layer concluded.
    """

    dominant_position_index: Optional[int]
    dominant_log10_nfa: float
    vote_map: Optional[np.ndarray]
    forged_windows: list
    window_count: int
    is_conclusive: bool
    note: str


@dataclass
class CfaComputation:
    """Everything the mathematical core produced, before scoring.

    Attributes:
        feature_map: Per-block feature L of Eq. 11.
        log_likelihood_ratio_map: log Lambda of Eq. 17, after the step-8 filter.
        tampering_map: Per-block 1 - Pr{M1|L}, the quantity that is reduced to
            raw_score.
        block_validity_mask: True where the block had enough texture to count.
        mixture: The fitted GMM.
        phase: The CFA phase estimate that set the acquired lattice.
        grid_consistency: Pipeline B's verdict, or None when it was not run.
        feature_block_size: Block size the feature was computed at.
        output_block_size: Block size of the maps returned, which differs from
            feature_block_size when cumulation is enabled.
        valid_block_count: Blocks that contributed to the statistic.
    """

    feature_map: np.ndarray
    log_likelihood_ratio_map: np.ndarray
    tampering_map: np.ndarray
    block_validity_mask: np.ndarray
    mixture: GaussianMixtureFit
    phase: CfaPhaseEstimate
    grid_consistency: Optional[GridConsistencyResult] = None
    feature_block_size: int = 0
    output_block_size: int = 0
    valid_block_count: int = 0


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


@dataclass
class FlaggedRegion:
    """One localised suspect area, in pixel coordinates of the cropped image.

    Attributes:
        top_left: (row, column) of the region's top-left pixel.
        height: Region height in pixels.
        width: Region width in pixels.
        mean_tampering_probability: Mean of 1 - Pr{M1|L} over the region.
        block_count: Number of feature blocks the region covers.
    """

    top_left: tuple
    height: int
    width: int
    mean_tampering_probability: float
    block_count: int
