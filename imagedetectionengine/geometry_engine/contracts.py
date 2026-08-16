"""Input/output dataclasses shared across all nine forensic engines.

The EngineOutput shape is FIXED. The fusion layer and the report generator both
bind to it by field name, so fields must not be added, removed or renamed here
without changing every engine in the system simultaneously.

Types below EngineOutput are internal to this engine and carry the intermediate
results of the vanishing-point and height-ratio modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class ImageMetadata:
    """Facts about the image established before any engine runs.

    Populated by the orchestrator's ingest stage (container parsing, EXIF read,
    and the JPEG-compression engine's quality-factor estimate).
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
            it is 1 - C_min over every evaluated object pair, where C is Yao
            et al.'s Eq. 8 consistency score. 0 = every pair obeys the
            vanishing-line height constraint, 1 = at least one pair violates it
            badly. Bounded to [0, 1] because C is.
        probability: Calibrated tampering probability in [0, 1], or None when
            the engine did not run or abstained.
        confidence: Self-assessed reliability weight in [0, 1] for the fusion
            layer. Independent of probability.
        is_reliable: Whether this vote should be counted at full weight.
        reliability_note: Human-readable justification, quoting the SKILL
            condition that fired. Read directly into the forensic report.
        evidence_map: Annotated BGR overlay showing the estimated vanishing
            line, its supporting line segments, and each tested region coloured
            by its consistency score. None when nothing was measured.
        flagged_regions: Object regions whose pair consistency fell below the
            paper's threshold. This engine localises, so this is populated
            whenever an inconsistent pair is found.
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
class LineSegment:
    """One detected straight edge, in both endpoint and homogeneous form.

    Attributes:
        start: (column, row) of the first endpoint, in pixels.
        end: (column, row) of the second endpoint.
        homogeneous: Length-3 homogeneous line coefficients, normalised so the
            first two entries form a unit vector. A point p lies on the line
            when homogeneous . [p_x, p_y, 1] is zero.
        direction: Unit direction vector (column, row) of the segment.
        length: Segment length in pixels.
    """

    start: tuple
    end: tuple
    homogeneous: np.ndarray
    direction: np.ndarray
    length: float


@dataclass
class ObjectRegion:
    """A candidate object presumed to rest on the reference plane.

    Attributes:
        identifier: Stable index for cross-referencing in the trace and report.
        top_row: Image row of the region's top edge, Yao's v_B1.
        bottom_row: Image row of the region's bottom edge, Yao's v_B2.
        left_column: Image column of the region's left edge.
        right_column: Image column of the region's right edge.
        appearance_signature: Normalised colour histogram used only to avoid
            pairing obviously unlike regions. Not a classifier.
        source: How the region was obtained, "supplied" or "slic".
    """

    identifier: int
    top_row: float
    bottom_row: float
    left_column: float
    right_column: float
    appearance_signature: Optional[np.ndarray] = None
    source: str = "slic"

    @property
    def image_height(self) -> float:
        """Height of the region in image rows.

        Returns:
            Non-negative pixel height, bottom_row minus top_row.
        """
        return float(self.bottom_row - self.top_row)


@dataclass
class PreparedScene:
    """Output of Stage 2, ready for the mathematical core.

    Attributes:
        colour_image: Float64 BGR image as supplied, unmodified.
        grayscale: Float64 single-plane image used for edges and keypoints.
        line_segments: Straight edges from the Hough transform, module A1.
        keypoint_positions: SIFT keypoint (column, row) coordinates, module A4.
        keypoint_scales: SIFT keypoint scales, needed by the scale score S_S.
        keypoint_descriptors: 128-dimensional SIFT descriptors, module A4.
        regions: Candidate object regions from SLIC, module D.
        original_shape: Shape of the image as received, for the trace.
    """

    colour_image: np.ndarray
    grayscale: np.ndarray
    line_segments: list
    keypoint_positions: np.ndarray
    keypoint_scales: np.ndarray
    keypoint_descriptors: np.ndarray
    regions: list
    original_shape: tuple


@dataclass
class VanishingPointEstimate:
    """Result of module A - a vanishing point and how much to trust it.

    Attributes:
        homogeneous_point: Length-3 homogeneous vanishing point. The third
            entry may be zero, meaning a point genuinely at infinity.
        vanishing_line_row: Image row v0 of the reference plane's vanishing
            line, or None when it could not be established.
        method: Which sub-module produced it, one of "A1_parallel_lines",
            "A4_recurrence", "A2_reference_objects" or "none".
        inlier_count: Lines agreeing with the estimate.
        total_line_count: Lines considered.
        inlier_fraction: inlier_count divided by total_line_count.
        line_fit_residual_pixels: Root-mean-square orthogonal distance from
            measured line endpoints to the refined lines, module A1's
            confidence indicator.
        is_at_infinity: True when the homogeneous scale component vanished.
        inlier_segments: The supporting line segments, for the evidence map.
        note: Human-readable account of how the estimate was reached.
    """

    homogeneous_point: Optional[np.ndarray]
    vanishing_line_row: Optional[float]
    method: str
    inlier_count: int
    total_line_count: int
    inlier_fraction: float
    line_fit_residual_pixels: float
    is_at_infinity: bool
    inlier_segments: list = field(default_factory=list)
    note: str = ""


@dataclass
class HeightRatioMeasurement:
    """One evaluated object pair under Yao et al. Eq. 7-8.

    Attributes:
        first_region_id: Identifier of the first region of the pair.
        second_region_id: Identifier of the second region.
        measured_ratio: Yao's beta, the real-world height ratio recovered from
            image coordinates via Eq. 7.
        expected_ratio: Yao's alpha, the ratio the pair was expected to show.
        sigma: The 0.1 * alpha standard deviation of Eq. 8.
        consistency: Yao's C from Eq. 8, in [0, 1]. 1 = perfectly consistent.
        is_consistent: Whether C reached the paper's threshold T = 0.5.
        expected_ratio_was_assumed: True when alpha came from the engine's
            default rather than from the caller.
        ratio_sensitivity_per_pixel: |d beta / d v0|, how much the measured
            ratio moves per pixel of error in the vanishing line. Derived from
            Eq. 7 itself by perturbation, not from any outside model. Pairs
            straddling very different depths have a large value here and their
            verdicts should be read with corresponding caution.
        tolerable_vanishing_line_error_pixels: How far v0 could be wrong before
            this pair's consistency would cross the paper's threshold.
    """

    first_region_id: int
    second_region_id: int
    measured_ratio: float
    expected_ratio: float
    sigma: float
    consistency: float
    is_consistent: bool
    expected_ratio_was_assumed: bool
    ratio_sensitivity_per_pixel: float = 0.0
    tolerable_vanishing_line_error_pixels: float = float("inf")


@dataclass
class HeightRatioAnalysis:
    """Everything module B produced, before scoring.

    Attributes:
        measurements: One HeightRatioMeasurement per evaluated pair.
        minimum_consistency: The smallest C observed, which drives raw_score.
        mean_consistency: Mean C across pairs, the averaging SKILL B step 5
            recommends.
        mean_measured_ratio: Mean beta across pairs.
        evaluated_pair_count: Pairs that produced a measurement.
        rejected_pair_count: Pairs discarded by the ground-plane sanity check.
        any_ratio_assumed: True when any pair used the default expected ratio.
    """

    measurements: list = field(default_factory=list)
    minimum_consistency: float = 1.0
    mean_consistency: float = 1.0
    mean_measured_ratio: float = 0.0
    evaluated_pair_count: int = 0
    rejected_pair_count: int = 0
    any_ratio_assumed: bool = False


@dataclass
class FlaggedRegion:
    """One localised suspect region, in pixel coordinates of the input image.

    Attributes:
        top_left: (row, column) of the region's top-left pixel.
        height: Region height in pixels.
        width: Region width in pixels.
        tampering_score: 1 - C for the worst pair this region took part in.
        paired_with: Identifier of the region it was least consistent with.
    """

    top_left: tuple
    height: int
    width: int
    tampering_score: float
    paired_with: int


@dataclass
class CalibrationSettings:
    """Optional calibration state supplied once by the orchestrator.

    Attributes:
        authentic_reference_scores: Raw scores previously measured on images
            known to be authentic, keyed by quality-factor bucket name. When a
            bucket holds enough samples the empirical-CDF route is used in
            place of the paper's own calibration.
        supplied_regions: Object regions the caller has already localised,
            bypassing SLIC proposal. Each entry is a dict with top_row,
            bottom_row, left_column and right_column.
        supplied_expected_ratios: Expected height ratios keyed by the
            (first_region_id, second_region_id) tuple, removing the engine's
            assumed-ratio penalty for those pairs.
        sigmoid_slope: Override for the optional logistic recalibration.
    """

    authentic_reference_scores: Optional[dict[str, Any]] = None
    supplied_regions: Optional[list] = None
    supplied_expected_ratios: Optional[dict] = None
    sigmoid_slope: Optional[float] = None
