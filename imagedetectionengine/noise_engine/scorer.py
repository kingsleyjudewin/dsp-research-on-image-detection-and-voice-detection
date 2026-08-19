"""Maps Pipeline A's aggregate anomaly scalar onto a [0, 1] probability.

SKILL, Output section: Pipeline A's calibration is explicitly "not specified
in corpus - engineering recommendation." The same two-route pattern used by
every other engine in this system applies: empirical CDF against known-
authentic reference scores when supplied, otherwise a flagged-uncalibrated
sigmoid.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from . import constants
from .contracts import CalibrationSettings
from .utils import clip_to_unit_interval

logger = logging.getLogger(__name__)

ScoringResult = tuple

UNCALIBRATED_ABSTENTION_NOTE = (
    "PROVISIONAL CALIBRATION: no known-authentic reference scores were "
    "supplied. The SKILL states Pipeline A's aggregation and calibration are "
    "an engineering recommendation, not a corpus value. Diagnostic testing "
    "measured the aggregate scalar moving +0.0340 to +0.0772 under a real "
    "local noise manipulation while six untampered images spanned 0.1222 - "
    "the between-image spread exceeds the manipulation effect, so no fixed "
    "cutoff on this scalar separates tampered from authentic. This engine "
    "therefore publishes no probability on the uncalibrated route; supply "
    "authentic_reference_scores to enable the empirical-CDF route. The "
    "raw_score, heatmap and flagged_regions are still published."
)


class NoiseScorer:
    """Turns Pipeline A's top-k% aggregate scalar into a probability."""

    def __init__(self, calibration: Optional[CalibrationSettings] = None) -> None:
        """Bind optional calibration state supplied once by the orchestrator.

        Args:
            calibration: Reference scores and/or sigmoid overrides.
        """
        self.calibration = calibration or CalibrationSettings()

    def to_probability(self, raw_score: float) -> ScoringResult:
        """Convert the aggregate anomaly scalar into a probability in [0, 1].

        Args:
            raw_score: Top-k% mean of Pipeline A's [0,1]-normalised heatmap.

        Returns:
            Tuple of (probability, route name, is_calibrated, note).
        """
        if not np.isfinite(raw_score):
            logger.warning("non-finite raw score %r; returning 0.0", raw_score)
            return (0.0, "none", False,
                    "Raw score was not finite; probability floored to 0.0.")

        reference_scores = self._lookup_reference_scores()
        if reference_scores is not None:
            probability = self._empirical_cdf_percentile(raw_score, reference_scores)
            return (probability, "empirical_cdf", True,
                    f"Calibrated by empirical CDF against "
                    f"{reference_scores.size} known-authentic reference "
                    f"aggregate scores.")

        return (self._provisional_sigmoid(raw_score), "provisional_sigmoid",
                False, UNCALIBRATED_ABSTENTION_NOTE)

    def _empirical_cdf_percentile(self, raw_score: float,
                                  reference_scores: np.ndarray) -> float:
        """Percentile rank of the score within a known-authentic distribution.

        Args:
            raw_score: Aggregate score measured on the image under test.
            reference_scores: Aggregate scores measured on authentic images.

        Returns:
            Fraction of reference scores at or below the test score, in [0, 1].
        """
        below_or_equal = np.count_nonzero(reference_scores <= raw_score)
        return clip_to_unit_interval(
            float(below_or_equal) / float(reference_scores.size))

    def _provisional_sigmoid(self, raw_score: float) -> float:
        """Logistic fallback mapping of the score onto [0, 1].

        Args:
            raw_score: Top-k% mean of Pipeline A's heatmap.

        Returns:
            Probability in [0, 1], monotonically increasing in raw_score.
        """
        slope = (self.calibration.sigmoid_slope
                 if self.calibration.sigmoid_slope is not None
                 else constants.PROVISIONAL_SIGMOID_SLOPE)
        midpoint = (self.calibration.sigmoid_midpoint
                    if self.calibration.sigmoid_midpoint is not None
                    else constants.PROVISIONAL_SIGMOID_MIDPOINT)
        exponent = float(np.clip(slope * (raw_score - midpoint),
                                 -constants.SIGMOID_EXPONENT_LIMIT,
                                 constants.SIGMOID_EXPONENT_LIMIT))
        return clip_to_unit_interval(1.0 / (1.0 + np.exp(-exponent)))

    def _lookup_reference_scores(self) -> Optional[np.ndarray]:
        """Fetch authentic reference scores, if enough were supplied.

        Returns:
            Array of reference scores, or None when absent or too small.
        """
        table = self.calibration.authentic_reference_scores
        if table is None:
            return None
        try:
            scores = np.asarray(table, dtype=np.float64).ravel()
        except (TypeError, ValueError) as error:
            logger.warning("calibration reference scores are unreadable: %s", error)
            return None
        if scores.size < constants.MINIMUM_CALIBRATION_REFERENCE_COUNT:
            return None
        return scores
