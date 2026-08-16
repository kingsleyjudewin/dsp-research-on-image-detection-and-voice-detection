"""Maps Pipeline B's D_max onto a [0, 1] probability.

Unlike most modules in this system, the SKILL supplies its own normalisation
here: "For fusion: (not specified in corpus) - engineering recommendation:
score = clip(D_max / Th, 0, 1)". That ratio route is used as the default,
which has the useful property that raw_score crosses 1.0 exactly where
D_max crosses the paper's own decision threshold Th = 0.19.

An empirical-CDF route against known-authentic reference scores is still
preferred when the orchestrator supplies one, matching the other engines.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from . import constants
from .contracts import CalibrationSettings
from .utils import clip_to_unit_interval

logger = logging.getLogger(__name__)

ScoringResult = tuple  # (probability, route, is_calibrated, note)


class GhostScorer:
    """Turns the maximised Bhattacharyya distance into a probability."""

    def __init__(self, calibration: Optional[CalibrationSettings] = None) -> None:
        """Bind optional calibration state supplied once by the orchestrator.

        Args:
            calibration: Reference scores and/or sigmoid overrides.
        """
        self.calibration = calibration or CalibrationSettings()

    @staticmethod
    def to_raw_score(max_distance: float) -> float:
        """Normalise D_max against the paper's decision threshold.

        Args:
            max_distance: D_max, the maximised Bhattacharyya distance.

        Returns:
            clip(D_max / Th, 0, 1); reaches 1.0 exactly at the paper's
            forged/authentic decision boundary.
        """
        if not np.isfinite(max_distance):
            return 0.0
        return clip_to_unit_interval(max_distance
                                     / constants.BHATTACHARYYA_THRESHOLD)

    def to_probability(self, raw_score: float) -> ScoringResult:
        """Convert the normalised ghost score into a probability in [0, 1].

        Args:
            raw_score: clip(D_max / Th, 0, 1) from to_raw_score.

        Returns:
            Tuple of (probability, route name, is_calibrated, note).
        """
        if not np.isfinite(raw_score):
            logger.warning("non-finite raw score %r; returning 0.0", raw_score)
            return (0.0, "none", False,
                    "Raw score was not finite; probability floored to 0.0.")

        reference_scores = self._lookup_reference_scores()
        if reference_scores is not None:
            probability = self._empirical_cdf_percentile(raw_score,
                                                          reference_scores)
            return (probability, "empirical_cdf", True,
                    f"Calibrated by empirical CDF against "
                    f"{reference_scores.size} known-authentic reference ghost "
                    f"scores.")

        return (clip_to_unit_interval(raw_score), "threshold_ratio", False,
                f"Scored by the SKILL's own recommended normalisation, "
                f"clip(D_max / Th, 0, 1) with Th="
                f"{constants.BHATTACHARYYA_THRESHOLD}. The threshold itself "
                f"is a corpus value fitted on 1000 original + 1000 tampered "
                f"UCID images, so a raw score of 1.0 marks the paper's own "
                f"forged/authentic boundary - but the mapping from that "
                f"ratio to a probability is not calibrated by the corpus, so "
                f"treat intermediate values as an ordering signal rather "
                f"than a likelihood.")

    def _empirical_cdf_percentile(self, raw_score: float,
                                  reference_scores: np.ndarray) -> float:
        """Percentile rank of the score within a known-authentic distribution.

        Args:
            raw_score: Normalised ghost score for the image under test.
            reference_scores: Scores previously measured on authentic images.

        Returns:
            Fraction of reference scores at or below the test score, in [0, 1].
        """
        below_or_equal = np.count_nonzero(reference_scores <= raw_score)
        return clip_to_unit_interval(
            float(below_or_equal) / float(reference_scores.size))

    def _provisional_sigmoid(self, raw_score: float) -> float:
        """Logistic fallback mapping, retained for interface parity.

        Args:
            raw_score: Normalised ghost score.

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
