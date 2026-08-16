"""Maps the raw divergence onto a [0, 1] probability for the fusion layer.

The SKILL file is explicit that the corpus contains no calibration function:
"none of the five papers specify a general-purpose calibration function mapping
their raw statistic to a probability." It then names two routes as engineering
recommendations, and both are implemented here:

  Route 1 - a logistic / Platt-style sigmoid.
  Route 2 - an empirical CDF percentile against a held-out calibration set of
            known-authentic images.

Route 2 is preferred whenever the orchestrator supplies reference scores,
because its parameters are measured rather than assumed. Route 1 is the
fallback, and any result produced by it is reported as uncalibrated.

Both routes are conditioned on the estimated quality factor, per the SKILL's
"Quality-factor conditioning" note that a single global cutoff is invalid.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from . import constants
from .contracts import CalibrationSettings
from .utils import clip_to_unit_interval

logger = logging.getLogger(__name__)

# (probability, calibration route used, is_calibrated, explanatory note)
ScoringResult = tuple[float, str, bool, str]


class BenfordScorer:
    """Turns a raw divergence into a calibrated tampering probability."""

    def __init__(self,
                 calibration: Optional[CalibrationSettings] = None) -> None:
        """Bind optional calibration state supplied once by the orchestrator.

        Args:
            calibration: Reference scores and/or sigmoid overrides. When None,
                the provisional sigmoid is used and every result is flagged
                uncalibrated.
        """
        self.calibration = calibration or CalibrationSettings()

    def to_probability(self,
                       raw_score: float,
                       estimated_quality_factor: Optional[float]) -> ScoringResult:
        """Convert a raw divergence into a probability in [0, 1].

        Args:
            raw_score: Mean symmetric-KL divergence from the computer.
            estimated_quality_factor: The image's quality factor, selecting
                which calibration bucket applies.

        Returns:
            Tuple of (probability, route name, is_calibrated, note).
        """
        if not np.isfinite(raw_score):
            logger.warning("non-finite raw score %r; returning 0.0", raw_score)
            return (0.0, "none", False,
                    "Raw score was not finite; probability floored to 0.0.")

        bucket = self._select_quality_factor_bucket(estimated_quality_factor)
        reference_scores = self._lookup_reference_scores(bucket)

        if reference_scores is not None:
            probability = self._empirical_cdf_percentile(raw_score,
                                                         reference_scores)
            return (probability, "empirical_cdf", True,
                    f"Calibrated by empirical CDF against {reference_scores.size} "
                    f"known-authentic reference scores in bucket '{bucket}'.")

        probability = self._platt_sigmoid(raw_score)
        return (probability, "provisional_sigmoid", False,
                "PROVISIONAL CALIBRATION: no known-authentic reference scores "
                "were supplied for bucket "
                f"'{bucket}', so a placeholder logistic curve was used. The "
                "SKILL file states no calibration function exists in the "
                "corpus for this statistic, so this probability expresses "
                "relative ordering only and must not be read as a literal "
                "likelihood. The raw score is the trustworthy quantity.")

    def _empirical_cdf_percentile(self,
                                  raw_score: float,
                                  reference_scores: np.ndarray) -> float:
        """Percentile rank of the score within a known-authentic distribution.

        SKILL "Calibrating to a [0,1] fusion-layer probability", route 2:
        "empirical CDF percentile against a held-out calibration set of
        known-authentic images at the deployment's expected QF".

        A score exceeding most authentic references is more suspicious, so the
        percentile maps directly onto tampering probability.

        Args:
            raw_score: Divergence measured on the image under test.
            reference_scores: Divergences previously measured on authentic
                images from the same quality-factor bucket.

        Returns:
            Fraction of reference scores at or below the test score, in [0, 1].
        """
        below_or_equal = np.count_nonzero(reference_scores <= raw_score)
        return clip_to_unit_interval(float(below_or_equal) /
                                     float(reference_scores.size))

    def _platt_sigmoid(self, raw_score: float) -> float:
        """Logistic fallback mapping of the raw score onto [0, 1].

        SKILL "Calibrating to a [0,1] fusion-layer probability", route 1:
        "fit a monotonic calibration (e.g. logistic/Platt-style sigmoid
        sigma(a*chi2 + b) ...)".

        The slope and midpoint are placeholders, not corpus values - the SKILL
        publishes no divergence magnitudes for this statistic. See the warning
        block in constants.py.

        Args:
            raw_score: Divergence measured on the image under test.

        Returns:
            Probability in [0, 1], monotonically increasing in raw_score.
        """
        slope = (self.calibration.platt_slope
                 if self.calibration.platt_slope is not None
                 else constants.PROVISIONAL_SIGMOID_SLOPE)
        midpoint = (self.calibration.platt_midpoint
                    if self.calibration.platt_midpoint is not None
                    else constants.PROVISIONAL_SIGMOID_MIDPOINT)

        exponent = slope * (raw_score - midpoint)
        # Bound the exponent so an extreme score cannot overflow exp().
        exponent = float(np.clip(exponent,
                                 -constants.SIGMOID_EXPONENT_LIMIT,
                                 constants.SIGMOID_EXPONENT_LIMIT))
        return clip_to_unit_interval(1.0 / (1.0 + np.exp(-exponent)))

    def _lookup_reference_scores(self, bucket: str) -> Optional[np.ndarray]:
        """Fetch authentic reference scores for a bucket, if usable.

        Args:
            bucket: Quality-factor bucket name.

        Returns:
            Array of reference scores, or None when absent or too small to be
            preferable to the sigmoid.
        """
        table = self.calibration.authentic_reference_scores
        if not table or bucket not in table:
            return None

        try:
            scores = np.asarray(table[bucket], dtype=np.float64).ravel()
        except (TypeError, ValueError) as error:
            logger.warning("calibration entry for bucket %r is unreadable: %s",
                           bucket, error)
            return None

        if scores.size < constants.MINIMUM_CALIBRATION_REFERENCE_COUNT:
            logger.debug("bucket %r holds only %d reference scores, below the "
                         "minimum of %d; falling back to the sigmoid",
                         bucket, scores.size,
                         constants.MINIMUM_CALIBRATION_REFERENCE_COUNT)
            return None

        return scores

    @staticmethod
    def _select_quality_factor_bucket(
            estimated_quality_factor: Optional[float]) -> str:
        """Map a quality factor onto its calibration bucket.

        SKILL "Implementation notes" -> "Quality-factor conditioning": "any
        deployed threshold must be conditioned on the image's estimated
        background quality factor ... rather than applied as one fixed global
        cutoff." Bucket edges reuse the two thresholds already sourced from the
        corpus (50 and 80) so no new cut points are introduced.

        Args:
            estimated_quality_factor: Quality factor, or None when unknown.

        Returns:
            One of the names in constants.QUALITY_FACTOR_BUCKET_NAMES.
        """
        if estimated_quality_factor is None:
            return constants.QUALITY_FACTOR_BUCKET_NAMES[-1]

        bucket_index = int(np.searchsorted(
            np.asarray(constants.QUALITY_FACTOR_BUCKET_EDGES, dtype=np.float64),
            float(estimated_quality_factor),
            side="right",
        ))
        return constants.QUALITY_FACTOR_BUCKET_NAMES[bucket_index]
