"""Reduces Ferrara's posterior map to a scalar and calibrates it.

Two separate jobs, both of which the SKILL file flags as engineering decisions
rather than corpus results:

  1. Reduction. SKILL "Output" -> Pipeline A: the native output is a heatmap,
     and one should "reduce to a whole-image scalar via max, 95th-percentile, or
     fraction-of-blocks-below-threshold, per this engine's fusion-layer contract
     (reduction rule not specified in the corpus - engineering recommendation)".
     All three named rules are implemented; the default was chosen by
     measurement, not preference, and is recorded in constants.

  2. Calibration. The corpus gives no mapping from the reduced scalar to a
     probability, so the same two routes the other engines use are offered: an
     empirical CDF against known-authentic reference scores when the
     orchestrator supplies them, and a provisional logistic curve otherwise.
     Anything produced by the second route is reported as uncalibrated.
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


class CfaScorer:
    """Turns the per-block tampering map into a calibrated probability."""

    def __init__(self,
                 calibration: Optional[CalibrationSettings] = None) -> None:
        """Bind optional calibration state supplied once by the orchestrator.

        Args:
            calibration: Reference scores and/or sigmoid overrides. When None,
                the provisional sigmoid is used and every result is flagged
                uncalibrated.
        """
        self.calibration = calibration or CalibrationSettings()

    @staticmethod
    def reduce_map_to_scalar(tampering_map: np.ndarray,
                             validity_mask: np.ndarray) -> tuple:
        """Reduce the per-block tampering map to one number.

        Args:
            tampering_map: Per-block 1 - Pr{M1|L}, in [0, 1].
            validity_mask: True where the block carried usable texture.

        Returns:
            Tuple of (raw score in [0, 1], name of the rule applied).

        Raises:
            ValueError: If the configured reduction rule is not recognised.
        """
        usable = np.asarray(tampering_map)[np.asarray(validity_mask, dtype=bool)]
        if usable.size == 0:
            return 0.0, constants.MAP_REDUCTION_RULE

        rule = constants.MAP_REDUCTION_RULE
        if rule == "max":
            return float(np.max(usable)), rule
        if rule == "percentile":
            return float(np.percentile(usable,
                                       constants.MAP_REDUCTION_PERCENTILE)), rule
        if rule == "fraction_below_threshold":
            # "Fraction of blocks below threshold" in the SKILL is phrased on
            # Pr{M1} (authenticity); on the tampering map the same set of blocks
            # is the fraction ABOVE the threshold.
            flagged = np.count_nonzero(
                usable > constants.TAMPERED_BLOCK_PROBABILITY_THRESHOLD)
            return float(flagged) / float(usable.size), rule
        raise ValueError(f"unknown map reduction rule {rule!r}")

    def to_probability(self,
                       raw_score: float,
                       estimated_quality_factor: Optional[float]) -> ScoringResult:
        """Convert the reduced score into a probability in [0, 1].

        Args:
            raw_score: Reduced tampering score from the posterior map.
            estimated_quality_factor: The image's JPEG quality factor,
                selecting which calibration bucket applies.

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
                    f"Calibrated by empirical CDF against "
                    f"{reference_scores.size} known-authentic reference scores "
                    f"in bucket '{bucket}'.")

        return (self._platt_sigmoid(raw_score), "provisional_sigmoid", False,
                "PROVISIONAL CALIBRATION: no known-authentic reference scores "
                f"were supplied for bucket '{bucket}', so a placeholder "
                "logistic curve was used. The SKILL file specifies no "
                "calibration function for this statistic and leaves the map "
                "reduction rule itself as an engineering recommendation, so "
                "this probability expresses relative ordering only and must "
                "not be read as a literal likelihood. The raw score and the "
                "per-block map are the trustworthy quantities.")

    @staticmethod
    def _empirical_cdf_percentile(raw_score: float,
                                  reference_scores: np.ndarray) -> float:
        """Percentile rank of the score within a known-authentic distribution.

        A score exceeding most authentic references is more suspicious, so the
        percentile maps directly onto tampering probability.

        Args:
            raw_score: Score measured on the image under test.
            reference_scores: Scores previously measured on authentic images
                from the same quality-factor bucket.

        Returns:
            Fraction of reference scores at or below the test score, in [0, 1].
        """
        below_or_equal = np.count_nonzero(reference_scores <= raw_score)
        return clip_to_unit_interval(float(below_or_equal)
                                     / float(reference_scores.size))

    def _platt_sigmoid(self, raw_score: float) -> float:
        """Logistic fallback mapping of the raw score onto [0, 1].

        The slope and midpoint are placeholders, not corpus values. See the
        warning block in constants.py.

        Args:
            raw_score: Score measured on the image under test.

        Returns:
            Probability in [0, 1], monotonically increasing in raw_score.
        """
        slope = (self.calibration.platt_slope
                 if self.calibration.platt_slope is not None
                 else constants.PROVISIONAL_SIGMOID_SLOPE)
        midpoint = (self.calibration.platt_midpoint
                    if self.calibration.platt_midpoint is not None
                    else constants.PROVISIONAL_SIGMOID_MIDPOINT)

        exponent = float(np.clip(slope * (raw_score - midpoint),
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
        """Map a JPEG quality factor onto its calibration bucket.

        The score distribution shifts sharply with compression - Ferrara's AUC
        falls from 0.9975 to chance between uncompressed and QF85 - so one
        global calibration would be invalid. Bucket edges reuse the two
        thresholds already sourced from the corpus, so no new cut points are
        introduced.

        Args:
            estimated_quality_factor: Quality factor, or None when unknown.

        Returns:
            One of constants.QUALITY_FACTOR_BUCKET_NAMES.
        """
        if estimated_quality_factor is None:
            return constants.QUALITY_FACTOR_BUCKET_NAMES[-1]

        bucket_index = int(np.searchsorted(
            np.asarray(constants.QUALITY_FACTOR_BUCKET_EDGES, dtype=np.float64),
            float(estimated_quality_factor),
            side="right",
        ))
        return constants.QUALITY_FACTOR_BUCKET_NAMES[bucket_index]
