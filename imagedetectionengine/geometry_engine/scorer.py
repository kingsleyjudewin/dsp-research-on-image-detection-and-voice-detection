"""Turns Yao et al.'s consistency score into the fusion layer's probability.

This engine is unusual in the system: its raw score is ALREADY calibrated by the
source paper. SKILL "Output" states of Eq. 8's C that "this is already a
calibrated, paper-defined [0,1] score, unlike most other modules in this engine
where a [0,1] mapping had to be recommended as an engineering addition", and
that "the natural tampering score is 1-C (or 1-C_min across all evaluated object
pairs, since one spliced object is typically inconsistent with several others)".

So the default route is a pass-through, not a fitted curve. The paper's own
threshold T = 0.5 is carried alongside as the decision point, which the SKILL
says "can seed (not replace) this engine's calibration". An empirical-CDF route
is available and takes precedence whenever the orchestrator supplies measured
reference scores, per that same instruction to eventually replace the seed.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from . import constants
from .contracts import CalibrationSettings, HeightRatioAnalysis
from .utils import clip_to_unit_interval

logger = logging.getLogger(__name__)

# (probability, calibration route used, is_calibrated, explanatory note)
ScoringResult = tuple[float, str, bool, str]


class GeometryScorer:
    """Converts height-ratio consistency into a calibrated probability."""

    def __init__(self,
                 calibration: Optional[CalibrationSettings] = None) -> None:
        """Bind optional calibration state supplied once by the orchestrator.

        Args:
            calibration: Reference scores and/or a sigmoid slope override. When
                None, the paper's own calibration is used unchanged.
        """
        self.calibration = calibration or CalibrationSettings()

    @staticmethod
    def reduce_analysis_to_scalar(analysis: HeightRatioAnalysis) -> float:
        """Reduce every evaluated object pair to one tampering score.

        SKILL "Output": "the natural tampering score is 1-C (or 1-C_min across
        all evaluated object pairs, since one spliced object is typically
        inconsistent with several others)". The minimum is used because a single
        spliced object need only be inconsistent with one partner to be caught,
        and averaging would dilute that evidence across every consistent pair.

        Args:
            analysis: The height-ratio analysis to reduce.

        Returns:
            Tampering score in [0, 1]; 0 = every pair consistent.
        """
        if analysis.evaluated_pair_count == 0:
            return 0.0
        return clip_to_unit_interval(1.0 - analysis.minimum_consistency)

    def to_probability(self,
                       raw_score: float,
                       estimated_quality_factor: Optional[float]) -> ScoringResult:
        """Convert the reduced score into a probability in [0, 1].

        Args:
            raw_score: 1 - C_min from reduce_analysis_to_scalar.
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
                    f"in bucket '{bucket}', replacing the paper's seed "
                    f"calibration as the SKILL file recommends.")

        return (clip_to_unit_interval(raw_score), "paper_calibration", False,
                self._paper_calibration_note())

    @staticmethod
    def _paper_calibration_note() -> str:
        """Explain the paper's own calibration and where its numbers come from.

        Returns:
            Note text for the reliability field and the trace.
        """
        return (f"Scored by Yao et al.'s own calibration: Eq. 8's consistency C "
                f"is already a paper-defined [0,1] quantity, so the tampering "
                f"probability is 1 - C_min directly, with no fitted curve "
                f"interposed. The paper's decision threshold is "
                f"T = {constants.CONSISTENCY_DECISION_THRESHOLD}, i.e. a score "
                f"above "
                f"{1.0 - constants.CONSISTENCY_DECISION_THRESHOLD} indicates at "
                f"least one object is inconsistent. For reference, Table I of "
                f"the source paper places authentic images at "
                f"{constants.YAO_AUTHENTIC_TAMPERING_SCORE_MINIMUM:.3f}-"
                f"{constants.YAO_AUTHENTIC_TAMPERING_SCORE_MAXIMUM:.3f} and "
                f"forged ones at "
                f"{constants.YAO_FORGED_TAMPERING_SCORE_MINIMUM:.3f}-"
                f"{constants.YAO_FORGED_TAMPERING_SCORE_MAXIMUM:.3f}. This "
                f"seed calibration rests on eight example images and should be "
                f"replaced with measured reference scores.")

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

    def _lookup_reference_scores(self, bucket: str) -> Optional[np.ndarray]:
        """Fetch authentic reference scores for a bucket, if usable.

        Args:
            bucket: Quality-factor bucket name.

        Returns:
            Array of reference scores, or None when absent or too small to be
            preferable to the paper's own calibration.
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
                         "minimum of %d; keeping the paper's calibration",
                         bucket, scores.size,
                         constants.MINIMUM_CALIBRATION_REFERENCE_COUNT)
            return None

        return scores

    @staticmethod
    def _select_quality_factor_bucket(
            estimated_quality_factor: Optional[float]) -> str:
        """Map a JPEG quality factor onto its calibration bucket.

        Only two buckets exist, unlike the compression-sensitive engines in this
        system, because the corpus documents this method as robust to
        low-quality recompression; a finer split would be modelling noise.

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
