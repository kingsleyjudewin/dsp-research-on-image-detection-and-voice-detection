"""Maps the gradient ratio onto a [0, 1] probability, capped at near-zero trust.

The SKILL file gives no calibration function for max_grad or any normalization
of it: "no normalization or calibration given". The two routes below - an
empirical CDF against measured authentic reference scores, and a provisional
logistic fallback - exist purely for interface consistency with the other
engines in this system. Neither route changes the fact that this engine's
underlying feature is not validated lighting-consistency evidence: SKILL
"Corpus gap" states this module "should carry the lowest reliability weight of
the nine detectors in the fusion layer". That instruction is enforced here as
an unconditional cap on confidence (constants.MAXIMUM_CONFIDENCE_CEILING),
applied regardless of which calibration route produced the probability.
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


class LightingScorer:
    """Turns the max_grad / median ratio into a capped tampering probability."""

    def __init__(self,
                 calibration: Optional[CalibrationSettings] = None) -> None:
        """Bind optional calibration state supplied once by the orchestrator.

        Args:
            calibration: Reference scores and/or sigmoid overrides. When None,
                the provisional sigmoid is used and every result is flagged
                uncalibrated.
        """
        self.calibration = calibration or CalibrationSettings()

    def to_probability(self, raw_score: float) -> ScoringResult:
        """Convert the gradient ratio into a probability in [0, 1].

        Args:
            raw_score: max_grad / median(gradient_magnitude) from the computer.

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
                    f"Calibrated by empirical CDF (percentile rank) against "
                    f"{reference_scores.size} known-authentic reference "
                    f"ratios. UNCONDITIONAL CAVEAT: even calibrated, this "
                    f"engine's underlying feature is an unvalidated generic "
                    f"edge-strength heuristic, not a lighting-consistency "
                    f"measurement - see the SKILL file's own Corpus Gap "
                    f"section. Confidence is capped accordingly.")

        return (self._provisional_sigmoid(raw_score), "provisional_sigmoid",
                False,
                "PROVISIONAL CALIBRATION: no known-authentic reference ratios "
                "were supplied, so a placeholder logistic curve was used. The "
                "SKILL file gives no calibration function for this statistic "
                "at all - unlike the other engines' provisional routes, there "
                "is no corpus-stated threshold this curve even approximates. "
                "This engine's underlying feature is furthermore an "
                "unvalidated generic edge-strength heuristic, not a "
                "lighting-consistency measurement. The raw score is a weak "
                "auxiliary signal at best, and both is_reliable and "
                "confidence should be read with that in mind regardless of "
                "this route.")

    def _empirical_cdf_percentile(self,
                                  raw_score: float,
                                  reference_scores: np.ndarray) -> float:
        """Percentile rank of the ratio within a known-authentic distribution.

        This is the non-degenerate reading of the SKILL Implementation Notes'
        suggestion to "normalize ... relative to the image's own
        gradient-magnitude distribution (e.g., a percentile rank)": ranking the
        max against its OWN image's pixel population is trivially 100% (it is
        that population's maximum by construction), so the percentile rank is
        instead taken against a reference population of other images' ratios.

        Args:
            raw_score: Ratio measured on the image under test.
            reference_scores: Ratios previously measured on authentic images.

        Returns:
            Fraction of reference scores at or below the test score, in [0, 1].
        """
        below_or_equal = np.count_nonzero(reference_scores <= raw_score)
        return clip_to_unit_interval(float(below_or_equal)
                                     / float(reference_scores.size))

    def _provisional_sigmoid(self, raw_score: float) -> float:
        """Logistic fallback mapping of the ratio onto [0, 1].

        The slope and midpoint are placeholders, not corpus values - the SKILL
        publishes no calibration at all for this statistic.

        Args:
            raw_score: Ratio measured on the image under test.

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
        """Fetch authentic reference ratios, if enough were supplied.

        Returns:
            Array of reference scores, or None when absent or too small to be
            preferable to the sigmoid.
        """
        table = self.calibration.authentic_reference_scores
        if table is None:
            return None

        try:
            scores = np.asarray(table, dtype=np.float64).ravel()
        except (TypeError, ValueError) as error:
            logger.warning("calibration reference scores are unreadable: %s",
                           error)
            return None

        if scores.size < constants.MINIMUM_CALIBRATION_REFERENCE_COUNT:
            logger.debug("only %d reference scores supplied, below the "
                         "minimum of %d; falling back to the sigmoid",
                         scores.size,
                         constants.MINIMUM_CALIBRATION_REFERENCE_COUNT)
            return None

        return scores
