"""Pre- and post-computation reliability gates for the wavelet-domain engine.

The SKILL's Input Requirements section lists three unreliability conditions
that are qualitative, not numeric thresholds: scaled/rotated copy-move
regions (undetectable by this pipeline's design, not a per-image check),
aggressively smoothed/denoised post-tampering content (defeats the
noise-residual path), and anti-forensic dithering (Pipeline B's documented
100% defeat, Stamm & Liu 2010). None of these are computable from
ImageMetadata alone, so they are surfaced as standing caveats in
reliability_note rather than invented numeric gates.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from . import constants
from .contracts import ConditionReport, ImageMetadata
from .utils import compose_confidence_penalties

logger = logging.getLogger(__name__)

CheckResult = tuple  # (passed: bool, confidence_penalty: float, note: str)

STANDING_LIMITATIONS_NOTE = (
    "Standing SKILL caveats: undetectable for scaled or rotated copy-move "
    "regions; the noise-residual path (Pipeline A) is defeated by "
    "aggressive post-tampering smoothing/denoising; the compression-history "
    "path (Pipeline B) is documented as 100%-defeatable by a knowledgeable "
    "adversary (Stamm & Liu 2010) and is therefore never scored, only "
    "reported as auxiliary evidence."
)


class ConditionChecker:
    """Decides whether the wavelet engine may run, and at what confidence."""

    def check(self, metadata: ImageMetadata,
              image: Optional[np.ndarray] = None,
              block_size: int = constants.DEFAULT_BLOCK_SIZE) -> ConditionReport:
        """Run the pre-computation conditions against the input.

        Args:
            metadata: Container and resolution facts.
            image: BGR uint8 or grayscale array, or None.
            block_size: R, the block side length Pipeline C will use.

        Returns:
            ConditionReport aggregating every check.
        """
        skip_reason = self._premise_failure_reason(image, block_size)
        if skip_reason:
            return ConditionReport(is_reliable=False,
                                   confidence_weight=constants.ZERO_CONFIDENCE,
                                   reliability_note=skip_reason, skip_engine=True)

        return ConditionReport(is_reliable=True,
                               confidence_weight=constants.FULL_CONFIDENCE,
                               reliability_note=STANDING_LIMITATIONS_NOTE,
                               skip_engine=False)

    def _premise_failure_reason(self, image: Optional[np.ndarray],
                                block_size: int) -> Optional[str]:
        """Explain why the engine cannot run at all, if it cannot.

        Args:
            image: BGR uint8 or grayscale array, or None.
            block_size: R, the block side length Pipeline C will use.

        Returns:
            Explanation string, or None when the engine may proceed.
        """
        if image is None:
            return "Engine skipped: no image array was supplied."

        array = np.asarray(image)
        if array.ndim not in (constants.GRAYSCALE_IMAGE_DIMENSION_COUNT,
                              constants.COLOUR_IMAGE_DIMENSION_COUNT):
            return (f"Engine skipped: expected a 2-D grayscale or 3-D colour "
                    f"image, received an array of {array.ndim} dimensions.")

        estimated_ll_height = (array.shape[0] + 1) // 2
        estimated_ll_width = (array.shape[1] + 1) // 2
        if (estimated_ll_height < block_size
                or estimated_ll_width < block_size):
            return (f"Engine skipped: LL subband is estimated at "
                    f"{estimated_ll_height}x{estimated_ll_width}, smaller "
                    f"than the {block_size}x{block_size} block size Pipeline "
                    f"C requires.")
        return None

    @staticmethod
    def assess_block_count_sufficiency(total_blocks: int) -> CheckResult:
        """Pre-computation check that enough blocks exist for a PCA fit.

        Args:
            total_blocks: Number of R x R blocks tiled from the LL subband.

        Returns:
            CheckResult reflecting whether block count was sufficient.
        """
        if total_blocks >= constants.MINIMUM_BLOCKS_FOR_PCA:
            return True, constants.FULL_CONFIDENCE, ""
        return (False, constants.ZERO_CONFIDENCE,
                f"Only {total_blocks} block(s) were tiled from the LL "
                f"subband, below the {constants.MINIMUM_BLOCKS_FOR_PCA} "
                f"needed for a PCA feature-space fit; copy-move search "
                f"cannot run meaningfully.")

    @staticmethod
    def assess_moment_degeneracy(mu00_values: list) -> CheckResult:
        """Post-computation check for blocks with degenerate total mass.

        A block with mu_00 near zero makes Eq. 17's contrast normalisation
        and the recursive Eq. 12 division numerically meaningless.

        Args:
            mu00_values: Zeroth central moment (total pixel mass) per block.

        Returns:
            CheckResult reflecting the fraction of degenerate blocks.
        """
        if not mu00_values:
            return False, constants.ZERO_CONFIDENCE, "No blocks to assess."
        degenerate = sum(1 for value in mu00_values
                         if abs(value) < constants.MOMENT_DEGENERACY_FLOOR)
        fraction = degenerate / len(mu00_values)
        if fraction == 0.0:
            return True, constants.FULL_CONFIDENCE, ""
        penalty = compose_confidence_penalties([1.0 - fraction])
        return (fraction < 1.0, penalty,
                f"{degenerate}/{len(mu00_values)} tiled blocks have near-zero "
                f"total mass (mu_00 below "
                f"{constants.MOMENT_DEGENERACY_FLOOR}); their "
                f"contrast-normalised features are numerically unstable.")

    @staticmethod
    def assess_texture_degeneracy(block_variances: list) -> CheckResult:
        """Post-computation check for flat/textureless block content.

        Flat blocks are not caught by assess_moment_degeneracy (mu_00 is
        their total intensity, which is nonzero for any non-black flat
        region) - but a flat block's contrast-normalised blur invariants are
        purely geometric (intensity cancels through the mu_00 division), so
        every flat block anywhere in the image maps to an identical feature
        vector. Eq. 30's minimum-separation rule only rejects NEARBY
        coincidental matches from local smoothness; it does not cover a
        globally flat/low-texture image where distant blocks are genuinely,
        trivially identical. No numeric variance floor is given in the
        SKILL for this - [ENGINEERING] gate, discovered by testing, in the
        same spirit as this system's other engines' false-positive gates.

        Args:
            block_variances: Pixel-intensity variance per tiled block.

        Returns:
            CheckResult reflecting the fraction of textureless blocks.
        """
        if not block_variances:
            return False, constants.ZERO_CONFIDENCE, "No blocks to assess."
        flat = sum(1 for value in block_variances
                  if value < constants.BLOCK_TEXTURE_VARIANCE_FLOOR)
        fraction = flat / len(block_variances)
        if fraction < constants.MAXIMUM_FLAT_BLOCK_FRACTION:
            return True, constants.FULL_CONFIDENCE, ""
        return (False, constants.ZERO_CONFIDENCE,
                f"{flat}/{len(block_variances)} tiled blocks are flat/"
                f"textureless (pixel variance below "
                f"{constants.BLOCK_TEXTURE_VARIANCE_FLOOR}); "
                f"blur-invariant features degenerate to purely geometric, "
                f"intensity-independent values, making unrelated flat "
                f"regions appear as spurious copy-move duplicates.")
