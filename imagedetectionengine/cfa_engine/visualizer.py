"""Renders the visual evidence for a CFA measurement.

Unlike the global-only engines in this system, Pipeline A IS a localisation
method - the SKILL calls it "most directly validated for fine-grained (down to
2x2 block) forgery localization". A spatial heatmap is therefore the honest form
of evidence here, and it is rendered at the true block resolution of the
measurement rather than smoothed, so a reader can see exactly how coarse the
finding is.

Blocks the engine declined to measure, because they were almost flat or
edge-dominated, are painted a flat grey rather than a heat colour. Ferrara names
both as conditions that mimic the CFA-absent signature, so colouring them as
"authentic" or "tampered" would assert something the corpus says the method
cannot support.

Everything here is presentation only and can never influence a score.
"""

from __future__ import annotations

import logging

import numpy as np

from . import constants
from .utils import apply_diverging_colormap, upscale_block_map

logger = logging.getLogger(__name__)


class CfaVisualizer:
    """Draws the per-block tampering posterior as a BGR heatmap."""

    def render_evidence_map(self,
                            tampering_map: np.ndarray,
                            validity_mask: np.ndarray,
                            block_size: int) -> np.ndarray:
        """Render the posterior map as a colour-coded image.

        Args:
            tampering_map: Per-block 1 - Pr{M1|L}, in [0, 1].
            validity_mask: True where the block carried usable texture.
            block_size: Pixels per block, used to restore the original scale.

        Returns:
            BGR uint8 image. Cool = confidently authentic, hot = confidently
            tampered, grey = not measured.

        Raises:
            ValueError: If the map and mask shapes disagree, or the map is
                empty.
        """
        posterior = np.asarray(tampering_map, dtype=np.float64)
        usable = np.asarray(validity_mask, dtype=bool)
        if posterior.shape != usable.shape:
            raise ValueError(f"tampering map shape {posterior.shape} does not "
                             f"match validity mask shape {usable.shape}")
        if posterior.size == 0:
            raise ValueError("cannot render an empty tampering map")

        coloured = apply_diverging_colormap(posterior,
                                            constants.EVIDENCE_COLOUR_AUTHENTIC,
                                            constants.EVIDENCE_COLOUR_NEUTRAL,
                                            constants.EVIDENCE_COLOUR_TAMPERED)
        coloured[~usable] = np.asarray(constants.EVIDENCE_COLOUR_EXCLUDED,
                                       dtype=np.uint8)
        return self._scale_for_display(coloured, block_size)

    @staticmethod
    def _scale_for_display(coloured_blocks: np.ndarray,
                           block_size: int) -> np.ndarray:
        """Expand the block image back toward the source resolution.

        Nearest-neighbour replication only, so no value is invented by
        interpolation and the block granularity of the measurement stays
        visible.

        Args:
            coloured_blocks: BGR uint8 image at block resolution.
            block_size: Pixels per block in the source image.

        Returns:
            BGR uint8 image, capped at the configured display size.
        """
        longest_block_edge = max(coloured_blocks.shape[0],
                                 coloured_blocks.shape[1])
        # Never enlarge past the display cap, and never shrink below one pixel
        # per block, so every measured block stays individually visible.
        affordable = max(1, constants.EVIDENCE_MAP_MAX_DIMENSION
                         // max(longest_block_edge, 1))
        return upscale_block_map(coloured_blocks, min(block_size, affordable))
