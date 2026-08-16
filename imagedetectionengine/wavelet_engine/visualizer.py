"""Renders the duplicate-region map as forensic evidence.

SKILL, Pipeline C step 10: "build a same-size binary matrix Q ... multiply
elementwise with the original image to visualize the flagged duplicated
regions map." This is a direct spatial localisation output, not a heatmap
requiring any score-to-colour conversion. Pipeline A's residual is exposed
separately as auxiliary evidence, never as the primary evidence_map, since it
does not drive this engine's score (see constants.py's SCOPE DECISION).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from . import constants

logger = logging.getLogger(__name__)


class WaveletVisualizer:
    """Draws the Pipeline C duplicate map as an overlay on the LL subband."""

    def render_duplicate_overlay(self, ll_subband: np.ndarray,
                                 duplicate_map: np.ndarray) -> np.ndarray:
        """Multiply the LL subband by the binary duplicate map, per the SKILL.

        Args:
            ll_subband: Float64 LL subband the blocks were tiled from.
            duplicate_map: Binary (0/1) map, same shape as ll_subband.

        Returns:
            BGR uint8 image: greyscale LL subband with duplicated regions
            highlighted in a distinct colour channel.

        Raises:
            ValueError: If the shapes do not match.
        """
        if ll_subband.shape != duplicate_map.shape:
            raise ValueError(f"LL subband shape {ll_subband.shape} does not "
                             f"match duplicate map shape {duplicate_map.shape}")

        normalised = self._normalise_for_display(ll_subband)
        base = cv2.cvtColor(normalised, cv2.COLOR_GRAY2BGR)
        highlighted = base.copy()
        highlighted[duplicate_map > 0] = (0, 0, constants.EIGHT_BIT_DISPLAY_MAXIMUM)
        overlay = cv2.addWeighted(base, 0.5, highlighted, 0.5, 0.0)
        return self._scale_for_display(overlay)

    @staticmethod
    def _normalise_for_display(subband: np.ndarray) -> np.ndarray:
        """Scale a float subband to uint8 for display.

        Args:
            subband: Float64 array.

        Returns:
            uint8 array of the same shape.
        """
        minimum, maximum = float(subband.min()), float(subband.max())
        if maximum <= minimum:
            return np.zeros(subband.shape, dtype=np.uint8)
        scaled = ((subband - minimum) / (maximum - minimum)
                 * constants.EIGHT_BIT_DISPLAY_MAXIMUM)
        return scaled.astype(np.uint8)

    @staticmethod
    def _scale_for_display(image: np.ndarray) -> np.ndarray:
        """Shrink the overlay to the configured display cap.

        Args:
            image: BGR uint8 image.

        Returns:
            BGR uint8 image whose longest edge is at most the display cap.
        """
        longest_edge = max(image.shape[0], image.shape[1])
        if longest_edge <= constants.EVIDENCE_MAP_MAX_DIMENSION:
            return image
        scale = constants.EVIDENCE_MAP_MAX_DIMENSION / float(longest_edge)
        target = (max(int(image.shape[1] * scale), 1),
                  max(int(image.shape[0] * scale), 1))
        return cv2.resize(image, target, interpolation=cv2.INTER_AREA)
