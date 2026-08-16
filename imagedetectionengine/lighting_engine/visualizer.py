"""Renders the visual evidence for a gradient-magnitude measurement.

The gradient magnitude map genuinely was computed (it is not invented for
display), so a heatmap of it is legitimate evidence to show - it just is NOT a
lighting-inconsistency localisation, and the SKILL gives no threshold for
turning it into discrete suspect regions. flagged_regions is therefore always
None; this visualisation is presented as what it is, a raw gradient-strength
map, not a forgery-probability overlay.

Everything here is presentation only and can never influence a score.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from . import constants

logger = logging.getLogger(__name__)


class LightingVisualizer:
    """Draws the gradient-magnitude map as a BGR heatmap."""

    def render_evidence_map(self, gradient_magnitude: np.ndarray) -> np.ndarray:
        """Render the gradient magnitude map as a colour-coded image.

        Args:
            gradient_magnitude: Per-pixel gradient magnitude, non-negative.

        Returns:
            BGR uint8 heatmap image.

        Raises:
            ValueError: If the map is empty.
        """
        magnitude = np.asarray(gradient_magnitude, dtype=np.float64)
        if magnitude.size == 0:
            raise ValueError("cannot render an empty gradient magnitude map")

        normalised = self._normalise_for_display(magnitude)
        coloured = cv2.applyColorMap(normalised, cv2.COLORMAP_INFERNO)
        return self._scale_for_display(coloured)

    @staticmethod
    def _normalise_for_display(magnitude: np.ndarray) -> np.ndarray:
        """Scale the magnitude map to uint8, clipping outliers first.

        A single very sharp edge can otherwise dominate the display range and
        wash out the rest of the map, so the clip percentile trades outlier
        visibility for overall contrast. Affects only the rendered pixels.

        Args:
            magnitude: Per-pixel gradient magnitude, non-negative.

        Returns:
            uint8 array of the same shape.
        """
        clip_value = float(np.percentile(
            magnitude, constants.EVIDENCE_DISPLAY_CLIP_PERCENTILE))
        if clip_value <= 0.0:
            return np.zeros(magnitude.shape, dtype=np.uint8)

        clipped = np.clip(magnitude, 0.0, clip_value)
        scaled = ((clipped / clip_value)
                 * constants.EIGHT_BIT_DISPLAY_MAXIMUM)
        return scaled.astype(np.uint8)

    @staticmethod
    def _scale_for_display(image: np.ndarray) -> np.ndarray:
        """Shrink the heatmap to the configured display cap.

        Args:
            image: BGR uint8 heatmap.

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
