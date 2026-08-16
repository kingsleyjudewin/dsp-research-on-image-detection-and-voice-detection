"""Renders Pipeline A's block-grid anomaly heatmap as forensic evidence.

SKILL, Output section: Pipeline A produces "an aggregable [0,1]-normalized
heatmap". This is the score-driving pipeline's own evidence; Pipelines B, C,
D never contribute pixels here since they are auxiliary/non-scoring.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from . import constants

logger = logging.getLogger(__name__)


class NoiseVisualizer:
    """Draws Pipeline A's per-block anomaly heatmap upsampled to image scale."""

    def render_heatmap(self, heatmap: np.ndarray, block_size: int) -> np.ndarray:
        """Upsample the block-grid heatmap to pixel resolution and colourise it.

        Args:
            heatmap: [0,1]-normalised per-block anomaly grid.
            block_size: Block side length in pixels, for upsampling scale.

        Returns:
            BGR uint8 heatmap image.

        Raises:
            ValueError: If the heatmap is empty.
        """
        if heatmap.size == 0:
            raise ValueError("cannot render an empty heatmap")

        pixel_scale = np.kron(heatmap, np.ones((block_size, block_size)))
        normalised = (np.clip(pixel_scale, 0.0, 1.0)
                     * constants.EIGHT_BIT_DISPLAY_MAXIMUM).astype(np.uint8)
        coloured = cv2.applyColorMap(normalised, cv2.COLORMAP_INFERNO)
        return self._scale_for_display(coloured)

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
