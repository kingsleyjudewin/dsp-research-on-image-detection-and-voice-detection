"""Renders the ghost segmentation mask as a spatial tamper overlay.

This is the one engine in this system whose localisation is corpus-backed
rather than an engineering extension. The SKILL states the segmentation mask
is "a direct localization heatmap (this is the strongest localization output
of any technique in this module - it is literally a segmented binary tamper
mask, not just a score)". flagged_regions is therefore populated here, where
the other modules must leave it None.

The mask drawn is the one belonging to the winning (q2, dx, dy) combination,
already cropped back out of its shifted frame into original coordinates.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from . import constants

logger = logging.getLogger(__name__)


class GhostVisualizer:
    """Draws the winning combination's binary tamper mask over the image."""

    def render_mask_overlay(self, image_rgb: np.ndarray,
                            mask: np.ndarray) -> np.ndarray:
        """Tint the flagged region over a greyscale rendering of the image.

        Args:
            image_rgb: uint8 H x W x 3 RGB image.
            mask: Binary (0/1) tamper mask in original image coordinates.

        Returns:
            BGR uint8 overlay image.

        Raises:
            ValueError: If the mask shape does not match the image.
        """
        if mask.shape[:2] != image_rgb.shape[:2]:
            raise ValueError(f"mask shape {mask.shape[:2]} does not match "
                             f"image shape {image_rgb.shape[:2]}")

        greyscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        base = cv2.cvtColor(greyscale, cv2.COLOR_GRAY2BGR)
        tinted = base.copy()
        tinted[mask > 0] = constants.MASK_OVERLAY_COLOUR
        weight = constants.MASK_OVERLAY_WEIGHT
        overlay = cv2.addWeighted(base, 1.0 - weight, tinted, weight, 0.0)
        return self._scale_for_display(overlay)

    @staticmethod
    def mask_to_regions(mask: np.ndarray) -> list:
        """Convert a binary mask into a list of connected-component boxes.

        Args:
            mask: Binary (0/1) tamper mask.

        Returns:
            List of region dicts with bounding box, area, and centroid.
        """
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=constants.CONNECTED_COMPONENT_CONNECTIVITY)
        regions = []
        # Label 0 is the background component, so it is skipped.
        for index in range(1, count):
            left, top, width, height, area = stats[index]
            regions.append({
                "x": int(left), "y": int(top),
                "width": int(width), "height": int(height),
                "area_pixels": int(area),
                "centroid_x": float(centroids[index][0]),
                "centroid_y": float(centroids[index][1]),
            })
        return regions

    @staticmethod
    def _scale_for_display(image: np.ndarray) -> np.ndarray:
        """Shrink the overlay to the configured display cap.

        Args:
            image: BGR uint8 overlay.

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
