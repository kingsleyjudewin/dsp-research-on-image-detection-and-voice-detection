"""Renders the visual evidence for a geometric-consistency measurement.

The evidence here is not a per-pixel heatmap: this engine's finding is a claim
about a handful of discrete objects and one horizon line, so the honest picture
is an annotated overlay showing exactly the geometry the decision rested on -
the estimated vanishing line, the line segments that voted for it, and each
tested object box coloured by the consistency score it achieved.

Drawing the supporting lines matters forensically. A reader can see at a glance
whether the vanishing point was fixed by genuine scene structure or by a handful
of unrelated edges, which is the difference between a trustworthy v0 and a
spurious one.

Everything here is presentation only and can never influence a score.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from . import constants
from .contracts import HeightRatioAnalysis, VanishingPointEstimate

logger = logging.getLogger(__name__)


class GeometryVisualizer:
    """Draws the estimated horizon, its supporting lines and the tested boxes."""

    def render_evidence_map(self,
                            colour_image: np.ndarray,
                            estimate: VanishingPointEstimate,
                            regions: list,
                            analysis: HeightRatioAnalysis) -> np.ndarray:
        """Render the geometry the decision rested on as an annotated image.

        Args:
            colour_image: BGR image the measurement was made on.
            estimate: The vanishing-point estimate to draw.
            regions: ObjectRegion objects that were considered.
            analysis: The height-ratio analysis, for per-region colouring.

        Returns:
            BGR uint8 annotated image.

        Raises:
            ValueError: If the image is empty or not a colour array.
        """
        base = np.ascontiguousarray(np.asarray(colour_image, dtype=np.uint8))
        if base.size == 0 or base.ndim != constants.COLOUR_IMAGE_DIMENSION_COUNT:
            raise ValueError(f"cannot render evidence on an array of shape "
                             f"{base.shape}")

        overlay = base.copy()
        self._draw_support_lines(overlay, estimate)
        self._draw_vanishing_line(overlay, estimate)
        self._draw_regions(overlay, regions, self._worst_score_by_region(analysis))

        blended = cv2.addWeighted(base, constants.EVIDENCE_BASE_IMAGE_WEIGHT,
                                  overlay, constants.EVIDENCE_OVERLAY_WEIGHT,
                                  0.0)
        return self._scale_for_display(blended)

    @staticmethod
    def _draw_support_lines(canvas: np.ndarray,
                            estimate: VanishingPointEstimate) -> None:
        """Draw the line segments that voted for the vanishing point.

        Args:
            canvas: Image to draw on, modified in place.
            estimate: The vanishing-point estimate whose inliers to draw.
        """
        for segment in estimate.inlier_segments:
            cv2.line(canvas,
                     (int(segment.start[0]), int(segment.start[1])),
                     (int(segment.end[0]), int(segment.end[1])),
                     constants.EVIDENCE_COLOUR_SUPPORT_LINE,
                     constants.EVIDENCE_LINE_THICKNESS)

    @staticmethod
    def _draw_vanishing_line(canvas: np.ndarray,
                             estimate: VanishingPointEstimate) -> None:
        """Draw the horizon row and, when visible, the vanishing point itself.

        Args:
            canvas: Image to draw on, modified in place.
            estimate: The vanishing-point estimate to draw.
        """
        if estimate.vanishing_line_row is None:
            return

        row = int(round(estimate.vanishing_line_row))
        if 0 <= row < canvas.shape[0]:
            cv2.line(canvas, (0, row), (canvas.shape[1] - 1, row),
                     constants.EVIDENCE_COLOUR_VANISHING_LINE,
                     constants.EVIDENCE_LINE_THICKNESS)

        point = estimate.homogeneous_point
        if point is None or estimate.is_at_infinity:
            return
        column = int(round(float(point[0]) / float(point[2])))
        if 0 <= column < canvas.shape[1] and 0 <= row < canvas.shape[0]:
            cv2.circle(canvas, (column, row),
                       constants.EVIDENCE_VANISHING_POINT_RADIUS,
                       constants.EVIDENCE_COLOUR_VANISHING_POINT, -1)

    @staticmethod
    def _worst_score_by_region(analysis: HeightRatioAnalysis) -> dict:
        """Map each region identifier to the worst tampering score it earned.

        Args:
            analysis: The height-ratio analysis to summarise.

        Returns:
            Dictionary of region identifier to 1 - C for its worst pair.
        """
        worst: dict = {}
        for measurement in analysis.measurements:
            score = 1.0 - measurement.consistency
            for identifier in (measurement.first_region_id,
                               measurement.second_region_id):
                worst[identifier] = max(worst.get(identifier, 0.0), score)
        return worst

    @staticmethod
    def _draw_regions(canvas: np.ndarray,
                      regions: list,
                      worst_scores: dict) -> None:
        """Draw each tested region, coloured by its worst consistency result.

        Args:
            canvas: Image to draw on, modified in place.
            regions: ObjectRegion objects that were considered.
            worst_scores: Region identifier to worst tampering score.
        """
        for region in regions:
            if region.identifier not in worst_scores:
                continue
            inconsistent = (worst_scores[region.identifier]
                            > 1.0 - constants.CONSISTENCY_DECISION_THRESHOLD)
            colour = (constants.EVIDENCE_COLOUR_INCONSISTENT_REGION
                      if inconsistent
                      else constants.EVIDENCE_COLOUR_CONSISTENT_REGION)
            cv2.rectangle(canvas,
                          (int(region.left_column), int(region.top_row)),
                          (int(region.right_column), int(region.bottom_row)),
                          colour, constants.EVIDENCE_BOX_THICKNESS)

    @staticmethod
    def _scale_for_display(image: np.ndarray) -> np.ndarray:
        """Shrink the annotated image to the configured display cap.

        Args:
            image: BGR uint8 annotated image.

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
