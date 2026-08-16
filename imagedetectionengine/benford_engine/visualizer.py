"""Renders the visual evidence for a Benford measurement.

Deliberately NOT a spatial heatmap. The SKILL file records an explicit negative
result for localisation: "Copy-move/cut-paste tampering within the same image
... first-digit statistics are aggregate/global and largely insensitive to
small localized splices (Singh 2015 explicitly demonstrates this: tampered vs.
original image Benford curves are visually near-identical)". Emitting a
per-pixel heatmap would advertise a capability the corpus says this method does
not have.

Instead the evidence map reproduces the form of visual evidence the source
papers themselves publish: the observed leading-digit histogram plotted against
the fitted Benford curve. Divergence between the two IS the finding.

Everything here is presentation only and can never influence a score.
"""

from __future__ import annotations

import logging

import numpy as np

from . import constants

logger = logging.getLogger(__name__)


class BenfordVisualizer:
    """Draws the observed-versus-fitted digit histogram as a BGR image."""

    def render_evidence_map(self,
                            empirical_pmf: np.ndarray,
                            fitted_pmf: np.ndarray) -> np.ndarray:
        """Render the digit histogram against its fitted Benford curve.

        Args:
            empirical_pmf: Observed pmf over digits 1..base-1.
            fitted_pmf: Fitted generalized-Benford curve, same length.

        Returns:
            BGR uint8 array of shape (height, width, 3). A blank canvas is
            returned rather than raising if the inputs cannot be plotted, so
            visualisation can never break an analysis.
        """
        canvas = np.full(
            (constants.EVIDENCE_MAP_HEIGHT,
             constants.EVIDENCE_MAP_WIDTH,
             constants.EXPECTED_CHANNEL_COUNT),
            constants.EVIDENCE_MAP_BACKGROUND_INTENSITY,
            dtype=np.uint8,
        )

        empirical = np.asarray(empirical_pmf, dtype=np.float64).ravel()
        fitted = np.asarray(fitted_pmf, dtype=np.float64).ravel()
        if empirical.size == 0 or empirical.size != fitted.size:
            logger.debug("evidence map skipped: pmf sizes %d and %d",
                         empirical.size, fitted.size)
            return canvas

        vertical_scale = self._compute_vertical_scale(empirical, fitted)
        self._draw_axes(canvas)
        self._draw_empirical_bars(canvas, empirical, vertical_scale)
        self._draw_fitted_markers(canvas, fitted, vertical_scale)
        return canvas

    @staticmethod
    def _compute_vertical_scale(empirical: np.ndarray,
                                fitted: np.ndarray) -> float:
        """Largest plottable value, with head-room so nothing is clipped.

        Args:
            empirical: Observed pmf.
            fitted: Fitted curve.

        Returns:
            Positive full-scale value for the vertical axis.
        """
        peak = float(max(np.max(empirical), np.max(fitted)))
        if peak <= 0.0:
            return 1.0
        return peak * constants.EVIDENCE_MAP_VERTICAL_HEADROOM

    @staticmethod
    def _plot_area() -> tuple[int, int, int, int]:
        """Pixel bounds of the drawable region inside the margins.

        Returns:
            Tuple of (top, bottom, left, right) in pixel coordinates.
        """
        margin = constants.EVIDENCE_MAP_MARGIN
        return (margin,
                constants.EVIDENCE_MAP_HEIGHT - margin,
                margin,
                constants.EVIDENCE_MAP_WIDTH - margin)

    def _draw_axes(self, canvas: np.ndarray) -> None:
        """Draw the left and bottom axis rules in place.

        Args:
            canvas: Image being drawn onto; mutated in place.
        """
        top, bottom, left, right = self._plot_area()
        thickness = constants.EVIDENCE_MAP_AXIS_THICKNESS
        intensity = constants.EVIDENCE_MAP_AXIS_INTENSITY

        canvas[bottom:bottom + thickness, left:right] = intensity  # baseline
        canvas[top:bottom, left:left + thickness] = intensity      # value axis

    def _draw_empirical_bars(self,
                             canvas: np.ndarray,
                             empirical: np.ndarray,
                             vertical_scale: float) -> None:
        """Draw one filled bar per digit for the observed histogram.

        Args:
            canvas: Image being drawn onto; mutated in place.
            empirical: Observed pmf.
            vertical_scale: Full-scale value of the vertical axis.
        """
        top, bottom, left, right = self._plot_area()
        plot_height = bottom - top
        slot_width = (right - left) / float(empirical.size)
        bar_width = slot_width * (1.0 - constants.EVIDENCE_MAP_BAR_GAP_FRACTION)

        for digit_index, probability in enumerate(empirical):
            bar_height = int(round((probability / vertical_scale) * plot_height))
            if bar_height <= 0:
                continue

            slot_start = left + digit_index * slot_width
            bar_left = int(round(slot_start + (slot_width - bar_width) / 2.0))
            bar_right = int(round(bar_left + bar_width))
            canvas[bottom - bar_height:bottom, bar_left:bar_right] = \
                constants.EVIDENCE_MAP_EMPIRICAL_BAR_COLOUR

    def _draw_fitted_markers(self,
                             canvas: np.ndarray,
                             fitted: np.ndarray,
                             vertical_scale: float) -> None:
        """Overlay the fitted curve as a horizontal tick above each digit.

        Args:
            canvas: Image being drawn onto; mutated in place.
            fitted: Fitted generalized-Benford curve.
            vertical_scale: Full-scale value of the vertical axis.
        """
        top, bottom, left, right = self._plot_area()
        plot_height = bottom - top
        slot_width = (right - left) / float(fitted.size)
        thickness = constants.EVIDENCE_MAP_FITTED_MARKER_THICKNESS

        for digit_index, probability in enumerate(fitted):
            marker_height = int(round((probability / vertical_scale) * plot_height))
            marker_row = int(np.clip(bottom - marker_height, top, bottom - thickness))

            slot_start = int(round(left + digit_index * slot_width))
            slot_end = int(round(left + (digit_index + 1) * slot_width))
            canvas[marker_row:marker_row + thickness, slot_start:slot_end] = \
                constants.EVIDENCE_MAP_FITTED_CURVE_COLOUR
