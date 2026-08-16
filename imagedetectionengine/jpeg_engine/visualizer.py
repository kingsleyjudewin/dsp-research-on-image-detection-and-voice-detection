"""Renders Pipeline B's per-frequency spectra as the engine's visual evidence.

This is deliberately NOT a spatial heatmap. The SKILL is explicit that
"none of Pipeline A/B natively produce a spatial heatmap", that Luo's and
Mahdian's methods are "validated at the whole-image or single-block level,
not as a sliding-window localizer", and that block-wise re-application for a
coarse map is "not itself validated end-to-end in the corpus". Painting a
spatial overlay here would present an unvalidated inference as localisation
evidence.

What is rendered instead is exactly what the detector actually looked at:
the ten trend-removed histogram-FFT magnitude spectra |H_i~|, with the
detected peaks marked. flagged_regions stays None for the same reason.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from . import constants

logger = logging.getLogger(__name__)


class JpegVisualizer:
    """Draws the ten Pipeline-B spectra as a stacked panel of line plots."""

    def render_spectra(self, spectra: list) -> np.ndarray:
        """Render every frequency's spectrum as one stacked evidence image.

        Args:
            spectra: List of FrequencySpectrum entries from Pipeline B.

        Returns:
            BGR uint8 image, one panel per frequency.

        Raises:
            ValueError: If no spectra were supplied.
        """
        if not spectra:
            raise ValueError("cannot render evidence from an empty spectrum list")

        panels = [self._render_panel(entry) for entry in spectra]
        canvas = np.vstack(panels)
        return self._scale_for_display(canvas)

    def _render_panel(self, entry) -> np.ndarray:
        """Draw one frequency's spectrum, peaks, and label into a panel.

        Args:
            entry: A FrequencySpectrum.

        Returns:
            BGR uint8 panel image.
        """
        height = constants.EVIDENCE_PANEL_HEIGHT_PIXELS
        width = constants.EVIDENCE_PANEL_WIDTH_PIXELS
        panel = np.zeros((height, width, constants.COLOUR_IMAGE_DIMENSION_COUNT),
                         dtype=np.uint8)

        if entry.was_excluded or entry.spectrum.size == 0:
            self._label(panel, f"H{entry.ordinal} {entry.frequency} EXCLUDED "
                               f"(zeros {entry.zero_coefficient_fraction:.0%})")
            return panel

        points = self._spectrum_to_points(entry.spectrum, height, width)
        cv2.polylines(panel, [points], isClosed=False,
                      color=constants.EVIDENCE_SPECTRUM_COLOUR,
                      thickness=constants.EVIDENCE_LINE_THICKNESS)
        self._mark_peaks(panel, entry, points)
        self._label(panel, f"H{entry.ordinal} {entry.frequency} "
                           f"peak={entry.strongest_prominence:.4f}")
        return panel

    @staticmethod
    def _spectrum_to_points(spectrum: np.ndarray, height: int,
                            width: int) -> np.ndarray:
        """Map a spectrum onto integer polyline coordinates within a panel.

        Args:
            spectrum: Trend-removed magnitude spectrum.
            height: Panel height in pixels.
            width: Panel width in pixels.

        Returns:
            Integer array of shape (n_points, 1, 2) for cv2.polylines.
        """
        margin = constants.EVIDENCE_PANEL_MARGIN_PIXELS
        peak_value = float(spectrum.max())
        scale = peak_value if peak_value > 0.0 else constants.FULL_CONFIDENCE

        x_coordinates = np.linspace(margin, width - margin, spectrum.size)
        usable_height = height - constants.SPECTRUM_HALF_DIVISOR * margin
        y_coordinates = (height - margin) - (spectrum / scale) * usable_height
        return np.stack([x_coordinates, y_coordinates],
                        axis=1).astype(np.int32).reshape(-1, 1, 2)

    @staticmethod
    def _mark_peaks(panel: np.ndarray, entry, points: np.ndarray) -> None:
        """Circle each detected peak on an already-drawn panel.

        Args:
            panel: BGR uint8 panel to draw onto, modified in place.
            entry: The FrequencySpectrum being drawn.
            points: Polyline coordinates produced by _spectrum_to_points.
        """
        for position, prominence in zip(entry.peak_positions,
                                        entry.peak_prominences):
            if position >= points.shape[0]:
                continue
            above_threshold = prominence >= constants.PEAK_PROMINENCE_THRESHOLD
            colour = (constants.EVIDENCE_PEAK_COLOUR if above_threshold
                      else constants.EVIDENCE_WEAK_PEAK_COLOUR)
            centre = (int(points[position, 0, 0]), int(points[position, 0, 1]))
            cv2.circle(panel, centre, constants.PEAK_MARKER_RADIUS_PIXELS,
                       colour, thickness=-1)

    @staticmethod
    def _label(panel: np.ndarray, text: str) -> None:
        """Write a caption into a panel's top-left corner.

        Args:
            panel: BGR uint8 panel to draw onto, modified in place.
            text: Caption text.
        """
        margin = constants.EVIDENCE_PANEL_MARGIN_PIXELS
        baseline = margin * constants.SPECTRUM_HALF_DIVISOR + margin
        cv2.putText(panel, text, (margin, baseline), cv2.FONT_HERSHEY_SIMPLEX,
                    constants.EVIDENCE_LABEL_FONT_SCALE,
                    constants.EVIDENCE_LABEL_COLOUR,
                    constants.EVIDENCE_LINE_THICKNESS, cv2.LINE_AA)

    @staticmethod
    def _scale_for_display(image: np.ndarray) -> np.ndarray:
        """Shrink the evidence canvas to the configured display cap.

        Args:
            image: BGR uint8 canvas.

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
