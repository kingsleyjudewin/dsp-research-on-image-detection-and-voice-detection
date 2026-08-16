"""Preprocessing for the perspective / geometric consistency engine.

Four jobs, all named in SKILL "Input requirements" -> "Preprocessing":

  1. "edge/line extraction (Hough transform) for the parallel-line
     vanishing-point method"  -> module A1's input.
  2. "SIFT feature extraction ... for the recurrence-based fallback"
                              -> module A4's input.
  3. "superpixel segmentation (SLIC recommended) for candidate object/region
     proposal when bounding boxes aren't manually available"
                              -> module D, feeding module B's object pairs.
  4. Grayscale conversion, since both edge detection and SIFT operate on a
     single plane and the SKILL admits "any RGB/grayscale image".

Regions supplied by the caller bypass step 3 entirely: Yao et al.'s method is
written around "manually- or superpixel-selected object bounding boxes", and a
caller who already has real object detections should use them.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from skimage.segmentation import slic

from . import constants
from .contracts import LineSegment, ObjectRegion, PreparedScene
from .utils import (compute_colour_histogram, describe_array_shape,
                    line_through_points)

logger = logging.getLogger(__name__)


class GeometryPreprocessor:
    """Extracts the lines, keypoints and candidate regions the modules need."""

    def __init__(self, supplied_regions: Optional[list] = None) -> None:
        """Bind optional caller-supplied object regions.

        Args:
            supplied_regions: Object boxes the caller has already localised.
                Each entry is a mapping with top_row, bottom_row, left_column
                and right_column. When present, SLIC proposal is skipped.
        """
        self.supplied_regions = supplied_regions

    def prepare(self, image: np.ndarray) -> PreparedScene:
        """Turn a raw image into the arrays the mathematical core needs.

        Args:
            image: BGR uint8 array of shape (H, W, 3), or a 2-D grayscale array.

        Returns:
            PreparedScene carrying line segments, SIFT keypoints and regions.

        Raises:
            ValueError: If the image is not a 2-D or 3-D array, or is smaller
                than the engine's minimum analysable side.
        """
        self._validate_image(image)
        original_shape = tuple(np.asarray(image).shape)

        colour_image = self._to_three_channel(np.asarray(image))
        grayscale = self._to_grayscale(colour_image)

        positions, scales, descriptors = self._extract_keypoints(grayscale)
        return PreparedScene(
            colour_image=colour_image.astype(np.float64),
            grayscale=grayscale.astype(np.float64),
            line_segments=self._extract_line_segments(grayscale),
            keypoint_positions=positions,
            keypoint_scales=scales,
            keypoint_descriptors=descriptors,
            regions=self._build_regions(colour_image),
            original_shape=original_shape,
        )

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        """Reject inputs the engine cannot analyse.

        Args:
            image: Candidate input array.

        Raises:
            ValueError: If the array is not 2-D or 3-D, has the wrong channel
                count, or is below the minimum analysable size.
        """
        array = np.asarray(image)
        if array.ndim not in (constants.GRAYSCALE_IMAGE_DIMENSION_COUNT,
                              constants.COLOUR_IMAGE_DIMENSION_COUNT):
            raise ValueError(
                f"expected a 2-D grayscale or 3-D colour image, received shape "
                f"{describe_array_shape(array)}")
        if (array.ndim == constants.COLOUR_IMAGE_DIMENSION_COUNT
                and array.shape[-1] != constants.EXPECTED_CHANNEL_COUNT):
            raise ValueError(
                f"expected {constants.EXPECTED_CHANNEL_COUNT} colour channels, "
                f"received {array.shape[-1]}")
        if min(array.shape[0], array.shape[1]) < constants.MINIMUM_IMAGE_SIDE_PIXELS:
            raise ValueError(
                f"image of shape {describe_array_shape(array)} is below the "
                f"{constants.MINIMUM_IMAGE_SIDE_PIXELS}-pixel minimum side")

    @staticmethod
    def _to_three_channel(image: np.ndarray) -> np.ndarray:
        """Promote a grayscale image to three identical BGR planes.

        The SKILL admits "any RGB/grayscale image", and SLIC plus the colour
        appearance signature both expect three planes, so a single-plane input
        is widened rather than rejected.

        Args:
            image: 2-D or 3-D uint8 array.

        Returns:
            uint8 array of shape (H, W, 3).
        """
        if image.ndim == constants.COLOUR_IMAGE_DIMENSION_COUNT:
            return np.ascontiguousarray(image.astype(np.uint8))
        widened = np.repeat(image[:, :, np.newaxis],
                            constants.EXPECTED_CHANNEL_COUNT, axis=2)
        return np.ascontiguousarray(widened.astype(np.uint8))

    @staticmethod
    def _to_grayscale(colour_image: np.ndarray) -> np.ndarray:
        """Reduce a BGR image to the single plane edges and SIFT operate on.

        Args:
            colour_image: uint8 BGR array.

        Returns:
            uint8 array of shape (H, W).
        """
        return cv2.cvtColor(colour_image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _extract_line_segments(grayscale: np.ndarray) -> list:
        """Detect straight edges for the parallel-line vanishing-point module.

        SKILL A1 step 1: "Extract straight edges via Hough transform on detected
        edges." The edge detector and its parameters are not specified by the
        corpus; see the constants module.

        Args:
            grayscale: uint8 single-plane image.

        Returns:
            List of LineSegment objects, longest first and capped in number.
        """
        edges = cv2.Canny(grayscale, constants.CANNY_LOW_THRESHOLD,
                          constants.CANNY_HIGH_THRESHOLD,
                          apertureSize=constants.CANNY_APERTURE_SIZE)
        raw = cv2.HoughLinesP(
            edges, constants.HOUGH_DISTANCE_RESOLUTION_PIXELS,
            constants.HOUGH_ANGLE_RESOLUTION_RADIANS,
            constants.HOUGH_VOTE_THRESHOLD,
            minLineLength=constants.HOUGH_MINIMUM_LINE_LENGTH_PIXELS,
            maxLineGap=constants.HOUGH_MAXIMUM_LINE_GAP_PIXELS)
        if raw is None:
            return []

        # OpenCV 4.x returns shape (N, 1, 4) and OpenCV 5.x returns (N, 4).
        # Flattening to (N, 4) accepts both rather than pinning a major version.
        endpoints = np.asarray(raw).reshape(-1, constants.HOUGH_ENDPOINT_COUNT)
        segments = [GeometryPreprocessor._build_segment(entry)
                    for entry in endpoints]
        usable = [segment for segment in segments if segment is not None]
        # Keep the longest segments: length is the most direct proxy for how
        # reliably a segment's direction is measured.
        usable.sort(key=lambda segment: segment.length, reverse=True)
        return usable[:constants.MAXIMUM_LINE_SEGMENT_COUNT]

    @staticmethod
    def _build_segment(endpoints: np.ndarray) -> Optional[LineSegment]:
        """Convert one Hough endpoint quadruple into a LineSegment.

        Args:
            endpoints: Length-4 array of (x1, y1, x2, y2).

        Returns:
            LineSegment, or None when the two endpoints coincide.
        """
        start = (float(endpoints[0]), float(endpoints[1]))
        end = (float(endpoints[2]), float(endpoints[3]))

        offset = np.array([end[0] - start[0], end[1] - start[1]],
                          dtype=np.float64)
        length = float(np.linalg.norm(offset))
        if length == 0.0:
            return None

        try:
            homogeneous = line_through_points(start, end)
        except ValueError as error:
            logger.debug("discarding degenerate Hough segment %s: %s",
                         endpoints, error)
            return None
        return LineSegment(start=start, end=end, homogeneous=homogeneous,
                           direction=offset / length, length=length)

    @staticmethod
    def _extract_keypoints(grayscale: np.ndarray) -> tuple:
        """Detect SIFT keypoints for the recurrence-based module.

        SKILL A4 step 1: "SIFT feature extraction over the whole image
        (128-dimensional descriptor per keypoint, standard DoG-pyramid SIFT)."

        Args:
            grayscale: uint8 single-plane image.

        Returns:
            Tuple of (positions (n,2), scales (n,), descriptors (n,128)).
        """
        detector = cv2.SIFT_create(
            nfeatures=constants.MAXIMUM_SIFT_KEYPOINT_COUNT)
        keypoints, descriptors = detector.detectAndCompute(grayscale, None)

        if not keypoints or descriptors is None:
            return (np.zeros((0, constants.CARTESIAN_VECTOR_LENGTH)),
                    np.zeros(0), np.zeros((0, constants.SIFT_DESCRIPTOR_LENGTH)))

        positions = np.array([keypoint.pt for keypoint in keypoints],
                             dtype=np.float64)
        scales = np.array([keypoint.size for keypoint in keypoints],
                          dtype=np.float64)
        return positions, scales, np.asarray(descriptors, dtype=np.float64)

    def _build_regions(self, colour_image: np.ndarray) -> list:
        """Assemble the candidate object regions module B will pair up.

        Args:
            colour_image: uint8 BGR array.

        Returns:
            List of ObjectRegion objects.
        """
        if self.supplied_regions:
            return self._regions_from_caller(colour_image)
        return self._propose_regions(colour_image)

    def _regions_from_caller(self, colour_image: np.ndarray) -> list:
        """Convert caller-supplied boxes into ObjectRegion objects.

        Args:
            colour_image: uint8 BGR array, for appearance signatures.

        Returns:
            List of ObjectRegion objects marked as supplied.
        """
        regions: list = []
        for identifier, box in enumerate(self.supplied_regions):
            region = ObjectRegion(
                identifier=identifier,
                top_row=float(box["top_row"]),
                bottom_row=float(box["bottom_row"]),
                left_column=float(box["left_column"]),
                right_column=float(box["right_column"]),
                source="supplied")
            region.appearance_signature = self._appearance_signature(
                colour_image, region)
            regions.append(region)
        return regions

    def _propose_regions(self, colour_image: np.ndarray) -> list:
        """Propose candidate object regions by superpixel segmentation.

        SKILL D: SLIC is the "Recommended default ... cited across the survey as
        the most widely-used method for its speed, memory efficiency, and strong
        boundary adherence", used "to generate candidate object/region masks
        automatically, avoiding the manual bounding-box selection Yao et al.'s
        original method relies on."

        Args:
            colour_image: uint8 BGR array.

        Returns:
            List of ObjectRegion objects, size-filtered and capped in number.
        """
        try:
            labels = slic(colour_image,
                          n_segments=constants.SLIC_SEGMENT_COUNT,
                          compactness=constants.SLIC_COMPACTNESS,
                          sigma=constants.SLIC_SIGMA,
                          start_label=constants.SLIC_START_LABEL,
                          channel_axis=-1)
        except (ValueError, MemoryError) as error:
            logger.warning("SLIC segmentation failed on image %s: %s",
                           describe_array_shape(colour_image), error)
            return []

        boxes = self._boxes_from_labels(labels, colour_image.shape[:2])
        return self._finalise_regions(boxes, colour_image)

    @staticmethod
    def _boxes_from_labels(labels: np.ndarray, shape: tuple) -> list:
        """Reduce each superpixel label to a size-filtered bounding box.

        Args:
            labels: Integer label image from SLIC.
            shape: (height, width) of the image.

        Returns:
            List of (top, bottom, left, right) tuples in pixels.
        """
        shorter_side = float(min(shape))
        minimum_side = constants.MINIMUM_REGION_SIDE_FRACTION * shorter_side
        maximum_side = constants.MAXIMUM_REGION_SIDE_FRACTION * shorter_side

        boxes: list = []
        for label in np.unique(labels):
            rows, columns = np.nonzero(labels == label)
            if rows.size < constants.MINIMUM_REGION_PIXEL_COUNT:
                continue
            top, bottom = float(rows.min()), float(rows.max())
            left, right = float(columns.min()), float(columns.max())
            if not (minimum_side <= bottom - top <= maximum_side):
                continue
            if not (minimum_side <= right - left <= maximum_side):
                continue
            boxes.append((top, bottom, left, right))
        return boxes

    def _finalise_regions(self, boxes: list, colour_image: np.ndarray) -> list:
        """Cap, index and describe the proposed boxes.

        Args:
            boxes: (top, bottom, left, right) tuples.
            colour_image: uint8 BGR array, for appearance signatures.

        Returns:
            List of ObjectRegion objects.
        """
        # Prefer the tallest boxes: Eq. 7 differences top and bottom rows, so a
        # taller region carries proportionally less segmentation noise.
        ordered = sorted(boxes, key=lambda box: box[1] - box[0], reverse=True)
        regions: list = []

        for identifier, (top, bottom, left, right) in enumerate(
                ordered[:constants.MAXIMUM_PROPOSED_REGION_COUNT]):
            region = ObjectRegion(identifier=identifier, top_row=top,
                                  bottom_row=bottom, left_column=left,
                                  right_column=right, source="slic")
            region.appearance_signature = self._appearance_signature(
                colour_image, region)
            regions.append(region)
        return regions

    @staticmethod
    def _appearance_signature(colour_image: np.ndarray,
                              region: ObjectRegion) -> Optional[np.ndarray]:
        """Coarse colour histogram of a region, used only to screen pairings.

        Args:
            colour_image: uint8 BGR array.
            region: The region to describe.

        Returns:
            Normalised histogram, or None when the region is empty.
        """
        top = max(int(region.top_row), 0)
        bottom = min(int(region.bottom_row) + 1, colour_image.shape[0])
        left = max(int(region.left_column), 0)
        right = min(int(region.right_column) + 1, colour_image.shape[1])

        patch = colour_image[top:bottom, left:right]
        if patch.size < constants.MINIMUM_REGION_PIXEL_COUNT:
            return None
        return compute_colour_histogram(
            patch.reshape(-1, colour_image.shape[2]),
            constants.APPEARANCE_HISTOGRAM_BINS_PER_CHANNEL)
