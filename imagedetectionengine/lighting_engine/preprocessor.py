"""Preprocessing for the lighting / illumination engine.

One job, named in SKILL "Input requirements" -> "Preprocessing": "grayscale
conversion for the Sobel-gradient cue (double(gray_img) per the source's own
MATLAB code)". The SKILL admits "any RGB image", and a grayscale array is
accepted unchanged as it needs no conversion at all.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import constants
from .contracts import PreparedImage
from .utils import describe_array_shape


class LightingPreprocessor:
    """Converts the input image to the grayscale plane Pipeline A operates on."""

    def prepare(self, image: np.ndarray) -> PreparedImage:
        """Turn a raw image into the grayscale array the computer needs.

        Args:
            image: BGR uint8 array of shape (H, W, 3), or a 2-D grayscale
                array.

        Returns:
            PreparedImage carrying the float64 grayscale plane.

        Raises:
            ValueError: If the image is not a 2-D or 3-D array, has the wrong
                channel count, or is smaller than numpy.gradient's own minimum.
        """
        self._validate_image(image)
        original_shape = tuple(np.asarray(image).shape)

        grayscale = self._to_grayscale(np.asarray(image))
        return PreparedImage(grayscale=grayscale.astype(np.float64),
                             original_shape=original_shape)

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        """Reject inputs the engine cannot compute a gradient on.

        Args:
            image: Candidate input array.

        Raises:
            ValueError: If the array is not 2-D or 3-D, has the wrong channel
                count, or is below numpy.gradient's minimum axis length.
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

        minimum = constants.MINIMUM_GRADIENT_AXIS_LENGTH
        if array.shape[0] < minimum or array.shape[1] < minimum:
            raise ValueError(
                f"image of shape {describe_array_shape(array)} is smaller than "
                f"the {minimum}x{minimum} minimum numpy.gradient requires on "
                f"each axis")

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        """Reduce a BGR image to the single plane the gradient is computed on.

        Args:
            image: uint8 array, 2-D grayscale or 3-D BGR.

        Returns:
            uint8 array of shape (H, W).
        """
        if image.ndim == constants.GRAYSCALE_IMAGE_DIMENSION_COUNT:
            return image
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
