"""Colour/grayscale views, grid shifting, and the JPEG recompression sweep.

The two pipelines want different views of the same image, and the SKILL is
explicit about both:

  * Pipeline B "operates on all 3 channels (R,G,B), summed (Azarian-Pour
    Eq. 5-6); no color-space conversion needed beyond having RGB available."
  * Pipeline A works on grayscale, downsampled x2 with nearest-neighbour
    "specifically to remove CFA-interpolation periodicity that would
    otherwise confound the resampling signal", then cropped to the centre
    256x256 block "to keep comparisons fair across parameter settings".
"""

from __future__ import annotations

import io
import logging

import cv2
import numpy as np
from PIL import Image

from . import constants
from .contracts import ImageMetadata, PreparedImage
from .utils import centre_crop

logger = logging.getLogger(__name__)


class GhostPreprocessor:
    """Builds the colour and resampling views and drives JPEG recompression."""

    def prepare(self, image: np.ndarray,
                metadata: ImageMetadata) -> PreparedImage:
        """Build both views this engine needs from one BGR input image.

        Args:
            image: BGR uint8 (H x W x 3) or grayscale (H x W) array.
            metadata: Container facts; not consulted for the conversions.

        Returns:
            PreparedImage holding the RGB view and the resampling window.

        Raises:
            ValueError: If the array has an unsupported number of dimensions.
        """
        array = np.asarray(image)
        if array.ndim == constants.COLOUR_IMAGE_DIMENSION_COUNT:
            colour_rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
            luminance = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        elif array.ndim == constants.GRAYSCALE_IMAGE_DIMENSION_COUNT:
            colour_rgb = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
            luminance = array
        else:
            raise ValueError(f"expected a 2-D or 3-D image array, got "
                             f"{array.ndim} dimensions")

        window, note = self.prepare_resampling_window(luminance)
        return PreparedImage(colour_rgb=colour_rgb.astype(np.uint8),
                             resampling_window=window,
                             original_shape=array.shape,
                             resampling_window_note=note)

    @staticmethod
    def prepare_resampling_window(luminance: np.ndarray) -> tuple:
        """Downsample x2 to strip CFA periodicity, then centre-crop.

        SKILL: Kirchner & Bohme downsample "x2 with nearest-neighbor from RAW
        specifically to remove CFA-interpolation periodicity", a step "found
        to be sufficient to reliably remove detectable traces of
        demosaicing", then "always crop to the center 256x256 block".

        Nearest-neighbour downsampling by exactly 2 keeps every other sample
        and interpolates nothing, so it introduces no periodicity of its own.

        Args:
            luminance: 2-D uint8 grayscale image.

        Returns:
            Tuple of (window as float64 or None, explanatory note).
        """
        factor = constants.CFA_SUPPRESSION_DOWNSAMPLE_FACTOR
        decimated = luminance[::factor, ::factor]
        window = centre_crop(decimated, constants.ANALYSIS_WINDOW_SIZE)

        if min(window.shape[:2]) < constants.PREDICTOR_NEIGHBOURHOOD_SIZE:
            return None, (f"Resampling window is {window.shape[0]}x"
                          f"{window.shape[1]} after the x{factor} "
                          f"CFA-suppression downsample, too small for the "
                          f"{constants.PREDICTOR_NEIGHBOURHOOD_SIZE}x"
                          f"{constants.PREDICTOR_NEIGHBOURHOOD_SIZE} "
                          f"predictor neighbourhood.")
        return window.astype(np.float64), (
            f"Analysed a {window.shape[0]}x{window.shape[1]} centre window "
            f"after the x{factor} nearest-neighbour downsample that strips "
            f"CFA-interpolation periodicity.")

    @staticmethod
    def shift_image(image_rgb: np.ndarray, shift_x: int,
                    shift_y: int) -> np.ndarray:
        """Zero-pad the image so its content moves off the 8x8 DCT grid.

        SKILL Step 1: "zero-pad/shift the dubious m x n image by
        (d_x, d_y) in {0,...,7}^2 to produce a padded (m+d_x) x (n+d_y)
        image I'", covering pastes the forger did not align to the grid.

        Args:
            image_rgb: uint8 H x W x 3 RGB image.
            shift_x: Horizontal pad in pixels, 0..7.
            shift_y: Vertical pad in pixels, 0..7.

        Returns:
            The padded uint8 image.
        """
        if shift_x == 0 and shift_y == 0:
            return image_rgb
        return np.pad(image_rgb, ((shift_y, 0), (shift_x, 0), (0, 0)),
                      mode="constant", constant_values=0)

    @staticmethod
    def recompress(image_rgb: np.ndarray, quality_factor: int) -> np.ndarray:
        """Encode at a candidate quality factor and decode straight back.

        Args:
            image_rgb: uint8 H x W x 3 RGB image.
            quality_factor: Candidate q2 in [1, 100].

        Returns:
            The decoded uint8 RGB image after the round trip.
        """
        buffer = io.BytesIO()
        Image.fromarray(image_rgb, mode="RGB").save(
            buffer, format="JPEG", quality=int(quality_factor))
        buffer.seek(0)
        return np.array(Image.open(buffer).convert("RGB"))
