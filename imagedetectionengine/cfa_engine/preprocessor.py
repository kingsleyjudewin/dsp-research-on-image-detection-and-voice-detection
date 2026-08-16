"""Preprocessing for the CFA / demosaicing engine.

Three jobs, all of them named in SKILL "Input requirements" -> "Preprocessing":

  1. Channel extraction - "Ferrara's method operates on the green channel only".
  2. Block alignment    - "partition into non-overlapping blocks whose size B is
     a multiple of the CFA period (minimum 2x2 for Bayer)". The image is cropped
     so both the Bayer period and the feature block size divide it exactly.
  3. Texture screening  - Ferrara's stated limitation that the method "is less
     effective in the presence of either almost flat areas or sharp edges".
     Blocks failing either test are marked so the mathematical core excludes
     them rather than scoring them as tampered.

The colour image is carried through alongside the green channel because
Pipelines B and C both need all three planes.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage

from . import constants
from .contracts import PreparedImage
from .utils import describe_array_shape, partition_into_blocks, upscale_block_map

logger = logging.getLogger(__name__)


class CfaPreprocessor:
    """Extracts the green channel, aligns the Bayer grid and screens texture."""

    def __init__(self, feature_block_size: int | None = None) -> None:
        """Fix the block geometry the crop must satisfy.

        Args:
            feature_block_size: Block size the mathematical core will use, so
                the crop can be made an exact multiple of it.
        """
        self.feature_block_size = int(feature_block_size
                                      or constants.FEATURE_BLOCK_SIZE)

    def prepare(self, image: np.ndarray) -> PreparedImage:
        """Turn a raw BGR image into the arrays the mathematical core needs.

        Args:
            image: BGR uint8 array of shape (H, W, 3).

        Returns:
            PreparedImage with the cropped green channel, the cropped colour
            image, a pixel-resolution texture mask and the excluded fraction.

        Raises:
            ValueError: If the image is not a three-channel colour array, or is
                too small to hold a single analysis block.
        """
        self._validate_image(image)
        original_shape = tuple(np.asarray(image).shape)

        colour_image = self._crop_to_analysis_grid(
            np.asarray(image, dtype=np.float64))
        green_channel = colour_image[..., constants.ANALYSIS_CHANNEL_INDEX]

        texture_mask, excluded_fraction = self._build_texture_mask(green_channel)
        return PreparedImage(
            green_channel=green_channel,
            colour_image=colour_image,
            texture_mask=texture_mask,
            excluded_block_fraction=excluded_fraction,
            original_shape=original_shape,
        )

    def _validate_image(self, image: np.ndarray) -> None:
        """Reject inputs the engine's premise does not cover.

        Args:
            image: Candidate input array.

        Raises:
            ValueError: If the array is not a three-channel colour image, or is
                smaller than one analysis block on either side.
        """
        array = np.asarray(image)
        if array.ndim != constants.EXPECTED_IMAGE_DIMENSION_COUNT:
            raise ValueError(
                f"expected a {constants.EXPECTED_IMAGE_DIMENSION_COUNT}-D "
                f"colour image, received shape {describe_array_shape(array)}. "
                f"CFA analysis needs all three planes.")
        if array.shape[-1] != constants.EXPECTED_CHANNEL_COUNT:
            raise ValueError(
                f"expected {constants.EXPECTED_CHANNEL_COUNT} colour channels, "
                f"received {array.shape[-1]}")

        step = self._crop_multiple()
        if min(array.shape[0], array.shape[1]) < step:
            raise ValueError(
                f"image of shape {describe_array_shape(array)} is smaller than "
                f"one {step}x{step} analysis block")

    def _crop_multiple(self) -> int:
        """Grid step that both the Bayer period and the block size must divide.

        Returns:
            The smallest crop multiple satisfying both constraints. The feature
            block size is already required to be a multiple of the Bayer
            period, so it is that multiple whenever it is the larger.
        """
        return max(self.feature_block_size, constants.BAYER_PERIOD)

    def _crop_to_analysis_grid(self, image: np.ndarray) -> np.ndarray:
        """Trim the image to a whole number of analysis blocks.

        The crop takes pixels off the bottom and right only. Trimming from the
        top-left would shift the Bayer phase, which would invert the acquired
        and interpolated lattices and with them the sign of Ferrara's feature.

        Args:
            image: Float64 BGR image.

        Returns:
            Float64 BGR image whose height and width are exact multiples of the
            analysis grid step.
        """
        step = self._crop_multiple()
        cropped_height = (image.shape[0] // step) * step
        cropped_width = (image.shape[1] // step) * step

        if (cropped_height, cropped_width) != image.shape[:2]:
            logger.debug("cropped %dx%d to %dx%d to align the Bayer grid",
                         image.shape[0], image.shape[1],
                         cropped_height, cropped_width)
        return image[:cropped_height, :cropped_width, :]

    def _build_texture_mask(self, green_channel: np.ndarray) -> tuple:
        """Mark the blocks with enough texture for the prediction error to talk.

        Args:
            green_channel: Float64 green plane, already cropped.

        Returns:
            Tuple of (pixel-resolution boolean mask, excluded block fraction).
        """
        flat_blocks = self._find_flat_blocks(green_channel)
        edge_blocks = self._find_edge_dominated_blocks(green_channel)

        usable_blocks = ~(flat_blocks | edge_blocks)
        excluded_fraction = 1.0 - (float(np.count_nonzero(usable_blocks))
                                   / float(usable_blocks.size))
        mask = upscale_block_map(usable_blocks, self.feature_block_size)
        return mask, excluded_fraction

    def _find_flat_blocks(self, green_channel: np.ndarray) -> np.ndarray:
        """Locate blocks too uniform to carry a prediction-error signal.

        SKILL "Unreliable / inapplicable when": "Flat/uniform and saturated
        regions - near-zero prediction error regardless of CFA presence
        (Ferrara, explicit limitation)".

        Args:
            green_channel: Float64 green plane.

        Returns:
            Boolean array at block resolution, True where the block is flat.
        """
        blocks = partition_into_blocks(green_channel, self.feature_block_size)
        block_variance = blocks.var(axis=(2, 3))
        return block_variance < constants.FLAT_BLOCK_VARIANCE_THRESHOLD

    def _find_edge_dominated_blocks(self, green_channel: np.ndarray) -> np.ndarray:
        """Locate blocks a sharp edge dominates.

        SKILL "Unreliable / inapplicable when": "Sharp edges - can mimic the
        CFA-absent statistical signature (false positives), same Ferrara
        limitation".

        Args:
            green_channel: Float64 green plane.

        Returns:
            Boolean array at block resolution, True where the block is
            edge-dominated.
        """
        # Sobel magnitude is a standard first-derivative edge response; the
        # engine only needs to know where the gradient is large, not its exact
        # calibration, so no corpus value is being substituted here.
        vertical = ndimage.sobel(green_channel, axis=0, mode="reflect")
        horizontal = ndimage.sobel(green_channel, axis=1, mode="reflect")
        # scipy returns the unnormalised Sobel response, four times the actual
        # gradient, so divide before comparing against a threshold expressed in
        # intensity units per pixel.
        magnitude = (np.hypot(vertical, horizontal)
                     / constants.SOBEL_NORMALISATION_FACTOR)

        is_edge = magnitude > constants.SHARP_EDGE_GRADIENT_THRESHOLD
        edge_blocks = partition_into_blocks(is_edge.astype(np.float64),
                                            self.feature_block_size)
        edge_fraction = edge_blocks.mean(axis=(2, 3))
        return edge_fraction > constants.SHARP_EDGE_PIXEL_FRACTION
