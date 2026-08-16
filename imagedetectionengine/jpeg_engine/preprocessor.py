"""Luminance extraction, 8x8 block-grid cropping, and block-DCT computation.

Both closed-form pipelines in this SKILL operate on the luminance channel
only: "grayscale or luminance (Y) channel only - every closed-form method
here (Luo, Mahdian) explicitly restricts to the luminance channel". Mahdian &
Saic's stated reason is that chrominance is subsampled and coarsely
quantized in-camera, leaving "little information valuable" for detection.

The block grid is assumed aligned at the image origin. The SKILL is explicit
that misaligned grids are unsolved here: "none of the closed-form methods in
this corpus solve" the misaligned-grid case.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from . import constants
from .contracts import BlockDctResult, ImageMetadata, PreparedImage
from .utils import split_into_tiles, tilewise_dct_two_dimensional

logger = logging.getLogger(__name__)


class JpegPreprocessor:
    """Turns a raw image into unsaturated 8x8 block-DCT coefficients."""

    def prepare(self, image: np.ndarray,
                metadata: ImageMetadata) -> PreparedImage:
        """Extract the luminance channel and crop it to the 8x8 block grid.

        Args:
            image: BGR uint8 (H x W x 3) or grayscale (H x W) array.
            metadata: Container facts; not consulted for luminance extraction.

        Returns:
            PreparedImage holding the cropped luminance array.

        Raises:
            ValueError: If the array has an unsupported number of dimensions,
                or is smaller than a single 8x8 block.
        """
        array = np.asarray(image)
        if array.ndim == constants.COLOUR_IMAGE_DIMENSION_COUNT:
            luminance = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        elif array.ndim == constants.GRAYSCALE_IMAGE_DIMENSION_COUNT:
            luminance = array
        else:
            raise ValueError(f"expected a 2-D or 3-D image array, got "
                             f"{array.ndim} dimensions")

        cropped = self.crop_to_block_grid(luminance)
        if cropped.size == 0:
            raise ValueError(f"image {array.shape[0]}x{array.shape[1]} is "
                             f"smaller than one "
                             f"{constants.DCT_BLOCK_SIZE}x"
                             f"{constants.DCT_BLOCK_SIZE} DCT block")
        return PreparedImage(luminance=cropped.astype(np.float64),
                             original_shape=array.shape,
                             cropped_shape=cropped.shape)

    @staticmethod
    def crop_to_block_grid(luminance: np.ndarray) -> np.ndarray:
        """Trim the image to an exact multiple of the DCT block size.

        Args:
            luminance: 2-D luminance array.

        Returns:
            The array trimmed at the right and bottom edges; empty if the
            image is smaller than one block in either dimension.
        """
        block = constants.DCT_BLOCK_SIZE
        usable_height = (luminance.shape[0] // block) * block
        usable_width = (luminance.shape[1] // block) * block
        return luminance[:usable_height, :usable_width]

    def extract_block_dct(self, luminance: np.ndarray) -> BlockDctResult:
        """Compute the 8x8 block DCT over every unsaturated block.

        Args:
            luminance: Float64 luminance array, cropped to the block grid.

        Returns:
            BlockDctResult holding the coefficients of unsaturated blocks only.
        """
        block = constants.DCT_BLOCK_SIZE
        pixel_tiles = split_into_tiles(luminance, block)
        keep = self.unsaturated_block_mask(pixel_tiles)

        # The level shift is definitional to the JPEG forward DCT: samples are
        # centred on zero before transforming.
        shifted = pixel_tiles[keep] - constants.JPEG_LEVEL_SHIFT
        coefficients = tilewise_dct_two_dimensional(shifted)

        return BlockDctResult(
            coefficients=coefficients,
            total_block_count=int(pixel_tiles.shape[0]),
            unsaturated_block_count=int(coefficients.shape[0]),
            blocks_per_row=luminance.shape[1] // block,
            blocks_per_column=luminance.shape[0] // block)

    @staticmethod
    def unsaturated_block_mask(pixel_tiles: np.ndarray) -> np.ndarray:
        """Mark blocks containing no clipped (saturated) pixel.

        SKILL: truncation error is "neglected in the rest of the analysis by
        restricting statistics to unsaturated 8x8 blocks". A clipped pixel
        sits exactly on a dynamic-range bound, so any block touching either
        bound is dropped.

        Args:
            pixel_tiles: Array of shape (n_blocks, size, size).

        Returns:
            Boolean array of length n_blocks, True for blocks to keep.
        """
        touches_floor = np.any(pixel_tiles <= constants.PIXEL_VALUE_MINIMUM,
                               axis=(1, 2))
        touches_ceiling = np.any(pixel_tiles >= constants.PIXEL_VALUE_MAXIMUM,
                                 axis=(1, 2))
        return ~(touches_floor | touches_ceiling)
