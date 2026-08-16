"""Grayscale conversion, wavelet-domain residual extraction, and block tiling.

The residual filter F is named ("Mihcak wavelet-based denoising filter") but
never mathematically specified in this SKILL - see constants.py's module
docstring. Implemented here as a single-level DWT reconstructed from the LL
subband only, the simplest concrete instantiation of F's stated role.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import pywt

from . import constants
from .contracts import ImageMetadata, NoiseBlock, PreparedImage

logger = logging.getLogger(__name__)


class NoisePreprocessor:
    """Prepares a raw image for residual extraction and block tiling."""

    def prepare(self, image: np.ndarray, metadata: ImageMetadata) -> PreparedImage:
        """Convert the input image to a float64 grayscale array.

        Args:
            image: BGR uint8 (H x W x 3) or grayscale (H x W) array.
            metadata: Container facts; not consulted for grayscale conversion.

        Returns:
            PreparedImage holding the grayscale array and original shape.

        Raises:
            ValueError: If the array has an unsupported number of dimensions.
        """
        array = np.asarray(image)
        if array.ndim == constants.COLOUR_IMAGE_DIMENSION_COUNT:
            gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        elif array.ndim == constants.GRAYSCALE_IMAGE_DIMENSION_COUNT:
            gray = array
        else:
            raise ValueError(f"expected a 2-D or 3-D image array, got "
                             f"{array.ndim} dimensions")
        return PreparedImage(grayscale=gray.astype(np.float64),
                             original_shape=array.shape)

    def extract_residual(self, grayscale: np.ndarray) -> np.ndarray:
        """Compute the noise residual W = I - F(I) via the wavelet low-pass F.

        Args:
            grayscale: Float64 grayscale image.

        Returns:
            Residual array, same shape as grayscale.
        """
        low_frequency_approximation, _ = pywt.dwt2(
            grayscale, constants.RESIDUAL_FILTER_WAVELET_FAMILY)
        reconstructed_low_pass = pywt.idwt2(
            (low_frequency_approximation, (None, None, None)),
            constants.RESIDUAL_FILTER_WAVELET_FAMILY)
        reconstructed_low_pass = reconstructed_low_pass[:grayscale.shape[0],
                                                        :grayscale.shape[1]]
        return grayscale - reconstructed_low_pass

    def resolution_scaled_block_size(self, height: int, width: int) -> int:
        """Pick a block size scaled to input resolution.

        SKILL: "start from Chen et al.'s 128x128 default for full-frame
        images and scale down proportionally for crops/small inputs,
        consistent with Debiasi et al.'s finding that finer fragmentation
        (up to a point - 8x8, not 10x10) improves sensitivity".

        Args:
            height: Image height in pixels.
            width: Image width in pixels.

        Returns:
            Block side length in pixels.
        """
        proposed = min(height, width) // constants.TARGET_BLOCK_GRID_CELLS
        bounded = max(proposed, constants.MINIMUM_BLOCK_SIZE_PIXELS)
        return min(bounded, constants.MAXIMUM_BLOCK_SIZE_PIXELS)

    def tile_blocks(self, residual: np.ndarray, intensity: np.ndarray,
                    block_size: int) -> list:
        """Tile the residual (and matching intensity) into non-overlapping blocks.

        Args:
            residual: Float64 noise-residual array W.
            intensity: Float64 original-intensity grayscale array, same shape.
            block_size: Block side length in pixels.

        Returns:
            List of NoiseBlock objects; empty if the image is smaller than
            one block in either dimension.
        """
        height, width = residual.shape
        blocks = []
        grid_row = 0
        for pixel_row in range(0, height - block_size + 1, block_size):
            grid_col = 0
            for pixel_col in range(0, width - block_size + 1, block_size):
                blocks.append(NoiseBlock(
                    residual_pixels=residual[pixel_row:pixel_row + block_size,
                                            pixel_col:pixel_col + block_size],
                    intensity_pixels=intensity[pixel_row:pixel_row + block_size,
                                              pixel_col:pixel_col + block_size],
                    row=grid_row, col=grid_col,
                    pixel_row=pixel_row, pixel_col=pixel_col))
                grid_col += 1
            grid_row += 1
        return blocks
