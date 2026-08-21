"""Grayscale conversion, Haar-DWT LL extraction, and block tiling.

All three pipelines in this SKILL operate on grayscale/single-channel data
("Format: grayscale or single-channel ... all three forensically-relevant
papers here ... operate on grayscale"). RGB input is converted once, here.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import pywt

from . import constants
from .contracts import Block, ImageMetadata, PreparedImage

logger = logging.getLogger(__name__)


class WaveletPreprocessor:
    """Prepares a raw image for wavelet decomposition and block tiling."""

    def prepare(self,
                image: np.ndarray,
                metadata: ImageMetadata) -> PreparedImage:
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

    def limit_resolution(self, image: np.ndarray) -> tuple:
        """Downscale so the long side fits the analysed-resolution cap.

        ENHANCEMENT 6. Pipeline C tiles at stride 1, so the candidate-pair
        count grows roughly with the square of the block count. At full
        resolution the six corpus images produced up to 11,842,381,512
        candidate pairs and the process was killed by the operating system;
        none could be scored. See constants.MAXIMUM_ANALYSED_LONG_SIDE_PIXELS
        for the measured basis of the cap.

        Args:
            image: BGR uint8 or grayscale array.

        Returns:
            Tuple of (possibly downscaled image, scale factor applied). A
            factor of 1.0 means the image was left untouched.
        """
        array = np.asarray(image)
        longest = max(array.shape[0], array.shape[1])
        cap = constants.MAXIMUM_ANALYSED_LONG_SIDE_PIXELS
        if longest <= cap:
            return array, 1.0
        scale = cap / float(longest)
        target = (max(int(array.shape[1] * scale), 1),
                  max(int(array.shape[0] * scale), 1))
        return cv2.resize(array, target, interpolation=cv2.INTER_AREA), scale

    def extract_ll_subband(self, grayscale: np.ndarray) -> np.ndarray:
        """Single-level Haar DWT, keeping only the coarse LL subband.

        Args:
            grayscale: Float64 grayscale image.

        Returns:
            The LL (approximation) subband as a float64 array.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: psi(x)=sum_k (-1)^k a_{N-1-k} sqrt(2) phi(2x-k) [Eq.1]
        #          f(x,y)=sum_j sum_k sum_l d_jkl psi_jk(x) psi_jl(y) [Eq.2]
        # Variables: psi=Haar wavelet function, phi=scaling function,
        #   d_jkl=decomposition coefficients, f=reconstructed 2-D signal.
        # Source: Kashyap & Joshi 2013, Eq. 1-2.
        # Expected range: LL values on the same scale as the input image.
        # ──────────────────────────────────────────────────────────
        ll_subband, _ = pywt.dwt2(grayscale, constants.PIPELINE_C_WAVELET_FAMILY)
        return ll_subband

    def tile_blocks(self,
                    ll_subband: np.ndarray,
                    block_size: int = constants.DEFAULT_BLOCK_SIZE,
                    stride: int = constants.BLOCK_STRIDE_PIXELS) -> list:
        """Slide overlapping R x R blocks across the LL subband.

        SKILL: "Tile the coarse subband into overlapping R x R blocks,
        sliding by 1 pixel horizontally then vertically ((M-R+1)x(N-R+1)
        total blocks for an M x N image)."

        Args:
            ll_subband: Float64 LL subband array.
            block_size: R, the block side length.
            stride: Pixel step between consecutive block origins.

        Returns:
            List of Block objects; empty if the subband is smaller than one
            block in either dimension.
        """
        height, width = ll_subband.shape
        blocks = []
        for row in range(0, height - block_size + 1, stride):
            for col in range(0, width - block_size + 1, stride):
                patch = ll_subband[row:row + block_size, col:col + block_size]
                blocks.append(Block(pixels=patch, row=row, col=col))
        return blocks
