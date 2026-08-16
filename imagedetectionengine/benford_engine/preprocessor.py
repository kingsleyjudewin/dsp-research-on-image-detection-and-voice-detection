"""Preprocessing chain turning a BGR image into analysable 8x8 blocks.

Implements the preprocessing steps of the SKILL file's "Input requirements":
single-channel selection, then partition into non-overlapping 8x8 blocks ready
for the block DCT.
"""

from __future__ import annotations

import logging

import numpy as np

from . import constants

logger = logging.getLogger(__name__)


class BenfordPreprocessor:
    """Converts a BGR image into the block array the computer consumes."""

    def prepare(self, image: np.ndarray) -> np.ndarray:
        """Run the full preprocessing chain.

        Args:
            image: BGR uint8 array of shape (height, width, 3).

        Returns:
            Float array of shape (block_count, 8, 8) holding every
            non-overlapping block of the analysis channel.

        Raises:
            ValueError: If the image is not a 3-channel array, or is too small
                to yield a single complete block.
        """
        channel = self._select_analysis_channel(image)
        cropped = self._crop_to_block_grid(channel)
        return self._partition_into_blocks(cropped)

    def _select_analysis_channel(self, image: np.ndarray) -> np.ndarray:
        """Extract the single colour plane the corpus analyses.

        SKILL "Input requirements" step 1: every paper "operates on grayscale
        or a single (green/luminance) channel". Moin et al. 2017 state the
        choice explicitly - "Extract the green channel" - so green is used.
        Selecting a plane rather than computing luma avoids introducing
        luma weights, which the SKILL flags as "(not specified in the corpus
        for color)".

        Args:
            image: BGR uint8 array of shape (height, width, 3).

        Returns:
            Float64 array of shape (height, width).

        Raises:
            ValueError: If the array is not a 3-channel colour image.
        """
        array = np.asarray(image)
        has_colour_axis = array.ndim == constants.EXPECTED_IMAGE_DIMENSION_COUNT
        if not has_colour_axis or \
                array.shape[-1] != constants.EXPECTED_CHANNEL_COUNT:
            raise ValueError(
                f"expected a {constants.EXPECTED_CHANNEL_COUNT}-channel BGR "
                f"image, received an array of shape {array.shape}"
            )

        # Index 1 of a BGR array is the green plane.
        return array[..., constants.ANALYSIS_CHANNEL_INDEX].astype(np.float64)

    def _crop_to_block_grid(self, channel: np.ndarray) -> np.ndarray:
        """Trim the channel so both dimensions are whole multiples of the block.

        Partial edge blocks are discarded rather than padded: padding would
        inject synthetic coefficient values into the digit histogram, which is
        the very statistic being measured.

        Args:
            channel: Single-channel array of shape (height, width).

        Returns:
            View of the channel cropped to a whole number of blocks.

        Raises:
            ValueError: If the channel cannot contain even one whole block.
        """
        block_size = constants.DCT_BLOCK_SIZE
        height, width = channel.shape[:2]
        usable_height = (height // block_size) * block_size
        usable_width = (width // block_size) * block_size

        if usable_height == 0 or usable_width == 0:
            raise ValueError(
                f"image of {height}x{width} is smaller than one "
                f"{block_size}x{block_size} block"
            )

        if (usable_height, usable_width) != (height, width):
            logger.debug("cropped %dx%d to %dx%d to fit the block grid",
                         height, width, usable_height, usable_width)
        return channel[:usable_height, :usable_width]

    def _partition_into_blocks(self, channel: np.ndarray) -> np.ndarray:
        """Split the channel into non-overlapping square blocks.

        SKILL "Input requirements" step 2, quoting Bonettini et al.: "K
        non-overlapping 8x8-pixel blocks". Non-overlapping is essential -
        overlapping tiles would count the same pixel many times and inflate the
        apparent sample size of the digit histogram.

        Note on DC level shift: libjpeg subtracts a constant from every pixel
        before the transform, which shifts only the DC coefficient. The SKILL's
        frequency selection excludes DC, so the shift cannot affect any
        coefficient this engine reads and is deliberately not applied.

        Args:
            channel: Array whose dimensions are exact multiples of the block size.

        Returns:
            Array of shape (block_count, block_size, block_size).
        """
        block_size = constants.DCT_BLOCK_SIZE
        height, width = channel.shape[:2]
        rows_of_blocks = height // block_size
        columns_of_blocks = width // block_size

        # Split each axis into (block index, within-block offset), then bring
        # the two block-index axes together so they can be flattened into one.
        return (channel
                .reshape(rows_of_blocks, block_size, columns_of_blocks, block_size)
                .swapaxes(1, 2)
                .reshape(rows_of_blocks * columns_of_blocks, block_size, block_size))
