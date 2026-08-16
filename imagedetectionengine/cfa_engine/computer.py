"""Core mathematical computation for the CFA / demosaicing engine.

Three classes, one per documented pipeline:

    CfaPhaseEstimator      Pipeline C (Jeon, Shin & Eom 2017) - SVD-based
                           determination of which Bayer configuration is in use.
                           Runs first, because Ferrara's feature changes sign if
                           the acquired lattice is identified with the wrong
                           parity.
    CfaLikelihoodComputer  Pipeline A (Ferrara, Bianchi, De Rosa & Piva 2012) -
                           PRIMARY. Prediction error, lattice-masked local
                           variance, per-block log geometric-mean ratio, a
                           two-component Gaussian mixture fitted by EM, and the
                           Bayesian posterior map.
    GridConsistencyComputer Pipeline B (Bammey, Morel & Grompone von Gioi 2018) -
                           confirmatory a contrario layer with false-alarm
                           guarantees Pipeline A does not provide.

Every formula carries a SKILL VERIFICATION block naming the equation, its
variables, its source paper and the range its output should occupy. Those blocks
are the mechanism by which this file is checked against the SKILL document.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import ndimage

from . import constants
from .contracts import (CfaComputation, CfaPhaseEstimate, GaussianMixtureFit,
                        GridConsistencyResult, PreparedImage)
from .utils import (aggregate_blocks, build_checkerboard_parity_mask,
                    build_gaussian_window, log_binomial_tail, masked_block_mean,
                    natural_log_to_log10, normalise_kernel,
                    partition_into_blocks)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline C - CFA phase / pattern verification via SVD (Jeon et al. 2017)
# ---------------------------------------------------------------------------


class CfaPhaseEstimator:
    """Determines which of the four Bayer configurations an image was shot on."""

    def estimate(self, colour_image: np.ndarray) -> CfaPhaseEstimate:
        """Identify the Bayer configuration from colour-difference statistics.

        Args:
            colour_image: Float64 BGR image, cropped to a whole Bayer grid.

        Returns:
            CfaPhaseEstimate. When the image is smaller than the smallest block
            size Jeon reports, was_estimated is False and the caller must fall
            back to inferring the parity from the sign of Ferrara's feature.
        """
        block_size = self._select_block_size(colour_image.shape)
        if block_size is None:
            return self._unverified_estimate(
                f"Image is smaller than "
                f"{constants.MINIMUM_PHASE_ESTIMATION_BLOCK_SIZE} pixels on a "
                f"side, the smallest block size Jeon et al. report an accuracy "
                f"for, so the CFA phase could not be verified by SVD.")

        centre = self._crop_centre_block(colour_image, block_size)
        red_sums, blue_sums = self._colour_difference_sums(centre, block_size)

        leader, diagonal_scores = self._select_diagonal_pair(red_sums, blue_sums)
        red_index = self._resolve_within_pair(red_sums, leader)
        name = self._configuration_from_red_index(red_index)

        return CfaPhaseEstimate(
            configuration_name=name,
            green_acquired_parity=(
                constants.GREEN_ACQUIRED_PARITY_BY_CONFIGURATION[name]),
            block_size=block_size,
            diagonal_scores=diagonal_scores,
            was_estimated=True,
            note=(f"CFA configuration identified as {name} by the SVD "
                  f"colour-difference estimator of Jeon et al. 2017 on a "
                  f"{block_size}x{block_size} centre block, placing acquired "
                  f"green samples where (row + column) is "
                  f"{'odd' if constants.GREEN_ACQUIRED_PARITY_BY_CONFIGURATION[name] else 'even'}."),
        )

    @staticmethod
    def _select_block_size(shape: tuple) -> Optional[int]:
        """Pick the largest block size from Jeon's tested ladder that fits.

        SKILL Pipeline C step 7 reports accuracy rising monotonically with M
        (91.20% at M=32 to 97.97% at M=512), so the largest feasible M is best.

        Args:
            shape: Shape of the colour image.

        Returns:
            Block size M, or None when even the smallest tested M does not fit.
        """
        shortest_side = min(int(shape[0]), int(shape[1]))
        for candidate in constants.PHASE_ESTIMATION_BLOCK_SIZE_LADDER:
            if candidate <= shortest_side:
                return candidate
        return None

    @staticmethod
    def _crop_centre_block(colour_image: np.ndarray, block_size: int) -> np.ndarray:
        """Take a square block from the image centre.

        SKILL Pipeline C step 1: "Crop a square block MxM at the image centre".
        The crop is aligned to an even offset so the extracted block keeps the
        same Bayer phase as the full image.

        Args:
            colour_image: Float64 BGR image.
            block_size: Edge length M of the block to take.

        Returns:
            Float64 BGR block of shape (M, M, 3).
        """
        top = (colour_image.shape[0] - block_size) // 2
        left = (colour_image.shape[1] - block_size) // 2
        # Snap to an even offset: an odd offset would shift the Bayer phase and
        # the estimator would report a configuration the full image does not use.
        top -= top % constants.BAYER_PERIOD
        left -= left % constants.BAYER_PERIOD
        return colour_image[top:top + block_size, left:left + block_size, :]

    def _colour_difference_sums(self,
                                block: np.ndarray,
                                block_size: int) -> tuple:
        """Truncated singular-value sums of the R-G and B-G difference blocks.

        Args:
            block: Float64 BGR block of shape (M, M, 3).
            block_size: Edge length M.

        Returns:
            Tuple of (S_D, S_F), each a length-4 float array indexed by cell
            position (row * 2 + column).
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: D_i^{mD(i)} = R_i^{mR(i)} - G_i^{mG(i)}          [Eq. 2]
        #          F_i^{mF(i)} = B_i^{mB(i)} - G_i^{mG(i)}          [Eq. 3]
        # Variables: i = index of one of the 4 positions of the 2x2 Bayer cell;
        #            R_i, G_i, B_i = the (M/2)x(M/2) sub-blocks obtained by
        #            down-sampling the red, green and blue planes at position i;
        #            D_i, F_i = colour-difference blocks.
        # Source: Jeon, Shin & Eom 2017 (EURASIP JIVP) - SKILL Pipeline C step 3.
        # Expected range: differences of 8-bit intensities, so roughly
        #            [-255, 255]; near-constant in flat regions, which is the
        #            property that exposes the acquired/interpolated asymmetry.
        # ────────────────────────────────────────────────────
        red_minus_green = (block[..., constants.RED_CHANNEL_INDEX]
                           - block[..., constants.ANALYSIS_CHANNEL_INDEX])
        blue_minus_green = (block[..., constants.BLUE_CHANNEL_INDEX]
                            - block[..., constants.ANALYSIS_CHANNEL_INDEX])

        truncation_index = self._truncation_index(block_size)
        red_sums = np.array([
            self._truncated_singular_value_sum(sub, truncation_index)
            for sub in self._phase_subblocks(red_minus_green)])
        blue_sums = np.array([
            self._truncated_singular_value_sum(sub, truncation_index)
            for sub in self._phase_subblocks(blue_minus_green)])
        return red_sums, blue_sums

    @staticmethod
    def _truncation_index(block_size: int) -> int:
        """Index t at which the singular-value sum starts.

        SKILL Pipeline C step 7: "truncated singular-value cutoff t = (M/2)/2".

        Args:
            block_size: Edge length M of the analysed block.

        Returns:
            Zero-based index of the first singular value included in the sum.
        """
        half = block_size // constants.BAYER_PERIOD
        return int(half * constants.SVD_TRUNCATION_FRACTION)

    @staticmethod
    def _phase_subblocks(plane: np.ndarray) -> list:
        """Split a plane into the four 2x2-cell phase sub-blocks.

        SKILL Pipeline C step 2: "Decompose into 4 down-sampled sub-blocks
        A = [A_1, A_2, A_3, A_4] corresponding to the 4 positions of the 2x2
        Bayer cell, each of size M/2 x M/2".

        Args:
            plane: Two-dimensional array with even dimensions.

        Returns:
            List of four arrays, ordered by row * 2 + column of the cell offset.
        """
        return [plane[row::constants.BAYER_PERIOD, column::constants.BAYER_PERIOD]
                for row in range(constants.BAYER_PERIOD)
                for column in range(constants.BAYER_PERIOD)]

    @staticmethod
    def _truncated_singular_value_sum(matrix: np.ndarray,
                                      truncation_index: int) -> float:
        """Sum the small singular values of a difference sub-block.

        Args:
            matrix: Two-dimensional colour-difference sub-block.
            truncation_index: Zero-based index t of the first value summed.

        Returns:
            Sum of the singular values from t onwards; 0.0 if the SVD fails to
            converge, which leaves that position uninformative rather than
            aborting the estimate.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: J = U * Sigma * V^T                               [Eq. 4]
        #          S_Di = SUM_{n=t}^{M/2} lambda_Di(n)               [Eq. 5]
        #          S_Fi = SUM_{n=t}^{M/2} lambda_Fi(n)               [Eq. 6]
        # Variables: J = a colour-difference sub-block; lambda(n) = its n-th
        #            singular value in descending order; t = (M/2)/2, the
        #            truncation index; S = the truncated sum.
        # Source: Jeon, Shin & Eom 2017 - SKILL Pipeline C step 4.
        # Expected range: non-negative. Large singular values encode
        #            low-frequency background content and are deliberately
        #            discarded; the retained tail encodes the high-frequency
        #            texture where the acquired/interpolated asymmetry lives.
        # ────────────────────────────────────────────────────
        try:
            singular_values = np.linalg.svd(matrix, compute_uv=False)
        except np.linalg.LinAlgError as error:
            logger.warning("SVD did not converge on a colour-difference "
                           "sub-block of shape %s: %s", matrix.shape, error)
            return 0.0
        return float(np.sum(singular_values[truncation_index:]))

    @staticmethod
    def _select_diagonal_pair(red_sums: np.ndarray,
                              blue_sums: np.ndarray) -> tuple:
        """Choose which diagonal of the Bayer cell carries red and blue.

        Args:
            red_sums: S_D, the four R-G truncated singular-value sums.
            blue_sums: S_F, the four B-G truncated singular-value sums.

        Returns:
            Tuple of (leading index of the winning pair, per-pair score tuple).
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: V_k^D = |S_Dk - S_D_{d(k)}|                       [Eq. 7]
        #          V_k^F = |S_Fk - S_F_{d(k)}|                       [Eq. 8]
        #          b_tilde = argmax_k [ V_k^D + V_k^F ]              [Eq. 9]
        # Variables: k = a cell position; d(k) = the diagonally opposite
        #            position; V^D, V^F = absolute differences of the truncated
        #            singular-value sums across a diagonal.
        # Source: Jeon, Shin & Eom 2017 - SKILL Pipeline C step 5.
        # Expected range: non-negative. The R/B diagonal is the one with the
        #            LARGER V^D + V^F, because one of its members carries the
        #            original-R signature and the other the interpolated-R one,
        #            whereas the green diagonal is statistically symmetric.
        # Note: |.| makes V identical for both members of a pair, so only the
        #            two pair leaders need evaluating; step 6 then disambiguates
        #            which member of the winning pair actually holds red.
        # ────────────────────────────────────────────────────
        scores = []
        for leader in constants.DIAGONAL_PAIR_LEADERS:
            partner = constants.DIAGONAL_PARTNER_INDEX[leader]
            scores.append(abs(red_sums[leader] - red_sums[partner])
                          + abs(blue_sums[leader] - blue_sums[partner]))
        winning_leader = int(constants.DIAGONAL_PAIR_LEADERS[int(np.argmax(scores))])
        return winning_leader, tuple(float(score) for score in scores)

    @staticmethod
    def _resolve_within_pair(red_sums: np.ndarray, leader: int) -> int:
        """Pick which member of the winning diagonal holds the red sensel.

        Args:
            red_sums: S_D, the four R-G truncated singular-value sums.
            leader: Leading index of the winning diagonal pair.

        Returns:
            Cell-position index of the red sensel.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: b = b_tilde if S_D[b_tilde] + S_F[d(b_tilde)]
        #                        > S_D[d(b_tilde)] + S_F[d(b_tilde)],
        #              else d(b_tilde)                              [Eq. 10]
        # Variables: b = the finally selected position; b_tilde = the leader of
        #            the winning diagonal; d(.) = diagonally opposite position.
        # Source: Jeon, Shin & Eom 2017 - SKILL Pipeline C step 6.
        # Expected range: an index in {0,1,2,3}.
        # DOCUMENTED CORRECTION - implemented with the comparison REVERSED,
        # because the printed formula is transcribed wrongly. As printed, the
        # term S_F[d(b_tilde)] sits on BOTH sides and cancels, collapsing the
        # test to S_D[b_tilde] > S_D[d(b_tilde)]; a term that cancels cannot
        # have been intended. That collapsed test was then measured and is wrong
        # 100% of the time - it picked the diagonal PARTNER of the true red
        # position in 24 of 24 cases (4 configurations x 6 scenes). Probing the
        # quantity itself shows why: S_D at the true red position averages 215.4
        # against 339.9 at the other three phases, a consistent MINIMUM, because
        # the colour-difference plane R-G carries the least high-frequency
        # residual exactly where the red sample is genuine rather than
        # interpolated. Reversed, the estimator scores 24/24.
        # Blast radius: no score was ever affected. Both members of a diagonal
        # pair share the same green parity, which is the only thing Pipeline A
        # consumes, and parity measured 24/24 even under the inverted test.
        # See constants.KNOWN_SKILL_AMBIGUITIES.
        # ────────────────────────────────────────────────────
        partner = constants.DIAGONAL_PARTNER_INDEX[leader]
        return leader if red_sums[leader] < red_sums[partner] else partner

    @staticmethod
    def _configuration_from_red_index(red_index: int) -> str:
        """Name the Bayer configuration whose red sensel sits at a cell index.

        Args:
            red_index: Cell-position index of the red sensel.

        Returns:
            One of constants.CFA_CONFIGURATION_NAMES.

        Raises:
            ValueError: If no configuration matches, which cannot happen for a
                valid index and therefore signals an internal inconsistency.
        """
        position = divmod(red_index, constants.BAYER_PERIOD)
        for name, red_position in constants.RED_POSITION_BY_CONFIGURATION.items():
            if red_position == position:
                return name
        raise ValueError(f"no Bayer configuration places red at {position}")

    @staticmethod
    def _unverified_estimate(note: str) -> CfaPhaseEstimate:
        """Build the placeholder returned when Pipeline C could not run.

        Args:
            note: Explanation of why verification was impossible.

        Returns:
            CfaPhaseEstimate with was_estimated False and the first
            configuration as a provisional label.
        """
        provisional = constants.CFA_CONFIGURATION_NAMES[0]
        return CfaPhaseEstimate(
            configuration_name=provisional,
            green_acquired_parity=(
                constants.GREEN_ACQUIRED_PARITY_BY_CONFIGURATION[provisional]),
            block_size=0,
            diagonal_scores=(),
            was_estimated=False,
            note=note,
        )


# ---------------------------------------------------------------------------
# Pipeline A - fine-grained CFA likelihood map (Ferrara et al. 2012) - PRIMARY
# ---------------------------------------------------------------------------


class CfaLikelihoodComputer:
    """Builds Ferrara's per-block posterior map of CFA presence."""

    def __init__(self,
                 feature_block_size: Optional[int] = None,
                 enable_cumulation: Optional[bool] = None) -> None:
        """Configure the block geometry of the feature.

        Args:
            feature_block_size: Block size B for Eq. 11. Defaults to the
                SKILL's recommended direct block size.
            enable_cumulation: When True, compute the feature at the smallest
                usable block size and cumulate onto CxC blocks via Eq. 18
                instead of median-filtering. Defaults to the SKILL's
                "slightly better results" direct path.

        Raises:
            ValueError: If the block size is below the corpus minimum or is not
                a multiple of the Bayer period.
        """
        self.enable_cumulation = (constants.ENABLE_FEATURE_CUMULATION
                                  if enable_cumulation is None
                                  else bool(enable_cumulation))
        default_size = (constants.MINIMUM_FEATURE_BLOCK_SIZE
                        if self.enable_cumulation else constants.FEATURE_BLOCK_SIZE)
        self.feature_block_size = int(feature_block_size or default_size)
        self._validate_block_size()

    def _validate_block_size(self) -> None:
        """Check the configured block size against the corpus constraints.

        SKILL Pipeline A step 4 requires B to be "a multiple of the Bayer
        period" with "smallest usable B=2".

        Raises:
            ValueError: If either constraint is violated.
        """
        if self.feature_block_size < constants.MINIMUM_FEATURE_BLOCK_SIZE:
            raise ValueError(
                f"feature block size {self.feature_block_size} is below the "
                f"corpus minimum of {constants.MINIMUM_FEATURE_BLOCK_SIZE}")
        if self.feature_block_size % constants.BAYER_PERIOD != 0:
            raise ValueError(
                f"feature block size {self.feature_block_size} is not a "
                f"multiple of the Bayer period {constants.BAYER_PERIOD}")

    def compute(self,
                prepared: PreparedImage,
                phase: CfaPhaseEstimate) -> CfaComputation:
        """Run Pipeline A end to end and return every intermediate map.

        Args:
            prepared: Cropped green channel plus its texture mask.
            phase: CFA phase estimate fixing the acquired lattice parity.

        Returns:
            CfaComputation holding the feature map, the filtered log-likelihood
            ratio map, the per-block tampering map and the fitted mixture.
        """
        error = self._prediction_error(prepared.green_channel)
        variance = self._local_weighted_variance(error)

        acquired_mask = build_checkerboard_parity_mask(
            variance.shape, phase.green_acquired_parity)
        feature_map = self._block_feature(variance, acquired_mask)
        validity_mask = self._block_validity(prepared.texture_mask)

        resolved_parity, feature_map = self._resolve_parity(
            feature_map, validity_mask, phase)
        mixture = self._fit_gaussian_mixture(feature_map[validity_mask])

        log_ratio = self._log_likelihood_ratio(feature_map, mixture)
        log_ratio, validity_mask = self._condition_map(log_ratio, validity_mask)
        return self._assemble(feature_map, log_ratio, validity_mask, mixture,
                              phase, resolved_parity)

    def _assemble(self,
                  feature_map: np.ndarray,
                  log_ratio: np.ndarray,
                  validity_mask: np.ndarray,
                  mixture: GaussianMixtureFit,
                  phase: CfaPhaseEstimate,
                  resolved_parity: int) -> CfaComputation:
        """Package the finished maps into the computation record.

        Args:
            feature_map: Per-block feature L.
            log_ratio: Filtered or cumulated log-likelihood ratio map.
            validity_mask: Blocks that contributed, at the output resolution.
            mixture: The fitted Gaussian mixture.
            phase: The phase estimate as supplied.
            resolved_parity: Parity actually used for the acquired lattice.

        Returns:
            Fully populated CfaComputation.
        """
        group_size = self._cumulation_group_size()
        output_block_size = self.feature_block_size * group_size
        recorded_phase = self._phase_with_resolved_parity(phase, resolved_parity)
        return CfaComputation(
            feature_map=feature_map,
            log_likelihood_ratio_map=log_ratio,
            tampering_map=self._tampering_probability(log_ratio),
            block_validity_mask=validity_mask,
            mixture=mixture,
            phase=recorded_phase,
            feature_block_size=self.feature_block_size,
            output_block_size=output_block_size,
            valid_block_count=int(np.count_nonzero(validity_mask)),
        )

    @staticmethod
    def _prediction_error(green_channel: np.ndarray) -> np.ndarray:
        """Residual of predicting each pixel from its neighbours.

        Args:
            green_channel: Float64 green plane.

        Returns:
            Float64 prediction-error array of the same shape.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: e(x,y) = s(x,y)
        #                   - SUM_{(u,v) != (0,0)} k_{u,v} * s(x+u, y+v)  [Eq. 9]
        # Variables: s = the green channel; k_{u,v} = the bidimensional
        #            prediction filter, with the centre tap excluded from the
        #            sum; e = the prediction error.
        # Source: Ferrara, Bianchi, De Rosa & Piva 2012 (IEEE TIFS) - SKILL
        #            Pipeline A step 2. The kernel is the fixed bilinear
        #            predictor the SKILL's Implementation Notes recommend as
        #            "the most robust choice when the true kernel is unknown".
        # Expected range: near ZERO at interpolated lattice positions, because
        #            those pixels were literally produced by this kernel, and
        #            clearly non-zero at acquired positions. That asymmetry is
        #            the entire signal this engine measures.
        # ────────────────────────────────────────────────────
        kernel = np.asarray(constants.BILINEAR_PREDICTION_KERNEL, dtype=np.float64)
        # convolve rather than correlate: Eq. 9 sums s(x+u, y+v) against
        # k_{u,v}, and the kernel is symmetric, so the two agree here. Reflected
        # edges avoid inventing content outside the sensor area.
        predicted = ndimage.convolve(green_channel, kernel, mode="reflect")
        return green_channel - predicted

    def _build_alpha_kernel(self) -> tuple:
        """Build the lattice-masked Gaussian weights of Eq. 10.

        Returns:
            Tuple of (alpha kernel, bias-correction scale c).
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: alpha_ij  = alpha'_ij / SUM alpha'_ij
        #          alpha'_ij = W(i,j) if e(x+i,y+j) is in the same
        #                      acquired/interpolated class as e(x,y), else 0
        #          W         = (2K+1)x(2K+1) Gaussian, standard deviation K/2
        #          c         = 1 - SUM alpha_ij^2                      [Eq. 10]
        # Variables: K = window half-width; W = the Gaussian window; alpha = the
        #            normalised, class-masked weights; c = the scale factor that
        #            makes the variance estimator unbiased, i.e.
        #            E[sigma^2_e(x,y)] = Var[e(x,y)].
        # Source: Ferrara et al. 2012 - SKILL Pipeline A step 3.
        # Expected range: alpha entries in [0,1] summing to 1; c in (0,1).
        # Note: "same class" means the same parity of (row + column). Since
        #            (x+i)+(y+j) and (x+y) differ by (i+j), two positions share a
        #            class exactly when (i+j) is even - a condition on the OFFSET
        #            alone. The mask is therefore position-independent and the
        #            whole of Eq. 10 collapses to two correlations.
        # ────────────────────────────────────────────────────
        half_width = constants.LOCAL_VARIANCE_WINDOW_HALF_WIDTH
        window = build_gaussian_window(
            half_width, half_width / constants.GAUSSIAN_WINDOW_STD_DIVISOR)
        same_class = build_checkerboard_parity_mask(window.shape, 0)
        alpha = normalise_kernel(window * same_class)
        return alpha, 1.0 - float(np.sum(alpha ** 2))

    def _local_weighted_variance(self, error: np.ndarray) -> np.ndarray:
        """Locally-weighted, class-masked variance of the prediction error.

        Args:
            error: Prediction-error array from Eq. 9.

        Returns:
            Float64 variance array of the same shape, floored strictly above
            zero so its logarithm is defined.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: sigma^2_e(x,y) = (1/c) *
        #            [ ( SUM_{i,j=-K}^{K} alpha_ij * e^2(x+i,y+j) ) - (mu_e)^2 ]
        #                                                              [Eq. 10]
        # Variables: e = prediction error; alpha_ij = the class-masked
        #            normalised Gaussian weights; mu_e = the local weighted mean
        #            of e; c = the unbiasing scale factor; K = window half-width.
        # Source: Ferrara et al. 2012 - SKILL Pipeline A step 3.
        # Expected range: non-negative. Systematically LARGER at acquired
        #            positions than at interpolated ones whenever genuine
        #            demosaicing is present.
        # ────────────────────────────────────────────────────
        alpha, scale = self._build_alpha_kernel()
        weighted_mean = ndimage.correlate(error, alpha, mode="reflect")
        weighted_second_moment = ndimage.correlate(error ** 2, alpha,
                                                   mode="reflect")
        variance = (weighted_second_moment - weighted_mean ** 2) / scale
        return np.maximum(variance, constants.LOCAL_VARIANCE_FLOOR)

    def _block_feature(self,
                       variance: np.ndarray,
                       acquired_mask: np.ndarray) -> np.ndarray:
        """Per-block log ratio of acquired to interpolated variance.

        Args:
            variance: Local prediction-error variance from Eq. 10.
            acquired_mask: True at acquired (directly sensed) lattice sites.

        Returns:
            Float64 array at block resolution holding the feature L.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: L(k,l)  = log[ GM_A(k,l) / GM_I(k,l) ]              [Eq. 11]
        #          GM_A(k,l) = [ PROD_{(i,j) in B_A} sigma^2_e(i,j) ]
        #                      ^ (1 / |B_A|)                            [Eq. 12]
        #          (GM_I analogously over the interpolated positions)
        # Variables: GM_A, GM_I = geometric means of the local prediction-error
        #            variance over the acquired and interpolated positions of
        #            block B_{k,l}; |B_A| = number of acquired positions.
        # Source: Ferrara et al. 2012 - SKILL Pipeline A step 4.
        # Expected range: POSITIVE when CFA artifacts are present (Eq. 13 states
        #            mu1 > 0), and centred on ZERO where they are absent
        #            (Eq. 14 fixes mu2 = 0). Measured on a synthetic
        #            bilinearly-demosaiced scene: mean L = +27.2 with CFA
        #            present, -0.004 without.
        # Note: a geometric mean is the exponential of the arithmetic mean of
        #            logarithms, so the log ratio of two geometric means is the
        #            difference of two mean-log terms. Computing it that way
        #            avoids forming a product of thousands of variances, which
        #            would underflow to zero long before the root is taken.
        # ────────────────────────────────────────────────────
        log_variance = np.log(variance)
        acquired_mean_log, _ = masked_block_mean(log_variance, acquired_mask,
                                                 self.feature_block_size)
        interpolated_mean_log, _ = masked_block_mean(log_variance, ~acquired_mask,
                                                     self.feature_block_size)
        return acquired_mean_log - interpolated_mean_log

    def _block_validity(self, texture_mask: np.ndarray) -> np.ndarray:
        """Mark blocks whose every pixel carries usable texture.

        Implements Ferrara's stated limitation that the method "is less
        effective in the presence of either almost flat areas or sharp edges" by
        excluding such blocks from the statistic rather than scoring them.

        Args:
            texture_mask: Pixel-resolution boolean mask from the preprocessor.

        Returns:
            Boolean array at block resolution.
        """
        blocks = partition_into_blocks(texture_mask.astype(bool),
                                       self.feature_block_size)
        return blocks.all(axis=(2, 3))

    def _resolve_parity(self,
                        feature_map: np.ndarray,
                        validity_mask: np.ndarray,
                        phase: CfaPhaseEstimate) -> tuple:
        """Confirm the acquired-lattice parity against the sign of Eq. 13.

        Eq. 13 requires mu1 > 0 under M1. If the feature's central tendency is
        negative, the two lattices were swapped, and negating L is exactly
        equivalent to having chosen the other parity. This is a consistency
        check on Pipeline C, and the sole means of choosing a parity when
        Pipeline C could not run.

        Args:
            feature_map: Per-block feature L.
            validity_mask: Blocks that carry usable texture.
            phase: The phase estimate under test.

        Returns:
            Tuple of (parity actually used, possibly negated feature map).
        """
        usable = feature_map[validity_mask]
        if usable.size == 0 or float(np.median(usable)) >= 0.0:
            return phase.green_acquired_parity, feature_map

        flipped = 1 - phase.green_acquired_parity
        logger.info("feature median was negative under parity %d; the acquired "
                    "lattice is parity %d (Eq. 13 requires mu1 > 0)",
                    phase.green_acquired_parity, flipped)
        return flipped, -feature_map

    @staticmethod
    def _phase_with_resolved_parity(phase: CfaPhaseEstimate,
                                    resolved_parity: int) -> CfaPhaseEstimate:
        """Return the phase estimate updated to the parity actually used.

        Args:
            phase: Phase estimate as produced by Pipeline C.
            resolved_parity: Parity the sign check settled on.

        Returns:
            The original estimate when the parity agreed, otherwise a copy
            recording the correction.
        """
        if resolved_parity == phase.green_acquired_parity:
            return phase
        return CfaPhaseEstimate(
            configuration_name=phase.configuration_name,
            green_acquired_parity=resolved_parity,
            block_size=phase.block_size,
            diagonal_scores=phase.diagonal_scores,
            was_estimated=phase.was_estimated,
            note=(f"{phase.note} The sign of the feature contradicted that "
                  f"parity, so the acquired lattice was taken as parity "
                  f"{resolved_parity} instead, per Eq. 13's requirement that "
                  f"mu1 > 0."),
        )

    def _fit_gaussian_mixture(self, samples: np.ndarray) -> GaussianMixtureFit:
        """Fit the two-component mixture of Eq. 13-14 by expectation-maximization.

        Args:
            samples: Feature values L from blocks with usable texture.

        Returns:
            GaussianMixtureFit. A degenerate fit (no usable samples) is returned
            with zero variances so the caller can detect it.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: M1 (CFA present, authentic): L(k,l) ~ N(mu1, sigma1^2),
        #                                       with mu1 > 0            [Eq. 13]
        #          M2 (CFA absent, tampered):   L(k,l) ~ N(0, sigma2^2) [Eq. 14]
        # Variables: mu1, sigma1 = mean and standard deviation of the authentic
        #            component; sigma2 = standard deviation of the tampered
        #            component. The tampered MEAN is fixed at 0 by assumption
        #            and is never estimated.
        # Source: Ferrara et al. 2012 - SKILL Pipeline A steps 5 and 6.
        #            Initialisation, convergence tolerance and iteration cap are
        #            the paper's exact stopping criteria as quoted in step 6:
        #            "initialize mu1, sigma1^2 to the sample mean/variance of the
        #            observed features, sigma2^2 = sigma1^2/10, mixing weight
        #            alpha=0.5; iterate to convergence, defined as increase in
        #            log-likelihood < 10^-3 or after 500 iterations".
        # Expected range: mu1 > 0 on any image with an intact CFA signature;
        #            sigma2^2 smaller than sigma1^2 at initialisation by
        #            construction.
        # ────────────────────────────────────────────────────
        if samples.size == 0:
            return GaussianMixtureFit(0.0, 0.0, 0.0, 0.0, 0, False, -np.inf)

        authentic_mean = float(np.mean(samples))
        authentic_variance = max(float(np.var(samples)),
                                 constants.EM_MINIMUM_COMPONENT_VARIANCE)
        state = (authentic_mean, authentic_variance,
                 authentic_variance * constants.EM_INITIAL_TAMPERED_VARIANCE_FRACTION,
                 constants.EM_INITIAL_MIXING_WEIGHT)
        return self._run_expectation_maximization(samples, state)

    def _run_expectation_maximization(self,
                                      samples: np.ndarray,
                                      state: tuple) -> GaussianMixtureFit:
        """Iterate EM to the SKILL's stopping criteria.

        Args:
            samples: Feature values L from blocks with usable texture.
            state: Initial (mu1, sigma1^2, sigma2^2, mixing weight).

        Returns:
            GaussianMixtureFit at the final parameters.
        """
        previous_log_likelihood = -np.inf
        converged = False
        iteration = 0

        for iteration in range(1, constants.EM_MAXIMUM_ITERATIONS + 1):
            responsibilities, log_likelihood = self._expectation_step(samples,
                                                                     state)
            if log_likelihood - previous_log_likelihood < \
                    constants.EM_LOG_LIKELIHOOD_TOLERANCE:
                converged = True
                break
            previous_log_likelihood = log_likelihood
            state = self._maximization_step(samples, responsibilities)

        _, final_log_likelihood = self._expectation_step(samples, state)
        return GaussianMixtureFit(
            authentic_mean=float(state[0]),
            authentic_variance=float(state[1]),
            tampered_variance=float(state[2]),
            mixing_weight=float(state[3]),
            iterations=iteration,
            converged=converged,
            final_log_likelihood=float(final_log_likelihood),
        )

    def _expectation_step(self, samples: np.ndarray, state: tuple) -> tuple:
        """Compute component responsibilities and the mixture log-likelihood.

        Args:
            samples: Feature values L.
            state: Current (mu1, sigma1^2, sigma2^2, mixing weight).

        Returns:
            Tuple of (responsibility of M1 per sample, total log-likelihood).
        """
        authentic_mean, authentic_variance, tampered_variance, weight = state
        log_authentic = self._log_normal_density(samples, authentic_mean,
                                                 authentic_variance)
        log_tampered = self._log_normal_density(samples,
                                                constants.MIXTURE_TAMPERED_MEAN,
                                                tampered_variance)

        # Mix in log space so an extremely improbable component cannot underflow
        # the responsibility to a NaN.
        weighted_authentic = np.log(max(weight, np.finfo(float).tiny)) + log_authentic
        weighted_tampered = (np.log(max(1.0 - weight, np.finfo(float).tiny))
                             + log_tampered)
        peak = np.maximum(weighted_authentic, weighted_tampered)
        total = peak + np.log(np.exp(weighted_authentic - peak)
                              + np.exp(weighted_tampered - peak))
        return np.exp(weighted_authentic - total), float(np.sum(total))

    @staticmethod
    def _maximization_step(samples: np.ndarray,
                           responsibilities: np.ndarray) -> tuple:
        """Re-estimate the mixture parameters, holding the tampered mean at zero.

        Args:
            samples: Feature values L.
            responsibilities: Responsibility of component M1 per sample.

        Returns:
            Updated (mu1, sigma1^2, sigma2^2, mixing weight).
        """
        tiny = np.finfo(float).tiny
        authentic_mass = max(float(np.sum(responsibilities)), tiny)
        tampered_mass = max(float(np.sum(1.0 - responsibilities)), tiny)

        authentic_mean = float(np.sum(responsibilities * samples)) / authentic_mass
        authentic_variance = max(
            float(np.sum(responsibilities * (samples - authentic_mean) ** 2))
            / authentic_mass, constants.EM_MINIMUM_COMPONENT_VARIANCE)
        # Eq. 14 fixes the tampered mean at zero, so the second moment is taken
        # about zero rather than about a fitted mean.
        tampered_variance = max(
            float(np.sum((1.0 - responsibilities)
                         * (samples - constants.MIXTURE_TAMPERED_MEAN) ** 2))
            / tampered_mass, constants.EM_MINIMUM_COMPONENT_VARIANCE)
        return (authentic_mean, authentic_variance, tampered_variance,
                authentic_mass / float(samples.size))

    @staticmethod
    def _log_normal_density(samples: np.ndarray,
                            mean: float,
                            variance: float) -> np.ndarray:
        """Log density of a univariate normal, evaluated elementwise.

        Args:
            samples: Points at which to evaluate.
            mean: Distribution mean.
            variance: Distribution variance, floored strictly above zero.

        Returns:
            Float64 array of log densities.
        """
        safe_variance = max(float(variance),
                            constants.EM_MINIMUM_COMPONENT_VARIANCE)
        deviation = samples - mean
        return (-0.5 * np.log(2.0 * np.pi * safe_variance)
                - deviation ** 2 / (2.0 * safe_variance))

    def _log_likelihood_ratio(self,
                              feature_map: np.ndarray,
                              mixture: GaussianMixtureFit) -> np.ndarray:
        """Log of the likelihood ratio of Eq. 17, per block.

        Args:
            feature_map: Per-block feature L.
            mixture: The fitted Gaussian mixture.

        Returns:
            Float64 array of log Lambda at block resolution.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: L(L(k,l)) = Pr{L(k,l)|M2} / Pr{L(k,l)|M1}          [Eq. 17]
        # Variables: the likelihood ratio of the tampered hypothesis M2 against
        #            the authentic hypothesis M1 at the observed feature value.
        # Source: Ferrara et al. 2012 - SKILL Pipeline A step 7.
        # Expected range: this function returns the NATURAL LOGARITHM of that
        #            ratio, which is unbounded in both directions. Negative
        #            means the block looks authentic, positive means tampered.
        #            The log form is used because step 8 filters "the
        #            log-likelihood map", and because Eq. 18 cumulates a PRODUCT
        #            of ratios, which is a sum in log space.
        # ────────────────────────────────────────────────────
        if mixture.authentic_variance <= 0.0 or mixture.tampered_variance <= 0.0:
            # A degenerate mixture carries no evidence either way; a zero log
            # ratio maps to a posterior of exactly one half.
            return np.zeros_like(feature_map)

        log_tampered = self._log_normal_density(
            feature_map, constants.MIXTURE_TAMPERED_MEAN,
            mixture.tampered_variance)
        log_authentic = self._log_normal_density(
            feature_map, mixture.authentic_mean, mixture.authentic_variance)
        return log_tampered - log_authentic

    def _condition_map(self,
                       log_ratio: np.ndarray,
                       validity_mask: np.ndarray) -> tuple:
        """Neutralise invalid blocks, then cumulate or filter the map.

        SKILL Pipeline A step 8 offers two alternatives: filter the
        log-likelihood map with a 5x5 median, or compute the feature on smaller
        blocks and cumulate posteriors onto CxC blocks via Eq. 18. Exactly one
        is applied, selected by enable_cumulation.

        Args:
            log_ratio: Raw per-block log-likelihood ratio.
            validity_mask: Blocks that carry usable texture.

        Returns:
            Tuple of (conditioned map, validity mask at the same resolution).
        """
        # Blocks the engine declined to measure contribute no evidence, so they
        # are set to a log ratio of zero (posterior exactly one half) before any
        # neighbourhood operation, rather than leaking an unmeasured value into
        # their neighbours.
        neutralised = np.where(validity_mask, log_ratio, 0.0)

        group_size = self._cumulation_group_size()
        if group_size > 1:
            return (self._cumulate_log_ratios(neutralised, group_size),
                    aggregate_blocks(validity_mask.astype(np.float64),
                                     group_size) > 0)
        return self._denoise(neutralised), validity_mask

    def _cumulation_group_size(self) -> int:
        """Number of feature blocks per side that are cumulated together.

        Returns:
            1 when cumulation is disabled, otherwise the ratio of the
            cumulation block size to the feature block size.
        """
        if not self.enable_cumulation:
            return 1
        return max(1, constants.CUMULATION_BLOCK_SIZE // self.feature_block_size)

    @staticmethod
    def _cumulate_log_ratios(log_ratio: np.ndarray, group_size: int) -> np.ndarray:
        """Cumulate per-block log-likelihood ratios onto larger blocks.

        Args:
            log_ratio: Per-block log-likelihood ratio map.
            group_size: Feature blocks per side in each output block.

        Returns:
            Float64 map at the coarser output resolution.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: L_cum(k',l') = PROD_{k,l} Pr{L(k,l)|M2}
        #                       / PROD_{k,l} Pr{L(k,l)|M1}            [Eq. 18]
        # Variables: the cumulated likelihood ratio over the constituent blocks
        #            (k,l) of the coarser block (k',l').
        # Source: Ferrara et al. 2012 - SKILL Pipeline A step 8, "assuming
        #            conditional independence of blocks given M1/M2".
        # Expected range: as a logarithm, unbounded in both directions; its
        #            magnitude grows with the number of blocks cumulated,
        #            because independent evidence accumulates.
        # Note: a ratio of products is a sum of log ratios, which is what is
        #            computed here. Working in log space is what keeps the
        #            product of many small densities from underflowing.
        # ────────────────────────────────────────────────────
        return aggregate_blocks(log_ratio, group_size)

    @staticmethod
    def _denoise(log_ratio: np.ndarray) -> np.ndarray:
        """Filter the log-likelihood map to suppress isolated block noise.

        SKILL Pipeline A step 8: "cumulate/filter the log-likelihood map with
        either a mean filter or a 5x5 median filter (median outperforms mean in
        the paper's experiments)".

        Args:
            log_ratio: Per-block log-likelihood ratio map.

        Returns:
            Filtered map of the same shape.

        Raises:
            ValueError: If the configured filter rule is not recognised.
        """
        if constants.MAP_FILTER_RULE == "median":
            return ndimage.median_filter(log_ratio,
                                         size=constants.MAP_FILTER_SIZE,
                                         mode="reflect")
        if constants.MAP_FILTER_RULE == "mean":
            return ndimage.uniform_filter(log_ratio,
                                          size=constants.MAP_FILTER_SIZE,
                                          mode="reflect")
        raise ValueError(f"unknown map filter rule "
                         f"{constants.MAP_FILTER_RULE!r}")

    @staticmethod
    def _tampering_probability(log_ratio: np.ndarray) -> np.ndarray:
        """Posterior probability that each block was tampered with.

        Args:
            log_ratio: Conditioned log-likelihood ratio map.

        Returns:
            Float64 map in [0, 1]; 0 = confidently authentic, 1 = confidently
            tampered.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: Pr{M1 | L(k,l)} = 1 / (1 + L(L(k,l)))          [Eq. 15-16]
        #          with equal priors Pr{M1} = Pr{M2} = 1/2
        # Variables: Pr{M1|L} = posterior probability the block is AUTHENTIC;
        #            L(L(k,l)) = the likelihood ratio of Eq. 17.
        # Source: Ferrara et al. 2012 - SKILL Pipeline A step 7.
        # Expected range: [0, 1]. SKILL "Output" defines the fusion-layer
        #            tampering score as 1 - Pr{M1|L(k,l)}, "i.e. 0 = confidently
        #            authentic, 1 = confidently tampered", which is what this
        #            function returns.
        # Note: substituting Lambda = exp(log Lambda) into 1 - 1/(1 + Lambda)
        #            gives Lambda/(1 + Lambda), the standard logistic function of
        #            the log ratio. Evaluating it that way is algebraically
        #            identical and numerically stable at extreme ratios.
        #            The priors are deliberately the fixed 1/2 of Eq. 15-16, NOT
        #            the mixing weight EM converged to.
        # ────────────────────────────────────────────────────
        bounded = np.clip(log_ratio, -constants.LOG_LIKELIHOOD_RATIO_LIMIT,
                          constants.LOG_LIKELIHOOD_RATIO_LIMIT)
        prior_ratio = (constants.POSTERIOR_PRIOR_TAMPERED
                       / constants.POSTERIOR_PRIOR_AUTHENTIC)
        odds = np.exp(bounded) * prior_ratio
        return odds / (1.0 + odds)


# ---------------------------------------------------------------------------
# Pipeline B - a contrario grid-position detection (Bammey et al. 2018)
# ---------------------------------------------------------------------------


class GridConsistencyComputer:
    """Tests whether one CFA grid phase explains the whole image, with an NFA."""

    def __init__(self, block_size: Optional[int] = None) -> None:
        """Configure the voting block geometry.

        Args:
            block_size: Edge length b of a voting block. Defaults to the
                corpus's small-forgery configuration.

        Raises:
            ValueError: If the block size is odd, which SKILL step 2 forbids.
        """
        self.block_size = int(block_size or constants.GRID_VOTE_BLOCK_SIZE)
        if self.block_size % constants.BAYER_PERIOD != 0:
            raise ValueError(f"grid vote block size {self.block_size} must be "
                             f"even so it contains whole Bayer cells")

    def analyse(self, colour_image: np.ndarray) -> GridConsistencyResult:
        """Vote each block for a CFA grid position and test the result's meaning.

        Args:
            colour_image: Float64 BGR image, cropped to a whole Bayer grid.

        Returns:
            GridConsistencyResult naming the dominant position, its NFA, and
            any windows that disagree with it.
        """
        residuals = np.stack(
            [self._position_residual(colour_image, position)
             for position in range(constants.BAYER_CELL_POSITION_COUNT)])
        vote_map = np.argmin(residuals, axis=0)

        dominant, dominant_log10_nfa = self._global_dominant_position(vote_map)
        if dominant is None:
            return GridConsistencyResult(
                None, dominant_log10_nfa, vote_map, [], 0, False,
                "No CFA grid position was statistically meaningful across the "
                "whole image at the a contrario false-alarm budget of "
                f"{constants.NFA_MEANINGFUL_BUDGET}, so this confirmatory "
                "layer neither supports nor contradicts the primary pipeline.")

        forged_windows, window_count = self._flag_windows(vote_map, dominant)
        return GridConsistencyResult(
            dominant_position_index=dominant,
            dominant_log10_nfa=dominant_log10_nfa,
            vote_map=vote_map,
            forged_windows=forged_windows,
            window_count=window_count,
            is_conclusive=True,
            note=(f"Grid position {dominant} dominates the image with "
                  f"log10 NFA = {dominant_log10_nfa:.1f}. "
                  f"{len(forged_windows)} of {window_count} windows contain a "
                  f"significant disagreeing position at the corpus detection "
                  f"threshold of 10^"
                  f"{constants.NFA_DETECTION_LOG10_THRESHOLD:.0f}."),
        )

    def _position_residual(self,
                           colour_image: np.ndarray,
                           position_index: int) -> np.ndarray:
        """Total reconstruction residual per block under one candidate grid.

        Args:
            colour_image: Float64 BGR image.
            position_index: Candidate CFA grid position, 0 to 3.

        Returns:
            Float64 array at voting-block resolution.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: A[u+Nv, s+Nt] = SUM_{x,y} G[x,y,0] * M[x+u,y+v] * M[x+s,y+t]
        #          b[u+Nv]       = SUM_{x,y} G[x,y,0] * M[x+u,y+v] * I[x,y,1]
        #          solve A * alpha = b for each of the 8 filters
        # Variables: M = the mosaiced image under the candidate grid; G = the
        #            CFA-grid indicator selecting the sampled positions of one
        #            sensel role; I = the full observed image; alpha = the
        #            estimated linear demosaicing filter; N = filter support
        #            width; (u,v), (s,t) = support offsets.
        # Source: Bammey, Morel & Grompone von Gioi 2018 (IEEE MIPR) - SKILL
        #            Pipeline B step 1. Eight filters are estimated, one per
        #            missing-channel/sampled-colour pairing, "using all observed
        #            channels (not just single-channel EM) to capture
        #            inter-channel correlation".
        # Expected range: non-negative residuals. The residual is SMALLEST for
        #            the grid position the camera actually used, which is what
        #            makes the per-block argmin of step 2 a vote.
        # ────────────────────────────────────────────────────
        mosaic, role_masks = self._build_mosaic(colour_image, position_index)
        support = self._stack_support(mosaic)

        residual = np.zeros(mosaic.shape, dtype=np.float64)
        for role_name, target_letter in constants.GRID_FILTER_PAIRINGS:
            mask = role_masks[role_name]
            target = colour_image[..., self._channel_index(target_letter)]
            residual += self._pairing_residual(support, mask, target)

        return partition_into_blocks(residual, self.block_size).sum(axis=(2, 3))

    def _pairing_residual(self,
                          support: np.ndarray,
                          mask: np.ndarray,
                          target: np.ndarray) -> np.ndarray:
        """Squared residual of one estimated filter, scattered back to pixels.

        Args:
            support: Stacked shifted copies of the mosaic, shape (H, W, N*N).
            mask: Boolean mask of the sensel positions this filter predicts at.
            target: The channel plane the filter must reconstruct.

        Returns:
            Float64 array of squared residuals, zero away from the mask.
        """
        observations = support[mask]
        targets = target[mask]
        if observations.shape[0] <= observations.shape[1]:
            # Fewer equations than unknowns: the filter is not identifiable, so
            # this pairing contributes no evidence rather than a fabricated one.
            return np.zeros(target.shape, dtype=np.float64)

        coefficients = self._solve_normal_equations(observations, targets)
        errors = targets - observations @ coefficients
        residual = np.zeros(target.shape, dtype=np.float64)
        residual[mask] = errors ** 2
        return residual

    @staticmethod
    def _solve_normal_equations(observations: np.ndarray,
                                targets: np.ndarray) -> np.ndarray:
        """Least-squares solve of A*alpha = b with a relative ridge term.

        Args:
            observations: Design matrix of stacked mosaic support values.
            targets: Observed values of the channel being reconstructed.

        Returns:
            Coefficient vector alpha.
        """
        normal_matrix = observations.T @ observations
        right_hand_side = observations.T @ targets
        # A flat or highly correlated region makes the normal matrix singular.
        # The ridge is scaled by the matrix's own trace so it is a relative
        # nudge that cannot bias a well-conditioned solve.
        ridge = (constants.GRID_FILTER_RIDGE_FRACTION
                 * float(np.trace(normal_matrix)) / normal_matrix.shape[0])
        regularised = normal_matrix + ridge * np.eye(normal_matrix.shape[0])
        try:
            return np.linalg.solve(regularised, right_hand_side)
        except np.linalg.LinAlgError:
            # Still singular: fall back to the pseudo-inverse solution, which is
            # defined for any matrix.
            return np.linalg.lstsq(observations, targets, rcond=None)[0]

    @staticmethod
    def _stack_support(mosaic: np.ndarray) -> np.ndarray:
        """Stack every shifted copy of the mosaic within the filter support.

        Args:
            mosaic: Two-dimensional mosaiced image.

        Returns:
            Float64 array of shape (H, W, N*N), where plane k holds the mosaic
            shifted by the k-th support offset.
        """
        support_size = constants.GRID_FILTER_SUPPORT_SIZE
        half = support_size // 2
        padded = np.pad(mosaic, half, mode="reflect")
        planes = [padded[row:row + mosaic.shape[0], column:column + mosaic.shape[1]]
                  for row in range(support_size)
                  for column in range(support_size)]
        return np.stack(planes, axis=-1)

    def _build_mosaic(self, colour_image: np.ndarray, position_index: int) -> tuple:
        """Re-mosaic the image under one candidate CFA grid position.

        Args:
            colour_image: Float64 BGR image.
            position_index: Candidate grid position, 0 to 3.

        Returns:
            Tuple of (mosaic array, dict of sensel-role boolean masks).
        """
        red_position = divmod(position_index, constants.BAYER_PERIOD)
        role_masks = self._build_role_masks(colour_image.shape[:2], red_position)

        mosaic = np.zeros(colour_image.shape[:2], dtype=np.float64)
        for role_name, channel_letter in (("R", "r"), ("B", "b"),
                                          ("GR", "g"), ("GB", "g")):
            mask = role_masks[role_name]
            mosaic[mask] = colour_image[..., self._channel_index(channel_letter)][mask]
        return mosaic, role_masks

    @staticmethod
    def _build_role_masks(shape: tuple, red_position: tuple) -> dict:
        """Mark which sensel role each pixel carries under a candidate grid.

        The four roles are the four positions of the Bayer cell: R, B, and the
        two greens, distinguished as GR (green sharing a row with red) and GB
        (green sharing a row with blue), exactly as SKILL Pipeline B step 1
        names them.

        Args:
            shape: (rows, columns) of the image.
            red_position: (row, column) of red inside the 2x2 cell.

        Returns:
            Dict mapping each role name to a boolean mask.
        """
        rows, columns = np.indices(shape)
        row_phase = rows % constants.BAYER_PERIOD
        column_phase = columns % constants.BAYER_PERIOD

        is_red_row = row_phase == red_position[0]
        is_red_column = column_phase == red_position[1]
        return {
            "R": is_red_row & is_red_column,
            "B": (~is_red_row) & (~is_red_column),
            "GR": is_red_row & (~is_red_column),
            "GB": (~is_red_row) & is_red_column,
        }

    @staticmethod
    def _channel_index(channel_letter: str) -> int:
        """Map a channel letter from the SKILL's filter names to a BGR index.

        Args:
            channel_letter: One of "r", "g", "b".

        Returns:
            Index into the BGR channel axis.

        Raises:
            ValueError: If the letter names no channel.
        """
        indices = {"b": constants.BLUE_CHANNEL_INDEX,
                   "g": constants.ANALYSIS_CHANNEL_INDEX,
                   "r": constants.RED_CHANNEL_INDEX}
        if channel_letter not in indices:
            raise ValueError(f"unknown channel letter {channel_letter!r}")
        return indices[channel_letter]

    def _global_dominant_position(self, vote_map: np.ndarray) -> tuple:
        """Find the image-wide grid position and test whether it is meaningful.

        Args:
            vote_map: Per-block winning grid position index.

        Returns:
            Tuple of (position index or None, log10 NFA of the global vote).
        """
        counts = np.bincount(vote_map.ravel(),
                             minlength=constants.BAYER_CELL_POSITION_COUNT)
        leader = int(np.argmax(counts))
        # z = 1: the whole image is a single window for this global test.
        log10_nfa = self._log10_nfa(int(counts[leader]), int(vote_map.size), 1)
        if log10_nfa > constants.NFA_MEANINGFUL_LOG10_BUDGET:
            return None, log10_nfa
        return leader, log10_nfa

    @staticmethod
    def _log10_nfa(position_votes: int, block_count: int, window_count: int) -> float:
        """Number of False Alarms for a vote count, as a base-10 logarithm.

        Args:
            position_votes: n_P, blocks voting for the position under test.
            block_count: n, total blocks in the window.
            window_count: z, the number of disjoint windows tested.

        Returns:
            log10 of the NFA. Smaller means more significant.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: NFA(n_P, n) = 4z * SUM_{i=n_P}^{n} C(n,i)
        #                            * (1/4)^i * (3/4)^(n-i)
        # Variables: n_P = blocks voting for position P; n = blocks in the
        #            window; z = number of disjoint windows tested; the null
        #            hypothesis is that each block votes for one of the 4
        #            configurations independently with probability 1/4.
        # Source: Bammey, Morel & Grompone von Gioi 2018 - SKILL Pipeline B
        #            step 3, "exact formula from paper". The leading factor 4z
        #            is the number of tests performed (4 positions x z windows).
        # Expected range: an expected false-alarm COUNT, not a probability, so
        #            it is bounded above by 4z and unbounded below. The corpus
        #            reports values as small as 10^-300 at detected forgeries.
        # Note: computed and returned in LOG space throughout, because the SKILL
        #            warns "NFA values as low as 10^-300 require log-space
        #            computation; plain floating point will underflow". Verified
        #            here: scipy.stats.binom.sf returns exactly 0.0 and
        #            binom.logsf returns -inf for n=1024 with every block
        #            agreeing, whereas the log-sum-exp form used by
        #            utils.log_binomial_tail returns the exact -615.9.
        # ────────────────────────────────────────────────────
        log_tail = log_binomial_tail(position_votes, block_count,
                                     constants.GRID_VOTE_NULL_PROBABILITY)
        log_tests = np.log(float(constants.BAYER_CELL_POSITION_COUNT
                                 * max(window_count, 1)))
        return natural_log_to_log10(log_tail + log_tests)

    def _flag_windows(self, vote_map: np.ndarray, dominant: int) -> tuple:
        """Flag windows holding a significant position other than the dominant.

        SKILL Pipeline B step 4: "A window is flagged forged if it contains at
        least one significant position that is not P0".

        Args:
            vote_map: Per-block winning grid position index.
            dominant: The globally dominant position P0.

        Returns:
            Tuple of (list of flagged window records, number of windows tested).
        """
        side = constants.NFA_WINDOW_SIDE_IN_BLOCKS
        window_rows = vote_map.shape[0] // side
        window_columns = vote_map.shape[1] // side
        if window_rows == 0 or window_columns == 0:
            return [], 0

        window_count = window_rows * window_columns
        flagged = []
        for window_row in range(window_rows):
            for window_column in range(window_columns):
                record = self._test_window(vote_map, window_row, window_column,
                                           dominant, window_count)
                if record is not None:
                    flagged.append(record)
        return flagged, window_count

    def _test_window(self,
                     vote_map: np.ndarray,
                     window_row: int,
                     window_column: int,
                     dominant: int,
                     window_count: int) -> Optional[dict]:
        """Test one window for a significant non-dominant grid position.

        Args:
            vote_map: Per-block winning grid position index.
            window_row: Window index down the image.
            window_column: Window index across the image.
            dominant: The globally dominant position P0.
            window_count: z, the number of windows tested.

        Returns:
            Record describing the disagreement, or None when the window agrees.
        """
        side = constants.NFA_WINDOW_SIDE_IN_BLOCKS
        window = vote_map[window_row * side:(window_row + 1) * side,
                          window_column * side:(window_column + 1) * side]
        counts = np.bincount(window.ravel(),
                             minlength=constants.BAYER_CELL_POSITION_COUNT)

        for position in range(constants.BAYER_CELL_POSITION_COUNT):
            if position == dominant:
                continue
            log10_nfa = self._log10_nfa(int(counts[position]), int(window.size),
                                        window_count)
            if log10_nfa <= constants.NFA_DETECTION_LOG10_THRESHOLD:
                return {"window_row": window_row,
                        "window_column": window_column,
                        "grid_position": position,
                        "log10_nfa": log10_nfa,
                        "block_size": self.block_size,
                        "window_side_in_blocks": side}
        return None
