"""Core mathematical computation for all three wavelet-domain pipelines.

Pipeline C (CopyMoveDetector) is the sole score-driving computation in this
engine - see the SCOPE DECISION note at the top of constants.py. Pipelines A
(NoiseResidualExtractor) and B (CompressionHistoryEstimator) are computed in
full and returned as auxiliary, non-scoring evidence.
"""

from __future__ import annotations

import logging

import numpy as np
import pywt
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

from . import constants
from .contracts import (Block, CompressionHistoryResult, CopyMoveResult,
                        DuplicatePair, NoiseResidualResult)
from .utils import (binomial_coefficient, generate_neighbour_offsets,
                    pixel_coordinate_grids, robust_sigma_from_mad,
                    solve_weighted_log_linear_fit)

logger = logging.getLogger(__name__)


class NoiseResidualExtractor:
    """Pipeline A: wavelet noise-residual extraction (auxiliary, non-scoring)."""

    def decompose(self, grayscale: np.ndarray) -> list:
        """Run the stationary wavelet transform to the configured level.

        Args:
            grayscale: Float64 grayscale image.

        Returns:
            SWT coefficient list, coarsest level first (pywt.swt2 order).
        """
        return pywt.swt2(grayscale, constants.PIPELINE_A_WAVELET_FAMILY,
                         level=constants.PIPELINE_A_DECOMPOSITION_LEVELS)

    def estimate_noise_sigma(self, diagonal_subband: np.ndarray) -> float:
        """Robust MAD-based noise standard-deviation estimate.

        Args:
            diagonal_subband: The HH detail subband at one decomposition level.

        Returns:
            Estimated noise sigma.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: sigma_W_Phi = median(|HH_i|) / 0.6745,  i = 1,...,k
        # Variables: HH_i=diagonal detail subband at level i, 0.6745=75th
        #   percentile point of the standard half-normal distribution.
        # Source: mgaga2019, Eq. 7.
        # Expected range: non-negative, on the scale of the input coefficients.
        # ──────────────────────────────────────────────────────────
        return robust_sigma_from_mad(diagonal_subband, constants.NOISE_MAD_CONSTANT)

    def clean_signal_sigma(self,
                           observed_subband: np.ndarray,
                           noise_sigma: float) -> float:
        """Separate signal variance from noise variance in a subband.

        Args:
            observed_subband: A detail subband of the (noisy) image under test.
            noise_sigma: Estimate of the noise component's sigma (Eq. 7).

        Returns:
            Estimated clean-signal sigma, floored at 0.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: sigma^2_WG(D,i) = (1/N(i)^2) * sum_x sum_y W_G(D,i)^2
        #   sigma^2_WG = sigma^2_WF + sigma^2_WPhi
        #   sigma_WF(D,i) = sqrt(max(sigma^2_WG(D,i) - sigma^2_WPhi(i), 0))
        # Variables: W_G=observed noisy subband, W_F=clean-signal component,
        #   W_Phi=noise component, N(i)=subband side length at level i.
        # Source: mgaga2019, Eq. 8-10.
        # Expected range: non-negative.
        # ──────────────────────────────────────────────────────────
        observed_variance = float(np.mean(np.square(observed_subband)))
        noise_variance = noise_sigma ** 2
        return float(np.sqrt(max(observed_variance - noise_variance, 0.0)))

    def select_threshold(self,
                         method: str,
                         noise_sigma: float,
                         signal_sigma: float,
                         num_coefficients: int,
                         weighted_median_subband: np.ndarray = None) -> float:
        """Compute the scalar threshold T for the visushrink/bayesshrink/
        golden_ratio methods.

        Args:
            method: One of THRESHOLD_METHODS (excluding "piecewise").
            noise_sigma: MAD-based noise sigma (Eq. 7).
            signal_sigma: Clean-signal sigma (Eq. 10), used by BayesShrink.
            num_coefficients: N, count of high-frequency coefficients.
            weighted_median_subband: HH subband, required for golden_ratio.

        Returns:
            Scalar threshold T.
        """
        if method == "visushrink":
            return self._visushrink_threshold(noise_sigma, num_coefficients)
        if method == "bayesshrink":
            return self._bayesshrink_threshold(noise_sigma, signal_sigma)
        if method == "golden_ratio":
            return self._golden_ratio_threshold(weighted_median_subband, num_coefficients)
        raise ValueError(f"unknown threshold method: {method}")

    def _visushrink_threshold(self, noise_sigma: float, num_coefficients: int) -> float:
        """VisuShrink universal threshold. Args/Returns: see select_threshold."""
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: T = sigma * sqrt(2 * log(N))
        # Variables: sigma=noise sigma, N=high-frequency coeff count.
        # Source: mgaga2019, Eq. 3 (citing Donoho 1995).
        # Expected range: non-negative, scales with sigma.
        # ──────────────────────────────────────────────────────────
        return noise_sigma * np.sqrt(
            constants.VISUSHRINK_LOG_COEFFICIENT * np.log(num_coefficients))

    def _bayesshrink_threshold(self, noise_sigma: float, signal_sigma: float) -> float:
        """BayesShrink threshold. Args/Returns: see select_threshold."""
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: T = sigma^2_noise / sigma_signal
        # Source: mgaga2019, Eq. 4.
        # Expected range: non-negative; undefined if signal_sigma is 0.
        # ──────────────────────────────────────────────────────────
        if signal_sigma <= 0.0:
            return noise_sigma ** 2
        return (noise_sigma ** 2) / signal_sigma

    def _golden_ratio_threshold(self, weighted_median_subband: np.ndarray,
                                num_coefficients: int) -> float:
        """Golden-ratio-modified universal threshold. See select_threshold."""
        weighted_sigma = self._weighted_median_sigma(weighted_median_subband)
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: T = sigma * sqrt(1.618 * log(N))
        # Source: mgaga2019, Eq. 13.
        # Expected range: non-negative.
        # ──────────────────────────────────────────────────────────
        return weighted_sigma * np.sqrt(
            constants.GOLDEN_RATIO_LOG_COEFFICIENT * np.log(num_coefficients))

    def _weighted_median_sigma(self, diagonal_subband: np.ndarray) -> float:
        """Weighted-median MAD sigma estimate for the golden-ratio method.

        Args:
            diagonal_subband: HH detail subband.

        Returns:
            Weighted-median-based sigma estimate.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: W(x,y) = 1 / exp(HH(x,y)), weighted median of |HH|
        # Source: mgaga2019, Eq. 14-16 (Sasirekha et al.).
        # Expected range: non-negative.
        # ──────────────────────────────────────────────────────────
        magnitudes = np.abs(diagonal_subband).ravel()
        weights = 1.0 / np.exp(diagonal_subband).ravel()
        order = np.argsort(magnitudes)
        sorted_magnitudes = magnitudes[order]
        cumulative_weight = np.cumsum(weights[order])
        halfway = 0.5 * cumulative_weight[-1]
        median_index = int(np.searchsorted(cumulative_weight, halfway))
        median_index = min(median_index, sorted_magnitudes.size - 1)
        return float(sorted_magnitudes[median_index]) / constants.NOISE_MAD_CONSTANT

    def compute_piecewise_lambdas(self,
                                  sigma: float,
                                  num_coefficients: int) -> tuple:
        """Compute lambda1/lambda2 for Zhu & Wang's piecewise shrinkage.

        Args:
            sigma: Noise sigma estimate.
            num_coefficients: N, count of high-frequency coefficients.

        Returns:
            Tuple of (lambda1, lambda2).
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: lambda1 = sigma*sqrt(2ln(N)+2ln(ln(N)))
        #          lambda2 = alpha*sigma*sqrt(2ln(N)+2ln(ln(N)))
        # Variables: alpha=0.5 (paper's own experiment), N=coefficient count.
        # Source: Zhu & Wang 2021, Eq. 3 (lambda1/lambda2 definitions).
        # Expected range: lambda2 <= lambda1, both non-negative.
        # ──────────────────────────────────────────────────────────
        base = sigma * np.sqrt(2.0 * np.log(num_coefficients)
                               + 2.0 * np.log(np.log(num_coefficients)))
        return float(base), float(constants.ZHU_WANG_ALPHA * base)

    def apply_threshold(self,
                        coefficients: np.ndarray,
                        threshold: float,
                        mode: str) -> np.ndarray:
        """Apply hard or soft thresholding to wavelet coefficients.

        Args:
            coefficients: Detail-subband coefficients.
            threshold: Scalar threshold T.
            mode: "hard" or "soft".

        Returns:
            Thresholded coefficients, same shape as input.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: hard f(x)={x,|x|>=T; 0,|x|<T}
        #          soft f(x)={x-T,x>T; 0,|x|<=T; x+T,x<-T}
        # Source: mgaga2019 Eq. 1-2 / Zhu & Wang Eq. 1-2.
        # Expected range: |f(x)| <= |x|.
        # ──────────────────────────────────────────────────────────
        if mode == "hard":
            return np.where(np.abs(coefficients) >= threshold, coefficients, 0.0)
        if mode == "soft":
            return np.sign(coefficients) * np.maximum(
                np.abs(coefficients) - threshold, 0.0)
        raise ValueError(f"unknown threshold mode: {mode}")

    def apply_piecewise_threshold(self,
                                  coefficients: np.ndarray,
                                  lambda1: float,
                                  lambda2: float) -> np.ndarray:
        """Apply Zhu & Wang's smooth piecewise shrinkage function.

        Args:
            coefficients: Detail-subband coefficients.
            lambda1: Upper threshold, above which coefficients pass through.
            lambda2: Lower threshold, below which coefficients are zeroed.

        Returns:
            Thresholded coefficients, same shape as input.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: w_hat = w, |w|>=lambda1
        #                = w*(1-|lambda2/w|^n), lambda2<=|w|<lambda1
        #                = 0, |w|<lambda2
        # Variables: n=2 (paper's own experiment).
        # Source: Zhu & Wang 2021, Eq. 3.
        # Expected range: |w_hat| <= |w|, continuous at both boundaries.
        # ──────────────────────────────────────────────────────────
        magnitude = np.abs(coefficients)
        safe_magnitude = np.where(magnitude > 0.0, magnitude, 1.0)
        middle_region = coefficients * (
            1.0 - np.abs(lambda2 / safe_magnitude) ** constants.ZHU_WANG_N)
        result = np.where(magnitude >= lambda1, coefficients, middle_region)
        return np.where(magnitude < lambda2, 0.0, result)

    def compute(self,
               grayscale: np.ndarray,
               method: str = constants.DEFAULT_THRESHOLD_METHOD,
               mode: str = constants.DEFAULT_THRESHOLD_MODE) -> NoiseResidualResult:
        """Run the full noise-residual extraction pipeline.

        Args:
            grayscale: Float64 grayscale image.
            method: Threshold-selection method.
            mode: Thresholding mode ("hard"/"soft", ignored for "piecewise").

        Returns:
            NoiseResidualResult with the residual map and diagnostics.
        """
        levels = self.decompose(grayscale)
        _, (cH, cV, cD) = levels[-1]  # finest level, i = 1
        noise_sigma = self.estimate_noise_sigma(cD)
        signal_sigma = self.clean_signal_sigma(cD, noise_sigma)
        num_coefficients = cD.size

        if method == "piecewise":
            lambda1, lambda2 = self.compute_piecewise_lambdas(
                noise_sigma, num_coefficients)
            thresholded = [self.apply_piecewise_threshold(band, lambda1, lambda2)
                          for band in (cH, cV, cD)]
            threshold_used = lambda1
        else:
            threshold_used = self.select_threshold(
                method, noise_sigma, signal_sigma, num_coefficients, cD)
            thresholded = [self.apply_threshold(band, threshold_used, mode)
                          for band in (cH, cV, cD)]

        residual = self.extract_residual((cH, cV, cD), thresholded)
        return NoiseResidualResult(residual=residual, sigma_estimate=noise_sigma,
                                   threshold_used=float(threshold_used),
                                   threshold_method=method, threshold_mode=mode)

    def extract_residual(self, original_bands: tuple, thresholded_bands: list) -> np.ndarray:
        """Combine per-direction residuals into one spatial residual map.

        Args:
            original_bands: (cH, cV, cD) before thresholding.
            thresholded_bands: Matching bands after thresholding.

        Returns:
            Single-channel residual magnitude map, same shape as the bands.
        """
        # SKILL: "denoising residual - original coefficients minus
        # thresholded coefficients - is the forensic signal passed to the
        # noise analysis module". Combined here across the three directions
        # via Euclidean magnitude for a single-map presentation. [ENGINEERING]
        residuals = [orig - thresh for orig, thresh
                    in zip(original_bands, thresholded_bands)]
        return np.sqrt(sum(r ** 2 for r in residuals))


class CompressionHistoryEstimator:
    """Pipeline B: wavelet-compression-history detection (auxiliary, low-trust)."""

    def decompose(self, grayscale: np.ndarray) -> np.ndarray:
        """Run a decimated N-level DWT and return the finest diagonal subband.

        SKILL fits the Laplacian model "for each subband"; this engine fits
        one representative subband (finest-level HH, the standard choice for
        quantization-artifact visibility) rather than exhaustively fitting
        every subband at every level. [ENGINEERING] scope simplification -
        the fitting formula itself (Eq. 4-7) is unchanged.

        Args:
            grayscale: Float64 grayscale image.

        Returns:
            Finest-level diagonal (HH) detail subband.
        """
        coeffs = pywt.wavedec2(grayscale, constants.PIPELINE_B_WAVELET_FAMILY,
                               level=constants.PIPELINE_B_DECOMPOSITION_LEVELS)
        _, _, finest_diagonal = coeffs[-1]
        return finest_diagonal

    def build_histogram(self, coefficients: np.ndarray) -> tuple:
        """Bin coefficients into an empirical histogram h_k at bin centers q_k.

        Args:
            coefficients: Detail-subband coefficients.

        Returns:
            Tuple of (bin_centers, counts) - q_k and h_k respectively.
        """
        counts, edges = np.histogram(
            coefficients.ravel(), bins=constants.PIPELINE_B_HISTOGRAM_BIN_COUNT)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, counts.astype(np.float64)

    def fit_laplacian(self, bin_centers: np.ndarray, histogram: np.ndarray) -> tuple:
        """Weighted least-squares fit of h_k to c * exp(-lambda_hat * |q_k|).

        Args:
            bin_centers: q_k values.
            histogram: h_k observed (or bias-corrected) counts.

        Returns:
            Tuple of (log_c, lambda_hat).
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: h_k = c * exp(-lambda_hat*|q_k|)                  [Eq.4]
        #   min_{lambda_hat,c} sum_k h_k*(log(h_k)-log(c)+lambda_hat*|q_k|)^2
        #                                                            [Eq.5]
        # Variables: h_k=histogram count at q_k, c=amplitude, lambda_hat=
        #   estimated pre-compression Laplacian parameter.
        # Source: Stamm & Liu 2010, Eq. 4-5.
        # Expected range: lambda_hat > 0.
        # DERIVED (Eq. 6, not printed in SKILL): weighted least squares
        # normal equations on y_k=log(h_k), weights w_k=h_k - see
        # constants.LAPLACIAN_FIT_IS_WEIGHTED_LOG_LINEAR.
        # ──────────────────────────────────────────────────────────
        valid = histogram > 0.0
        if np.count_nonzero(valid) < 2:
            return 0.0, 0.0
        log_c, negative_lambda = solve_weighted_log_linear_fit(
            np.abs(bin_centers[valid]), np.log(histogram[valid]),
            histogram[valid])
        return log_c, -negative_lambda

    def correct_bitplane_bias(self, bin_centers: np.ndarray,
                              histogram: np.ndarray) -> CompressionHistoryResult:
        """Iteratively refit lambda_hat, correcting for bitplane-truncation bias.

        Args:
            bin_centers: q_k values.
            histogram: Observed h_k counts.

        Returns:
            CompressionHistoryResult with the converged fit.
        """
        zero_index = int(np.argmin(np.abs(bin_centers)))
        neighbour_indices = [i for i in (zero_index - 1, zero_index + 1)
                             if 0 <= i < bin_centers.size]

        working = histogram.copy()
        log_c, lambda_hat = self.fit_laplacian(bin_centers, working)
        converged = False
        iterations_run = 0
        for iterations_run in range(1, constants.BIAS_CORRECTION_MAX_ITERATIONS + 1):
            working = self._apply_bias_correction(
                histogram, working, zero_index, neighbour_indices, log_c)
            new_log_c, new_lambda = self.fit_laplacian(bin_centers, working)
            denominator = abs(new_lambda) if new_lambda != 0.0 else 1.0
            relative_change = abs(new_lambda - lambda_hat) / denominator
            log_c, lambda_hat = new_log_c, new_lambda
            if relative_change < constants.BIAS_CORRECTION_CONVERGENCE_TOLERANCE:
                converged = True
                break

        residual = self._fit_residual(bin_centers, working, log_c, lambda_hat)
        return CompressionHistoryResult(lambda_hat=lambda_hat, log_c_hat=log_c,
                                        iterations_run=iterations_run,
                                        converged=converged, fit_residual=residual)

    def _apply_bias_correction(self, original_histogram, working_histogram,
                               zero_index, neighbour_indices, log_c) -> np.ndarray:
        """One iteration of Eq. 7's zero-bin/neighbour-bin correction.

        Args:
            original_histogram: Untouched observed h_k.
            working_histogram: Histogram from the previous iteration.
            zero_index: Bin index closest to q_k=0.
            neighbour_indices: Bin indices adjacent to zero_index.
            log_c: log(c) from the previous iteration's fit.

        Returns:
            Corrected histogram for the next fit.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: h_hat_k^(i) = c^(i), k=0
        #                      = h_k + 0.5*(h_0 - c^(i)), k=+-1
        #                      = h_k, otherwise
        # Source: Stamm & Liu 2010, Eq. 7.
        # Expected range: non-negative counts.
        # ──────────────────────────────────────────────────────────
        # Numerical safety clip only (not a SKILL value): an ill-conditioned
        # fit on near-random data can drive log_c beyond float64's exp range.
        c_current = np.exp(np.clip(log_c, -constants.LOG_C_EXPONENT_CLIP,
                                   constants.LOG_C_EXPONENT_CLIP))
        corrected = original_histogram.copy()
        corrected[zero_index] = c_current
        for index in neighbour_indices:
            corrected[index] = original_histogram[index] + 0.5 * (
                original_histogram[zero_index] - c_current)
        return corrected

    def _fit_residual(self, bin_centers, histogram, log_c, lambda_hat) -> float:
        """Weighted sum-of-squares residual of the final Laplacian fit.

        Args:
            bin_centers: q_k values.
            histogram: Final (bias-corrected) h_k.
            log_c: Fitted log(c).
            lambda_hat: Fitted lambda_hat.

        Returns:
            Non-negative weighted residual, per Eq. 5's objective.
        """
        valid = histogram > 0.0
        if not np.any(valid):
            return 0.0
        predicted = log_c - lambda_hat * np.abs(bin_centers[valid])
        errors = np.log(histogram[valid]) - predicted
        return float(np.sum(histogram[valid] * errors ** 2))

    def compute(self, grayscale: np.ndarray) -> CompressionHistoryResult:
        """Run the full compression-history estimation pipeline.

        Args:
            grayscale: Float64 grayscale image.

        Returns:
            CompressionHistoryResult, always reported as auxiliary/low-trust.
        """
        subband = self.decompose(grayscale)
        bin_centers, histogram = self.build_histogram(subband)
        return self.correct_bitplane_bias(bin_centers, histogram)


class CopyMoveDetector:
    """Pipeline C: Haar-DWT blur-invariant copy-move block matching.

    The sole score-driving pipeline in this engine - see constants.py's
    SCOPE DECISION note.
    """

    def compute_moments(self, block_pixels: np.ndarray, max_order: int) -> dict:
        """Compute all central moments mu_pq with p+q <= max_order.

        cv2.moments only reaches order 3, insufficient for this module's
        7th-order blur invariants, so moments are computed manually here per
        Eq. 6-8's direct double-sum definitions. Vectorised over the block's
        pixel grid rather than looped, for speed across many blocks.

        Args:
            block_pixels: R x R float64 block.
            max_order: Highest p+q to compute.

        Returns:
            Dict mapping (p, q) -> central moment value, for all 0<=p+q<=max_order.
        """
        x_grid, y_grid = pixel_coordinate_grids(block_pixels.shape[0])
        raw_00 = float(np.sum(block_pixels))
        raw_10 = float(np.sum(x_grid * block_pixels))
        raw_01 = float(np.sum(y_grid * block_pixels))
        centroid_x = raw_10 / raw_00 if raw_00 != 0.0 else 0.0
        centroid_y = raw_01 / raw_00 if raw_00 != 0.0 else 0.0

        centred_x = x_grid - centroid_x
        centred_y = y_grid - centroid_y
        moments = {}
        for p in range(max_order + 1):
            for q in range(max_order + 1 - p):
                # ── SKILL VERIFICATION ──────────────────────────
                # Formula: mu_pq = sum_x sum_y (x-x_t)^p (y-y_t)^q f(x,y)
                #   (x_t,y_t) = (m10/m00, m01/m00)
                # Source: Kashyap & Joshi 2013, Eq. 6-8.
                # Expected range: real-valued, mu_00 = raw m00 (total mass).
                # ──────────────────────────────────────────────
                moments[(p, q)] = float(np.sum(
                    (centred_x ** p) * (centred_y ** q) * block_pixels))
        return moments

    def compute_blur_invariant(self, moments: dict, p: int, q: int,
                               mu00: float, cache: dict) -> float:
        """Recursively compute the blur-invariant feature B(p, q).

        Args:
            moments: Central moments dict from compute_moments.
            p: First order index.
            q: Second order index.
            mu00: Zeroth central moment (total block mass).
            cache: Memoisation dict, shared across one block's computation.

        Returns:
            B(p, q), the blur-invariant value.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: B(p,q) = mu_pq - alpha*mu_pq*(1/mu00)*
        #   sum_{n=0}^{k} sum_{i=m1}^{m2} C(t-2i,2i)*C(q,2i)*
        #   B(p-t+2i,q-2i)*mu_{t-2i,2i}
        #   k=floor((p+q-4)/2), t=2(k-n+1), m1=max(0,floor((t-p+1)/2)),
        #   m2=min(t//2, q//2), alpha=1 if p,q both even else 0.
        # Source: Kashyap & Joshi 2013, Eq. 12-15 (subscript interpreted per
        #   constants.BLUR_INVARIANT_SUBSCRIPT_INTERPRETATION).
        # Expected range: real-valued, same scale as mu_pq.
        # ──────────────────────────────────────────────────────────
        if (p, q) in cache:
            return cache[(p, q)]
        order = p + q
        if order < constants.BLUR_INVARIANT_ORDER_OFFSET or p % 2 == 1 or q % 2 == 1:
            value = moments.get((p, q), 0.0)
            cache[(p, q)] = value
            return value

        total = self._blur_invariant_correction_sum(moments, p, q, cache)
        value = moments[(p, q)] - moments[(p, q)] * (total / mu00 if mu00 != 0.0 else 0.0)
        cache[(p, q)] = value
        return value

    def _blur_invariant_correction_sum(self, moments, p, q, cache) -> float:
        """The double summation term inside Eq. 12.

        Args:
            moments: Central moments dict.
            p: First order index (both p, q even, p+q >= 4 guaranteed by caller).
            q: Second order index.
            cache: Memoisation dict for the recursive B() calls.

        Returns:
            The raw (unweighted-by-alpha/mu00) correction sum.
        """
        order = p + q
        k = (order - constants.BLUR_INVARIANT_ORDER_OFFSET) // 2
        total = 0.0
        for n in range(k + 1):
            t = 2 * (k - n + 1)
            m1 = max(0, (t - p + 1) // 2)
            m2 = min(t // 2, q // 2)
            for i in range(m1, m2 + 1):
                sub_p, sub_q = p - t + 2 * i, q - 2 * i
                if sub_p < 0 or sub_q < 0:
                    continue
                coefficient = (binomial_coefficient(t - 2 * i, 2 * i)
                              * binomial_coefficient(q, 2 * i))
                sub_invariant = self.compute_blur_invariant(
                    moments, sub_p, sub_q, moments.get((0, 0), 0.0), cache)
                mu_term = moments.get((t - 2 * i, 2 * i), 0.0)
                total += coefficient * sub_invariant * mu_term
        return total

    def build_feature_vector(self, block_pixels: np.ndarray) -> np.ndarray:
        """Build the contrast-normalised blur-invariant feature vector for a block.

        Args:
            block_pixels: R x R float64 block.

        Returns:
            1-D feature vector, one entry per (p, q) with
            MINIMUM_MOMENT_ORDER <= p+q <= MAXIMUM_MOMENT_ORDER.
        """
        moments = self.compute_moments(block_pixels, constants.MAXIMUM_MOMENT_ORDER)
        mu00 = moments.get((0, 0), 0.0)
        cache = {}
        block_size = block_pixels.shape[0]
        values = []
        for order in range(constants.MINIMUM_MOMENT_ORDER,
                           constants.MAXIMUM_MOMENT_ORDER + 1):
            for p in range(order + 1):
                q = order - p
                invariant = self.compute_blur_invariant(moments, p, q, mu00, cache)
                values.append(self._contrast_normalise(invariant, block_size, order, mu00))
        return np.array(values, dtype=np.float64)

    def _contrast_normalise(self, invariant: float, block_size: int,
                            order: int, mu00: float) -> float:
        """Contrast-normalise one blur-invariant value.

        Args:
            invariant: Raw B(p, q) value.
            block_size: R, the block side length.
            order: r = p + q.
            mu00: Zeroth central moment.

        Returns:
            Contrast-normalised invariant B'_i.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: B'_i = B_i / ((R/2)^r * mu_00)
        # Variables: R=block size, r=order of B_i.
        # Source: Kashyap & Joshi 2013, Eq. 17.
        # Expected range: dimensionless, contrast-invariant.
        # ──────────────────────────────────────────────────────────
        denominator = ((block_size / 2.0) ** order) * mu00
        if denominator == 0.0:
            return 0.0
        return invariant / denominator

    def reduce_dimensionality(self, feature_matrix: np.ndarray) -> np.ndarray:
        """PCA-reduce block feature vectors, keeping the target variance.

        Args:
            feature_matrix: n_blocks x n_features array.

        Returns:
            n_blocks x m_0 reduced feature matrix.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: eigen-decomposition of R=E[X X^T], keep eigenvectors with
        #   the largest eigenvalues lambda_i, project X onto top m_0 << m.
        # Source: Kashyap & Joshi 2013, Eq. 18-26.
        # Expected range: reduced-dimension real vectors.
        # ──────────────────────────────────────────────────────────
        max_components = min(feature_matrix.shape[0], feature_matrix.shape[1])
        pca = PCA(n_components=max_components)
        pca.fit(feature_matrix)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        n_components = int(np.searchsorted(
            cumulative, constants.PCA_EXPLAINED_VARIANCE_TARGET) + 1)
        n_components = max(n_components, constants.PCA_MINIMUM_COMPONENTS)
        n_components = min(n_components, max_components)
        return PCA(n_components=n_components).fit_transform(feature_matrix)

    def find_candidate_pairs(self, reduced_vectors: np.ndarray) -> set:
        """Find all block-index pairs whose similarity meets the threshold.

        Args:
            reduced_vectors: n_blocks x m_0 PCA-reduced feature matrix.

        Returns:
            Set of (index_a, index_b) tuples with index_a < index_b.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: S(Bi,Bj) = 1/(1+rho(Bi,Bj)), rho = Euclidean distance.
        #   Candidate duplicate if S(Bi,Bj) >= T.
        # Source: Kashyap & Joshi 2013, Eq. 27-28.
        # Expected range: S in (0, 1].
        # ──────────────────────────────────────────────────────────
        max_distance = (1.0 / constants.SIMILARITY_THRESHOLD) - 1.0
        tree = cKDTree(reduced_vectors)
        return tree.query_pairs(r=max_distance)

    def confirm_neighbour_consistency(self, candidates: set, positions: list,
                                      reduced_vectors: np.ndarray,
                                      position_lookup: dict,
                                      block_size: int) -> list:
        """Apply the 16-neighbour consistency and minimum-separation checks.

        Args:
            candidates: Candidate (index_a, index_b) pairs from Eq. 27-28.
            positions: List of (row, col) per block index.
            reduced_vectors: n_blocks x m_0 PCA-reduced feature matrix.
            position_lookup: Dict mapping (row, col) -> block index.
            block_size: R, the block side length.

        Returns:
            List of confirmed DuplicatePair objects.
        """
        offsets = generate_neighbour_offsets(
            constants.NEIGHBOUR_CHECK_MAX_OFFSET_PIXELS,
            constants.NEIGHBOUR_CHECK_COUNT)
        min_separation = constants.MINIMUM_SEPARATION_BLOCK_MULTIPLE * block_size

        confirmed = []
        for index_a, index_b in candidates:
            row_a, col_a = positions[index_a]
            row_b, col_b = positions[index_b]
            separation = float(np.hypot(row_a - row_b, col_a - col_b))
            if separation <= min_separation:
                continue
            if self._all_neighbours_consistent(row_a, col_a, row_b, col_b,
                                               offsets, position_lookup,
                                               reduced_vectors):
                similarity = 1.0 / (1.0 + np.linalg.norm(
                    reduced_vectors[index_a] - reduced_vectors[index_b]))
                confirmed.append(DuplicatePair(
                    block_a_row=row_a, block_a_col=col_a,
                    block_b_row=row_b, block_b_col=col_b,
                    similarity=float(similarity)))
        return confirmed

    def _all_neighbours_consistent(self, row_a, col_a, row_b, col_b,
                                   offsets, position_lookup,
                                   reduced_vectors) -> bool:
        """Check every one of the 16 neighbour-offset pairs meets the threshold.

        SKILL Eq. 29 requires ALL 16 neighbour pairs to satisfy S>=T; a
        candidate whose shifted position falls outside the tiled block grid
        (near an image edge) cannot be evaluated and is therefore not
        confirmed. [ENGINEERING] boundary-handling choice, not a relaxation
        of the threshold itself.

        Args:
            row_a: Candidate block A's row origin.
            col_a: Candidate block A's column origin.
            row_b: Candidate block B's row origin.
            col_b: Candidate block B's column origin.
            offsets: List of (dx, dy) neighbour offsets.
            position_lookup: Dict mapping (row, col) -> block index.
            reduced_vectors: n_blocks x m_0 PCA-reduced feature matrix.

        Returns:
            True if all 16 neighbour pairs are consistent.
        """
        for dx, dy in offsets:
            key_a = (row_a + dx, col_a + dy)
            key_b = (row_b + dx, col_b + dy)
            if key_a not in position_lookup or key_b not in position_lookup:
                return False
            distance = np.linalg.norm(
                reduced_vectors[position_lookup[key_a]]
                - reduced_vectors[position_lookup[key_b]])
            if 1.0 / (1.0 + distance) < constants.SIMILARITY_THRESHOLD:
                return False
        return True

    def build_duplicate_map(self, confirmed_pairs: list, ll_shape: tuple,
                            block_size: int) -> np.ndarray:
        """Mark every confirmed pair's block footprint in a binary map.

        Args:
            confirmed_pairs: List of confirmed DuplicatePair objects.
            ll_shape: Shape of the LL subband.
            block_size: R, the block side length.

        Returns:
            Binary (0/1) map, shape ll_shape.
        """
        duplicate_map = np.zeros(ll_shape, dtype=np.uint8)
        for pair in confirmed_pairs:
            duplicate_map[pair.block_a_row:pair.block_a_row + block_size,
                         pair.block_a_col:pair.block_a_col + block_size] = 1
            duplicate_map[pair.block_b_row:pair.block_b_row + block_size,
                         pair.block_b_col:pair.block_b_col + block_size] = 1
        return duplicate_map

    def compute(self, blocks: list, ll_shape: tuple,
               block_size: int) -> CopyMoveResult:
        """Run the full copy-move detection pipeline over pre-tiled blocks.

        Args:
            blocks: List of Block objects from the preprocessor.
            ll_shape: Shape of the LL subband the blocks were tiled from.
            block_size: R, the block side length.

        Returns:
            CopyMoveResult with the duplicate map and summary scalar.
        """
        feature_matrix = np.stack(
            [self.build_feature_vector(block.pixels) for block in blocks])
        reduced = self.reduce_dimensionality(feature_matrix)

        positions = [(block.row, block.col) for block in blocks]
        position_lookup = {position: index for index, position in enumerate(positions)}

        candidates = self.find_candidate_pairs(reduced)
        confirmed = self.confirm_neighbour_consistency(
            candidates, positions, reduced, position_lookup, block_size)

        duplicate_map = self.build_duplicate_map(confirmed, ll_shape, block_size)
        flagged_blocks = {(p.block_a_row, p.block_a_col) for p in confirmed}
        flagged_blocks |= {(p.block_b_row, p.block_b_col) for p in confirmed}

        return CopyMoveResult(
            duplicate_map=duplicate_map, confirmed_pairs=confirmed,
            total_blocks=len(blocks), flagged_block_count=len(flagged_blocks),
            fraction_flagged=len(flagged_blocks) / len(blocks) if blocks else 0.0,
            block_size=block_size)
