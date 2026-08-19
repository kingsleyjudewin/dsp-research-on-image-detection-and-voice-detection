"""Core mathematical computation for all four noise-pattern pipelines.

Pipeline A (LocalNoiseInconsistencyComputer) is the sole score-driving
computation in this engine - see the SCOPE DECISION note at the top of
constants.py. Pipelines B, C, and D are computed when possible and returned
as auxiliary, non-scoring evidence (D modulates confidence only, never
raw_score).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from scipy.signal import wiener as scipy_wiener_filter

from . import constants
from .contracts import (LocalInconsistencyResult, NoiseTriageResult,
                        ReferencePRNUResult, SpectralAnalysisResult)
from .utils import grid_neighbourhood_median, top_k_fraction_mean

logger = logging.getLogger(__name__)


class LocalNoiseInconsistencyComputer:
    """Pipeline A: blind local noise-level inconsistency (PRIMARY, scored)."""

    def block_residual_variance(self, blocks: list) -> np.ndarray:
        """Compute each block's residual variance, arranged as a 2-D grid.

        Args:
            blocks: List of NoiseBlock objects from the preprocessor.

        Returns:
            2-D array of shape (max_row+1, max_col+1) holding per-block
            residual variance.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: local noise-level statistic = block's residual variance
        #   (option (a) of two offered).
        # Variables: W=noise residual (I-F(I)) within a block.
        # Source: SKILL, Pipeline A step 3.
        # Expected range: non-negative.
        # ──────────────────────────────────────────────────────────
        max_row = max(block.row for block in blocks) + 1
        max_col = max(block.col for block in blocks) + 1
        grid = np.zeros((max_row, max_col), dtype=np.float64)
        for block in blocks:
            grid[block.row, block.col] = float(np.var(block.residual_pixels))
        return grid

    def block_texture_energy(self, blocks: list) -> np.ndarray:
        """Per-block scene-texture energy, measured on the intensity plane.

        Args:
            blocks: List of NoiseBlock objects from the preprocessor.

        Returns:
            2-D array matching the statistic grid, holding the mean absolute
            Laplacian of each block's original intensity pixels.
        """
        # ENHANCEMENT 1: diagnostic testing measured Spearman(block residual
        # variance, block Laplacian energy) between 0.7726 and 0.9912 on all
        # six corpus images, so what step 3's "local noise-level statistic"
        # actually ranks is scene texture. Texture is measured here from the
        # INTENSITY plane, never from the residual: a feature derived from the
        # residual moves with a denoising attack and cancels the signal being
        # looked for - SKILL Eq. 20's f_T was tested and scored 0.076
        # top-10%-inside on denoised regions, BELOW the 0.111 chance rate.
        # See REJECTED_ENHANCEMENTS in constants.py.
        max_row = max(block.row for block in blocks) + 1
        max_col = max(block.col for block in blocks) + 1
        grid = np.zeros((max_row, max_col), dtype=np.float64)
        for block in blocks:
            laplacian = cv2.Laplacian(block.intensity_pixels, cv2.CV_64F)
            grid[block.row, block.col] = float(np.mean(np.abs(laplacian)))
        return grid

    def _texture_conditioned_statistic(self, statistic_grid: np.ndarray,
                                       texture_grid: np.ndarray) -> np.ndarray:
        """Regress the texture-explained part out of the block statistic.

        Args:
            statistic_grid: Per-block residual-variance grid.
            texture_grid: Per-block scene-texture energy grid.

        Returns:
            Grid of fit residuals: the part of log2(statistic) that block
            texture does not account for.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: least-squares fit of log2(block statistic) on
        #   log2(block texture feature); the fit residual is what texture
        #   does not explain.
        # Variables: statistic=per-block residual variance, texture=per-block
        #   scene-texture energy.
        # Source: SKILL, Pipeline A step 4's own stated rationale - "the same
        #   value of a raw statistic means different things in different
        #   texture/intensity contexts" - which it attributes directly to
        #   Chen et al.'s correlation predictor, Pipeline B.4. B.4 predicts
        #   rho_b from block features by "polynomial multivariate
        #   least-squares fitting", for which the SKILL's Implementation
        #   Notes name numpy.linalg.lstsq. Only B.4's texture direction is
        #   used: its intensity and signal-flattening features need the
        #   camera-specific constants I_crit, tau and c that the SKILL states
        #   "do not transfer directly across camera models".
        # Expected range: real-valued, centred near 0.
        # ──────────────────────────────────────────────────────────
        floor = constants.MINIMUM_STATISTIC_FLOOR
        texture = np.log2(np.maximum(texture_grid.ravel(), floor))
        statistic = np.log2(np.maximum(statistic_grid.ravel(), floor))
        if (texture.size < constants.MINIMUM_BLOCKS_FOR_TEXTURE_FIT
                or float(np.std(texture)) == 0.0):
            return statistic.reshape(statistic_grid.shape)
        slope, intercept = np.polyfit(texture, statistic,
                                      constants.TEXTURE_FIT_POLYNOMIAL_DEGREE)
        return (statistic - (slope * texture + intercept)).reshape(
            statistic_grid.shape)

    def deviation_field(self, statistic_grid: np.ndarray,
                        texture_grid: np.ndarray) -> np.ndarray:
        """Each block's departure from what its neighbourhood predicts.

        Args:
            statistic_grid: Per-block residual-variance grid.
            texture_grid: Per-block scene-texture energy grid.

        Returns:
            Signed deviation grid, same shape as statistic_grid.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: compare each block's statistic against its local
        #   neighborhood median (not a fixed global threshold).
        # Source: SKILL, Pipeline A step 4.
        # Expected range: symmetric around 0.
        # ──────────────────────────────────────────────────────────
        conditioned = (
            self._texture_conditioned_statistic(statistic_grid, texture_grid)
            if constants.TEXTURE_CONDITIONING_ENABLED
            else np.log2(np.maximum(statistic_grid,
                                    constants.MINIMUM_STATISTIC_FLOOR)))
        return conditioned - grid_neighbourhood_median(
            conditioned, constants.LOCAL_NEIGHBOURHOOD_WINDOW_BLOCKS)

    @staticmethod
    def deviation_scale(field: np.ndarray) -> float:
        """Flag threshold, taken from the deviation field's own robust spread.

        Args:
            field: Signed deviation grid.

        Returns:
            The absolute deviation beyond which a block counts as flagged.
        """
        # ENHANCEMENT 2: the fixed factor-of-2 cutoff was measured to sit
        # BELOW the ordinary block-to-block spread of authentic photographs -
        # median |log2 ratio| ran 0.4192 to 0.7725 across the six corpus
        # images, i.e. the typical authentic block already differs from its
        # neighbours by 1.34x to 1.71x - so 24% to 42% of every image's cells
        # clipped at the 1.0 ceiling and raw_score was exactly 1.000000 on all
        # six. The SKILL's step 4 itself insists the comparison must not use
        # "a fixed global threshold"; this reads the scale off the image's own
        # deviation distribution instead.
        absolute = np.abs(field - np.median(field))
        sigma = (float(np.median(absolute))
                 / constants.MEDIAN_ABSOLUTE_DEVIATION_TO_SIGMA)
        return max(constants.DEVIATION_ROBUST_SPREAD_MULTIPLE * sigma,
                   constants.MINIMUM_STATISTIC_FLOOR)

    def flag_deviant_blocks(self, statistic_grid: np.ndarray,
                            texture_grid: np.ndarray) -> np.ndarray:
        """Render the deviation field as a [0,1]-normalised anomaly heatmap.

        Args:
            statistic_grid: Per-block residual-variance grid.
            texture_grid: Per-block scene-texture energy grid.

        Returns:
            [0,1]-normalised anomaly grid.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: flag blocks whose statistic deviates significantly from
        #   their neighborhood; aggregate into a [0,1]-normalized heatmap.
        # Source: SKILL, Pipeline A steps 4-5.
        # Expected range: [0, 1].
        # "Deviates" is bidirectional in the SKILL's own wording, and the
        # SKILL's single most emphasised failure mode - a denoising/
        # smoothing attack - produces LOWER local variance than the
        # surroundings, not higher; a one-directional (over-variance-only)
        # flag would silently miss exactly that case. [ENGINEERING]
        # resolution of "deviates significantly" as a symmetric deviation.
        # ──────────────────────────────────────────────────────────
        field = self.deviation_field(statistic_grid, texture_grid)
        return np.clip(np.abs(field) / self.deviation_scale(field), 0.0, 1.0)

    def compute(self, blocks: list, block_size: int) -> LocalInconsistencyResult:
        """Run the full local noise-inconsistency pipeline.

        Args:
            blocks: List of NoiseBlock objects from the preprocessor.
            block_size: Block side length in pixels.

        Returns:
            LocalInconsistencyResult with the heatmap and aggregate scalar.
        """
        statistic_grid = self.block_residual_variance(blocks)
        texture_grid = self.block_texture_energy(blocks)
        field = self.deviation_field(statistic_grid, texture_grid)
        scale = self.deviation_scale(field)
        heatmap = np.clip(np.abs(field) / scale, 0.0, 1.0)

        deviant = np.abs(field) > scale
        flagged = [block for block in blocks if deviant[block.row, block.col]]
        # ENHANCEMENT 2: SKILL step 5 - "Aggregate flagged blocks into a
        # heatmap and a scalar summary". The Output section's alternative
        # top-k% mean of the heatmap was measured to have exactly zero dynamic
        # range: 1.000000 on all six corpus images, on all six global nuisance
        # transforms of each, and on all 18 ground-truth manipulations of a
        # real photograph (delta +0.000000 every time). The flagged-block
        # fraction moved in the correct direction on 17 of those 18.
        aggregate_scalar = (float(np.count_nonzero(deviant))
                            / float(deviant.size))

        return LocalInconsistencyResult(
            heatmap=heatmap, flagged_blocks=flagged, total_blocks=len(blocks),
            flagged_block_count=len(flagged), aggregate_scalar=aggregate_scalar,
            block_size=block_size,
            legacy_top_k_scalar=top_k_fraction_mean(
                heatmap, constants.SCALAR_AGGREGATION_TOP_K_FRACTION))


class BlindSpectralAnalyser:
    """Pipeline C: blind cell-based PRNU spectral analysis (auxiliary)."""

    def split_into_cells(self, residual: np.ndarray, grid_cells: int) -> list:
        """Divide the residual into grid_cells x grid_cells rectangular cells.

        Args:
            residual: Float64 noise-residual array.
            grid_cells: N, the per-axis cell count (e.g. 8 for 8x8=64 cells).

        Returns:
            List of 2-D cell arrays.
        """
        height, width = residual.shape
        row_edges = np.linspace(0, height, grid_cells + 1, dtype=int)
        col_edges = np.linspace(0, width, grid_cells + 1, dtype=int)
        cells = []
        for row_index in range(grid_cells):
            for col_index in range(grid_cells):
                cell = residual[row_edges[row_index]:row_edges[row_index + 1],
                               col_edges[col_index]:col_edges[col_index + 1]]
                if cell.size > 0:
                    cells.append(cell)
        return cells

    def dft_magnitude_histogram(self, cell: np.ndarray) -> np.ndarray:
        """2-D DFT magnitude spectrum, histogrammed over the paper's range.

        Args:
            cell: One rectangular residual cell.

        Returns:
            Histogram counts array, length PIPELINE_C_HISTOGRAM_BIN_COUNT.
        """
        magnitude_spectrum = np.abs(np.fft.fft2(cell))
        counts, _ = np.histogram(magnitude_spectrum,
                                 bins=constants.PIPELINE_C_HISTOGRAM_BIN_COUNT,
                                 range=constants.PIPELINE_C_DFT_MAGNITUDE_RANGE)
        return counts.astype(np.float64)

    def peak_features(self, histogram: np.ndarray) -> tuple:
        """Extract P_val, P_pos, and P_pv from one cell's DFT histogram.

        Args:
            histogram: Histogram counts array for one cell.

        Returns:
            Tuple of (P_val, P_pos, P_pv).
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: P_val = max_n H(n)          [Eq. 2]
        #          P_pos = argmax_n H(n)       [Eq. 3]
        #          P_pv  = P_val * P_pos       [Eq. 4]
        # Variables: H=DFT-magnitude histogram of one cell.
        # Source: Debiasi et al. 2018, Eq. 2-4.
        # Expected range: P_val>=0, P_pos in [0, bin_count-1], P_pv>=0.
        # ──────────────────────────────────────────────────────────
        p_val = float(np.max(histogram))
        p_pos = float(np.argmax(histogram))
        return p_val, p_pos, p_val * p_pos

    def aggregate_cells(self, values: np.ndarray) -> tuple:
        """Aggregate a per-cell feature array into S_mean and S_rms.

        Args:
            values: 1-D array of one feature (e.g. all cells' P_val).

        Returns:
            Tuple of (S_mean, S_rms).
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: S_mean = (1/N) sum_n P_n          [Eq. 5]
        #          S_rms  = sqrt((1/N) sum_n P_n^2)  [Eq. 6]
        # Source: Debiasi et al. 2018, Eq. 5-6.
        # Expected range: unbounded, dataset/camera-dependent scale.
        # ──────────────────────────────────────────────────────────
        s_mean = float(np.mean(values))
        s_rms = float(np.sqrt(np.mean(np.square(values))))
        return s_mean, s_rms

    def compute(self, residual: np.ndarray) -> SpectralAnalysisResult:
        """Run the full blind spectral-analysis pipeline.

        Args:
            residual: Float64 noise-residual array W.

        Returns:
            SpectralAnalysisResult with S_mean/S_rms per feature.
        """
        grid_cells = constants.PIPELINE_C_DEFAULT_GRID_CELLS
        cells = self.split_into_cells(residual, grid_cells)
        p_vals, p_poss, p_pvs = [], [], []
        for cell in cells:
            histogram = self.dft_magnitude_histogram(cell)
            p_val, p_pos, p_pv = self.peak_features(histogram)
            p_vals.append(p_val)
            p_poss.append(p_pos)
            p_pvs.append(p_pv)

        mean_val, rms_val = self.aggregate_cells(np.array(p_vals))
        mean_pos, rms_pos = self.aggregate_cells(np.array(p_poss))
        mean_pv, rms_pv = self.aggregate_cells(np.array(p_pvs))
        return SpectralAnalysisResult(
            s_mean_p_val=mean_val, s_mean_p_pos=mean_pos, s_mean_p_pv=mean_pv,
            s_rms_p_val=rms_val, s_rms_p_pos=rms_pos, s_rms_p_pv=rms_pv,
            grid_cells=grid_cells)


class ReferencePRNUEstimator:
    """Pipeline B: reference-based PRNU (auxiliary, calibration-gated).

    Only the fully-specified, camera-agnostic pieces are implemented - see
    constants.py's SCOPE DECISION and KNOWN_UNIMPLEMENTED_MODULES.
    """

    def estimate_prnu(self, residuals: list, intensities: list) -> np.ndarray:
        """Maximum-likelihood PRNU factor estimate from reference images.

        Args:
            residuals: List of W_k residual arrays, one per reference image.
            intensities: List of matching I_k intensity arrays.

        Returns:
            K_hat, the estimated PRNU factor array.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: K_hat = sum_{k=1}^N (W_k * I_k) / sum_{k=1}^N (I_k)^2
        # Variables: W_k=noise residual of reference image k, I_k=its
        #   intensity, N=reference image count.
        # Source: Chen, Fridrich, Goljan & Lukas 2008, Eq. 6.
        # Expected range: real-valued, zero-mean in aggregate.
        # ──────────────────────────────────────────────────────────
        numerator = sum(w * i for w, i in zip(residuals, intensities))
        denominator = sum(i ** 2 for i in intensities)
        safe_denominator = np.where(denominator != 0.0, denominator, 1.0)
        return numerator / safe_denominator

    def zero_mean_preprocessing(self, prnu_estimate: np.ndarray) -> np.ndarray:
        """Enforce zero mean in every row and every column of K_hat.

        Args:
            prnu_estimate: Raw K_hat estimate.

        Returns:
            ZM(K_hat), with row and column means removed.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: ZM(K_hat) = K_hat with zero mean enforced in every row
        #   and every column (subtract column average per column, then row
        #   average per row).
        # Source: Chen et al. 2008, B.2 step 5.
        # Expected range: same scale as K_hat, row/column means now zero.
        # ──────────────────────────────────────────────────────────
        column_centred = prnu_estimate - np.mean(prnu_estimate, axis=0, keepdims=True)
        return column_centred - np.mean(column_centred, axis=1, keepdims=True)

    def wiener_preprocessing(self, zero_mean_estimate: np.ndarray) -> np.ndarray:
        """Suppress structured Fourier-domain artifacts via a Wiener filter.

        Args:
            zero_mean_estimate: ZM(K_hat).

        Returns:
            WF(ZM(K_hat)), the preprocessed PRNU factor.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: WF(ZM(K_hat)) = F^-1{ F(ZM(K_hat)) - W(F(ZM(K_hat))) }
        #   W = 3x3 Wiener filter, variance from the sample variance of
        #   the magnitude of F(ZM(K_hat)).
        # Source: Chen et al. 2008, B.2 step 6.
        # Expected range: same scale as ZM(K_hat).
        # ──────────────────────────────────────────────────────────
        frequency_domain = np.fft.fft2(zero_mean_estimate)
        magnitude = np.abs(frequency_domain)
        noise_variance = float(np.var(magnitude))
        filtered_magnitude = scipy_wiener_filter(
            magnitude, mysize=constants.WIENER_FILTER_KERNEL_SIZE,
            noise=noise_variance)
        cleaned_frequency_domain = frequency_domain - (filtered_magnitude
                                                       - magnitude)
        return np.real(np.fft.ifft2(cleaned_frequency_domain))

    def block_correlation(self, prnu_term: np.ndarray, residual: np.ndarray) -> float:
        """Normalised correlation between the PRNU term and a residual block.

        The weighted aggregate detector (Eq. 11) and its beta_b shaping
        factor are not implemented - see constants.py's SCOPE DECISION. This
        is the raw, unweighted rho_b only.

        Args:
            prnu_term: X_b = I*K_hat for the block under test.
            residual: W_b, the block's own noise residual.

        Returns:
            Correlation coefficient in [-1, 1].
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: rho_b = corr(X_b, W_b)
        # Source: Chen et al. 2008, B.3 step 9 (normalized correlation
        #   within block b; the beta_b weighting is not implemented).
        # Expected range: [-1, 1].
        # ──────────────────────────────────────────────────────────
        flat_prnu, flat_residual = prnu_term.ravel(), residual.ravel()
        if np.std(flat_prnu) == 0.0 or np.std(flat_residual) == 0.0:
            return 0.0
        return float(np.corrcoef(flat_prnu, flat_residual)[0, 1])

    def compute(self, reference_residuals: list, reference_intensities: list,
               test_residual: np.ndarray, test_intensity: np.ndarray) -> ReferencePRNUResult:
        """Run the implemented subset of Pipeline B.

        Args:
            reference_residuals: W_k for each same-camera reference image.
            reference_intensities: I_k for each same-camera reference image.
            test_residual: W for the image under test.
            test_intensity: I for the image under test.

        Returns:
            ReferencePRNUResult with the mean block correlation.
        """
        if len(reference_residuals) < constants.MINIMUM_REFERENCE_IMAGES_FOR_PRNU:
            return ReferencePRNUResult(
                ran=False,
                note=f"Fewer than "
                     f"{constants.MINIMUM_REFERENCE_IMAGES_FOR_PRNU} "
                     f"same-camera reference images supplied; Pipeline B "
                     f"skipped.")
        k_hat = self.estimate_prnu(reference_residuals, reference_intensities)
        k_hat_zm = self.zero_mean_preprocessing(k_hat)
        k_hat_clean = self.wiener_preprocessing(k_hat_zm)
        prnu_term = test_intensity * k_hat_clean[:test_intensity.shape[0],
                                                :test_intensity.shape[1]]
        correlation = self.block_correlation(prnu_term, test_residual)
        return ReferencePRNUResult(
            ran=True, mean_block_correlation=correlation,
            reference_image_count=len(reference_residuals),
            note="Raw unweighted correlation only; the paper's weighted "
                 "detector and correlation predictor are not implemented "
                 "(camera-specific unsourced constants) - see "
                 "KNOWN_UNIMPLEMENTED_MODULES.")


class NoiseTriageClassifier:
    """Pipeline D: FFT-spectrum noise-type triage (auxiliary, confidence-only)."""

    def fft_magnitude_spectrum(self, grayscale: np.ndarray) -> np.ndarray:
        """Compute the 2-D FFT magnitude spectrum of a grayscale image.

        Args:
            grayscale: Float64 grayscale image.

        Returns:
            Magnitude spectrum array, same shape as grayscale.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: F[u,v] = sum_x sum_y f(x,y) exp(-j2pi(ux/N+vy/M))
        # Source: Jain & Arolkar 2024, Pipeline D step 2.
        # Expected range: non-negative magnitudes.
        # ──────────────────────────────────────────────────────────
        return np.abs(np.fft.fft2(grayscale))

    def classify(self, magnitude_spectrum: np.ndarray,
                reference_mean_magnitude: float) -> NoiseTriageResult:
        """Classify the image's noise type via the SKILL's resolved z-score rule.

        Args:
            magnitude_spectrum: FFT magnitude spectrum of the image.
            reference_mean_magnitude: Training-set baseline mean magnitude,
                must be supplied by the orchestrator (no SKILL default).

        Returns:
            NoiseTriageResult with the resolved label and z-score.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: z_score = (mean_magnitude - avg_mean_magnitude_train)
        #   / std_dev
        #   Decision (SKILL's own resolved ambiguity, not the literal
        #   contradictory printed pseudocode): z<-1.0 -> Gaussian blur;
        #   z>+1.0 -> Impulse; -1.0<=z<=+1.0 -> No noise.
        # Source: Jain & Arolkar 2024, Pipeline D steps 3-5.
        # Expected range: label in {Gaussian blur, Impulse, No noise}.
        # ──────────────────────────────────────────────────────────
        mean_magnitude = float(np.mean(magnitude_spectrum))
        std_dev = float(np.std(magnitude_spectrum))
        if std_dev == 0.0:
            return NoiseTriageResult(ran=False,
                                     note="Magnitude spectrum has zero "
                                          "standard deviation; triage "
                                          "z-score undefined.")
        z_score = (mean_magnitude - reference_mean_magnitude) / std_dev
        if z_score < constants.NOISE_TRIAGE_GAUSSIAN_THRESHOLD:
            label = "Gaussian blur"
        elif z_score > constants.NOISE_TRIAGE_IMPULSE_THRESHOLD:
            label = "Impulse"
        else:
            label = "No noise"
        return NoiseTriageResult(ran=True, label=label, z_score=z_score,
                                 note="Weak triage signal per the SKILL's "
                                      "own low-accuracy benchmark (40-image "
                                      "test set); used only to modulate "
                                      "confidence, never raw_score.")
