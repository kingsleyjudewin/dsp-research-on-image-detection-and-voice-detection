"""Core mathematical computation for the ghost and resampling pipelines.

GhostDetector (Pipeline B) is the sole score-driving computation - see the
SCOPE DECISION note at the top of constants.py. ResamplingDetector
(Pipeline A) runs and is reported, but cannot be scored: its threshold rho_T
is never given a numeric value in this SKILL, and the gamma contrast
function its spectrum enhancement calls for is explicitly not printed there.
Pipeline C is [ML - excluded] and absent by design.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from skimage.filters import threshold_multiotsu, threshold_otsu
from skimage.segmentation import slic

from . import constants
from .contracts import GhostCandidate, GhostResult, ResamplingResult
from .utils import (class_mean_and_standard_deviation,
                    min_max_normalise_along_axis, radial_frequency_grid,
                    two_class_split_by_threshold)

logger = logging.getLogger(__name__)


class GhostDetector:
    """Pipeline B: automated JPEG-ghost detection (SCORE-DRIVING)."""

    def __init__(self, preprocessor) -> None:
        """Bind the preprocessor that performs shifting and recompression.

        Args:
            preprocessor: A GhostPreprocessor instance.
        """
        self.preprocessor = preprocessor

    @staticmethod
    def difference_energy(original_rgb: np.ndarray,
                          recompressed_rgb: np.ndarray) -> np.ndarray:
        """Smoothed per-pixel difference energy between an image and its recompression.

        Args:
            original_rgb: uint8 H x W x 3 RGB image (possibly grid-shifted).
            recompressed_rgb: The same image after a q2 round trip.

        Returns:
            Float64 H x W map of smoothed difference energy.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: delta_{(q2,dx,dy)}(x,y)
        #   = (1/3w^2) * sum_{c in {R,G,B}} sum_{i,j=0}^{w-1}
        #     ( I'(x+i,y+j,c) - I'_{q2}(x+i,y+j,c) )^2
        # Variables: I' = the (grid-shifted) dubious image, I'_q2 = the same
        #   image recompressed at candidate quality q2, w = smoothing window
        #   (w=16, the paper's stated default following the original method),
        #   c ranges over the three RGB channels, and 3w^2 normalises by the
        #   channel count times the window area.
        # Source: Azarian-Pour, Babaie-Zadeh & Sadri 2016, Eq. 6 / Eq. 8.
        # Expected range: non-negative; a region already quantized at q2
        #   loses little further information and so dips to a local minimum.
        # ──────────────────────────────────────────────────────────
        squared = (original_rgb.astype(np.float64)
                   - recompressed_rgb.astype(np.float64)) ** 2
        channel_sum = np.sum(squared, axis=2)
        window = constants.GHOST_SMOOTHING_WINDOW
        smoothed = cv2.boxFilter(channel_sum, ddepth=-1, ksize=(window, window),
                                 normalize=True)
        return smoothed / constants.GHOST_CHANNEL_COUNT

    @staticmethod
    def normalise_across_qualities(energy_stack: np.ndarray) -> np.ndarray:
        """Min-max normalise every pixel across the candidate-quality sweep.

        Args:
            energy_stack: Float64 (n_qualities, H, W) difference energies for
                one grid shift.

        Returns:
            Float64 array of the same shape, each pixel rescaled to [0, 1].
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: d_{(q2,dx,dy)}(x,y) =
        #   ( delta_{(q2,dx,dy)}(x,y) - min_q[delta_{(q2,dx,dy)}(x,y)] )
        #   / ( max_q[...] - min_q[...] )
        # Variables: the minimum and maximum are taken over the candidate
        #   quality sweep q at each pixel independently, for a fixed grid
        #   shift. This turns the absolute energy into a per-pixel profile
        #   whose minimum marks that pixel's own original quality.
        # Source: Azarian-Pour 2016, Eq. 7 / Eq. 9.
        # Expected range: [0, 1] per pixel, reaching 0 at its ghost quality.
        # ──────────────────────────────────────────────────────────
        return min_max_normalise_along_axis(energy_stack, axis=0)

    @staticmethod
    def segment_difference_map(difference_map: np.ndarray) -> np.ndarray:
        """Split a difference map into ghost (class 0) and rest (class 1).

        Half of the SE-MinCut substitute. SLIC groups the map into spatially
        compact superpixels and the split is taken over their MEANS, so
        isolated noisy pixels cannot form a class on their own. The other
        half - deciding whether the resulting split describes a real pasted
        region at all - is is_valid_ghost_candidate, and both halves are
        needed: see constants.MINIMUM_GHOST_SPATIAL_COHERENCE.

        The map is segmented at reduced scale because the w=16 smoothing
        already sets its usable detail, and the reduction additionally damps
        the fractal noise the segmenter has to resist. Labels are upsampled
        back so Eq. 10 is evaluated at full resolution.

        Args:
            difference_map: Float64 H x W normalised difference map.

        Returns:
            Integer H x W array of 0/1 class labels.
        """
        factor = constants.SEGMENTATION_DOWNSAMPLE_FACTOR
        small = cv2.resize(difference_map,
                           (max(difference_map.shape[1] // factor, 1),
                            max(difference_map.shape[0] // factor, 1)),
                           interpolation=cv2.INTER_AREA)
        superpixels = slic(small, n_segments=constants.SEGMENTATION_SUPERPIXEL_COUNT,
                           channel_axis=None,
                           compactness=constants.SEGMENTATION_COMPACTNESS,
                           start_label=constants.SEGMENTATION_START_LABEL)
        labels_small = GhostDetector._group_superpixels(small, superpixels)
        return cv2.resize(labels_small.astype(np.uint8),
                          (difference_map.shape[1], difference_map.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(np.int64)

    @staticmethod
    def _group_superpixels(difference_map: np.ndarray,
                           superpixels: np.ndarray) -> np.ndarray:
        """Assign each superpixel to one of two classes by its mean value.

        A genuine ghost makes the map bimodal - the pasted region collapses
        toward zero at its own quality while the background stays higher -
        so the split point is placed between those two modes. Deciding
        WHETHER that split describes a real region is deliberately not this
        method's job: it is handled by is_valid_ghost_candidate, because a
        split point alone cannot tell a bimodal ghost from a unimodal noise
        map (see constants.MINIMUM_GHOST_SPATIAL_COHERENCE).

        Args:
            difference_map: Float64 reduced-scale difference map.
            superpixels: Integer superpixel label image of the same shape.

        Returns:
            Integer array of 0/1 class labels at the reduced scale.
        """
        count = int(superpixels.max()) + 1
        sums = np.bincount(superpixels.ravel(), weights=difference_map.ravel(),
                           minlength=count)
        sizes = np.bincount(superpixels.ravel(), minlength=count)
        means = sums / np.maximum(sizes, 1)

        if float(np.ptp(means)) <= 0.0:
            return np.zeros(difference_map.shape, dtype=np.int64)
        # ENHANCEMENT 1: the cut comes from a THREE-class threshold and the
        # ghost is the lowest band. A two-class cut spends itself separating
        # the textured-background tail and leaves the ghost inside a majority
        # class. See constants.SEGMENTATION_CLASS_COUNT for the measurement.
        return two_class_split_by_threshold(
            means, GhostDetector._lowest_band_threshold(means))[superpixels]

    @staticmethod
    def _lowest_band_threshold(means: np.ndarray) -> float:
        """Find the cut below which superpixels form the lowest of three bands.

        Args:
            means: 1-D array of per-superpixel mean difference values.

        Returns:
            The lowest multi-Otsu cut, falling back to the two-class cut when
            the values cannot support three distinct classes.
        """
        try:
            cuts = threshold_multiotsu(means,
                                       classes=constants.SEGMENTATION_CLASS_COUNT)
        except ValueError:
            # Too few distinct values for three classes; the two-class cut is
            # the only one defined, and the validity test still guards it.
            return float(threshold_otsu(means))
        return float(cuts[0])

    @staticmethod
    def is_valid_ghost_candidate(labels: np.ndarray) -> bool:
        """Check that class-0 looks like a pasted region rather than noise.

        Stands in for SE-MinCut's spatial regularisation, which this corpus
        has no available implementation of. A genuine ghost is one contiguous
        minority region; thresholded fractal noise is scattered confetti, and
        a uniform map drifting wholesale across the split point produces a
        "ghost" covering most of the frame. Both are rejected here.

        Args:
            labels: Integer 0/1 class labels at full resolution.

        Returns:
            True when class-0 is coherent enough and small enough to be a
            plausible tampered region.
        """
        ghost_mask = (labels == 0).astype(np.uint8)
        ghost_area = int(ghost_mask.sum())
        if ghost_area < constants.MINIMUM_CLASS_PIXEL_COUNT:
            return False

        area_fraction = ghost_area / float(labels.size)
        if not (constants.MINIMUM_GHOST_AREA_FRACTION <= area_fraction
                <= constants.MAXIMUM_GHOST_AREA_FRACTION):
            return False

        component_count, _, stats, _ = cv2.connectedComponentsWithStats(
            ghost_mask, connectivity=constants.CONNECTED_COMPONENT_CONNECTIVITY)
        if component_count <= 1:
            return False
        # Label 0 is the background component, so the foreground starts at 1.
        largest = float(stats[1:, cv2.CC_STAT_AREA].max())
        coherence = largest / float(ghost_area)
        return coherence >= constants.MINIMUM_GHOST_SPATIAL_COHERENCE

    def bhattacharyya_distance(self, values: np.ndarray,
                               labels: np.ndarray) -> float:
        """Measure how distinguishable the two segmented classes are.

        Args:
            values: Float64 difference-map pixel values.
            labels: Integer 0/1 class labels of the same shape.

        Returns:
            Bhattacharyya distance B, or 0.0 when either class is degenerate.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: B = (1/2)*ln[ (sigma0^2 + sigma1^2) / (2*sigma0*sigma1) ]
        #              + (mu0 - mu1)^2 / ( 4*(sigma0^2 + sigma1^2) )
        # Variables: mu0, sigma0^2 = mean and variance of class-0 (the ghost
        #   region candidate) pixel values; mu1, sigma1^2 = the same for
        #   class-1 (the rest of the image).
        # Source: Azarian-Pour 2016, Eq. 10.
        # Expected range: >= 0, growing with class separability; compared
        #   against Th = 0.19 after maximisation over all (q2, dx, dy).
        # ──────────────────────────────────────────────────────────
        if not self.is_valid_ghost_candidate(labels):
            return 0.0
        statistics = self._class_statistics(values, labels)
        if statistics is None:
            return 0.0
        return self._bhattacharyya_from_statistics(*statistics)

    @staticmethod
    def _class_statistics(values: np.ndarray, labels: np.ndarray):
        """Extract each class's mean and spread, or None if either is degenerate.

        Args:
            values: Float64 difference-map pixel values.
            labels: Integer 0/1 class labels of the same shape.

        Returns:
            Tuple of (mean0, sigma0, mean1, sigma1), or None when a class is
            too small or has no spread for Eq. 10 to be defined on.
        """
        ghost_values = values[labels == 0]
        rest_values = values[labels == 1]
        if (ghost_values.size < constants.MINIMUM_CLASS_PIXEL_COUNT
                or rest_values.size < constants.MINIMUM_CLASS_PIXEL_COUNT):
            return None

        mean_zero, sigma_zero = class_mean_and_standard_deviation(ghost_values)
        mean_one, sigma_one = class_mean_and_standard_deviation(rest_values)
        floor = constants.MINIMUM_CLASS_STANDARD_DEVIATION
        if sigma_zero < floor or sigma_one < floor:
            return None
        return mean_zero, sigma_zero, mean_one, sigma_one

    @staticmethod
    def _bhattacharyya_from_statistics(mean_zero: float, sigma_zero: float,
                                       mean_one: float, sigma_one: float) -> float:
        """Evaluate Eq. 10 from two classes' means and standard deviations.

        Args:
            mean_zero: Class-0 mean.
            sigma_zero: Class-0 standard deviation.
            mean_one: Class-1 mean.
            sigma_one: Class-1 standard deviation.

        Returns:
            The Bhattacharyya distance B.
        """
        variance_sum = sigma_zero ** 2 + sigma_one ** 2
        spread_term = constants.BHATTACHARYYA_SPREAD_COEFFICIENT * np.log(
            variance_sum / (constants.BHATTACHARYYA_VARIANCE_PRODUCT_FACTOR
                            * sigma_zero * sigma_one))
        mean_term = ((mean_zero - mean_one) ** 2
                     / (constants.BHATTACHARYYA_MEAN_TERM_DIVISOR * variance_sum))
        return float(spread_term + mean_term)

    def _sweep_one_shift(self, image_rgb: np.ndarray, shift_x: int,
                         shift_y: int, quality_factors: list) -> list:
        """Evaluate every candidate quality at one grid shift.

        Args:
            image_rgb: uint8 H x W x 3 RGB image.
            shift_x: Horizontal grid shift, 0..7.
            shift_y: Vertical grid shift, 0..7.
            quality_factors: Candidate q2 values to sweep.

        Returns:
            List of GhostCandidate, one per candidate quality.
        """
        maps = self._normalised_maps(image_rgb, shift_x, shift_y,
                                     quality_factors)
        candidates = []
        for index, quality in enumerate(quality_factors):
            difference_map = maps[index]
            if difference_map.size == 0:
                continue
            labels = self.segment_difference_map(difference_map)
            distance = self.bhattacharyya_distance(difference_map, labels)
            # ENHANCEMENT 2: a separable split is not yet a ghost. The SKILL
            # identifies the ghost as a LOCAL MINIMUM in q2, so a candidate
            # that is merely separable at this quality - and no lower here
            # than at the qualities either side of it - is discarded.
            if distance > 0.0 and not self._is_quality_local_minimum(
                    maps, index, labels):
                distance = 0.0
            candidates.append(GhostCandidate(
                quality_factor=quality, shift_x=shift_x, shift_y=shift_y,
                bhattacharyya_distance=distance,
                mask=self._place_mask(labels, image_rgb.shape[:2])))
        return candidates

    def _normalised_maps(self, image_rgb: np.ndarray, shift_x: int,
                         shift_y: int, quality_factors: list) -> list:
        """Build the analysis-region difference map for every candidate quality.

        Args:
            image_rgb: uint8 H x W x 3 RGB image.
            shift_x: Horizontal grid shift, 0..7.
            shift_y: Vertical grid shift, 0..7.
            quality_factors: Candidate q2 values to sweep.

        Returns:
            List of float64 maps, one per candidate quality, each already
            cropped to the analysis region.
        """
        shifted = self.preprocessor.shift_image(image_rgb, shift_x, shift_y)
        energies = np.stack([
            self.difference_energy(shifted, self.preprocessor.recompress(
                shifted, quality))
            for quality in quality_factors])
        normalised = self.normalise_across_qualities(energies)
        return [self._crop_to_analysis_region(normalised[index], shift_x,
                                              shift_y, image_rgb.shape[:2])
                for index in range(len(quality_factors))]

    @staticmethod
    def _is_quality_local_minimum(maps: list, index: int,
                                  labels: np.ndarray) -> bool:
        """Check the candidate region dips at this quality relative to its neighbours.

        SKILL B1: "A genuinely double-quantized region shows a local minimum
        in d at q2 equal to its true original quality q0." The comparison is
        made over the candidate's OWN class-0 pixels, so it asks whether that
        region turns over here rather than whether the frame as a whole does.

        Args:
            maps: The analysis-region difference map for every swept quality.
            index: Position of the candidate quality within that list.
            labels: Integer 0/1 class labels for the candidate.

        Returns:
            True when class-0's mean is strictly lower here than at the
            qualities either side; False at the sweep endpoints, where a local
            minimum is undefined for want of a neighbour.
        """
        if not constants.REQUIRE_GHOST_LOCAL_MINIMUM:
            return True
        offset = constants.LOCAL_MINIMUM_NEIGHBOUR_OFFSET
        if index - offset < 0 or index + offset >= len(maps):
            return False
        selection = (labels == 0)
        if not selection.any():
            return False
        before = float(maps[index - offset][selection].mean())
        here = float(maps[index][selection].mean())
        after = float(maps[index + offset][selection].mean())
        return here < before and here < after

    @staticmethod
    def _crop_to_analysis_region(difference_map: np.ndarray, shift_x: int,
                                 shift_y: int, original_shape: tuple) -> np.ndarray:
        """Drop the zero-padding and its smoothing bleed from a difference map.

        The Eq. 8 grid shift pads the image so its content moves off the 8x8
        DCT grid; that padding is not content. Analysed, its border shows up
        as a coherent minority region and is mistaken for a ghost - see
        constants.ANALYSIS_BORDER_MARGIN for the measurement.

        Args:
            difference_map: Float64 map in the padded frame.
            shift_x: Horizontal pad that was applied.
            shift_y: Vertical pad that was applied.
            original_shape: (height, width) of the unpadded image.

        Returns:
            Float64 map covering only the interior of the original content.
        """
        margin = constants.ANALYSIS_BORDER_MARGIN
        top = shift_y + margin
        left = shift_x + margin
        bottom = shift_y + original_shape[0] - margin
        right = shift_x + original_shape[1] - margin
        if bottom <= top or right <= left:
            return np.empty((0, 0), dtype=np.float64)
        return difference_map[top:bottom, left:right]

    @staticmethod
    def _place_mask(labels: np.ndarray, original_shape: tuple) -> np.ndarray:
        """Insert an analysis-region label map back into full image coordinates.

        Args:
            labels: Integer 0/1 labels over the analysis region.
            original_shape: (height, width) of the unpadded image.

        Returns:
            uint8 binary mask marking class-0 (ghost) in original coordinates.
        """
        margin = constants.ANALYSIS_BORDER_MARGIN
        mask = np.zeros(original_shape, dtype=np.uint8)
        mask[margin:margin + labels.shape[0],
             margin:margin + labels.shape[1]] = (labels == 0).astype(np.uint8)
        return mask

    def compute(self, image_rgb: np.ndarray, quality_factors: list,
                grid_shifts: list) -> GhostResult:
        """Run the ghost sweep and take the most separable combination.

        Args:
            image_rgb: uint8 H x W x 3 RGB image.
            quality_factors: Candidate q2 values to sweep.
            grid_shifts: (dx, dy) pairs to sweep.

        Returns:
            GhostResult holding D_max and its winning candidate.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: take the parameter triple (q2_m, dx_m, dy_m) that
        #   MAXIMIZES B; call that maximum D_max; classify as forged if
        #   D_max > Th, with Th = 0.19.
        # Variables: q2_m is reported as the tampered region's estimated
        #   quality factor and (dx_m, dy_m) as the detected DCT grid
        #   misalignment.
        # Source: Azarian-Pour 2016, Step 4 (automatic decision).
        # Expected range: D_max >= 0, unbounded above; Th = 0.19.
        # ──────────────────────────────────────────────────────────
        candidates = []
        for shift_x, shift_y in grid_shifts:
            candidates.extend(self._sweep_one_shift(image_rgb, shift_x, shift_y,
                                                    quality_factors))

        degenerate = sum(1 for entry in candidates
                         if entry.bhattacharyya_distance <= 0.0)
        best = (max(candidates, key=lambda entry: entry.bhattacharyya_distance)
                if candidates else None)
        return GhostResult(
            max_distance=best.bhattacharyya_distance if best else 0.0,
            best_candidate=best, combinations_evaluated=len(candidates),
            quality_factors_swept=len(quality_factors),
            grid_shifts_swept=len(grid_shifts),
            degenerate_segmentation_count=degenerate)


class ResamplingDetector:
    """Pipeline A: EM resampling-periodicity detector (AUXILIARY, unscorable).

    Reported but never scored: this SKILL prints no numeric value for the
    decision threshold rho_T, and states the gamma contrast function's
    "exact formula [is] not printed in the extracted text of this paper".
    """

    @staticmethod
    def build_predictor_system(window: np.ndarray) -> tuple:
        """Stack every pixel's neighbourhood row vector and its own value.

        Args:
            window: Float64 2-D analysis window.

        Returns:
            Tuple of (Y, y, map_shape): the neighbourhood design matrix with
            the self-predicting centre column removed, the centre-pixel
            vector, and the shape of the resulting p-map.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: y_i = P^{alpha,i} . y + eps_i                    [Eq. 7]
        #          P^{alpha,i} = 1^{1xK^2} . ( (1^{1xdim(y)} (x) alpha)
        #                                      (.) N^i )            [Eq. 8]
        # Variables: alpha = K^2 unobservable predictor weights with the
        #   centre weight alpha_{floor(K^2/2)} := 0 (a pixel does not predict
        #   itself); N^i = the neighbourhood indicator matrix selecting pixel
        #   i's K^2 local neighbours; K = 5 throughout the paper's main
        #   experiments. The centre column is dropped here rather than zeroed,
        #   because a zero column makes Y'WY singular in Eq. 11 while giving
        #   the identical constrained fit.
        # Source: Kirchner & Bohme 2008, Eq. 7-8.
        # Expected range: Y entries are pixel values; y is the centre pixels.
        # ──────────────────────────────────────────────────────────
        size = constants.PREDICTOR_NEIGHBOURHOOD_SIZE
        patches = sliding_window_view(window, (size, size))
        map_shape = patches.shape[:2]
        design = patches.reshape(-1, size * size)

        centre_index = (size * size) // 2
        centre_values = design[:, centre_index].copy()
        neighbours = np.delete(design, centre_index, axis=1)
        return neighbours, centre_values, map_shape

    @staticmethod
    def expectation_step(residuals: np.ndarray, sigma: float) -> np.ndarray:
        """Posterior probability that each pixel belongs to the resampled set.

        Args:
            residuals: Float64 prediction residuals eps_i.
            sigma: Current M1 residual standard deviation.

        Returns:
            Float64 array of posteriors p_i in [0, 1].
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: p_i = Prob(y_i in M1 | y_i)
        #   = Prob(y_i|y_i in M1) * Prob(y_i in M1)
        #     / sum_{k=1}^{2} Prob(y_i|y_i in M_k) * Prob(y_i in M_k)
        # Variables: M1 = the high-linear-dependence (interpolated) set, with
        #   y_i ~ N(P^{alpha,i} y, sigma_M1); M2 = the genuinely acquired set,
        #   with y_i ~ U(0, 2^l - 1). The prior is uniform,
        #   Prob(y_i in M1) = Prob(y_i in M2), so it cancels from the ratio.
        # Source: Kirchner & Bohme 2008, Eq. 10.
        # Expected range: [0, 1].
        # ──────────────────────────────────────────────────────────
        safe_sigma = max(sigma, constants.EM_MINIMUM_RESIDUAL_SIGMA)
        gaussian = (np.exp(-(residuals ** 2) / (2.0 * safe_sigma ** 2))
                    / (safe_sigma * np.sqrt(2.0 * np.pi)))
        uniform = 1.0 / constants.UNIFORM_MODEL_MAXIMUM_VALUE
        prior = constants.CLASS_PRIOR_PROBABILITY
        denominator = gaussian * prior + uniform * prior
        return np.where(denominator > 0.0, gaussian * prior / denominator, 0.0)

    @staticmethod
    def maximisation_step(design: np.ndarray, centre_values: np.ndarray,
                          posteriors: np.ndarray) -> tuple:
        """Re-estimate the predictor weights and the M1 residual spread.

        Args:
            design: Float64 (n_pixels, K^2 - 1) neighbourhood matrix.
            centre_values: Float64 centre-pixel vector.
            posteriors: Float64 per-pixel posteriors from the E-step.

        Returns:
            Tuple of (weights, sigma).
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: alpha = (Y'WY)^-1 Y'Wy                          [Eq. 11]
        #          sigma_M1 = sqrt( sum_i p_i*eps_i^2 / sum_i p_i ) [Eq. 12]
        # Variables: Y stacks all K^2-length local-neighbourhood row vectors
        #   for every pixel, W = diag(p) holds the E-step posteriors, and
        #   eps_i is pixel i's prediction residual. Eq. 11 is a weighted
        #   least-squares solve; Eq. 12 is a posterior-weighted RMS.
        # Source: Kirchner & Bohme 2008, Eq. 11-12.
        # Expected range: sigma_M1 >= 0.
        # ──────────────────────────────────────────────────────────
        weighted_design = design * posteriors[:, None]
        normal_matrix = design.T @ weighted_design
        normal_vector = weighted_design.T @ centre_values
        try:
            weights = np.linalg.solve(normal_matrix, normal_vector)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(normal_matrix, normal_vector, rcond=None)[0]

        residuals = centre_values - design @ weights
        posterior_mass = float(np.sum(posteriors))
        sigma = (float(np.sqrt(np.sum(posteriors * residuals ** 2)
                               / posterior_mass))
                 if posterior_mass > 0.0 else constants.EM_INITIAL_RESIDUAL_SIGMA)
        return weights, sigma

    def build_probability_map(self, window: np.ndarray) -> tuple:
        """Iterate EM to convergence and reshape the posteriors into a p-map.

        Args:
            window: Float64 2-D analysis window.

        Returns:
            Tuple of (p_map, iterations_run, converged).
        """
        design, centre_values, map_shape = self.build_predictor_system(window)
        weights = np.zeros(design.shape[1], dtype=np.float64)
        sigma = constants.EM_INITIAL_RESIDUAL_SIGMA
        posteriors = np.ones(centre_values.shape, dtype=np.float64)
        converged, iterations = False, 0

        for iterations in range(1, constants.EM_MAXIMUM_ITERATIONS + 1):
            previous_weights = weights
            weights, sigma = self.maximisation_step(design, centre_values,
                                                     posteriors)
            residuals = centre_values - design @ weights
            posteriors = self.expectation_step(residuals, sigma)
            # SKILL's own stopping recommendation: iterate until the change
            # in alpha falls below a small tolerance, e.g. 1e-4.
            if float(np.max(np.abs(weights - previous_weights))) \
                    < constants.EM_CONVERGENCE_TOLERANCE:
                converged = True
                break

        # p = diag(W), the converged posteriors reshaped to image dimensions.
        return posteriors.reshape(map_shape), iterations, converged

    @staticmethod
    def enhanced_spectrum(probability_map: np.ndarray) -> np.ndarray:
        """DFT the p-map and radially high-pass it to expose periodic peaks.

        SKILL step 6 pairs the radial high-pass with a gamma contrast
        function whose "exact formula [is] not printed in the extracted text
        of this paper". Only the high-pass is implemented; the enhancement
        below is this module's construction, not the paper's gamma.

        Args:
            probability_map: Float64 p-map in [0, 1].

        Returns:
            Float64 magnitude spectrum, DC-centred and high-pass weighted.
        """
        spectrum = np.fft.fftshift(np.fft.fft2(probability_map))
        magnitude = np.abs(spectrum)
        radius = radial_frequency_grid(magnitude.shape)
        # Suppress the DC/low-frequency bulk that would otherwise swamp the
        # periodic peaks the whole method depends on.
        weight = np.clip(
            (radius - constants.RADIAL_HIGHPASS_CUTOFF_FRACTION)
            / max(1.0 - constants.RADIAL_HIGHPASS_CUTOFF_FRACTION,
                  constants.RADIAL_HIGHPASS_DENOMINATOR_FLOOR),
            0.0, 1.0) ** constants.RADIAL_HIGHPASS_EXPONENT
        return magnitude * weight

    @staticmethod
    def synthetic_map(transform: np.ndarray, shape: tuple) -> np.ndarray:
        """Build the synthetic periodic map for one candidate transformation.

        Args:
            transform: 2x2 transformation matrix A.
            shape: (height, width) of the analysis window.

        Returns:
            Float64 array of per-position distance to the nearest lattice point.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: s_t^A = || A.nu_{m_s}^-1(i)
        #                    - floor( A.nu_{m_s}^-1(i) + (1/2).1^{2x1} ) ||
        # Variables: A = a candidate transformation matrix (scaling factor
        #   and/or rotation angle) from the search set; nu_{m_s}^-1(i) maps a
        #   linear index to its 2-D coordinate, implemented here as the
        #   analysis window's pixel coordinate grid. The expression is the
        #   distance from the transformed position to its nearest integer
        #   lattice point, which is periodic in the resampling factor.
        # Source: Kirchner & Bohme 2008, Eq. 18.
        # Expected range: [0, sqrt(2)/2].
        # ──────────────────────────────────────────────────────────
        rows, columns = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
        mapped_x = transform[0, 0] * columns + transform[0, 1] * rows
        mapped_y = transform[1, 0] * columns + transform[1, 1] * rows
        offset = constants.LATTICE_ROUNDING_OFFSET
        residual_x = mapped_x - np.floor(mapped_x + offset)
        residual_y = mapped_y - np.floor(mapped_y + offset)
        return np.sqrt(residual_x ** 2 + residual_y ** 2)

    @staticmethod
    def build_search_set(step: int) -> list:
        """Enumerate the candidate transformation matrices.

        Args:
            step: Take every `step`-th candidate; 1 gives the paper's full
                692-entry set.

        Returns:
            List of (kind, value, 2x2 matrix) tuples.
        """
        scalings = np.linspace(constants.SCALING_SEARCH_MINIMUM,
                               constants.SCALING_SEARCH_MAXIMUM,
                               constants.SCALING_SEARCH_COUNT)[::step]
        rotations = np.linspace(constants.ROTATION_SEARCH_MINIMUM,
                                constants.ROTATION_SEARCH_MAXIMUM,
                                constants.ROTATION_SEARCH_COUNT)[::step]
        candidates = [("scaling", float(factor),
                       np.array([[factor, 0.0], [0.0, factor]]))
                      for factor in scalings]
        candidates.extend(
            ("rotation", float(angle),
             np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]]))
            for angle in rotations)
        return candidates

    def decision_statistic(self, spectrum: np.ndarray, step: int) -> tuple:
        """Correlate the enhanced spectrum against the synthetic-map bank.

        Args:
            spectrum: Float64 enhanced magnitude spectrum of the p-map.
            step: Search-set stride; 1 gives the paper's full 692 maps.

        Returns:
            Tuple of (rho, best kind, best value, maps evaluated).
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: rho = max_{A in calA}
        #                || ( |gamma(DFT(p))| (.) |DFT(s^A)| )^{1/2} ||
        # Variables: p = the converged p-map, s^A = the synthetic periodic
        #   map for candidate transformation A, (.) = elementwise product,
        #   and the search set calA holds 692 candidates (601 scalings,
        #   91 rotations). gamma is the unprinted contrast function; the
        #   radial high-pass of enhanced_spectrum stands in for it.
        # Source: Kirchner & Bohme 2008, Eq. 19.
        # Expected range: rho >= 0, UNBOUNDED - it is compared against an
        #   empirically calibrated rho_T that this SKILL never gives a
        #   numeric value for, which is why rho cannot be scored here.
        # ──────────────────────────────────────────────────────────
        best_value, best_kind, best_parameter = 0.0, "", 0.0
        candidates = self.build_search_set(step)
        for kind, parameter, transform in candidates:
            synthetic = self.synthetic_map(transform, spectrum.shape)
            synthetic_spectrum = np.abs(np.fft.fftshift(np.fft.fft2(synthetic)))
            statistic = float(np.linalg.norm(
                np.sqrt(spectrum * synthetic_spectrum)))
            if statistic > best_value:
                best_value, best_kind, best_parameter = statistic, kind, parameter
        return best_value, best_kind, best_parameter, len(candidates)

    def compute(self, window: np.ndarray, step: int) -> ResamplingResult:
        """Run the EM p-map and its spectral detection statistic.

        Args:
            window: Float64 2-D analysis window.
            step: Search-set stride for the synthetic-map bank.

        Returns:
            ResamplingResult, always flagged as uncalibrated.
        """
        probability_map, iterations, converged = self.build_probability_map(window)
        spectrum = self.enhanced_spectrum(probability_map)
        statistic, kind, parameter, evaluated = self.decision_statistic(
            spectrum, step)
        return ResamplingResult(
            ran=True, probability_map=probability_map,
            decision_statistic=statistic, best_transform_kind=kind,
            best_transform_value=parameter, iterations_run=iterations,
            converged=converged, synthetic_maps_evaluated=evaluated,
            note="rho is UNCALIBRATED: this SKILL gives no numeric value for "
                 "the decision threshold rho_T, and the gamma contrast "
                 "function its spectrum enhancement calls for is not printed "
                 "in the source either. Reported as evidence only; it does "
                 "not contribute to raw_score.")
