"""Core mathematical computation for the JPEG-compression-artifact pipelines.

DoubleCompressionDetector (Pipeline B) is the sole score-driving computation
- see the SCOPE DECISION note at the top of constants.py. JpegHistoryIdentifier
(A.1) gates the engine, QuantizationStepEstimator (A.2) and
QualityFactorDetector (A.3) supply conditioning parameters and a confidence
input. Pipelines C and D are [ML - excluded] and absent by design.
"""

from __future__ import annotations

import io
import logging

import numpy as np
from PIL import Image
from scipy.signal import find_peaks

from . import constants
from .contracts import (DoubleCompressionResult, FrequencySpectrum,
                        JpegHistoryResult, QualityFactorResult,
                        QuantizationStepResult)
from .utils import (clip_to_unit_interval, integer_bin_histogram,
                    moving_average, trailing_minimum, unit_norm)

logger = logging.getLogger(__name__)


class JpegHistoryIdentifier:
    """Pipeline A.1: decides whether a bitmap was ever JPEG-compressed (gate)."""

    @staticmethod
    def pooled_ac_coefficients(coefficients: np.ndarray) -> np.ndarray:
        """Pool every AC coefficient across all frequencies and blocks.

        Args:
            coefficients: Block-DCT array of shape (n_blocks, 8, 8).

        Returns:
            1-D array of all AC coefficients (the DC term is excluded).
        """
        flattened = coefficients.reshape(coefficients.shape[0], -1)
        # Position 0 of the flattened 8x8 block is the DC term (0,0); the
        # SKILL's p_ac is the pdf of AC coefficients only.
        return flattened[:, 1:].ravel()

    @staticmethod
    def _region_counts(magnitudes: np.ndarray) -> tuple:
        """Count AC coefficients falling in each of Eq. 7's two regions.

        Args:
            magnitudes: Absolute values of the pooled AC coefficients.

        Returns:
            Tuple of (count in R1, count in R2).
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: R1 = (-1, +1),  R2 = (-2,-1) U (+1,+2)
        # Variables: the two disjoint regions of the real line over which
        #   Eq. 7 integrates the AC-coefficient pdf. Quantization pulls
        #   probability mass INTO R1 and OUT OF R2.
        # Source: Luo, Huang & Qiu 2010, Eq. 7 region definitions.
        # Expected range: non-negative integer counts.
        # ──────────────────────────────────────────────────────────
        in_region_one = int(np.count_nonzero(
            magnitudes < constants.REGION_R1_OUTER_BOUND))
        in_region_two = int(np.count_nonzero(
            (magnitudes > constants.REGION_R2_INNER_BOUND)
            & (magnitudes < constants.REGION_R2_OUTER_BOUND)))
        return in_region_one, in_region_two

    def compute_history_feature(self, coefficients: np.ndarray,
                                threshold: float) -> JpegHistoryResult:
        """Compute the JPEG-history feature s and compare it to its threshold.

        Args:
            coefficients: Block-DCT array of shape (n_blocks, 8, 8).
            threshold: Decision threshold t for s.

        Returns:
            JpegHistoryResult holding s and the resulting gate verdict.
        """
        magnitudes = np.abs(self.pooled_ac_coefficients(coefficients))
        in_region_one, in_region_two = self._region_counts(magnitudes)

        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: s = integral_{R1} p_ac(y) dy / integral_{R2} p_ac(y) dy
        # Variables: p_ac = empirical pdf of all AC coefficients of the test
        #   image, pooled across frequencies and blocks. Both integrals share
        #   that pdf, so their ratio is the ratio of the raw region counts.
        #   A JPEG-compressed image has a LARGER s than an uncompressed one.
        # Source: Luo, Huang & Qiu 2010, Eq. 7.
        # Expected range: s >= 0, unbounded above. NOTE: the paper's quoted
        #   thresholds (t ~ 0.29-0.38) are NOT on the scale this formula
        #   produces here - measured s is ~1.05 never-compressed and 26-2000
        #   compressed. The direction holds; the absolute cut does not
        #   transfer. See constants.DEFAULT_HISTORY_THRESHOLD.
        # ──────────────────────────────────────────────────────────
        history_feature = (float(in_region_one) / float(in_region_two)
                           if in_region_two > 0 else float("inf"))

        return JpegHistoryResult(
            history_feature=history_feature, threshold=threshold,
            is_jpeg_derived=history_feature > threshold,
            region_one_count=in_region_one, region_two_count=in_region_two)


class QuantizationStepEstimator:
    """Pipeline A.2: per-frequency quantization step (conditioning parameter)."""

    @staticmethod
    def rounded_absolute_histogram(frequency_values: np.ndarray) -> np.ndarray:
        """Histogram the absolute rounded coefficients of one frequency.

        Args:
            frequency_values: All blocks' coefficients at one DCT frequency.

        Returns:
            Array H where H[k] is the count of coefficients with |[d2]| == k.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: P([d2(i,j)] = d1'(i,j)) = P(eps(i,j) = 0)
        #            = integral_{-0.5}^{+0.5} p_eps(y) dy >= 91.50%
        # Variables: d2 = re-DCT'd coefficient of the decompressed image,
        #   d1' = true dequantized (step-multiple) coefficient, eps = the
        #   approximately Gaussian rounding error with variance 1/12. So
        #   simple rounding recovers d1' with probability >= 91.50%.
        # Source: Luo, Huang & Qiu 2010, Eq. 11.
        # Expected range: counts >= 0. Bin index k is |[d2]|, merging the
        #   symmetric +-k bins as the SKILL specifies.
        # ──────────────────────────────────────────────────────────
        rounded = np.abs(np.rint(frequency_values)).astype(np.int64)
        return np.bincount(rounded).astype(np.float64)

    @staticmethod
    def estimate_step(histogram: np.ndarray) -> int:
        """Estimate one frequency's quantization step from its histogram.

        Args:
            histogram: Array H where H[k] counts coefficients with |[d2]| == k.

        Returns:
            Estimated step q_hat, or 0 when the histogram cannot support an
            estimate (no bin at k >= 2 carries any mass).
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: if H(1)/H(0) > t AND H(1) > H_max  ->  q_hat = 1   [Eq.12]
        #          else q_hat = argmin_k ( k | H(k) = H_max, k >= 2 ) [Eq.13]
        # Variables: H = histogram of |[d2(i,j)]|; t = 0.3, tuned in the
        #   paper over [0.10, 0.35]; H_max = max_{k>=2} H(k) - Eq. 13 takes
        #   its argmin against that same H_max under the k>=2 restriction,
        #   and only this reading leaves both of Eq. 12's conditions
        #   non-redundant (see KNOWN_SKILL_AMBIGUITIES). Eq. 12 guards the
        #   "ghost at 1" false estimate caused by rounding-error spillover.
        # Source: Luo, Huang & Qiu 2010, Eq. 12-13.
        # Expected range: q_hat >= 1 (integer), or 0 when unusable.
        # ──────────────────────────────────────────────────────────
        minimum_step = constants.MINIMUM_QUANTIZATION_STEP
        if histogram.size <= minimum_step:
            return 0

        upper_bins = histogram[minimum_step:]
        maximum_count = float(upper_bins.max())
        if maximum_count <= 0.0:
            return 0

        zero_count, one_count = float(histogram[0]), float(histogram[1])
        ratio_exceeds = (zero_count > 0.0
                         and one_count / zero_count
                         > constants.GHOST_AT_ONE_RATIO_THRESHOLD)
        if ratio_exceeds and one_count > maximum_count:
            return 1

        return minimum_step + int(np.argmax(upper_bins == maximum_count))

    def compute(self, coefficients: np.ndarray) -> QuantizationStepResult:
        """Estimate the quantization step at every DCT frequency.

        Args:
            coefficients: Block-DCT array of shape (n_blocks, 8, 8).

        Returns:
            QuantizationStepResult with an 8x8 integer step map.
        """
        block = constants.DCT_BLOCK_SIZE
        steps = np.zeros((block, block), dtype=np.int64)
        for row in range(block):
            for column in range(block):
                # Eq. 12/13 select a bin by its count, which assumes a
                # zero-peaked population. DC carries block mean brightness
                # and violates that; see SKIP_DC_IN_STEP_ESTIMATION.
                if (constants.SKIP_DC_IN_STEP_ESTIMATION
                        and row == 0 and column == 0):
                    continue
                histogram = self.rounded_absolute_histogram(
                    coefficients[:, row, column])
                steps[row, column] = self.estimate_step(histogram)
        return QuantizationStepResult(
            steps=steps, usable_frequency_count=int(np.count_nonzero(steps)))


class QualityFactorDetector:
    """Pipeline A.3: quality-factor recovery by recompression search."""

    @staticmethod
    def standard_quantization_table(quality_factor: int) -> np.ndarray:
        """Build a candidate quantization table from the IJG base table.

        Args:
            quality_factor: Candidate QF in [1, 100].

        Returns:
            8x8 integer quantization table.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: Table_QF = floor(t * 50/QF + 0.5),        1 <= QF < 50
        #                   = floor(t * (2 - QF/50) + 0.5),  50 <= QF <= 100
        #          (values less than 1 are floored up to 1)
        # Variables: t = the IJG standard base luminance table, transcribed
        #   in constants.IJG_BASE_LUMINANCE_TABLE exactly as the SKILL prints
        #   it; QF = candidate quality factor.
        # Source: Luo, Huang & Qiu 2010, Eq. 26.
        # Expected range: integers >= 1. NOTE: Eq. 26's real arithmetic
        #   exceeds 255 for every QF below ~24, which a baseline 8-bit DQT
        #   marker cannot encode, and differs by +-1 from the integer
        #   arithmetic real encoders use at 35 of 100 quality factors
        #   (measured). This method implements Eq. 26 exactly as printed and
        #   is reported for transparency; the sweep below drives the encoder
        #   itself, because Eq. 24 scores by exact pixel identity.
        # ──────────────────────────────────────────────────────────
        base = constants.IJG_BASE_LUMINANCE_TABLE.astype(np.float64)
        if quality_factor < constants.QUALITY_FACTOR_PIVOT:
            scaled = base * (constants.TABLE_SCALE_NUMERATOR / quality_factor)
        else:
            scaled = base * (constants.TABLE_SCALE_OFFSET - quality_factor
                            / constants.TABLE_SCALE_NUMERATOR)
        floored = np.floor(scaled + constants.TABLE_ROUNDING_OFFSET)
        return np.maximum(floored,
                          constants.QUANTIZATION_VALUE_MINIMUM).astype(np.int64)

    @staticmethod
    def encoder_quantization_table(quality_factor: int) -> np.ndarray:
        """Build the table a real (libjpeg-family) encoder produces for a QF.

        This is Eq. 26 evaluated the way encoders actually evaluate it: the
        scale factor is truncated to an integer first, and the result is
        clamped to the 8-bit range a DQT marker can store. Measured to
        reproduce Pillow's own tables exactly for all QF in 1..100.

        Args:
            quality_factor: Candidate QF in [1, 100].

        Returns:
            8x8 integer quantization table.
        """
        base = constants.IJG_BASE_LUMINANCE_TABLE
        if quality_factor < constants.QUALITY_FACTOR_PIVOT:
            scale = (constants.LIBJPEG_LOW_QUALITY_SCALE_NUMERATOR
                    // quality_factor)
        else:
            scale = (constants.LIBJPEG_HIGH_QUALITY_SCALE_BASE
                    - constants.LIBJPEG_HIGH_QUALITY_SCALE_FACTOR
                    * quality_factor)
        scaled = ((base * scale + constants.LIBJPEG_TABLE_ROUNDING_OFFSET)
                 // constants.LIBJPEG_TABLE_DIVISOR)
        return np.clip(scaled, constants.QUANTIZATION_VALUE_MINIMUM,
                       constants.QUANTIZATION_VALUE_MAXIMUM)

    @staticmethod
    def pixel_match_ratio(first: np.ndarray, second: np.ndarray) -> float:
        """Fraction of positions where two images are exactly pixel-identical.

        Args:
            first: Reference image J1, uint8.
            second: Recompressed candidate J2, uint8, same shape.

        Returns:
            R in [0, 1].
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: R(J1,J2) = |E| / (M*N),
        #          E = {(x,y) : J1(x,y) = J2(x,y), 1<=x<=M, 1<=y<=N}
        # Variables: J1 = the decompressed bitmap under test, J2 = J1
        #   recompressed at a candidate table, M*N = pixel count. Equality
        #   is EXACT, not approximate: the correct table reproduces the same
        #   rounding/truncation behaviour that generated J1.
        # Source: Luo, Huang & Qiu 2010, Eq. 24.
        # Expected range: [0, 1]; peaks at the true table.
        # ──────────────────────────────────────────────────────────
        return float(np.mean(first == second))

    @staticmethod
    def recompress(luminance_uint8: np.ndarray, quality_factor: int) -> np.ndarray:
        """Encode an image at a candidate quality factor and decode it back.

        Args:
            luminance_uint8: Single-channel uint8 image.
            quality_factor: Candidate QF in [1, 100].

        Returns:
            The decoded uint8 image after the round trip.
        """
        buffer = io.BytesIO()
        Image.fromarray(luminance_uint8, mode="L").save(
            buffer, format="JPEG", quality=int(quality_factor))
        buffer.seek(0)
        return np.array(Image.open(buffer).convert("L"))

    def detect(self, luminance: np.ndarray) -> QualityFactorResult:
        """Recover the quality factor by maximising the exact-pixel-match rate.

        Args:
            luminance: Float64 luminance array of the image under test.

        Returns:
            QualityFactorResult with the winning QF and its match ratio.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: Q_hat_F = argmax_i ( R(J1, J2(i)) ),  i = 1,...,100
        # Variables: J2(i) = J1 recompressed with the candidate table for
        #   quality factor i. The correct table produces the highest
        #   exact-pixel-match rate.
        # Source: Luo, Huang & Qiu 2010, Eq. 25.
        # Expected range: Q_hat_F in {1,...,100}.
        # ──────────────────────────────────────────────────────────
        reference = np.clip(luminance, constants.PIXEL_VALUE_MINIMUM,
                            constants.PIXEL_VALUE_MAXIMUM).astype(np.uint8)
        ratios = np.array([
            self.pixel_match_ratio(reference, self.recompress(reference, candidate))
            for candidate in range(constants.QUALITY_FACTOR_MINIMUM,
                                   constants.QUALITY_FACTOR_MAXIMUM + 1)])

        best_index = int(np.argmax(ratios))
        ordered = np.sort(ratios)[::-1]
        return QualityFactorResult(
            quality_factor=best_index + constants.QUALITY_FACTOR_MINIMUM,
            pixel_match_ratio=float(ratios[best_index]),
            runner_up_match_ratio=float(ordered[1]) if ordered.size > 1 else 0.0,
            sweep_ran=True)


class DoubleCompressionDetector:
    """Pipeline B: double-quantization Fourier periodicity (SCORE-DRIVING).

    The underlying model is Mahdian & Saic 2009 Eq. 4,
    F^{Q_beta}(u,v) = round(F^{Q_alpha}(u,v) * Q_alpha(u,v) / Q_beta(u,v)),
    which is descriptive: neither quantization matrix is known for an image
    under test. What is measurable is its consequence - re-binning a
    coefficient population quantized at one step into bins of another is a
    non-injective periodic remapping, so the histogram acquires periodicity
    that shows up as peaks in its Fourier magnitude.
    """

    @staticmethod
    def zero_mean_histogram(frequency_values: np.ndarray) -> np.ndarray:
        """Build one frequency's integer-binned, mean-removed histogram.

        SKILL Implementation Notes require integer-valued bins ("do not apply
        continuous/KDE binning, which would blur exactly the periodic bin
        structure both methods depend on"). "Zero-mean" is therefore read as
        removing the histogram's own mean bin count - shifting the
        coefficients instead would break the integer binning the same note
        mandates. Removing that mean suppresses the f=0 term that would
        otherwise dominate the magnitude spectrum.

        Args:
            frequency_values: All blocks' coefficients at one DCT frequency.

        Returns:
            Mean-removed histogram counts.
        """
        _, counts = integer_bin_histogram(frequency_values)
        if counts.size == 0:
            return counts
        return counts - float(np.mean(counts))

    @staticmethod
    def normalised_spectrum(histogram: np.ndarray) -> np.ndarray:
        """Magnitude of the histogram's 1-D FFT, normalised to unit length.

        Args:
            histogram: Zero-mean histogram counts for one frequency.

        Returns:
            Unit-L2 magnitude spectrum over the independent (positive-
            frequency) half.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: compute the magnitude of the 1-D FFT of each frequency's
        #   zero-mean coefficient histogram: |H_1|,...,|H_10| (normalized to
        #   unit length). A doubly-quantized position exhibits specific
        #   periodic peaks; a singly-quantized one shows a smooth decaying
        #   trend instead.
        # Variables: H_i = histogram-FFT magnitude for the i-th of the 10
        #   selected low-frequency DCT positions.
        # Source: Mahdian & Saic 2009, step 3.
        # Expected range: non-negative, unit L2 norm, so any single
        #   component - and hence any peak prominence - is bounded by 1.
        # ──────────────────────────────────────────────────────────
        if histogram.size == 0:
            return histogram
        magnitude = np.abs(np.fft.fft(histogram))
        half_length = max(1, magnitude.size // constants.SPECTRUM_HALF_DIVISOR)
        return unit_norm(magnitude[:half_length], constants.SPECTRUM_NORM_FLOOR)

    @staticmethod
    def remove_trend(spectrum: np.ndarray, window_length: int) -> np.ndarray:
        """Subtract the trailing local minimum to flatten the decaying trend.

        Args:
            spectrum: Unit-norm magnitude spectrum |H_i|, already smoothed.
            window_length: n, the trailing-window length.

        Returns:
            Trend-removed spectrum |H_i~|.
        """
        # ── SKILL VERIFICATION ──────────────────────────────────────
        # Formula: |H_i~|(f) = |H_i|(f) - M_i(f)                    [Eq. 5]
        #          M_i(f) = min{ |H_i|(f), ..., |H_i|(f-n) }
        # Variables: n = trailing minimum-filter length. The SKILL states n
        #   is "determined per quantization step in a training process - the
        #   paper does not give a single fixed default value"; see
        #   constants.TREND_REMOVAL_WINDOW_LENGTH. This local-minimum
        #   subtraction is the paper's stated improvement over Popescu &
        #   Farid's generalized-Laplace curve fit, which it cites as heavier
        #   and more false-positive-prone on real image histograms.
        # Source: Mahdian & Saic 2009, Eq. 5.
        # Expected range: >= 0, since M_i(f) <= |H_i|(f) by construction.
        # ──────────────────────────────────────────────────────────
        return spectrum - trailing_minimum(spectrum, window_length)

    @staticmethod
    def extract_peaks(spectrum: np.ndarray) -> tuple:
        """Locate local peaks of a trend-removed spectrum and their prominences.

        SKILL step 6 replaces the paper's [ML-excluded] Gaussian-kernel SVM
        with a direct peak-prominence threshold; the prominences returned
        here are that substitute's raw material.

        Args:
            spectrum: Trend-removed spectrum |H_i~|.

        Returns:
            Tuple of (peak_positions, peak_prominences).
        """
        if spectrum.size < constants.DCT_BLOCK_SIZE:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        positions, properties = find_peaks(spectrum, prominence=0.0)
        return positions, properties.get("prominences",
                                         np.array([], dtype=np.float64))

    @staticmethod
    def to_quantized_domain(values: np.ndarray, step: int) -> np.ndarray:
        """Convert dequantized coefficients back to their integer domain.

        Eq. 4 is written over the integer bitstream coefficients, and a
        dequantized coefficient is a multiple of its quantization step, so
        histogramming it directly measures the step rather than the
        double-quantization artifact - see the extended note at
        constants.NORMALISE_HISTOGRAM_BY_QUANTIZATION_STEP for the measured
        consequence (the score separates in the wrong direction without this).

        Args:
            values: One frequency's dequantized coefficients across blocks.
            step: That frequency's quantization step q; 0 or 1 leaves the
                values untouched.

        Returns:
            Coefficients divided by the step.
        """
        if step <= 1:
            return values
        return values / float(step)

    def analyse_frequency(self, coefficients: np.ndarray, frequency: tuple,
                          ordinal: int, window_length: int,
                          step: int) -> FrequencySpectrum:
        """Run the full Pipeline-B chain for one DCT frequency position.

        Args:
            coefficients: Block-DCT array of shape (n_blocks, 8, 8).
            frequency: (u, v) position within the 8x8 grid.
            ordinal: 1-based index i matching the SKILL's H_1 ... H_10.
            window_length: n, the Eq. 5 trailing-window length.
            step: This frequency's quantization step q, used to recover the
                integer domain Eq. 4 operates in. 0 marks it unknown, which
                excludes the frequency.

        Returns:
            FrequencySpectrum describing this position's evidence.
        """
        raw = coefficients[:, frequency[0], frequency[1]]
        values = self.to_quantized_domain(raw, step)
        zero_fraction = float(np.mean(np.rint(values) == 0)) if values.size else 1.0
        excluded = (step <= 0
                    or zero_fraction
                    >= constants.ZERO_COEFFICIENT_EXCLUSION_FRACTION)

        spectrum = self.normalised_spectrum(self.zero_mean_histogram(values))
        if not excluded and spectrum.size:
            spectrum = self._flatten_spectrum(spectrum, ordinal, window_length)
        positions, prominences = (self.extract_peaks(spectrum)
                                  if not excluded
                                  else (np.array([]), np.array([])))

        return FrequencySpectrum(
            frequency=frequency, ordinal=ordinal, spectrum=spectrum,
            peak_positions=positions, peak_prominences=prominences,
            strongest_prominence=float(prominences.max()) if prominences.size else 0.0,
            was_excluded=excluded, zero_coefficient_fraction=zero_fraction)

    @staticmethod
    def _flatten_spectrum(spectrum: np.ndarray, ordinal: int,
                          window_length: int) -> np.ndarray:
        """Smooth and detrend a spectrum, leaving the DC position untouched.

        SKILL step 4: the averaging filter applies to i = 2,...,10, and
        "i=1/DC is treated as a special case because it alone shows a clear
        peak under double compression rather than a decaying trend under
        single compression" - so H_1 has no trend to remove and is used as-is.

        Args:
            spectrum: Unit-norm magnitude spectrum |H_i|.
            ordinal: 1-based index i.
            window_length: n, the Eq. 5 trailing-window length.

        Returns:
            The spectrum ready for peak extraction.
        """
        if ordinal == constants.DC_FREQUENCY_ORDINAL + 1:
            return spectrum
        smoothed = moving_average(spectrum, constants.AVERAGING_FILTER_LENGTH)
        return DoubleCompressionDetector.remove_trend(smoothed, window_length)

    def compute(self, coefficients: np.ndarray, window_length: int,
                quantization_table: np.ndarray) -> DoubleCompressionResult:
        """Run Pipeline B across all 10 selected low-frequency positions.

        Args:
            coefficients: Block-DCT array of shape (n_blocks, 8, 8).
            window_length: n, the Eq. 5 trailing-window length.
            quantization_table: 8x8 steps used to recover the integer
                coefficient domain; entries of 0 exclude their frequency.

        Returns:
            DoubleCompressionResult holding the per-frequency spectra and the
            aggregate double-compression score.
        """
        spectra = [
            self.analyse_frequency(coefficients, frequency, ordinal,
                                   window_length,
                                   int(quantization_table[frequency[0],
                                                          frequency[1]]))
            for ordinal, frequency in enumerate(
                constants.DOUBLE_QUANTIZATION_FREQUENCIES, start=1)]

        usable = [entry for entry in spectra if not entry.was_excluded]
        peak_bearing = [entry for entry in usable
                        if entry.strongest_prominence
                        >= constants.PEAK_PROMINENCE_THRESHOLD]

        return DoubleCompressionResult(
            spectra=spectra, aggregate_score=self._aggregate(usable),
            usable_frequency_count=len(usable),
            peak_bearing_frequency_count=len(peak_bearing))

    @staticmethod
    def _aggregate(usable_spectra: list) -> float:
        """Reduce per-frequency peak prominences to one [0, 1] scalar.

        SKILL Output: Pipeline B yields "per-frequency peak-prominence
        values, aggregable into a scalar double-compression score" - no
        aggregation formula is given. The mean of each usable frequency's
        strongest prominence is used, which integrates evidence across the
        10 positions rather than trusting any single one; the SKILL notes
        the training-free substitute otherwise "sacrifices the SVM's ability
        to integrate evidence across all 10 frequencies simultaneously".
        [ENGINEERING]

        Args:
            usable_spectra: FrequencySpectrum entries that were not excluded.

        Returns:
            Aggregate score in [0, 1].
        """
        if not usable_spectra:
            return 0.0
        strongest = [entry.strongest_prominence for entry in usable_spectra]
        return clip_to_unit_interval(float(np.mean(strongest)))
