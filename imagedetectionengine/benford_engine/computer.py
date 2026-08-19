"""Core mathematical computation for the Benford engine.

Every mathematical operation in this module carries a SKILL VERIFICATION block
naming the exact formula, its variables, its source paper and its expected
output range. That block is the audit mechanism: an implementation may only be
trusted to the extent each block matches the SKILL file line for line.

This module contains no I/O, no scoring policy and no presentation logic.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from functools import partial
from scipy.fft import dctn
from scipy.optimize import curve_fit
from scipy.stats import entropy

from . import constants
from .contracts import BenfordComputation, DivergenceSample
from .utils import (compute_probability_mass_function,
                    extract_first_significant_digit,
                    generate_zigzag_indices,
                    load_standard_jpeg_quantization_table,
                    safe_normalise_distribution)

logger = logging.getLogger(__name__)


def generalized_benford_probability(digits: np.ndarray,
                                    beta: float,
                                    gamma: float,
                                    delta: float,
                                    base: float) -> np.ndarray:
    """Evaluate the generalized Benford model at the given digits.

    Module-level and pure so scipy.optimize can bind it directly.

    Args:
        digits: Digit values to evaluate at, typically 1..base-1.
        beta: Outer scale factor.
        gamma: Additive shape parameter inside the reciprocal.
        delta: Exponent applied to the digit.
        base: Logarithm base.

    Returns:
        Model probabilities at each digit.
    """
    # ── SKILL VERIFICATION ──────────────────────────────
    # Formula: p(d) = beta * log_b( 1 + 1 / (gamma + d**delta) )
    # Variables: d = leading digit in {1..b-1}; beta = scale factor;
    #            gamma, delta = shape parameters of the logarithmic curve;
    #            b = digit base.
    # Source: Bonettini et al. 2021 (ICPR), Eq. 5 / 11 - SKILL family A step 4.
    # Expected range: strictly positive; near log_b(2) ~ 0.30 at d=1 for b=10
    #            with beta=1, decaying monotonically as d grows.
    # Note: log_b(x) is computed as log(x)/log(b) since numpy has no
    #            arbitrary-base logarithm.
    # ────────────────────────────────────────────────────
    inner_reciprocal = 1.0 / (gamma + np.power(digits.astype(np.float64), delta))
    return beta * (np.log1p(inner_reciprocal) / np.log(float(base)))


def benford_model_for_base(base: float,
                           digits: np.ndarray,
                           beta: float,
                           gamma: float,
                           delta: float) -> np.ndarray:
    """Adapter putting `base` first so it can be bound with functools.partial.

    scipy.optimize.curve_fit requires a callable of the form f(x, *parameters).
    Binding the base through partial keeps the model a named module-level
    function instead of an undocumented closure.

    Args:
        base: Logarithm base to bind.
        digits: Digit values to evaluate at.
        beta: Outer scale factor.
        gamma: Additive shape parameter.
        delta: Exponent applied to the digit.

    Returns:
        Model probabilities at each digit.
    """
    return generalized_benford_probability(digits, beta, gamma, delta, base)


class BenfordComputer:
    """Runs the block DCT and the (base, frequency, quality factor) sweep."""

    def __init__(self,
                 digit_bases: tuple[int, ...] = constants.DEFAULT_DIGIT_BASES,
                 zigzag_frequency_indices: tuple[int, ...] =
                 constants.DEFAULT_ZIGZAG_FREQUENCY_INDICES,
                 quality_factor_sweep: tuple[int, ...] =
                 constants.DEFAULT_QUALITY_FACTOR_SWEEP) -> None:
        """Configure the sweep grid.

        Args:
            digit_bases: Digit bases to evaluate.
            zigzag_frequency_indices: Zig-zag indices, DC (0) excluded.
            quality_factor_sweep: JPEG quality factors supplying quantization
                tables.

        Raises:
            ValueError: If any grid axis is empty or outside its validated range.
        """
        self._validate_grid(digit_bases, zigzag_frequency_indices,
                            quality_factor_sweep)
        self.digit_bases = digit_bases
        self.zigzag_frequency_indices = zigzag_frequency_indices
        self.quality_factor_sweep = quality_factor_sweep
        self.zigzag_coordinates = generate_zigzag_indices(constants.DCT_BLOCK_SIZE)

    def compute(self,
                blocks: np.ndarray,
                estimated_quality_factor: Optional[float] = None
                ) -> BenfordComputation:
        """Run the whole mathematical pipeline over the prepared blocks.

        Args:
            blocks: Float array of shape (block_count, 8, 8).
            estimated_quality_factor: The image's own quality factor. Used both
                to pick the comparable chi-square and, per ENHANCEMENT 2, to
                add the image's own quantization to the sweep.

        Returns:
            BenfordComputation holding every grid sample and the aggregates.
        """
        dct_coefficients = self._forward_dct(blocks)
        native = self._native_quality_factor(estimated_quality_factor)
        sweep = self._effective_sweep(native)
        samples, clamp_events = self._sweep_grid(dct_coefficients, sweep,
                                                 native)

        scored = [sample for sample in samples if not sample.is_finer_than_native]
        result = BenfordComputation(
            samples=samples,
            total_block_count=int(blocks.shape[0]),
            digit_clamp_event_count=clamp_events,
            skipped_configuration_count=self._grid_size(sweep) - len(samples),
            excluded_configuration_count=len(samples) - len(scored),
            residual_lattice_configuration_count=sum(
                1 for sample in scored if sample.has_residual_lattice),
            scored_configuration_count=len(scored),
            native_quality_factor=native,
        )
        if not scored:
            logger.warning("every sweep configuration applied a step finer than "
                           "the image's own quantization (%d cells); nothing "
                           "could be scored", len(samples))
            return result
        return self._aggregate(result, scored, estimated_quality_factor)

    def _aggregate(self,
                   result: BenfordComputation,
                   scored: list[DivergenceSample],
                   estimated_quality_factor: Optional[float]
                   ) -> BenfordComputation:
        """Reduce the scored cells to the grid aggregates, in place.

        Args:
            result: Computation being filled in.
            scored: Cells that survived exclusion.
            estimated_quality_factor: Selects the comparable chi-square.

        Returns:
            The same result, with its aggregate fields populated.
        """
        divergences = np.array([sample.divergence for sample in scored])
        result.mean_divergence = float(np.mean(divergences))
        result.max_divergence = float(np.max(divergences))
        result.overall_zero_coefficient_rate = float(
            np.mean([sample.zero_coefficient_rate for sample in scored])
        )
        result.representative_chi_square = self._select_representative_chi_square(
            scored, estimated_quality_factor
        )
        return result

    @staticmethod
    def _native_quality_factor(
            estimated_quality_factor: Optional[float]) -> Optional[int]:
        """Round the metadata quality factor into an encoder-valid integer.

        Args:
            estimated_quality_factor: Quality factor from metadata, or None.

        Returns:
            Integer quality factor inside the encoder's range, or None.
        """
        if estimated_quality_factor is None or \
                not np.isfinite(estimated_quality_factor):
            return None
        rounded = int(round(float(estimated_quality_factor)))
        if not (constants.MINIMUM_JPEG_QUALITY_FACTOR <= rounded
                <= constants.MAXIMUM_JPEG_QUALITY_FACTOR):
            return None
        return rounded

    def _effective_sweep(self, native: Optional[int]) -> tuple[int, ...]:
        """Add the image's own quality factor to the corpus sweep.

        ENHANCEMENT 2, forced by diagnostic testing. The corpus sweep
        {80, 85, 90, 95, 100} is Bonettini et al.'s, chosen for a corpus of
        UNCOMPRESSED images where that quantization is the image's first. On an
        image that is already JPEG, every one of those factors is a SECOND
        quantization at a mismatched step. Measured on campic.jpeg: divergence
        0.1143 at its own native QF74 against 2.60-3.83 at every swept factor.
        Including the native factor guarantees the sweep contains at least one
        cell where the engine is reading the codec's own coefficients.

        This is what the SKILL's "Quality-factor conditioning" note asks for:
        "any deployed threshold must be conditioned on the image's estimated
        background quality factor ... rather than applied as one fixed global
        cutoff."

        Args:
            native: The image's own quality factor, or None when unknown.

        Returns:
            Sweep quality factors in ascending order.
        """
        if native is None:
            return tuple(sorted(self.quality_factor_sweep))
        return tuple(sorted(set(self.quality_factor_sweep) | {native}))

    def _sweep_grid(self,
                    dct_coefficients: np.ndarray,
                    quality_factor_sweep: tuple[int, ...],
                    native_quality_factor: Optional[int]
                    ) -> tuple[list[DivergenceSample], int]:
        """Evaluate every (quality factor, base, frequency) cell of the grid.

        Args:
            dct_coefficients: Float array of shape (block_count, 8, 8).
            quality_factor_sweep: Quality factors to apply.
            native_quality_factor: The image's own quality factor, used to
                attribute any lattice found in a cell.

        Returns:
            Tuple of the successful samples and the total digit-clamp count.
        """
        samples: list[DivergenceSample] = []
        clamp_events = 0

        for quality_factor in quality_factor_sweep:
            quantized = self._quantize_at_quality(dct_coefficients,
                                                  int(quality_factor))
            for frequency_index in self.zigzag_frequency_indices:
                row, column = self.zigzag_coordinates[frequency_index]
                coefficients_at_frequency = quantized[:, row, column]

                for digit_base in self.digit_bases:
                    sample, clamped = self._evaluate_configuration(
                        coefficients_at_frequency, digit_base,
                        frequency_index, int(quality_factor),
                        native_quality_factor
                    )
                    clamp_events += clamped
                    if sample is not None:
                        samples.append(sample)

        return samples, clamp_events

    def _quantize_at_quality(self,
                             dct_coefficients: np.ndarray,
                             quality_factor: int) -> np.ndarray:
        """Quantize the whole block stack with the table at one quality factor.

        Args:
            dct_coefficients: Float array of shape (block_count, 8, 8).
            quality_factor: JPEG quality factor selecting the standard table.

        Returns:
            Quantized coefficients of the same shape.
        """
        quantization_table = np.asarray(
            load_standard_jpeg_quantization_table(quality_factor,
                                                  constants.DCT_BLOCK_SIZE),
            dtype=np.float64,
        )
        return self._quantize(dct_coefficients, quantization_table)

    def _evaluate_configuration(self,
                                coefficients: np.ndarray,
                                digit_base: int,
                                frequency_index: int,
                                quality_factor: int,
                                native_quality_factor: Optional[int] = None
                                ) -> tuple[Optional[DivergenceSample], int]:
        """Turn one frequency's coefficients into a divergence measurement.

        Args:
            coefficients: Quantized coefficients, one per block.
            digit_base: Digit base for extraction.
            frequency_index: Zig-zag index these coefficients came from.
            quality_factor: Quality factor whose table was applied.
            native_quality_factor: The image's own quality factor.

        Returns:
            The sample (or None when too few coefficients survived) and the
            digit-clamp count.
        """
        empirical_pmf, sample_count, zero_rate, clamp_events = \
            self._empirical_pmf(coefficients, digit_base)

        if sample_count < constants.MINIMUM_DIGIT_SAMPLE_COUNT:
            return None, clamp_events

        cell = (digit_base, frequency_index, quality_factor,
                native_quality_factor)
        histogram = (empirical_pmf, sample_count, zero_rate)
        return self._build_sample(coefficients, histogram, cell), clamp_events

    def _build_sample(self,
                      coefficients: np.ndarray,
                      histogram: tuple,
                      cell: tuple) -> DivergenceSample:
        """Fit the curve and assemble one grid cell's result record.

        Args:
            coefficients: Quantized coefficients, one per block.
            histogram: (empirical pmf, non-zero sample count, zero fraction).
            cell: (digit base, zig-zag frequency index, applied quality factor,
                the image's own quality factor).

        Returns:
            The populated DivergenceSample.
        """
        empirical_pmf, sample_count, zero_rate = histogram
        digit_base, frequency_index, quality_factor, native_quality_factor = cell
        excess = self._sublattice_excess(coefficients, digit_base)
        fitted_pmf, fit_parameters = self._fit_generalized_benford(empirical_pmf,
                                                                  digit_base)
        return DivergenceSample(
            digit_base=digit_base,
            zigzag_frequency_index=frequency_index,
            quality_factor=quality_factor,
            divergence=self._symmetric_kl_divergence(empirical_pmf, fitted_pmf,
                                                     sample_count),
            chi_square=self._chi_square_divergence(empirical_pmf, digit_base),
            sample_count=sample_count,
            zero_coefficient_rate=zero_rate,
            empirical_pmf=empirical_pmf,
            fitted_pmf=fitted_pmf,
            fit_parameters=fit_parameters,
            sublattice_excess=excess,
            is_finer_than_native=self._is_finer_than_native(
                quality_factor, native_quality_factor),
            has_residual_lattice=excess >= constants.SUBLATTICE_EXCESS_TOLERANCE,
        )

    @staticmethod
    def _is_finer_than_native(quality_factor: int,
                              native_quality_factor: Optional[int]) -> bool:
        """Decide whether this cell re-quantized the image more finely than its
        own encoder did, which makes its digit histogram unusable.

        Quantizing an already-JPEG image more finely than its own encoder did
        cannot recover discarded information; it only re-expresses the surviving
        lattice, distorting the digit histogram. Excluding on the ratio's
        DIRECTION catches every such case without a threshold, and leaves cells
        at or below the native factor alone - which is where Singh et al.'s
        genuine double-compression evidence lives. Full derivation and the four
        controlled cases are in constants.py.

        Args:
            quality_factor: Quality factor whose table was applied here.
            native_quality_factor: The image's own quality factor, or None.

        Returns:
            True when this cell must be excluded from the score.
        """
        if not constants.CONTAMINATION_REQUIRES_FINER_THAN_NATIVE:
            return False
        # With no native factor nothing can be attributed, so every cell is kept
        # and the ambiguity is reported rather than silently resolved.
        if native_quality_factor is None:
            return False
        return quality_factor > native_quality_factor

    @staticmethod
    def _sublattice_excess(coefficients: np.ndarray, digit_base: int) -> float:
        """Measure how strongly the quantized coefficients lie on a sublattice.

        ENHANCEMENT 1, forced by diagnostic testing. This is NOT a forensic
        statistic and never contributes to a score - it detects an artifact
        this engine creates itself.

        SKILL family A step 1 quantizes with a standard JPEG table at a chosen
        quality factor. For Bonettini et al.'s uncompressed corpus that is the
        image's FIRST quantization. For an image that is already JPEG it is a
        SECOND one, and when the applied step is a rational fraction of the step
        already in the image the coefficients are forced onto a sublattice.
        Measured on the authentic photo campic.jpeg: embedded step 6, applied
        step 3 at QF85, so every quantized value became even and digit 2 took
        0.674 of the mass against Benford's 0.176.

        Under any population NOT confined to a sublattice, a fraction 1/m of
        integers is divisible by m, so (observed fraction) x m is about 1. On a
        lattice of period m the product approaches m.

        Args:
            coefficients: Quantized coefficients, one per block.
            digit_base: Digit base; sets how far the divisor search runs.

        Returns:
            Largest divisibility excess over divisors 2..base-1; 1.0 means no
            sublattice was detected.
        """
        magnitudes = np.abs(np.asarray(coefficients)).astype(np.int64).ravel()
        magnitudes = magnitudes[magnitudes > 0]
        if magnitudes.size < constants.SUBLATTICE_MINIMUM_SAMPLE_COUNT:
            return constants.SUBLATTICE_NEUTRAL_EXCESS

        largest_divisor = digit_base - constants.SUBLATTICE_MAXIMUM_DIVISOR_OFFSET
        excesses = [
            (float(np.count_nonzero(magnitudes % divisor == 0))
             / float(magnitudes.size)) * float(divisor)
            for divisor in range(2, largest_divisor + 1)
        ]
        return max(excesses + [constants.SUBLATTICE_NEUTRAL_EXCESS])

    def _forward_dct(self, blocks: np.ndarray) -> np.ndarray:
        """Apply the JPEG-normalized two-dimensional DCT to every block.

        Args:
            blocks: Float array of shape (block_count, 8, 8).

        Returns:
            DCT coefficients, same shape as the input.

        Raises:
            ValueError: If scipy rejects the array.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: I_ij = (1/4) * T_i * T_j *
        #                 SUM_m SUM_n Z(m,n) * cos((2m+1) i pi / 16)
        #                                    * cos((2n+1) j pi / 16)
        #          with T_k = 1/sqrt(2) for k = 0, else 1.
        # Variables: Z(m,n) = pixel at (m,n) of the 8x8 block; I_ij = DCT
        #            coefficient at frequency (i,j); T = normalisation factor.
        # Source: Thai, Retraint & Cogranne 2012 (ICIP), Eq. 8 - quoted in the
        #            SKILL "Implementation notes" -> "DCT convention" as the
        #            exact form that reproduces libjpeg-compatible magnitudes.
        # Expected range: for 8-bit input, roughly +/-2040 at DC and rapidly
        #            smaller at higher frequencies.
        # Equivalence: scipy.fft.dctn(type=2, norm="ortho") was verified against
        #            a literal transcription of Eq. 8 and agrees to 3.4e-13,
        #            i.e. to floating-point noise, so the library call IS Eq. 8.
        # ────────────────────────────────────────────────────
        try:
            return dctn(np.asarray(blocks, dtype=np.float64),
                        type=constants.DCT_TYPE,
                        norm=constants.DCT_NORMALISATION,
                        axes=(-2, -1))
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"block DCT failed for an array of shape "
                f"{np.asarray(blocks).shape}: {error}"
            ) from error

    def _quantize(self,
                  dct_coefficients: np.ndarray,
                  quantization_table: np.ndarray) -> np.ndarray:
        """Quantize DCT coefficients with a standard JPEG table.

        Args:
            dct_coefficients: Array of shape (block_count, 8, 8).
            quantization_table: Steps of shape (8, 8).

        Returns:
            Integer-valued float array of the same shape as the input.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: c_{n,Delta} = round( I_n / Delta_n )
        # Variables: I_n = DCT coefficient at zig-zag frequency n;
        #            Delta_n = quantization step for that frequency, read from
        #            a standard JPEG quantization matrix at a chosen quality
        #            factor; c = the quantized coefficient whose leading digit
        #            is measured.
        # Source: Bonettini et al. 2021 - SKILL family A step 1: "Quantize each
        #            coefficient at frequency n ... using a quantization step
        #            Delta taken from a standard JPEG quantization matrix at a
        #            chosen quality factor". The SKILL stresses the digit
        #            statistic is computed on quantized coefficients, not raw
        #            DCT output.
        # Expected range: small signed integers; many exactly 0 at high
        #            frequencies, which is the documented zero-coefficient
        #            problem handled in _empirical_pmf.
        # ────────────────────────────────────────────────────
        return np.round(dct_coefficients / quantization_table)

    def _empirical_pmf(self,
                       coefficients: np.ndarray,
                       digit_base: int) -> tuple[np.ndarray, int, float, int]:
        """Build the leading-digit histogram for one frequency.

        A quantized coefficient of 0 has no leading digit (log_b(0) is
        undefined), so per the SKILL file these are EXCLUDED and K becomes
        "count of nonzero coefficients observed", not "count of blocks".

        Args:
            coefficients: Quantized coefficients, one per block.
            digit_base: Digit base.

        Returns:
            Tuple of (pmf, sample count, zero fraction, digit-clamp count).
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: d_{b,n,Delta}(k) = floor( |c_{n,Delta}(k)| /
        #                                    b**floor(log_b |c_{n,Delta}(k)|) )
        #          p_hat(d) = (1/K) * SUM_{k=1..K} 1_x( d(k) ),  d in {1..b-1}
        # Variables: c(k) = quantized coefficient of block k; b = digit base;
        #            d = leading digit; K = contributing coefficient count;
        #            1_x(y) = 1 if y == x else 0.
        # Source: Bonettini et al. 2021, Eq. 2-4 - SKILL family A steps 2 and 3.
        # Expected range: pmf in [0,1] summing to 1, decreasing in d.
        # ────────────────────────────────────────────────────
        flattened = np.asarray(coefficients, dtype=np.float64).ravel()
        total_count = int(flattened.size)
        if total_count == 0:
            return np.zeros(digit_base - 1, dtype=np.float64), 0, 0.0, 0

        # The 0 sentinel marks a zero-valued coefficient, excluded per above.
        digits = extract_first_significant_digit(flattened, digit_base)
        surviving = digits[digits > 0]
        zero_rate = 1.0 - (float(surviving.size) / float(total_count))

        clamped, clamp_events = self._clamp_digits_to_support(surviving,
                                                              digit_base)
        pmf = compute_probability_mass_function(clamped, digit_base)
        return pmf, int(clamped.size), zero_rate, clamp_events

    @staticmethod
    def _clamp_digits_to_support(digits: np.ndarray,
                                 digit_base: int) -> tuple[np.ndarray, int]:
        """Constrain extracted digits to the valid support, counting any fixes.

        Floating-point guard, not a change to the formula: log(1000)/log(10) can
        evaluate just below 3, which would yield a digit of 10 for a base-10
        extraction. Clamping restores the mathematically correct support, and
        the event count is surfaced so a reviewer can confirm whether it ever
        actually fired.

        Args:
            digits: Extracted leading digits, zero sentinels already removed.
            digit_base: Digit base defining the support {1..base-1}.

        Returns:
            Tuple of the clamped digits and the number of values corrected.
        """
        clamped = np.clip(digits, 1, digit_base - 1)
        return clamped, int(np.count_nonzero(clamped != digits))

    def _fit_generalized_benford(self,
                                 empirical_pmf: np.ndarray,
                                 digit_base: int) -> tuple[np.ndarray, tuple]:
        """Least-squares fit the generalized Benford curve to a histogram.

        Args:
            empirical_pmf: Observed pmf over digits 1..base-1.
            digit_base: Digit base.

        Returns:
            Tuple of the fitted curve (same length as the pmf) and the fitted
            (beta, gamma, delta). Falls back to the initial guess if the
            optimiser fails to converge.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: p_fit = argmin_p SUM_d ( p_hat(d) - p(d) )**2
        #          over the family p(d) = beta * log_b(1 + 1/(gamma + d**delta))
        # Variables: p_hat = empirical pmf; beta, gamma, delta = free parameters.
        # Source: Bonettini et al. 2021, Eq. 5 / 11 - SKILL family A step 4.
        # Expected range: fitted curve strictly positive and close to p_hat for
        #            authentic content.
        # Initial guess: Wang et al. 2009 Table 1 Laplacian row (alpha1 = 1.05,
        #            alpha2 = 1.352) supplies beta and delta; gamma starts at 0,
        #            which collapses the model to Wang's shape. Starting point
        #            only - it does not constrain the optimum.
        # DISCREPANCY, surfaced not silently resolved: SKILL step 4 writes the
        #            residual sum as "SUM_{d=0}^{b-1}", while the pmf (step 3)
        #            and the divergences (step 5) are both defined over
        #            d in {1..b-1}. Digit 0 is not a leading digit in any base,
        #            so the d=0 term is read as a transcription artifact and the
        #            fit is performed over {1..b-1}, consistent with every other
        #            step in the corpus.
        # ────────────────────────────────────────────────────
        digits = np.arange(1, digit_base, dtype=np.float64)
        model = partial(benford_model_for_base, float(digit_base))
        parameters = self._solve_fit(model, digits, empirical_pmf, digit_base)
        fitted = model(digits, *parameters)
        return fitted, tuple(float(value) for value in parameters)

    @staticmethod
    def _solve_fit(model,
                   digits: np.ndarray,
                   empirical_pmf: np.ndarray,
                   digit_base: int) -> np.ndarray:
        """Run the bounded least-squares solve, falling back on non-convergence.

        Args:
            model: Callable of the form f(digits, beta, gamma, delta).
            digits: Digit support the curve is evaluated on.
            empirical_pmf: Observed pmf being fitted.
            digit_base: Digit base, for the log message only.

        Returns:
            Fitted (beta, gamma, delta), or the published Laplacian starting
            values when the optimiser fails to converge.
        """
        initial_guess = (constants.BENFORD_FIT_INITIAL_BETA,
                         constants.BENFORD_FIT_INITIAL_GAMMA,
                         constants.BENFORD_FIT_INITIAL_DELTA)
        try:
            parameters, _ = curve_fit(
                model, digits, empirical_pmf,
                p0=initial_guess,
                bounds=(constants.BENFORD_FIT_LOWER_BOUNDS,
                        constants.BENFORD_FIT_UPPER_BOUNDS),
                maxfev=constants.BENFORD_FIT_MAX_FUNCTION_EVALUATIONS,
            )
            return parameters
        except (RuntimeError, ValueError) as error:
            # Non-convergence is expected on pathological histograms; falling
            # back to the published Laplacian curve keeps the divergence finite
            # and is recorded rather than hidden.
            logger.debug("generalized Benford fit did not converge for base %d "
                         "(%s); falling back to the Wang Laplacian curve",
                         digit_base, error)
            return np.asarray(initial_guess, dtype=np.float64)

    def _symmetric_kl_divergence(self,
                                 empirical_pmf: np.ndarray,
                                 fitted_pmf: np.ndarray,
                                 sample_count: int) -> float:
        """Measure divergence between the observed histogram and its fit.

        Args:
            empirical_pmf: Observed pmf.
            fitted_pmf: Fitted generalized-Benford curve.
            sample_count: Number of coefficients the pmf was estimated from,
                which sets the smallest probability the sample can resolve.

        Returns:
            Non-negative divergence; 0.0 indicates perfect agreement.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: D_JS(p_hat || p) = D_KL(p_hat || p) + D_KL(p || p_hat)
        #          D_KL(p_hat || p) = SUM_{d=1}^{b-1} p_hat(d) log( p_hat(d)/p(d) )
        # Variables: p_hat = empirical pmf; p = fitted curve.
        # Source: Bonettini et al. 2021, Eq. 6 - SKILL family A step 5.
        # Expected range: [0, inf); 0 = perfect Benford conformance (most
        #            natural), larger = greater deviation (more suspicious).
        # CRITICAL NORMALISATION NOTE: the SKILL warns this is NOT the
        #            conventional averaged Jensen-Shannon divergence -
        #            "Bonettini's Eq. 6 ... unaveraged, no factor of 0.5 - is a
        #            different normalisation - implement their exact unaveraged
        #            form if reproducing their reported numbers". The two KL
        #            terms are SUMMED, with no 0.5 and no midpoint distribution.
        # Flooring: see _probability_floor - both inputs are floored and
        #            renormalised so no term can be infinite.
        # ────────────────────────────────────────────────────
        floor = self._probability_floor(sample_count)
        safe_empirical = safe_normalise_distribution(empirical_pmf, floor)
        safe_fitted = safe_normalise_distribution(fitted_pmf, floor)

        forward = float(entropy(safe_empirical, safe_fitted))
        reverse = float(entropy(safe_fitted, safe_empirical))
        divergence = forward + reverse

        return divergence if np.isfinite(divergence) else 0.0

    @staticmethod
    def _probability_floor(sample_count: int) -> float:
        """Smallest probability the sample can resolve, for KL flooring.

        The fitted curve has a free scale beta so does not sum to 1, and an
        empty histogram bin would make the reverse KL term infinite, so both
        distributions are floored and renormalised before the logarithms.

        ENHANCEMENT 3: the floor is 1/K rather than a fixed 1e-12. The step-4
        least-squares fit can collapse to numerically zero on digits that still
        carry real empirical mass, and under the old floor those bins supplied
        97% of the reported divergence at a magnitude set by log(1/1e-12)
        rather than by anything in the image. A probability below 1/K is not
        distinguishable from zero given K draws. Full measurement in
        constants.USE_RESOLVABLE_PROBABILITY_FLOOR.

        Args:
            sample_count: Number of coefficients contributing to the pmf.

        Returns:
            1/K when the resolvable floor is enabled and K is positive, else
            the fixed absolute floor.
        """
        if not constants.USE_RESOLVABLE_PROBABILITY_FLOOR or sample_count <= 0:
            return constants.DIVERGENCE_PROBABILITY_FLOOR
        return max(1.0 / float(sample_count),
                   constants.DIVERGENCE_PROBABILITY_FLOOR)

    def _chi_square_divergence(self,
                               empirical_pmf: np.ndarray,
                               digit_base: int) -> float:
        """Pearson chi-square against the classical Benford curve.

        Args:
            empirical_pmf: Observed pmf.
            digit_base: Digit base.

        Returns:
            Non-negative chi-square statistic.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: chi2 = SUM_d ( p_hat(d) - p(d) )**2 / p(d)
        #          against the classical curve p(d) = log10(1 + 1/d)
        # Variables: p_hat = empirical pmf; p = classical Benford probability.
        # Source: Moin & Islam 2017 (ICIIP) - SKILL family B1 step 3.
        # Expected range: [0, inf). Published anchors, the ONLY numeric
        #            reference values for any statistic in this SKILL file:
        #            unaltered mean 0.0112-0.0126 across QF 50/70/90;
        #            contrast-enhanced 0.0051-0.0791 depending on gamma and QF.
        # Base generalisation: Moin writes log10 because the paper works in
        #            base 10 throughout. The classical law generalises to
        #            log_b(1 + 1/d), which is a proper pmf for any base since
        #            SUM_{d=1}^{b-1} log_b((d+1)/d) telescopes to log_b(b) = 1.
        #            At the default base 10 this reduces exactly to Moin's form.
        # ────────────────────────────────────────────────────
        classical_pmf = self._classical_benford_pmf(digit_base)
        difference = np.asarray(empirical_pmf, dtype=np.float64) - classical_pmf
        chi_square = float(np.sum((difference ** 2) / classical_pmf))
        return chi_square if np.isfinite(chi_square) else 0.0

    @staticmethod
    def _classical_benford_pmf(digit_base: int) -> np.ndarray:
        """Classical Benford probabilities over digits 1..base-1.

        Args:
            digit_base: Digit base.

        Returns:
            Float array of length base-1 summing to 1.0.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: p(d) = log10(1 + 1/d),  d = 1..9   (base-10 classical form)
        # Variables: d = leading digit; p(d) = expected probability.
        # Source: Wang et al. 2009, derived at SKILL "Core mathematical
        #            principle" from a log-uniform variable (Eq. 1-7); reused as
        #            the reference curve by Moin & Islam 2017.
        # Expected range: p(1) = 0.301 down to p(9) = 0.0458 at base 10, summing
        #            to exactly 1.
        # ────────────────────────────────────────────────────
        digits = np.arange(1, digit_base, dtype=np.float64)
        return np.log1p(1.0 / digits) / np.log(float(digit_base))

    def _select_representative_chi_square(
            self,
            samples: list[DivergenceSample],
            estimated_quality_factor: Optional[float]) -> float:
        """Pick the chi-square measured nearest the image's own quality factor.

        SKILL "Implementation notes" -> "Quality-factor conditioning": the clean
        baseline itself shifts with quality factor, so a chi-square is only
        comparable against published values at a matching factor.

        Args:
            samples: Successful grid samples.
            estimated_quality_factor: The image's quality factor, if known.

        Returns:
            Mean chi-square across the chosen quality factor, or across all
            samples when the factor is unknown.
        """
        if not samples:
            return 0.0
        if estimated_quality_factor is None:
            return float(np.mean([sample.chi_square for sample in samples]))

        available = {sample.quality_factor for sample in samples}
        nearest = min(available,
                      key=lambda factor: abs(factor - estimated_quality_factor))
        matching = [sample.chi_square for sample in samples
                    if sample.quality_factor == nearest]
        return float(np.mean(matching))

    def _grid_size(self, quality_factor_sweep: tuple[int, ...]) -> int:
        """Total number of cells in the sweep grid actually evaluated.

        Args:
            quality_factor_sweep: Quality factors that were applied, which may
                include the image's own factor as well as the corpus set.

        Returns:
            Product of the three grid axis lengths.
        """
        return (len(self.digit_bases) *
                len(self.zigzag_frequency_indices) *
                len(quality_factor_sweep))

    @staticmethod
    def _validate_grid(digit_bases: tuple[int, ...],
                       zigzag_frequency_indices: tuple[int, ...],
                       quality_factor_sweep: tuple[int, ...]) -> None:
        """Reject sweep grids outside the corpus-validated ranges.

        Args:
            digit_bases: Requested digit bases.
            zigzag_frequency_indices: Requested zig-zag indices.
            quality_factor_sweep: Requested quality factors.

        Raises:
            ValueError: If any axis is empty or leaves the validated range.
        """
        if not digit_bases or not zigzag_frequency_indices or not quality_factor_sweep:
            raise ValueError("every sweep axis must contain at least one value")

        for base in digit_bases:
            if base < 2:
                raise ValueError(f"digit base must be at least 2, got {base}")

        for index in zigzag_frequency_indices:
            # Index 0 is DC, which the SKILL excludes; beyond index 9 the corpus
            # offers no evidence at all.
            if not 1 <= index <= constants.MAX_VALIDATED_ZIGZAG_FREQUENCY_INDEX:
                raise ValueError(
                    f"zig-zag frequency index {index} is outside the validated "
                    f"range 1..{constants.MAX_VALIDATED_ZIGZAG_FREQUENCY_INDEX} "
                    f"(0 is DC and is excluded by the SKILL file)"
                )

        for quality_factor in quality_factor_sweep:
            if not (constants.MINIMUM_JPEG_QUALITY_FACTOR <= quality_factor
                    <= constants.MAXIMUM_JPEG_QUALITY_FACTOR):
                raise ValueError(
                    f"quality factor {quality_factor} is outside the encoder "
                    f"range {constants.MINIMUM_JPEG_QUALITY_FACTOR}.."
                    f"{constants.MAXIMUM_JPEG_QUALITY_FACTOR}"
                )
