"""Every tunable parameter, threshold and magic number for the Benford engine.

RULE: no numeric literal with forensic meaning may appear anywhere else in this
package. Each constant below records its exact value, the paper it comes from
(via the SKILL file), and why this value rather than another.

Provenance tags used throughout:
    [CORPUS]      - value is printed explicitly in the SKILL file / source paper.
    [DERIVED]     - value is computed from a [CORPUS] value; derivation shown.
    [ENGINEERING] - value is NOT in the corpus. The SKILL file either flags the
                    quantity as unspecified/ambiguous or is silent. These are the
                    only values a reviewer needs to challenge, and every one of
                    them is listed in KNOWN_UNSOURCED_PARAMETERS at the bottom.
    [STRUCTURAL]  - fixed by a file format or by arithmetic, not a free choice
                    (e.g. 8-bit images saturate at 255).
    [PRESENTATION]- affects only the rendered evidence image, never the score.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

ENGINE_NAME: str = "benford"

# Version of the SKILL document this implementation was written against. Bump
# this whenever SKILL(Benford's Law Forgery Detection).md changes, so a stored
# forensic report can be traced back to the exact spec that produced it.
SKILL_VERSION: str = "SKILL(Benford's Law Forgery Detection).md@2026-08-15"


# ---------------------------------------------------------------------------
# Block transform  -  SKILL section "Step-by-step algorithm", family A step 1
# ---------------------------------------------------------------------------

# Value: 8. [CORPUS] Bonettini et al. 2021, SKILL preprocessing step 2:
# "Partition into non-overlapping 8x8 blocks (DCT) - Bonettini et al. explicitly
# state K non-overlapping 8x8-pixel blocks."
# Why not another value: 8x8 is the JPEG transform block size; the entire
# premise of the engine is that it reads the SAME grid the codec quantized on.
DCT_BLOCK_SIZE: int = 8

# Values: type-II DCT, orthonormal scaling. [CORPUS] SKILL "Implementation notes"
# -> "DCT convention": I_ij = (1/4)*T_i*T_j*SUM SUM Z(m,n)*cos(...)*cos(...) with
# T_k = 1/sqrt(2) for k=0 else 1 (Thai et al. 2012, Eq. 8).
# Verified numerically: scipy.fft.dctn(type=2, norm="ortho") reproduces that
# expression to 3.4e-13 (floating point noise), so the library call IS Eq. 8.
DCT_TYPE: int = 2
DCT_NORMALISATION: str = "ortho"

# Value: 1 (green plane of a BGR uint8 array). [CORPUS] Moin et al. 2017,
# SKILL family B1 step 1: "Extract the green channel".
# Why not luma: the SKILL flags colour handling as "(not specified in the corpus
# for color)". Green is the only channel choice any paper in the corpus states
# explicitly, and choosing it introduces no unsourced luma weights.
ANALYSIS_CHANNEL_INDEX: int = 1
ANALYSIS_CHANNEL_NAME: str = "green"

# Number of colour planes a valid BGR input must carry. [STRUCTURAL]
EXPECTED_CHANNEL_COUNT: int = 3

# Number of array dimensions a colour image must have (height, width, channel).
# [STRUCTURAL]
EXPECTED_IMAGE_DIMENSION_COUNT: int = 3

# Milliseconds per second, for reporting processing_time_ms. [STRUCTURAL]
MILLISECONDS_PER_SECOND: float = 1000.0

# Ordinal positions of each entry in the computation trace. [STRUCTURAL] -
# these are sequence labels for the report generator, not forensic parameters.
COMPUTATION_STEP_CONDITION_CHECK: int = 1
COMPUTATION_STEP_PREPROCESSING: int = 2
COMPUTATION_STEP_BLOCK_DCT: int = 3
COMPUTATION_STEP_DIVERGENCE_SWEEP: int = 4
COMPUTATION_STEP_CALIBRATION: int = 5
COMPUTATION_STEP_FAILURE: int = 1


# ---------------------------------------------------------------------------
# Sweep grid  -  SKILL family A steps 1-6
# ---------------------------------------------------------------------------

# Values: {10, 20, 40, 60}. [CORPUS] SKILL family A step 6: "Bases tested:
# B subset of {10, 20, 40, 60}".
SUPPORTED_DIGIT_BASES: tuple[int, ...] = (10, 20, 40, 60)

# Value: (10,). [CORPUS] SKILL "Implementation notes" -> "Base selection":
# "b=10 alone captures most of the achievable accuracy; adding more bases from
# {20,40,60} gives only marginal further improvement (Bonettini Fig. 6c/6d) -
# not worth the added computation for a first implementation."
DEFAULT_DIGIT_BASES: tuple[int, ...] = (10,)

# Values: zig-zag indices 1..9. [CORPUS] SKILL family A step 6: "Frequencies
# tested: N subset of {1,...,9} in zig-zag order after DC".
# Index 0 (DC) is deliberately absent - SKILL "Implementation notes" ->
# "Frequency selection": "use zig-zag order excluding DC (frequency index 0)".
DEFAULT_ZIGZAG_FREQUENCY_INDICES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9)

# Value: 9. [CORPUS] SKILL "Frequency selection": "there is no evidence in the
# corpus about frequencies beyond index 9." Guards against a caller sweeping
# into an unvalidated region.
MAX_VALIDATED_ZIGZAG_FREQUENCY_INDEX: int = 9

# Values: {80, 85, 90, 95, 100}. [CORPUS] SKILL family A step 6: "Quantization
# steps: derived from JPEG quality factors J subset of {80,85,90,95,100}".
DEFAULT_QUALITY_FACTOR_SWEEP: tuple[int, ...] = (80, 85, 90, 95, 100)

# Value: 100. [CORPUS] SKILL family A step 7 names the best-performing single
# configuration as "b=10, all 9 frequencies, QF=100 per the paper's own
# ablation". Reported alongside the swept mean for transparency.
BEST_SINGLE_CONFIGURATION_QUALITY_FACTOR: int = 100

# PIL accepts JPEG quality in 1..100. [STRUCTURAL]
MINIMUM_JPEG_QUALITY_FACTOR: int = 1
MAXIMUM_JPEG_QUALITY_FACTOR: int = 100


# ---------------------------------------------------------------------------
# Generalized-Benford curve fit  -  SKILL family A step 4
# ---------------------------------------------------------------------------
# Model:  p(d) = beta * log_b( 1 + 1 / (gamma + d**delta) )
#
# Initial guess is taken from the closest published fit in the corpus: Wang et
# al. 2009 Table 1, Laplacian row, quoted in the SKILL benchmark table as
# "Laplacian: alpha1=1.05, alpha2=1.352, alpha3=1.061".
#
# Mapping Wang's form  B_g(d) = alpha1 * log10(alpha3 + 1/d**alpha2)
# onto Bonettini's     p(d)   = beta   * log_b (1     + 1/(gamma + d**delta)):
#   beta  <- alpha1   (both are the outer multiplicative scale)
#   delta <- alpha2   (both are the exponent applied to the digit d)
#   gamma <- 0.0      Wang's alpha3 occupies a DIFFERENT slot (it is added to
#                     the whole reciprocal term, not to d**alpha2), so it has no
#                     counterpart. Setting gamma = 0 collapses Bonettini's inner
#                     term to 1/d**delta, which is exactly Wang's shape - making
#                     0.0 the neutral starting point rather than an invention.
# The Laplacian row is used (not the Gaussian row) because the SKILL states the
# Laplacian model is the special case alpha*=1 of the DCT coefficient
# distribution derived by Thai et al. 2012.
#
# These affect only where the optimiser STARTS, never where it converges.
BENFORD_FIT_INITIAL_BETA: float = 1.05     # [CORPUS] Wang 2009 Table 1, alpha1
BENFORD_FIT_INITIAL_DELTA: float = 1.352   # [CORPUS] Wang 2009 Table 1, alpha2
BENFORD_FIT_INITIAL_GAMMA: float = 0.0     # [DERIVED] see mapping note above

# Gaussian row of the same table, retained for reference / alternative seeding.
# [CORPUS] SKILL benchmark table: "Gaussian: alpha1=1.08, alpha2=2.55, alpha3=1.15".
WANG_GAUSSIAN_FIT_ALPHA1: float = 1.08
WANG_GAUSSIAN_FIT_ALPHA2: float = 2.55
WANG_GAUSSIAN_FIT_ALPHA3: float = 1.15

# Parameter bounds. [STRUCTURAL] - these are the domain of definition of the
# model, not tuning choices. For digits d >= 1 and delta > 0 we have d**delta >= 1,
# so gamma >= 0 guarantees (gamma + d**delta) >= 1 > 0, keeping the reciprocal
# finite and the logarithm's argument > 1. Violating them makes p(d) undefined.
BENFORD_FIT_LOWER_BOUNDS: tuple[float, float, float] = (0.0, 0.0, 0.0)
BENFORD_FIT_UPPER_BOUNDS: tuple[float, float, float] = (float("inf"),
                                                        float("inf"),
                                                        float("inf"))

# Value: 10000. [ENGINEERING] scipy's default max function evaluations is often
# too low for a 3-parameter fit on a 9-point curve to converge. Raising the cap
# changes only whether the optimiser is allowed to finish, never the optimum.
BENFORD_FIT_MAX_FUNCTION_EVALUATIONS: int = 10000


# ---------------------------------------------------------------------------
# Divergence  -  SKILL family A step 5
# ---------------------------------------------------------------------------

# The SKILL offers three symmetrized divergences and recommends one:
# "All three divergences were compared; the paper does not report one being
# categorically superior - JS is the simplest to implement and has no free
# parameter, making it the recommended default."
#
# Renyi and Tsallis are NOT implemented, deliberately: both require an order
# parameter alpha which the SKILL records as unresolvable - "(ambiguous in the
# corpus - the paper states alpha is 'removed as a dependency... kept constant'
# without printing the numeric value used)". Implementing them would require
# inventing alpha, which this engine refuses to do.
PRIMARY_DIVERGENCE_NAME: str = "symmetric_kl_unaveraged"

# Value: 1e-12. [ENGINEERING] Absolute floor applied to probabilities before the
# log in the KL sum, so an empty histogram bin yields a large-but-finite
# contribution instead of inf/NaN. Retained only as a last-resort guard; the
# operative floor is now the resolvable floor below.
DIVERGENCE_PROBABILITY_FLOOR: float = 1e-12

# ENHANCEMENT 3: use 1/K - the smallest probability K samples can resolve - as
# the flooring level, instead of the fixed 1e-12 above.
#
# Test evidence (diagnostic run, campic.jpeg, worst cell QF85 frequency 4):
# the least-squares fit escaped to beta=3.56e16, gamma=4.07e16, delta=36.3,
# producing a fitted curve of [0.380, 0.380, 0.060, 0, 0, 0, 0, 0, 0]. Digits
# 4-9 hold real empirical mass (0.063, 0.050, 0.009, 0.031, 0.005, 0.021) but
# the fitted curve there is ~1e-17. Per-digit forward-KL terms measured
# [-0.145, 0.252, -0.012, 0.643, 0.904, 0.211, 0.737, 0.115, 0.504]: digits 4-9
# contribute 3.114 of the total 3.210, i.e. 97% of the reported divergence came
# from bins where the FIT collapsed, and their magnitude was set by
# log(p_hat / 1e-12) ~ 25 nats - a number chosen by this file, not measured
# from the image.
#
# A probability below 1/K is not distinguishable from zero given K draws, so
# flooring there states exactly what the sample can support. Measured effect on
# the same image: raw score 3.8317 -> 2.3560, with the ordering of cells intact.
USE_RESOLVABLE_PROBABILITY_FLOOR: bool = True

# Value: "max". [CORPUS-SANCTIONED, resolved by measurement]
# The SKILL offers two aggregations on equal footing for the training-free
# substitute: "threshold a single scalar - e.g. the mean or max divergence
# across the swept (b,n,Delta) grid". It does not say which to prefer, so the
# choice was made empirically during implementation.
#
# Measured over 5 synthetic scenes x 4 manipulations (contrast enhancement at
# gamma 0.5 and 2.0, double compression at QF 75->95 and 60->95), counting how
# often a manipulated image outscored its own matched authentic baseline:
#       mean : 10/20   (indistinguishable from chance)
#       max  : 18/20
# The mean yields larger ratios for double compression, which perturbs every
# grid cell at once, but INVERTS on contrast enhancement (0.95x and 0.61x, i.e.
# the tampered image scored lower). Contrast enhancement distorts only a few
# (frequency, quality factor) cells, so averaging over ~45 cells - most of them
# unaffected - buries the signal. The max asks the correct question for a
# detector: did ANY configuration break Benford conformance?
#
# Both values remain available on BenfordComputation; only which one becomes
# raw_score is governed here.
DIVERGENCE_AGGREGATION_RULE: str = "max"


# ---------------------------------------------------------------------------
# Zero-coefficient handling  -  SKILL "Implementation notes", first bullet
# ---------------------------------------------------------------------------

# [CORPUS-SANCTIONED] SKILL: "Engineering recommendation: exclude zero
# coefficients from the per-block digit count (i.e. K ... should be reinterpreted
# as 'count of nonzero coefficients observed', not 'count of blocks')".
# Required because d = floor(|c| / b**floor(log_b|c|)) is undefined at c = 0.
EXCLUDE_ZERO_COEFFICIENTS: bool = True

# Value: 0.95. [ENGINEERING] The SKILL mandates TRACKING the zero rate - "track
# what fraction of coefficients were excluded as a data-quality signal, since a
# very high zero-rate (heavy quantization) will make the pmf estimate noisy/
# unreliable regardless" - but prints no cutoff. 0.95 means "19 of every 20
# coefficients vanished", at which point the surviving sample is not a
# meaningful population. Flagged as unsourced; tune on calibration data.
MAX_ACCEPTABLE_ZERO_COEFFICIENT_RATE: float = 0.95

# Value: 100. [ENGINEERING] Minimum surviving non-zero coefficients required
# before a single (base, frequency, QF) pmf is considered estimable at all.
# Not in the corpus; a pmf over 9 bins from fewer than ~100 draws is dominated
# by counting noise. Configurations below this are skipped, not scored.
MINIMUM_DIGIT_SAMPLE_COUNT: int = 100


# ---------------------------------------------------------------------------
# ENHANCEMENT 1 - self-inflicted double-quantization detection
# ---------------------------------------------------------------------------
# THE SINGLE LARGEST DEFECT FOUND IN DIAGNOSTIC TESTING.
#
# SKILL family A step 1 says to quantize with "a standard JPEG quantization
# matrix at a chosen quality factor". In Bonettini et al.'s corpus the images
# are uncompressed GAN output, so that quantization is the image's FIRST. Every
# real-world input is already JPEG, so the same operation is a SECOND
# quantization, and when the applied step is a rational fraction of the step
# already baked into the image, the result is forced onto a sublattice.
#
# Measured on campic.jpeg (an authentic camera photo):
#   embedded quantization steps, zig-zag 1..9 : [6, 6, 6, 7, 6, 7, 8, 8, 7]
#   IJG steps the engine applied at QF85      : [3, 4, 4, 4, 3, 5, 4, 4, 4]
#   ratio                                     : [2.0, 1.5, 1.5, 1.75, 2.0, ...]
# At ratio exactly 2.0 every quantized coefficient becomes even, so the leading
# digit can only be 2, 4, 6, 8 or (via 10-19) 1. The observed pmf at that cell
# was [0.086, 0.674, 0.060, 0.063, 0.050, 0.009, 0.031, 0.005, 0.021] - a 0.674
# spike on digit 2 against Benford's 0.176.
#
# Consequence measured across the sweep for that one authentic image:
#   native QF74 : divergence 0.1143      <- uncontaminated
#   QF80        : 2.6043
#   QF85        : 3.8317                 <- 33x inflation, became the raw score
#   QF90/95/100 : 3.5276 / 3.3140 / 3.3140
# The engine was reading an artifact of its own arithmetic as evidence of
# tampering, and did so most strongly on authentic photographs.
#
# The gate below detects the symptom directly on the integers, with no need to
# know the image's true table: under any population NOT confined to a
# sublattice, a fraction 1/m of integers is divisible by m, so the ratio
# (observed divisible fraction) x m is ~1. On a sublattice of period m it
# approaches m.

# Value: base - 1, i.e. 9 at base 10. [STRUCTURAL] The leading-digit support is
# {1..base-1}; a lattice whose period reaches the base shifts the digit wholesale
# rather than distorting the histogram shape, so divisors are tested up to the
# top of the support.
SUBLATTICE_MAXIMUM_DIVISOR_OFFSET: int = 1

# Value: 1.3. [ENGINEERING - test-derived] Measured excess ratios on
# campic.jpeg, frequencies 1-9, by swept quality factor:
#   QF60/70/74 (at or below native) : 1.00 at all nine frequencies
#   QF80                            : one frequency at 1.43
#   QF85                            : six of nine at 1.46-1.99
#   QF90                            : five of nine at 1.17-2.24
#   QF95 / QF100                    : nine of nine at 1.34-3.74
# Clean cells sit at exactly 1.00 and contaminated ones at 1.17 and above, so
# 1.3 falls inside the observed empty gap rather than splitting a population.
# Flagged unsourced; re-measure on a wider corpus before forensic use.
SUBLATTICE_EXCESS_TOLERANCE: float = 1.3

# THE ATTRIBUTION RULE, and the reason this gate does not destroy the engine's
# one genuinely working detection. A lattice in the quantized coefficients has
# two possible authors, and they must not be treated alike:
#
#   * THIS ENGINE created it, by applying a step FINER than the one already in
#     the image. Quantizing an already-JPEG image more finely than its own
#     encoder did cannot recover information the encoder destroyed; it can only
#     re-express the surviving lattice. This is the artifact.
#   * THE IMAGE'S OWN HISTORY created it - a genuine earlier compression at a
#     coarser step, which is precisely Singh et al.'s double-compression
#     signal and one of the three manipulations the SKILL says this method
#     detects best.
#
# Measured separation between the two, over four controlled cases:
#   single-compressed synthetic, native QF90 : 0 cells flagged at ANY swept QF
#   authentic photo campic.jpeg, native QF74 : 0 flagged at QF60/70/74,
#                                              30 flagged across QF80-100
#                                              -> artifact lives ABOVE native
#   double-compressed QF30 then QF95         : 15 flagged at QF70/80/85/90,
#                                              0 at QF95 and QF100
#   double-compressed QF60 then QF95         : 9 flagged at QF80/85/90,
#                                              0 at QF100
#                                              -> signal lives AT OR BELOW native
#
# The two populations do not overlap in a single one of these cases, so a cell
# is discarded ONLY when it is both lattice-bearing AND swept above the image's
# own quality factor. Below and at the native factor a lattice is evidence
# about the image and is kept.
CONTAMINATION_REQUIRES_FINER_THAN_NATIVE: bool = True

# Value: 32. [ENGINEERING] Minimum non-zero coefficients before a divisibility
# fraction is estimated at all; below this the fraction is counting noise.
SUBLATTICE_MINIMUM_SAMPLE_COUNT: int = 32

# Neutral excess ratio, meaning "no sublattice detected". [STRUCTURAL]
SUBLATTICE_NEUTRAL_EXCESS: float = 1.0


# ---------------------------------------------------------------------------
# Condition checking  -  SKILL "Input requirements"
# ---------------------------------------------------------------------------

# Value: 50. [CORPUS] SKILL "Unreliable / inapplicable when": "Very low quality
# factor (QF~50 or below) where high-frequency coefficients quantize to zero,
# removing their leading digit entirely."
MINIMUM_RELIABLE_QUALITY_FACTOR: float = 50.0

# Value: 80. [CORPUS] SKILL "Reliable when": "quality factor >= ~80 for the
# strongest chi-square separation in Moin et al.'s tests (though their method
# still detects down to QF50, just with reduced deviation)."
STRONG_SEPARATION_QUALITY_FACTOR: float = 80.0

# Value: 256. [CORPUS] SKILL "Bit depth / resolution": "Bonettini et al.'s
# corpus is 256x256". Smallest image size any Benford paper in the corpus
# validates against.
MINIMUM_VALIDATED_IMAGE_DIMENSION: int = 256

# Value: 1024. [DERIVED] (MINIMUM_VALIDATED_IMAGE_DIMENSION / DCT_BLOCK_SIZE)**2
# = (256/8)**2 = 1024 non-overlapping blocks, i.e. the number of samples per
# frequency available in Bonettini's smallest validated image.
MINIMUM_ANALYSIS_BLOCK_COUNT: int = 1024

# Value: 255. [STRUCTURAL] Maximum intensity of an 8-bit unsigned channel;
# needed to measure the "near-saturated/degenerate" state the SKILL excludes.
SATURATION_INTENSITY_LEVEL: int = 255
MINIMUM_INTENSITY_LEVEL: int = 0

# Value: 0.5. [ENGINEERING] SKILL "Reliable when" requires the image be "not
# near-saturated/degenerate" but quantifies neither term. 0.5 = "half the pixels
# are pinned at an extreme". Flagged as unsourced.
MAX_ACCEPTABLE_SATURATION_FRACTION: float = 0.5

# Value: 1e-6. [ENGINEERING] A channel whose intensity variance falls below this
# is constant (a blank/degenerate plate); its DCT is all zeros and no digit
# statistic exists. Numerical-degeneracy guard, not a forensic threshold.
MINIMUM_CHANNEL_VARIANCE: float = 1e-6

# Container formats that evidence a block-transform quantization history.
# [CORPUS] SKILL "Reliable when": "image has passed through JPEG (or JPEG2000)
# block-transform quantization at least once (this is what imposes/preserves
# Benford-fitting structure in the first place)".
JPEG_FORMAT_NAMES: frozenset[str] = frozenset({"JPEG", "JPG", "JPEG2000", "JP2"})

# Formats decoded from a wavelet codec. [CORPUS] SKILL "Unreliable /
# inapplicable when": for JPEG2000 double-compression "the DWT-domain
# double-compression detector essentially does not work" (Singh 2015 Table II,
# deviation ratios 0 to 2.2204e-16). This engine implements the DCT path only.
WAVELET_CODEC_FORMAT_NAMES: frozenset[str] = frozenset({"JPEG2000", "JP2", "J2K"})


# ---------------------------------------------------------------------------
# Confidence weighting  -  all [ENGINEERING]
# ---------------------------------------------------------------------------
# The SKILL specifies no fusion weights. These multipliers express the relative
# severity of each documented degradation so the fusion layer can down-weight
# this engine's vote. They compose multiplicatively and are clamped to [0, 1].

FULL_CONFIDENCE: float = 1.0
ZERO_CONFIDENCE: float = 0.0

# Applied when the container is not JPEG-derived but compression evidence exists.
CONFIDENCE_PENALTY_UNKNOWN_COMPRESSION_HISTORY: float = 0.40
# Applied when QF is between MINIMUM_RELIABLE and STRONG_SEPARATION (50..80):
# the SKILL says detection still works here, "just with reduced deviation".
CONFIDENCE_PENALTY_WEAK_QUALITY_FACTOR: float = 0.60
# Applied when the decoded container is a wavelet codec (DCT path is off-domain).
CONFIDENCE_PENALTY_WAVELET_CODEC: float = 0.30
# Applied when the image is smaller than Bonettini's validated 256x256.
CONFIDENCE_PENALTY_BELOW_VALIDATED_RESOLUTION: float = 0.50
# Applied when metadata reports prior resizing. [CORPUS-motivated] SKILL
# "Documented failure cases": Wang 2009 show a compensating operation
# (equalization, RESCALING) can restore Benford conformance without undoing the
# tampering - i.e. resizing raises false-negative risk.
CONFIDENCE_PENALTY_RESAMPLED_INPUT: float = 0.70
# Applied when the surviving non-zero sample is thin but still above the floor.
CONFIDENCE_PENALTY_SPARSE_SAMPLE: float = 0.70
# Applied whenever the probability came from the provisional sigmoid rather
# than a fitted empirical-CDF calibration set.
CONFIDENCE_PENALTY_UNCALIBRATED: float = 0.50


# ---------------------------------------------------------------------------
# Calibration  -  SKILL "Output" -> "Calibrating to a [0,1] fusion-layer probability"
# ---------------------------------------------------------------------------
# The SKILL is explicit that NO calibration function exists in the corpus:
# "none of the five papers specify a general-purpose calibration function
# mapping their raw statistic to a probability. (Not specified in the corpus -
# engineering recommendation): fit a monotonic calibration (e.g. logistic/
# Platt-style sigmoid sigma(a*chi2 + b), or empirical CDF percentile against a
# held-out calibration set of known-authentic images ...) per-statistic, per
# estimated-QF-bucket".
#
# Both routes named there are implemented. Route 2 (empirical CDF) is preferred
# and is used whenever the caller supplies reference scores. Route 1 is the
# fallback and its two parameters are the least-defensible numbers in this file.

# Published chi-square anchors - the ONLY numeric reference values for any
# statistic in this entire SKILL file. [CORPUS] SKILL "Output" family B1 and the
# benchmark table: unaltered mean chi2 0.0112-0.0126 across QF 50/70/90;
# contrast-enhanced 0.0051-0.0791 depending on gamma and QF.
MOIN_UNALTERED_CHI_SQUARE_MINIMUM: float = 0.0112
MOIN_UNALTERED_CHI_SQUARE_MAXIMUM: float = 0.0126
MOIN_ALTERED_CHI_SQUARE_MINIMUM: float = 0.0051
MOIN_ALTERED_CHI_SQUARE_MAXIMUM: float = 0.0791

# ENHANCEMENT 4: report the distance from the published chi-square range, and
# penalise confidence by it - but do NOT gate on it.
#
# This was first implemented as a hard reliability gate and the gate was
# WITHDRAWN after measurement, because this pipeline does not reproduce Moin et
# al.'s chi-square scale and therefore cannot borrow their numbers as a
# threshold. Measured on a single-compressed synthetic image with no
# manipulation whatsoever, against Moin's published unaltered 0.0112-0.0126:
#     QF90  per-frequency mean 0.10212   pooled over frequencies 0.02475
#     QF70  per-frequency mean 0.65603   pooled over frequencies 0.24106
#     QF50  per-frequency mean 1.24708   pooled over frequencies 0.52219
# Pooling the frequencies into one histogram, which is Moin's simpler setup
# ("it does not sweep multiple frequencies the way Bonettini does"), moves the
# QF90 figure from 9x the published value to 2x - but the gap then grows to 19x
# at QF70 and 41x at QF50. The discrepancy is not a constant factor, so no
# rescaling recovers comparability, and a threshold taken from the published
# number would be arbitrary.
#
# What survives as honest is the ORDER OF MAGNITUDE of the gap: a chi-square
# many times the largest value any paper reports for any class tells the fusion
# layer that this image's coefficient population is unlike the corpus material,
# without pretending to a calibrated boundary.
CHI_SQUARE_VALIDITY_CEILING: float = MOIN_ALTERED_CHI_SQUARE_MAXIMUM

# Applied when the measured chi-square sits far outside every published value.
# [ENGINEERING] Set equal to the existing harshest non-zero penalty in this
# file (CONFIDENCE_PENALTY_WAVELET_CODEC) so the severity ladder stays
# consistent rather than introducing a new level.
CONFIDENCE_PENALTY_OUTSIDE_PUBLISHED_REGIME: float = 0.30

# Per-quality-factor unaltered baseline. [CORPUS] SKILL "Calibrating to a [0,1]
# ...": "Moin's Table I: unaltered mean chi2 is 0.0112 at QF90 vs 0.0126 at
# QF50 - not constant". The QF70 value is not individually printed in the SKILL,
# so it is absent here rather than interpolated.
# NOTE - internal inconsistency in the source document: the SKILL quotes the
# unaltered range as "0.0112-0.0126" in two places and as "0.0109-0.0126" in
# "Implementation notes" -> "Quality-factor conditioning". The twice-stated
# 0.0112 lower bound is used; the discrepancy is surfaced, not silently resolved.
MOIN_UNALTERED_CHI_SQUARE_BY_QUALITY_FACTOR: dict[int, float] = {
    50: 0.0126,
    90: 0.0112,
}

# Quality-factor buckets for conditioned calibration. [CORPUS-motivated] SKILL
# "Quality-factor conditioning": "any deployed threshold must be conditioned on
# the image's estimated background quality factor ... rather than applied as one
# fixed global cutoff." Bucket edges mirror the two thresholds already sourced
# above (50 and 80) so no new cut points are invented.
QUALITY_FACTOR_BUCKET_EDGES: tuple[float, ...] = (MINIMUM_RELIABLE_QUALITY_FACTOR,
                                                  STRONG_SEPARATION_QUALITY_FACTOR)
QUALITY_FACTOR_BUCKET_NAMES: tuple[str, ...] = ("qf_low", "qf_medium", "qf_high")

# Provisional Platt sigmoid: probability = 1 / (1 + exp(-(a * raw + b))).
#
# *** THESE TWO NUMBERS ARE NOT FROM THE CORPUS. ***
# The SKILL publishes no divergence magnitudes whatsoever for family A, so no
# honest data-driven midpoint exists.
#
# ENHANCEMENT 5 - re-anchored after the previous values were shown to saturate.
# The former settings (midpoint 0.30, slope 10.0) had been chosen on SYNTHETIC
# imagery where the statistic sat near 0.21 for authentic content. On real
# photographs the statistic is an order of magnitude larger, so the sigmoid
# saturated: measured probabilities on the six diagnostic images were
# 1.0000, 1.0000, 0.0891, 1.0000, 1.0000, 1.0000 - five of six pinned at
# exactly 1.0, INCLUDING both authentic camera photographs. A probability that
# takes two values is not a probability; the fusion layer received a hard vote
# of "certainly forged" on genuine photographs.
#
# The values below are placed against the statistic's measured range once
# ENHANCEMENTS 1-3 remove the artifacts:
#   single-compressed synthetic, QF80-100 : 0.0764 - 0.0979
#   authentic camera photographs          : 0.0298 - 0.1271
#   double-compressed QF75 then QF95      : 0.1991
#   double-compressed QF60 then QF95      : 1.6389
#   double-compressed QF45 then QF95      : 2.3682
#   double-compressed QF30 then QF95      : 3.6207
# A midpoint of 0.70 sits in the gap between the single-compression cluster and
# the clearly double-compressed cases, and a slope of 2.5 grades the response
# across it instead of stepping: the same settings map a clean image to 0.18
# and a QF30-then-QF95 double compression to 0.999.
#
# This is a statement about the DYNAMIC RANGE of the statistic, NOT a validated
# forensic decision threshold, and it must never be reported as one. Any run
# using these values is flagged uncalibrated, has its confidence multiplied by
# CONFIDENCE_PENALTY_UNCALIBRATED, and says so in reliability_note. Replace by
# fitting on labelled data before forensic use.
PROVISIONAL_SIGMOID_MIDPOINT: float = 0.70   # [ENGINEERING - UNSOURCED]
PROVISIONAL_SIGMOID_SLOPE: float = 2.5       # [ENGINEERING - UNSOURCED]

# Guard for the exponent in the logistic function, preventing overflow warnings
# on extreme inputs. [STRUCTURAL] numerical safety only.
SIGMOID_EXPONENT_LIMIT: float = 60.0

# Minimum number of reference scores before the empirical-CDF route is trusted
# in preference to the sigmoid. [ENGINEERING] Below this the percentile estimate
# is coarser than the sigmoid it would replace.
MINIMUM_CALIBRATION_REFERENCE_COUNT: int = 30


# ---------------------------------------------------------------------------
# Evidence map rendering  -  all [PRESENTATION], never affect any score
# ---------------------------------------------------------------------------

EVIDENCE_MAP_HEIGHT: int = 360
EVIDENCE_MAP_WIDTH: int = 640
EVIDENCE_MAP_MARGIN: int = 40
EVIDENCE_MAP_BACKGROUND_INTENSITY: int = 255
EVIDENCE_MAP_AXIS_INTENSITY: int = 60
EVIDENCE_MAP_AXIS_THICKNESS: int = 2
EVIDENCE_MAP_BAR_GAP_FRACTION: float = 0.25
EVIDENCE_MAP_FITTED_MARKER_THICKNESS: int = 3
# BGR triples, matching the BGR convention of the input contract.
EVIDENCE_MAP_EMPIRICAL_BAR_COLOUR: tuple[int, int, int] = (200, 130, 60)
EVIDENCE_MAP_FITTED_CURVE_COLOUR: tuple[int, int, int] = (60, 60, 220)
# Head-room above the tallest plotted probability so bars are not clipped.
EVIDENCE_MAP_VERTICAL_HEADROOM: float = 1.15


# ---------------------------------------------------------------------------
# Audit aid
# ---------------------------------------------------------------------------

# Every [ENGINEERING] value above, gathered in one place. A reviewer challenging
# this engine's numbers only has to argue with this list - everything else is
# traceable to a printed value in the SKILL file.
KNOWN_UNSOURCED_PARAMETERS: tuple[str, ...] = (
    "MAX_ACCEPTABLE_ZERO_COEFFICIENT_RATE",
    "MINIMUM_DIGIT_SAMPLE_COUNT",
    "MAX_ACCEPTABLE_SATURATION_FRACTION",
    "MINIMUM_CHANNEL_VARIANCE",
    "BENFORD_FIT_MAX_FUNCTION_EVALUATIONS",
    "DIVERGENCE_PROBABILITY_FLOOR",
    "PROVISIONAL_SIGMOID_MIDPOINT",
    "PROVISIONAL_SIGMOID_SLOPE",
    "MINIMUM_CALIBRATION_REFERENCE_COUNT",
    "SUBLATTICE_EXCESS_TOLERANCE",
    "SUBLATTICE_MINIMUM_SAMPLE_COUNT",
    "all CONFIDENCE_PENALTY_* multipliers",
)

# Changes made in response to diagnostic testing rather than to the SKILL file.
# Each names the measurement that forced it, so a reviewer can re-run that
# measurement and challenge the change on its own evidence.
TEST_DERIVED_ENHANCEMENTS: tuple[tuple[str, str], ...] = (
    ("exclude cells quantized finer than the image's own encoder",
     "authentic camera photo campic.jpeg scored 3.8317 at swept QF85 versus "
     "0.1143 at its own native quantization; the applied step was exactly half "
     "the embedded step, forcing every coefficient even and producing a 0.674 "
     "spike on digit 2. Excluding by the ratio's DIRECTION rather than by "
     "detecting a sublattice also strengthened the one detection that does "
     "work: double compression QF30-then-QF95 moved from 13x to 40x the "
     "single-compression score, because the genuine lattice below the native "
     "factor is now scored instead of discarded"),
    ("native quality factor added to the sweep",
     "every swept quality factor differed from the image's own, so no "
     "uncontaminated cell existed for the sweep to fall back on"),
    ("resolvable probability floor 1/K",
     "97% of the reported divergence came from digits where the least-squares "
     "fit had collapsed to ~1e-17, with magnitude set by log(1/1e-12)"),
    ("chi-square distance from the published range, as a confidence penalty",
     "measured chi-square 0.0494-0.9444 against the corpus's published "
     "0.0051-0.0791 for altered images and 0.0112-0.0126 for unaltered; "
     "reported and penalised but NOT gated on, because this pipeline does not "
     "reproduce the published scale even on unmanipulated images"),
    ("sigmoid re-anchored",
     "five of six probabilities pinned at exactly 1.0, including both "
     "authentic photographs"),
)

# Changes that were implemented, measured, and REJECTED because the measurement
# did not support them. Recorded so they are not proposed again.
REJECTED_ENHANCEMENTS: tuple[tuple[str, str], ...] = (
    ("gate the sweep on a hard chi-square ceiling taken from Moin et al.",
     "withdrawn after measurement. On a single-compressed synthetic image with "
     "NO manipulation the measured chi-square was 0.10212 at QF90, 0.65603 at "
     "QF70 and 1.24708 at QF50 against a published 0.0112-0.0126 at all three. "
     "Pooling frequencies into one histogram, which is Moin's setup, narrows "
     "the QF90 gap from 9x to 2x but widens it to 19x at QF70 and 41x at QF50. "
     "The discrepancy is not a constant factor, so the published number cannot "
     "serve as a threshold. Kept as a confidence penalty instead"),
    ("exclude cells by detecting a sublattice rather than by direction",
     "superseded. The divisibility test only fires on integer step ratios, and "
     "the ratios measured on campic.jpeg were [2.0, 1.5, 1.5, 1.75, 2.0, 1.4, "
     "2.0, 2.0, 1.75]; ratio 1.5 maps coefficients onto 2, 3, 5, 6, 8, 9 with "
     "no common divisor, so it is invisible to that test yet just as "
     "distorting. The test left campic at 1.1829 against 0.1143 at its native "
     "factor. Retained as a reported diagnostic only"),
    ("bound the fit exponent delta at Wang's published maximum 2.55",
     "removed the degenerate fit but made class separation worse, from -0.230 "
     "to -0.905 across nine images; the resolvable-floor fix addresses the "
     "same pathology without constraining the model family"),
    ("gate cells on decades of magnitude spanned (Benford's log-uniformity "
     "precondition)",
     "the precondition is not in fact violated: 456 of 477 measured cells span "
     "more than 1.1 decades, so the gate never fired and separation was "
     "unchanged at -0.749"),
    ("tighten the zero-coefficient-rate gate from 0.95",
     "gating at 0.75 moved separation from -0.749 to -0.747, i.e. no "
     "measurable effect; the correlation with the score is real but is a "
     "symptom of the texture confound, not a fixable defect"),
)
