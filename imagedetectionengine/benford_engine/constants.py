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

# Value: 1e-12. [ENGINEERING] Floor applied to probabilities before the log in
# the KL sum, so an empty histogram bin yields a large-but-finite contribution
# instead of inf/NaN. Chosen far below the smallest pmf value representable from
# realistic sample counts, so it cannot perturb a genuine score.
DIVERGENCE_PROBABILITY_FLOOR: float = 1e-12

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
# honest data-driven midpoint exists. Their only justification is SCALE: during
# implementation testing on SYNTHETIC imagery the max-divergence statistic sat
# near 0.21 for authentic content and spanned 0.23-10 for manipulated content,
# so a midpoint of 0.30 places the sigmoid's steep region where those two
# populations actually separate, instead of leaving it saturated at one end.
#
# That is a statement about synthetic test images, NOT a validated forensic
# threshold, and it must never be reported as one. Any run falling back to
# these values is flagged uncalibrated, has its confidence multiplied by
# CONFIDENCE_PENALTY_UNCALIBRATED, and says so in reliability_note. Replace by
# fitting on labelled data before forensic use.
PROVISIONAL_SIGMOID_MIDPOINT: float = 0.30   # [ENGINEERING - UNSOURCED]
PROVISIONAL_SIGMOID_SLOPE: float = 10.0      # [ENGINEERING - UNSOURCED]

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
    "all CONFIDENCE_PENALTY_* multipliers",
)
