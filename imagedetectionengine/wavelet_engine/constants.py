"""All parameters, thresholds, and provenance tags for the wavelet-domain engine.

Provenance tags used throughout this file:
    [CORPUS]      - value/formula printed verbatim in the SKILL file.
    [CORPUS-RANGE]- SKILL gives a validated *range*, not one exact value; the
                    specific point picked within that range is documented below.
    [DERIVED]     - not printed in the SKILL, but algebraically derivable from
                    a formula the SKILL DOES print (shown in the comment).
    [ENGINEERING] - the SKILL explicitly states no value is given ("user-defined",
                    "user/image-characteristic-dependent") and this module must
                    supply a working default to run at all.
    [STRUCTURAL]  - shape/type constants with no forensic meaning of their own.
    [PRESENTATION]- display-only, never affects a score.

SCOPE DECISION (applies to every constant below): this SKILL describes three
independent pipelines. Pipeline A's own output is explicitly "not a scalar
score on its own ... an intermediate signal consumed by the noise analysis
module" (SKILL, Output section). Pipeline B's own output is explicitly
"low-trust by default, not high-confidence evidence" with no formula anywhere
for combining it with Pipeline C's score. Inventing that combination weight
would itself be a value not present in the SKILL file. Therefore this engine's
raw_score is driven by Pipeline C alone; Pipelines A and B are fully computed
per their SKILL algorithms and reported as auxiliary, non-scoring evidence.
"""

from __future__ import annotations

ENGINE_NAME = "wavelet_domain_forgery"  # [STRUCTURAL]
SKILL_VERSION = "1.0.0"  # [STRUCTURAL]

# ── Structural / contract constants ─────────────────────────────────────────
GRAYSCALE_IMAGE_DIMENSION_COUNT = 2  # [STRUCTURAL]
COLOUR_IMAGE_DIMENSION_COUNT = 3  # [STRUCTURAL]
FULL_CONFIDENCE = 1.0  # [STRUCTURAL]
ZERO_CONFIDENCE = 0.0  # [STRUCTURAL]
TRACE_DECIMAL_PLACES = 4  # [PRESENTATION]
TRACE_COARSE_DECIMAL_PLACES = 2  # [PRESENTATION]
MILLISECONDS_PER_SECOND = 1000.0  # [STRUCTURAL]

# ENHANCEMENT 1 (test-derived): SWT requires both image dimensions to be
# divisible by 2**levels. 3 of the 6 supplied corpus images are not, and each
# raised ValueError out of Pipeline A - an explicitly NON-SCORING pipeline -
# which analyse() caught and turned into a whole-engine failure, discarding a
# completed Pipeline C result. The SKILL's Implementation Notes prescribe the
# remedy directly: "use symmetric/reflect padding, PyWavelets' default
# 'symmetric' mode". [CORPUS] for the mode, [STRUCTURAL] for the base.
SWT_DIMENSION_MULTIPLE_BASE = 2
BOUNDARY_EXTENSION_MODE = "symmetric"

# ── Pipeline A: wavelet noise-residual extraction ───────────────────────────
# Formula: sigma_W_Phi = median(|HH_i|) / 0.6745
# Source: mgaga2019, Eq. 7. 0.6745 is the 75th-percentile point of the
# standard half-normal distribution, converting a MAD into an unbiased
# Gaussian sigma estimate. [CORPUS]
NOISE_MAD_CONSTANT = 0.6745

# SKILL: "2-3 levels recurs as the practically-validated range across sources
# ... treat 2-3 as a reasonable starting default, not a rigorSKILL(Wavelet-Domain Forgery Detection)ously-derived
# optimum". [CORPUS-RANGE]; 3 is picked as the upper, more-cited end of that
# range (also the level Zhu & Wang's own experiment uses). [ENGINEERING]
PIPELINE_A_DECOMPOSITION_LEVELS = 3

# SKILL: "db2/sym4 are recommended by mgaga2019's review as 'smoother ...
# better for speckle noise' for the noise-estimation path". [CORPUS]
PIPELINE_A_WAVELET_FAMILY = "sym4"

# SKILL Implementation Notes: "Use SWT ... for any step requiring precise
# pixel-level localization" (Pipeline A feeds spatial noise-inconsistency
# maps). [CORPUS]
PIPELINE_A_USE_SWT = True

# Formula: T = sigma * sqrt(2 * log(N))
# Source: mgaga2019, Eq. 3, citing Donoho 1995. [CORPUS]
VISUSHRINK_LOG_COEFFICIENT = 2.0

# Formula: T = sigma * sqrt(1.618 * log(N))
# Source: mgaga2019, Eq. 13, reviewing Sasirekha et al. - golden ratio
# replaces VisuShrink's constant 2. [CORPUS]
GOLDEN_RATIO_LOG_COEFFICIENT = 1.618

# Formula: W(x,y) = 1 / exp(HH(x,y)), applied before the golden-ratio
# method's weighted-median sigma estimate.
# Source: mgaga2019, Eq. 14-16 (Sasirekha et al.). [CORPUS]
GOLDEN_RATIO_WEIGHT_IS_INVERSE_EXP = True

# Zhu & Wang 2021 Eq. 3 piecewise shrinkage: paper's own experiment uses
# alpha=0.5, n=2, 3-level sym4. [CORPUS]
ZHU_WANG_ALPHA = 0.5
ZHU_WANG_N = 2
ZHU_WANG_DECOMPOSITION_LEVELS = 3
ZHU_WANG_WAVELET_FAMILY = "sym4"

# The SKILL presents four threshold formulas as "genuine disagreement - use
# per the stated tradeoff, not a single 'best' value" and never nominates one
# as the overall default. A default must exist for the engine to run
# unconfigured; VisuShrink is picked because it is the only one of the four
# that is a plain closed-form function of sigma and N with no extra tunable
# constants of its own. [ENGINEERING]
DEFAULT_THRESHOLD_METHOD = "visushrink"
THRESHOLD_METHODS = ("visushrink", "bayesshrink", "golden_ratio", "piecewise")

# Formula: hard f(x)={x,|x|>=T;0,|x|<T}; soft f(x)={x-T,x>T;0,|x|<=T;x+T,x<-T}
# Source: mgaga2019 Eq. 1-2 / Zhu & Wang Eq. 1-2 (identical definitions
# across both sources). [CORPUS]
# The SKILL gives both forms without nominating a default; soft thresholding
# is picked because it is the form Zhu & Wang's own comparison (Table 1)
# shows outperforming hard thresholding on both SNR and MSE. [ENGINEERING]
DEFAULT_THRESHOLD_MODE = "soft"
THRESHOLD_MODES = ("hard", "soft", "piecewise")

# ── Pipeline B: wavelet-compression-history / Laplacian fit ────────────────
# SKILL does not give an explicit decomposition level for Pipeline B; reuses
# the same corpus-validated 2-3 range as Pipeline A. [CORPUS-RANGE] /
# [ENGINEERING]
PIPELINE_B_DECOMPOSITION_LEVELS = 3
PIPELINE_B_WAVELET_FAMILY = "sym4"

# Formula: P(X=x) = (lambda/2) * exp(-lambda*|x|)
# Source: Stamm & Liu 2010, Eq. 2. [CORPUS]
# Formula: h_k = c * exp(-lambda_hat * |q_k|), fit via weighted least squares
# on min_{lambda_hat,c} sum_k h_k * (log(h_k) - log(c) + lambda_hat*|q_k|)^2
# Source: Stamm & Liu 2010, Eq. 4-5. [CORPUS]
#
# Eq. 6 ("a closed-form 2x2 linear system") is referenced but its explicit
# form is never printed in the SKILL. [DERIVED] below: writing y_k=log(h_k),
# the weighted least-squares normal equations for y_k = log(c) - lambda*|q_k|
# with weights w_k=h_k are the standard 2x2 system
#   [ sum(w)          -sum(w*|q_k|)   ] [log(c) ]   [ sum(w*y_k)        ]
#   [ -sum(w*|q_k|)    sum(w*|q_k|^2) ] [lambda ] = [ -sum(w*|q_k|*y_k) ]
# solved directly with numpy.linalg.solve. This is algebra applied to a
# formula the SKILL does print (Eq. 5), not a new parameter.
LAPLACIAN_FIT_IS_WEIGHTED_LOG_LINEAR = True

# Formula: h_hat_k^(i) = {c^(i), k=0; h_k+0.5*(h_0-c^(i)), k=+-1; h_k, else}
# Source: Stamm & Liu 2010, Eq. 7 (iterative bitplane-truncation-bias
# correction). [CORPUS]
# Convergence rule: terminate when |lambda_hat^(i)-lambda_hat^(i-1)|
#   / lambda_hat^(i) < tau, "tau a user-defined tolerance" - SKILL gives no
# numeric tau. [ENGINEERING] / KNOWN_UNSOURCED_PARAMETER
BIAS_CORRECTION_CONVERGENCE_TOLERANCE = 1.0e-3
BIAS_CORRECTION_MAX_ITERATIONS = 50  # SKILL gives no cap. [ENGINEERING]

# The SKILL fits h_k = "the observed count at quantized value q_k" - for an
# image under test, the quantization bin boundaries of whatever encoder (if
# any) produced it are not known a priori, and the SKILL gives no blind
# bin-width estimator. The empirical coefficient histogram is used directly
# as h_k, binned at this resolution. [ENGINEERING] / KNOWN_UNSOURCED_PARAMETER
PIPELINE_B_HISTOGRAM_BIN_COUNT = 51

# Numerical safety clip on log(c) before exponentiating in the bias-
# correction loop - not a SKILL value, purely prevents float64 overflow on
# an ill-conditioned fit. [STRUCTURAL]
LOG_C_EXPONENT_CLIP = 700.0

# ── Pipeline C: Haar-DWT copy-move block matching ───────────────────────────
# Source: Kashyap & Joshi 2013. [CORPUS]
PIPELINE_C_WAVELET_FAMILY = "haar"  # Eq. 1-2, chosen by the paper for speed.

# SKILL benchmark table runs both R=8 and R=16; 16 is picked as the default
# because it is the larger (more stable-moment) of the two paper-tested
# sizes. [CORPUS], value selected among two attested options [ENGINEERING]
DEFAULT_BLOCK_SIZE = 16

# "sliding by 1 pixel horizontally then vertically". [CORPUS]
BLOCK_STRIDE_PIXELS = 1

# "24 blur invariants up to 7th order" - the SKILL states the construction is
# "a recursive construction rule, not a closed enumerated list", i.e. it does
# not name which (p,q) pairs make the 24. This engine enumerates all (p,q)
# with 2 <= p+q <= MAXIMUM_MOMENT_ORDER systematically (see computer.py),
# rather than guessing the paper's specific subset. [CORPUS] order bound,
# [ENGINEERING] enumeration.
MAXIMUM_MOMENT_ORDER = 7
MINIMUM_MOMENT_ORDER = 2

# Eq. 12's recursive term contains "mu_{t=2i,2i}"; every other index pair in
# the same formula (e.g. B(p-t+2i, q-2i)) is a subtracted pair, and a lone
# "=" inside a subscript is a common PDF-text-extraction artifact for "-".
# Interpreted as mu_{t-2i, 2i}. KNOWN_SKILL_AMBIGUITY, documented, not a
# silent guess.
BLUR_INVARIANT_SUBSCRIPT_INTERPRETATION = "t-2i,2i"

# Formula: k = floor((p+q-4)/2) - the order below which B(p,q)=mu_pq with no
# recursive correction term. Source: Eq. 12-15. [CORPUS]
BLUR_INVARIANT_ORDER_OFFSET = 4

# Formula: B'_i = B_i / ((R/2)^r * mu_00), R=block size, r=order of B_i.
# Source: Eq. 17 (contrast-normalized invariants). [CORPUS]
CONTRAST_NORMALISATION_APPLIES = True

# "keeping only m_0 << m components" - SKILL gives no explicit m_0.
# [ENGINEERING] / KNOWN_UNSOURCED_PARAMETER: retain the smallest number of
# principal components whose cumulative explained-variance ratio reaches
# this fraction.
PCA_EXPLAINED_VARIANCE_TARGET = 0.95
PCA_MINIMUM_COMPONENTS = 1

# Formula: S(Bi,Bj) = 1 / (1 + rho(Bi,Bj)); rho = Euclidean distance.
# Source: Eq. 27-28. [CORPUS]
# "S(Bi,Bj) >= T (a user/image-characteristic-dependent similarity
# threshold)" - SKILL gives no numeric T. [ENGINEERING] /
# KNOWN_UNSOURCED_PARAMETER
SIMILARITY_THRESHOLD = 0.95

# "examine 16 neighboring blocks within maximum distance 4 pixels".
# [CORPUS]: the offset radius (4) and count (16) are both stated; which
# specific 16 of the 80 non-zero lattice points within [-4,4]x[-4,4] is not
# stated. Resolved deterministically (row-major scan order, see
# utils.generate_neighbour_offsets) rather than an arbitrary/random choice.
# [CORPUS] radius+count, [ENGINEERING] selection order.
NEIGHBOUR_CHECK_MAX_OFFSET_PIXELS = 4
NEIGHBOUR_CHECK_COUNT = 16

# "spatial separation between the original candidate blocks must exceed a
# minimum distance D" - SKILL gives no numeric D, only the shape of the rule
# (Eq. 30). [ENGINEERING] / KNOWN_UNSOURCED_PARAMETER: tied to block size
# (2 block-widths) rather than an arbitrary pixel count, since the rule
# exists specifically to reject smooth-region self-similarity between
# overlapping/adjacent blocks.
MINIMUM_SEPARATION_BLOCK_MULTIPLE = 2.0

# ── Condition-checker constants ─────────────────────────────────────────────
# SKILL, Input requirements: unreliable when copy-move regions are scaled or
# rotated (undetectable by this pipeline, not a numeric gate); when content
# has been aggressively smoothed/denoised after tampering (defeats the
# noise-residual path); when anti-forensic dithering has been applied
# (Pipeline B's documented 100% defeat). None of these are quantified
# thresholds in the SKILL - they are structural/qualitative conditions,
# checked as such in condition.py. [CORPUS]
MOMENT_DEGENERACY_FLOOR = 1.0e-6  # mu_00 below this treated as near-zero mass.
# [ENGINEERING]: no formula-level floor is given; this is the point below
# which Eq. 17's division by mu_00 becomes numerically meaningless.

# A flat/textureless block's contrast-normalised blur invariants are purely
# geometric (intensity cancels through the mu_00 division in Eq. 17), so
# every flat block in the image - however far apart - maps to the same
# feature vector and Eq. 27's similarity threshold trivially confirms them
# as "duplicates". Eq. 30's minimum-separation rule only rejects nearby
# coincidental matches, not this global case. No numeric variance floor is
# given in the SKILL. [ENGINEERING] - discovered via testing (see condition.py
# assess_texture_degeneracy), same pattern as this system's other engines'
# testing-discovered false-positive gates.
BLOCK_TEXTURE_VARIANCE_FLOOR = 1.0
MAXIMUM_FLAT_BLOCK_FRACTION = 0.5  # [ENGINEERING]

MINIMUM_BLOCKS_FOR_PCA = 2  # [STRUCTURAL] - PCA needs at least 2 samples.

# ── Scorer constants (provisional-route placeholders, same pattern as the
# other engines in this system; SKILL gives no calibration for the Pipeline C
# fraction-of-flagged-blocks scalar - it is explicitly an "engineering
# recommendation", not a corpus value: "(not explicitly defined in the
# source paper as a summary scalar - engineering recommendation: sum of Q,
# or largest connected-component size, for a fusion-layer scalar)".
# [ENGINEERING]
MINIMUM_CALIBRATION_REFERENCE_COUNT = 10
PROVISIONAL_SIGMOID_SLOPE = 12.0
PROVISIONAL_SIGMOID_MIDPOINT = 0.05
SIGMOID_EXPONENT_LIMIT = 60.0

# ── Visualisation constants ─────────────────────────────────────────────────
EVIDENCE_DISPLAY_CLIP_PERCENTILE = 99.0  # [PRESENTATION]
EIGHT_BIT_DISPLAY_MAXIMUM = 255.0  # [STRUCTURAL]
EVIDENCE_MAP_MAX_DIMENSION = 2048  # [PRESENTATION]

# ── Audit-aid tuples ─────────────────────────────────────────────────────────
KNOWN_UNSOURCED_PARAMETERS = (
    "BIAS_CORRECTION_CONVERGENCE_TOLERANCE (tau) - SKILL: 'a user-defined "
    "tolerance', no numeric value given.",
    "BIAS_CORRECTION_MAX_ITERATIONS - no cap given for Eq. 7's iteration.",
    "PIPELINE_B_HISTOGRAM_BIN_COUNT - SKILL assumes known quantization bin "
    "boundaries {b_k}; no blind bin-width estimator is given for an "
    "arbitrary image under test.",
    "PCA_EXPLAINED_VARIANCE_TARGET / PCA_MINIMUM_COMPONENTS - SKILL: "
    "'keeping only m_0 << m components', no numeric m_0.",
    "SIMILARITY_THRESHOLD (T) - SKILL: 'a user/image-characteristic-"
    "dependent similarity threshold', no numeric value given.",
    "MINIMUM_SEPARATION_BLOCK_MULTIPLE (D) - SKILL: 'a minimum distance D', "
    "no numeric value given.",
    "MOMENT_DEGENERACY_FLOOR - no formula-level floor given for mu_00.",
    "BLOCK_TEXTURE_VARIANCE_FLOOR / MAXIMUM_FLAT_BLOCK_FRACTION - no "
    "numeric floor given for detecting textureless blocks; discovered as "
    "necessary via testing, see condition.py assess_texture_degeneracy.",
    "PROVISIONAL_SIGMOID_SLOPE / MIDPOINT - placeholder fallback route only, "
    "SKILL gives no calibration at all for the fraction-of-flagged-blocks "
    "scalar (itself an engineering recommendation, not a corpus value).",
)

KNOWN_SKILL_AMBIGUITIES = (
    "Eq. 6 (Pipeline B's 2x2 linear system) is referenced but never printed "
    "in the SKILL; derived here via standard weighted least-squares normal "
    "equations applied to Eq. 5 (see the comment above "
    "LAPLACIAN_FIT_IS_WEIGHTED_LOG_LINEAR).",
    "Eq. 12's 'mu_{t=2i,2i}' subscript interpreted as 'mu_{t-2i,2i}' - see "
    "BLUR_INVARIANT_SUBSCRIPT_INTERPRETATION.",
    "The exact (p,q) pairs making up 'the 24 blur invariants up to 7th "
    "order' are not enumerated in the SKILL ('a recursive construction "
    "rule, not a closed enumerated list'); this engine enumerates all "
    "(p,q) with 2<=p+q<=7 systematically instead of guessing the paper's "
    "specific subset - see MAXIMUM_MOMENT_ORDER.",
    "The specific 16 of 80 candidate lattice offsets for the neighbour-"
    "consistency check (Eq. 29) are not named in the SKILL; resolved via a "
    "deterministic row-major scan order - see NEIGHBOUR_CHECK_COUNT.",
)

KNOWN_UNIMPLEMENTED_MODULES = (
    "RGB 72-dimensional per-channel-concatenated feature vectors (Kashyap "
    "& Joshi's stated RGB extension) - grayscale-only 24-per-order feature "
    "vectors are implemented; the paper's primary pipeline is grayscale.",
    "Pipeline A's four reviewed noise-estimation studies (Yinping et al., "
    "Zaki et al., Iqbal, Sasirekha et al.) are drawn from non-directly-"
    "comparable datasets per the SKILL's own caveat; only the formulas "
    "themselves are implemented, not any single study's specific benchmark "
    "reproduction.",
    "Anti-forensic dither construction (Stamm & Liu Eq. 9-13) is fully "
    "specified in the SKILL for red-team/robustness testing of this "
    "engine's own Pipeline B, but is out of scope for a detection engine - "
    "not implemented here.",
)
