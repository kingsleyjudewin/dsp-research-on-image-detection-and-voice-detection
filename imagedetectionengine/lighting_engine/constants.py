"""Every tunable parameter, threshold and magic number for the lighting engine.

RULE: no numeric literal with forensic meaning may appear anywhere else in this
package. Each constant below records its exact value, the paper it comes from
(via the SKILL file), and why this value rather than another.

READ THIS BEFORE TRUSTING THIS ENGINE'S OUTPUT. The SKILL file's own Corpus Gap
section states this module has "the thinnest evidentiary base of all nine" in
the system: "Zero papers in this folder ... perform actual photometric lighting
analysis". The only implementable technique (Pipeline A, a Sobel-gradient
magnitude heuristic) is described in the source paper as a single unvalidated
paragraph with an internally inconsistent decision rule, and the SKILL states
outright that this module "should carry the lowest reliability weight of the
nine detectors in the fusion layer". MAXIMUM_CONFIDENCE_CEILING below encodes
that instruction as a hard cap, unconditionally, regardless of what else this
engine computes.

Provenance tags used throughout:
    [CORPUS]      - value is printed explicitly in the SKILL file / source paper.
    [DERIVED]     - value is computed from a [CORPUS] value; derivation shown.
    [ENGINEERING] - value is NOT in the corpus. The SKILL file either flags the
                    quantity as unspecified/ambiguous or is silent. These are the
                    only values a reviewer needs to challenge, and every one of
                    them is listed in KNOWN_UNSOURCED_PARAMETERS at the bottom.
    [STRUCTURAL]  - fixed by a file format, a library requirement, or by
                    arithmetic, not a free choice.
    [PRESENTATION]- affects only the rendered evidence image, never the score.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

ENGINE_NAME: str = "illumination_inconsistency"

# Version of the SKILL document this implementation was written against. Bump
# whenever SKILL(Illumination Inconsistency Forgery Detection).md changes, so a
# stored forensic report can be traced back to the exact spec that produced it.
SKILL_VERSION: str = ("SKILL(Illumination Inconsistency Forgery Detection)"
                      ".md@2026-08-15")


# ---------------------------------------------------------------------------
# Array layout of the input
# ---------------------------------------------------------------------------

# BGR channel indices of the input array. [STRUCTURAL] OpenCV channel order,
# fixed by the EngineInput contract.
BLUE_CHANNEL_INDEX: int = 0
GREEN_CHANNEL_INDEX: int = 1
RED_CHANNEL_INDEX: int = 2

# Number of colour planes a valid BGR input carries. [STRUCTURAL]
EXPECTED_CHANNEL_COUNT: int = 3

# Array dimension counts for colour and single-plane images. [STRUCTURAL] SKILL
# "Input requirements": "any RGB image" - a grayscale array is also accepted,
# since the very first preprocessing step converts to grayscale anyway.
COLOUR_IMAGE_DIMENSION_COUNT: int = 3
GRAYSCALE_IMAGE_DIMENSION_COUNT: int = 2

# Milliseconds per second, for reporting processing_time_ms. [STRUCTURAL]
MILLISECONDS_PER_SECOND: float = 1000.0

# Smallest image side numpy.gradient can operate on at all. Value: 2.
# [STRUCTURAL] Verified empirically: numpy.gradient raises ValueError
# ("Shape of array too small to calculate a numerical gradient") on any axis of
# length 1, and succeeds at length 2, for both axes independently. This is a
# hard library requirement, not a design choice.
MINIMUM_GRADIENT_AXIS_LENGTH: int = 2

# Ordinal positions of each entry in the computation trace. [STRUCTURAL] -
# sequence labels for the report generator, not forensic parameters.
COMPUTATION_STEP_CONDITION_CHECK: int = 1
COMPUTATION_STEP_PREPROCESSING: int = 2
COMPUTATION_STEP_GRADIENT_COMPUTATION: int = 3
COMPUTATION_STEP_CALIBRATION: int = 4
COMPUTATION_STEP_FAILURE: int = 1


# ---------------------------------------------------------------------------
# Pipeline A - Sobel-gradient magnitude heuristic
# SKILL "Step-by-step algorithm" section A, steps 1-4
# Rao, Ghanekar, Chitnis, Dawkhar & Mishra 2025
# ---------------------------------------------------------------------------

# The MATLAB source code, transcribed exactly in the SKILL, computes
# [Gx, Gy] = gradient(double(gray_img)) then gradient_mag = sqrt(Gx.^2+Gy.^2).
# [STRUCTURAL] numpy.gradient(A) on a 2-D array returns (dA/d(axis0),
# dA/d(axis1)) - i.e. (vertical, horizontal) - whereas MATLAB's gradient(A)
# returns (FX, FY) = (horizontal, vertical), the opposite order. Verified this
# cannot matter here: gradient_mag = sqrt(Gx^2 + Gy^2) is symmetric in its two
# arguments, so sqrt(g0^2+g1^2) == sqrt(g1^2+g0^2) to floating-point identity
# regardless of which of numpy's two returned arrays is "Gx" and which is "Gy".
# No swap is therefore needed and none is performed.
NUMPY_GRADIENT_ORDER_MATCHES_MATLAB_MAGNITUDE: bool = True

# Divisor used to convert an unbounded max_grad into a scale-invariant ratio.
# Value: "median". [CORPUS-INFORMED] SKILL "Output" -> Pipeline A: "if used,
# normalize per-image (e.g. divide by the image's own median gradient
# magnitude) rather than using an absolute cutoff, since the source gives no
# calibration at all." This is the normalization actually implemented; the
# SKILL's Implementation Notes separately suggest "a percentile rank" as an
# alternative phrasing, but rank of the array's OWN maximum within itself is
# trivially 100% and therefore uninformative as a raw statistic - the
# percentile-style comparison this engine actually performs is instead pushed
# to the SCORING stage (empirical CDF of this ratio against a reference
# population of authentic images), which is where "percentile rank" is a
# well-defined, non-degenerate operation. See scorer.py.
GRADIENT_RATIO_DIVISOR: str = "median"

# Floor added to the median gradient magnitude before dividing. Value: 1e-9.
# [ENGINEERING] A perfectly flat region has zero median gradient, which would
# make the ratio divide by zero. This floor is far below any median produced by
# a real 8-bit image with non-zero variance, so it changes no genuine
# measurement - it exists only to keep the ratio finite. A median at or near
# this floor is instead caught explicitly by the degenerate-image condition
# check, which reports is_reliable=False rather than a huge or infinite ratio.
MEDIAN_GRADIENT_FLOOR: float = 1e-9

# Smallest median gradient magnitude, in per-pixel grey-level units, at which
# the image is considered to have enough texture for the ratio to be a
# meaningful statistic rather than noise divided by near-zero noise. Value:
# 0.01. [ENGINEERING - UNSOURCED] The SKILL gives no operating envelope for
# this module at all ("cannot be meaningfully characterized for this module
# from the current corpus"). This threshold exists only to distinguish a
# genuinely degenerate (constant or near-constant) image from one with real
# texture, and is set two orders of magnitude below the smallest median gradient
# a naturally textured 8-bit photograph produces.
MINIMUM_MEDIAN_GRADIENT_FOR_RATIO: float = 0.01


# ---------------------------------------------------------------------------
# ENHANCEMENT 1 (test-derived): degenerate images must not publish a score
# ---------------------------------------------------------------------------
# MEDIAN_GRADIENT_FLOOR keeps the ratio finite, but it does not keep it
# meaningful. On four Lambertian renders whose median gradient is exactly zero,
# the engine published raw_score = 96,873,629,022.56 - max_grad divided by
# 1e-9 - while correctly reporting is_reliable=False. The gate worked; the
# number it published alongside the abstention did not. A fusion layer reading
# raw_score without checking is_reliable would ingest 10^11.
#
# The skip and failure paths already publish raw_score 0.0. This makes the
# degenerate path agree with them, so an abstention looks like an abstention on
# every field, not just one. [DERIVED]
DEGENERATE_IMAGE_RAW_SCORE: float = 0.0

# ---------------------------------------------------------------------------
# ENHANCEMENT 2 (test-derived): abstain rather than emit a constant vote
# ---------------------------------------------------------------------------
# The provisional sigmoid has midpoint 3.0. The statistic it maps has an
# observed operating range of 18 to 392 across every image tested - the six
# supplied photographs (114 to 392) and the synthetic ground truth (18 to 36).
# Every one of those saturates the curve. Measured probabilities: 1.0000 on all
# six supplied images, giving a separation of EXACTLY 0.0, and 0.9996 to 1.0000
# on ground truth.
#
# A constant function carries no information. Emitting it as a FAKE vote at
# probability 1.0 can only add bias to the fusion layer, never evidence, and it
# is a confident FAKE on authentic photographs. The SKILL's Corpus Gap section
# names the alternative explicitly: "a near-zero or explicitly 'abstain' weight
# is more honest than presenting an unvalidated gradient heuristic as
# equivalent evidence".
#
# So the uncalibrated route no longer votes: is_reliable becomes False and
# probability None, while raw_score is still published for the record. The
# empirical-CDF route is unaffected - if an orchestrator supplies at least
# MINIMUM_CALIBRATION_REFERENCE_COUNT authentic reference ratios, the
# percentile rank against them is a real measurement and the engine votes
# normally. This makes voting conditional on calibration existing, which is the
# only circumstance under which the corpus supports a number at all. [DERIVED]
ABSTAIN_WHEN_UNCALIBRATED: bool = True

# ---------------------------------------------------------------------------
# ENHANCEMENT 3 (test-derived): exclude the one-sided-difference border
# ---------------------------------------------------------------------------
# numpy.gradient computes a centred difference in the interior but a ONE-SIDED
# difference on the first and last row and column, so boundary positions are
# produced by a different estimator than everywhere else and are not comparable
# with them. That is not a theoretical concern here: on the supplied image
# "fake .jpeg" the maximum gradient sits at row 0 - the array's own boundary -
# with 172 pixels tied at that value, so max_grad was reading the edge of the
# array rather than any edge in the scene. Excluding one row and column on each
# side moves that image's ratio by -15.50%.
#
# This narrows WHERE the SKILL's max(gradient_mag(:)) is evaluated; it does not
# change the formula, which is still the maximum of sqrt(Gx^2+Gy^2). The full
# map is still rendered into the evidence image. [DERIVED]
GRADIENT_BORDER_EXCLUSION_WIDTH: int = 1

# Smallest image side for which excluding the border still leaves an interior
# to measure. Two borders plus at least one interior row. [STRUCTURAL]
MINIMUM_SIDE_FOR_BORDER_EXCLUSION: int = 2 * GRADIENT_BORDER_EXCLUSION_WIDTH + 1


# ---------------------------------------------------------------------------
# Grayscale conversion
# SKILL "Input requirements" -> "Preprocessing"
# ---------------------------------------------------------------------------

# ITU-R BT.601 luma weights (R, G, B). [ENGINEERING - UNSOURCED, BUT NOT
# ARBITRARY] The SKILL states only that grayscale conversion is required
# ("double(gray_img) per the source's own MATLAB code") without printing the
# RGB-to-gray weighting the source paper's MATLAB call actually used. These are
# not a free engineering choice, however: they are MATLAB's own rgb2gray()
# default coefficients, and cv2.cvtColor(..., COLOR_BGR2GRAY) was verified here
# to reproduce them exactly (a synthetic BGR=(10,20,30) pixel converts to gray
# value 22 under both the closed-form 0.299R+0.587G+0.114B formula and
# cv2's own conversion). Implementing grayscale conversion via cv2 therefore
# matches what the source paper's own toolchain would have produced, even
# though the SKILL text itself does not print the coefficients.
LUMA_WEIGHT_RED: float = 0.299
LUMA_WEIGHT_GREEN: float = 0.587
LUMA_WEIGHT_BLUE: float = 0.114


# ---------------------------------------------------------------------------
# Confidence weights
# ---------------------------------------------------------------------------

# Neutral and disqualifying weights. [STRUCTURAL] endpoints of the [0,1] range.
FULL_CONFIDENCE: float = 1.0
ZERO_CONFIDENCE: float = 0.0

# THE central constant of this engine. Value: 0.05. [CORPUS-INFORMED, exact
# number is ENGINEERING] SKILL "Corpus gap": "this module should carry the
# lowest reliability weight of the nine detectors in the fusion layer ... a
# near-zero or explicitly 'abstain' weight is more honest than presenting an
# unvalidated gradient heuristic as equivalent evidence to, e.g., the CFA or
# Benford modules' well-validated statistics." 0.05 is not zero, because the
# module is still capable of producing a well-formed, reproducible number when
# the input is structurally valid, and an explicit is_reliable=False /
# skip_engine path already exists for genuinely unusable input; but it is
# capped an order of magnitude below the lightest penalty any other engine in
# this system applies, so the fusion layer can never mistake this vote for
# comparable evidence. This ceiling is applied UNCONDITIONALLY - every other
# confidence penalty in this engine multiplies further downward from it, never
# upward past it.
MAXIMUM_CONFIDENCE_CEILING: float = 0.05

# Additional penalty applied when the calibration route falls back to the
# provisional sigmoid rather than measured reference data. [ENGINEERING] Kept
# for interface consistency with the other engines' scorers, but its effect is
# small relative to MAXIMUM_CONFIDENCE_CEILING, which already dominates.
CONFIDENCE_PENALTY_UNCALIBRATED: float = 0.5


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

# Value: 30. [ENGINEERING] Fewest known-authentic reference scores that make an
# empirical CDF preferable to the provisional sigmoid. Same convention as the
# other engines in this system.
MINIMUM_CALIBRATION_REFERENCE_COUNT: int = 30

# WARNING - placeholders, not corpus values. The SKILL gives no calibration
# function for max_grad or any normalization of it whatsoever ("no
# normalization or calibration given"). These centre and scale the sigmoid only
# so it is not degenerate before real reference data exists; any result
# produced through this route is reported as uncalibrated AND is already capped
# by MAXIMUM_CONFIDENCE_CEILING regardless.
PROVISIONAL_SIGMOID_MIDPOINT: float = 3.0
PROVISIONAL_SIGMOID_SLOPE: float = 0.5

# Bound on the sigmoid exponent so an extreme ratio cannot overflow exp().
# [STRUCTURAL]
SIGMOID_EXPONENT_LIMIT: float = 700.0


# ---------------------------------------------------------------------------
# Presentation  -  affects the evidence image only, never a score
# ---------------------------------------------------------------------------

# Longest edge of the rendered evidence map, in pixels. [PRESENTATION]
EVIDENCE_MAP_MAX_DIMENSION: int = 512

# Percentile used to clip the gradient-magnitude map before colour-mapping it,
# so a single outlier pixel does not wash out the whole visualisation.
# [PRESENTATION] Affects only the rendered image, never raw_score.
EVIDENCE_DISPLAY_CLIP_PERCENTILE: float = 99.0

# Maximum value of an 8-bit display channel, used to scale the normalised
# gradient map before colour-mapping. [STRUCTURAL]
EIGHT_BIT_DISPLAY_MAXIMUM: float = 255.0

# Decimal places used when rounding values into the computation trace.
# [PRESENTATION] Affects only how numbers are displayed to the report
# generator and can never change a score.
TRACE_DECIMAL_PLACES: int = 4
TRACE_SCORE_DECIMAL_PLACES: int = 6


# ---------------------------------------------------------------------------
# Audit aid
# ---------------------------------------------------------------------------

# Every value in this file that is NOT traceable to the SKILL document. A
# reviewer checking this engine against the corpus only has to argue with these.
KNOWN_UNSOURCED_PARAMETERS: tuple = (
    "MEDIAN_GRADIENT_FLOOR",
    "MINIMUM_MEDIAN_GRADIENT_FOR_RATIO",
    "LUMA_WEIGHT_RED",
    "LUMA_WEIGHT_GREEN",
    "LUMA_WEIGHT_BLUE",
    "MAXIMUM_CONFIDENCE_CEILING",
    "CONFIDENCE_PENALTY_UNCALIBRATED",
    "MINIMUM_CALIBRATION_REFERENCE_COUNT",
    "PROVISIONAL_SIGMOID_MIDPOINT",
    "PROVISIONAL_SIGMOID_SLOPE",
)

# Parts of the SKILL document this engine deliberately does NOT implement, with
# the reason. Collected here so a reviewer can see the scope boundary at a
# glance rather than inferring it from absence.
TEST_DERIVED_ENHANCEMENTS: tuple = (
    "ENHANCEMENT 1 - DEGENERATE_IMAGE_RAW_SCORE. Four Lambertian renders with "
    "a median gradient of exactly zero published raw_score = 96,873,629,022.56 "
    "while reporting is_reliable=False. The degenerate path now publishes 0.0, "
    "agreeing with the skip and failure paths.",
    "ENHANCEMENT 2 - ABSTAIN_WHEN_UNCALIBRATED. The provisional sigmoid is a "
    "constant over the entire observed operating range: probability 1.0000 on "
    "all six supplied images (separation exactly 0.0) and 0.9996-1.0000 on "
    "ground truth. A constant vote is bias, not evidence, so the uncalibrated "
    "route now abstains - which the SKILL's Corpus Gap section names as the "
    "more honest option in as many words.",
    "ENHANCEMENT 3 - GRADIENT_BORDER_EXCLUSION_WIDTH. numpy.gradient uses a "
    "one-sided difference on the array boundary. On 'fake .jpeg' the maximum "
    "sat at row 0 with 172 tied pixels; excluding the border moves that "
    "image's ratio by -15.50%.",
)

REJECTED_ENHANCEMENTS: tuple = (
    "Replacing max_grad with a high percentile (p99, p99.9, p99.99) or with "
    "mean/std. Measured on Lambertian ground truth where the lighting "
    "inconsistency is exact by construction, NO reduction responds to lighting. "
    "Sensitivity to a genuine inconsistency against sensitivity to transforms "
    "that change no lighting at all: max/median 0.000000 against 14.96; "
    "p99.99/median 0.134 against 14.53; p99.9/median 0.888 against 13.20; "
    "p99/median 0.004 against 5.27; mean/median 0.000128 against 0.204. The "
    "best nuisance-to-lighting ratio of any candidate is 13.8x, and the "
    "shipped max is literally 0 sensitivity. A percentile would therefore buy "
    "stability without buying validity, at the cost of changing a formula the "
    "SKILL prints verbatim (max_grad = max(gradient_mag(:))). Not worth it.",
    "Re-fitting PROVISIONAL_SIGMOID_MIDPOINT to the observed operating range. "
    "It would turn a constant into a curve fitted to six unlabelled corpus "
    "images and a synthetic scene, which is inventing the calibration the "
    "SKILL states does not exist ('no normalization or calibration given'). "
    "Abstaining is the honest response to an absent calibration, not guessing "
    "one. See ENHANCEMENT 2.",
    "Implementing a block-wise or region-wise version of the gradient "
    "statistic to give the engine some spatial sensitivity. This is Pipeline B "
    "in disguise, and the SKILL tags every step of Pipeline B '(not specified "
    "in the corpus)' and calls it 'entirely engineering synthesis'. Building "
    "it would mean inventing a technique, not implementing one.",
)

KNOWN_UNIMPLEMENTED_MODULES: tuple = (
    "Pipeline B (Spherical-Harmonics photometric consistency): every single "
    "step is explicitly tagged '(not specified in the corpus)' in the SKILL "
    "file itself. There is no formula, threshold, or parameter to implement - "
    "the SKILL calls it 'entirely engineering synthesis ... included because it "
    "is the technically correct approach ... so that the module has a target "
    "to build toward'. Building it would mean inventing a face-landmark "
    "detector, a 3D morphable model fit, and an SH least-squares solve with no "
    "source basis for any of the three.",
    "Pipeline C (dual-attention DeepFake localizer, Li et al. 2021): tagged "
    "'[ML - excluded from the no-ML engine]' in the SKILL file, and the SKILL "
    "itself states it 'is not a lighting-modeling technique at all' - its only "
    "connection to illumination is one motivating sentence in the source "
    "paper's introduction about a single generator (FaceSwap). Excluded on "
    "both the ML constraint and lack of topical relevance.",
    "The 'multiple light directions' half of Pipeline A step 3's decision "
    "rule: the SKILL flags this itself as an unresolved internal "
    "inconsistency - 'the paper does not describe how a single scalar "
    "max_grad (a magnitude, with no directional/angular component) could "
    "detect multiple light directions'. No directional gradient decomposition "
    "is implemented, because none is specified; only the magnitude half of the "
    "rule is computable at all.",
    "PBCA and IINC (Pipeline C's localization-quality metrics): their formal "
    "formulas are stated in the SKILL to be 'not present in the extracted "
    "pages of this paper', and Pipeline C itself is excluded, so these metrics "
    "have no role in this engine.",
)
