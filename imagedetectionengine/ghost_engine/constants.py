"""All parameters and thresholds for the Fourier-domain / JPEG-ghost engine.

Provenance tags used throughout this file:
    [CORPUS]        - value/formula printed verbatim in the SKILL file.
    [CORPUS-REC]    - the SKILL itself supplies this as its own explicit
                      engineering recommendation, flagged there as not being
                      a printed value from the source paper.
    [DERIVED]       - not printed, but forced by a constraint the SKILL's own
                      formulas operate under (shown in the comment).
    [ENGINEERING]   - the SKILL states no value is given, and this module must
                      supply a working default to run at all.
    [STRUCTURAL]    - shape/type constants with no forensic meaning of their own.
    [PRESENTATION]  - display-only, never affects a score.

SCOPE DECISION (applies to every constant below).

This SKILL contains three pipelines, and labels BOTH A and B "PRIMARY" while
giving no formula anywhere for combining them. Rather than invent a fusion
weight, the split follows which pipeline the corpus actually equips with a
complete decision rule:

  * Pipeline B (JPEG ghost) DRIVES raw_score. It is the only pipeline with a
    numeric decision threshold (Th=0.19, fitted FP=FN on 1000 original +
    1000 tampered UCID images), published benchmarks (97.73% accuracy /
    91.01% precision), and an explicit fusion recommendation in the SKILL's
    Output section: "score = clip(D_max / Th, 0, 1)". It also produces the
    segmentation mask the SKILL calls "the strongest localization output of
    any technique in this module - it is literally a segmented binary tamper
    mask, not just a score", which is why this engine populates
    flagged_regions where the other engines in this system cannot.

  * Pipeline A (resampling periodicity) runs and is REPORTED, but cannot
    score. Its threshold rho_T is "calibrated empirically per desired
    false-acceptance rate" with no numeric value printed anywhere, and its
    contrast function gamma is explicitly "not printed in the extracted text
    of this paper". An uncalibrated rho compared against an absent threshold
    is not a probability, so rho is surfaced as auxiliary evidence with that
    stated plainly.

  * Pipeline C is NOT implemented. The SKILL calls it "fusion-layer pattern
    reference, not a Fourier/ghost-specific technique"; its decision stage is
    tagged [ML - excluded]; its Sobel lighting cue belongs to this system's
    lighting module (already built, under a hard confidence ceiling); and the
    SKILL's own corpus-honesty note states the source "reports no
    quantitative accuracy/precision/F1/AUC numbers anywhere in the extracted
    text ... Do not cite this paper as evidence of validated detection
    performance." See KNOWN_UNIMPLEMENTED_MODULES.
"""

from __future__ import annotations

import numpy as np

ENGINE_NAME = "fourier_jpeg_ghost"  # [STRUCTURAL]
SKILL_VERSION = "1.0.0"  # [STRUCTURAL]

# ── Structural / contract constants ─────────────────────────────────────────
GRAYSCALE_IMAGE_DIMENSION_COUNT = 2  # [STRUCTURAL]
COLOUR_IMAGE_DIMENSION_COUNT = 3  # [STRUCTURAL]
FULL_CONFIDENCE = 1.0  # [STRUCTURAL]
ZERO_CONFIDENCE = 0.0  # [STRUCTURAL]
TRACE_DECIMAL_PLACES = 4  # [PRESENTATION]
MILLISECONDS_PER_SECOND = 1000.0  # [STRUCTURAL]

# ── Pipeline B: JPEG ghost detection (SCORE-DRIVING) ───────────────────────
# Formula: delta = (1/3w^2) * sum_{c in {R,G,B}} sum_{i,j=0}^{w-1}
#                  ( I(x+i,y+j,c) - I_q2(x+i,y+j,c) )^2
# Source: Azarian-Pour, Babaie-Zadeh & Sadri 2016, Eq. 6 (and Eq. 8 for the
# grid-shifted form). "w = 16" is the paper's stated default, "following the
# original method" (Farid's). [CORPUS]
GHOST_SMOOTHING_WINDOW = 16

# The 3 in the 1/(3w^2) normaliser is the RGB channel count Eq. 6 sums over.
# [CORPUS]
GHOST_CHANNEL_COUNT = 3

# "Full sweep: q2 = 1,...,100 x (d_x,d_y) in {0,...,7}^2 = 6400 total
# (q2, dx, dy) combinations". Source: Azarian-Pour 2016, Step 1. [CORPUS]
QUALITY_FACTOR_MINIMUM = 1
QUALITY_FACTOR_MAXIMUM = 100
GRID_SHIFT_MAXIMUM = 7
FULL_SEARCH_COMBINATION_COUNT = 6400

# Formula: B = (1/2)*ln[ (s0^2 + s1^2) / (2*s0*s1) ]
#              + (m0 - m1)^2 / (4*(s0^2 + s1^2))
# Source: Azarian-Pour 2016, Eq. 10. [CORPUS]
#
# "Classify the image as forged if D_max > Th. Threshold Th = 0.19,
# determined by minimizing the classification error rate (FP = FN) on a
# training set of 1000 original + 1000 tampered UCID images." [CORPUS]
BHATTACHARYYA_THRESHOLD = 0.19

# Eq. 10's own numeric coefficients, named rather than inlined so no bare
# constant of the formula lives outside this file. [CORPUS]
BHATTACHARYYA_SPREAD_COEFFICIENT = 0.5      # the leading (1/2) on the ln term
BHATTACHARYYA_VARIANCE_PRODUCT_FACTOR = 2.0  # the 2 in 2*sigma0*sigma1
BHATTACHARYYA_MEAN_TERM_DIVISOR = 4.0        # the 4 in 4*(sigma0^2+sigma1^2)

# "exceeds 95% average sensitivity once Delta_q > 22". Reported for the
# record and surfaced in the reliability note; not computable per-image,
# since q0 (the spliced region's original quality) is exactly what is
# unknown. [CORPUS]
RELIABLE_QUALITY_FACTOR_GAP = 22

# ── Pipeline B pruning (sanctioned by the SKILL, values are ours) ──────────
# SKILL Implementation Notes: "The 6400-run ghost sweep is the dominant cost
# in Pipeline B - for a practical engine, prune it: (a) skip grid shifts if
# the paste region is already known/suspected to be block-aligned ...
# reducing 64->1; (b) coarsen the q2 sweep to every 2-5 steps rather than
# every integer quality factor, since ghost minima are broad, not
# needle-sharp (not explicitly validated in the corpus, but consistent with
# the smoothing window w=16 already blurring fine q2 resolution)."
#
# The pruning is therefore corpus-sanctioned; the specific step sizes are
# ours. Both are exposed on CalibrationSettings so a caller can request the
# full 6400-combination sweep. [ENGINEERING] within a [CORPUS] sanction.
DEFAULT_QUALITY_FACTOR_STEP = 5  # top of the SKILL's stated 2-5 range
DEFAULT_GRID_SHIFT_STEP = 2  # {0,2,4,6}^2 = 16 of the 64 shifts
#
# BOTH VALUES ARE UNCHANGED FROM THE BUILD, AND THAT IS A TESTED RESULT
# RATHER THAN AN UNTESTED DEFAULT. A finer quality sweep was implemented,
# measured and reverted; see REJECTED_ENHANCEMENTS for the numbers.

# ── Pipeline B segmentation (SE-MinCut substitute) ─────────────────────────
# SKILL Step 2 calls for SE-MinCut (Estrada & Jepson), "chosen specifically
# for its robustness against the fractal-noise-like texture typical of
# ghost/difference maps". SKILL Implementation Notes: "no direct SE-MinCut
# implementation; skimage.segmentation.slic + graph-cut ... or maxflow/
# PyMaxflow for a direct min-cut formulation ... (the exact SE-MinCut
# algorithm is not available as a standard Python package; this is an
# engineering substitution, not a like-for-like reproduction)."
#
# The substitute used here is the SKILL's own first-named fallback: SLIC
# superpixels (which impose the spatial coherence that makes the segmenter
# noise-robust, the property SE-MinCut was chosen for), then a two-class
# split of the superpixel MEANS. Eq. 10 is then evaluated on the raw pixel
# values of each class, not on the superpixel means - deliberately, so that
# the segmenter's own separation objective does not inflate B for every
# image including authentic ones. [ENGINEERING], flagged by the SKILL itself.
SEGMENTATION_SUPERPIXEL_COUNT = 150
SEGMENTATION_COMPACTNESS = 0.1  # low: follow value, not spatial squareness
SEGMENTATION_START_LABEL = 0

# ENHANCEMENT 1 (test-derived): the split is taken over THREE classes, not
# two, and the ghost is the lowest band.
#
# A normalised difference map holds three populations, not two: the ghost
# (near zero at its own original quality), flat background (low at every
# quality because it has little detail to lose), and textured background
# (high). Two-class Otsu spends its single cut separating the textured tail,
# which leaves the ghost buried inside a majority class. Measured on the six
# supplied photographs before this change, class-0 held a median 0.607 to
# 0.781 of the frame and 18-20 of every 20 candidates were thrown out by the
# area test as "too large" - so the only candidates that ever survived were
# the degenerate ones at the extreme low-quality end of the sweep, and the
# engine's winning q2 was 1 or 6 on all six images.
#
# On ground truth the effect is direct. For a q0=50 splice at its true ghost
# quality q2=51, the two-class split returned class-0 covering 0.740 of the
# frame with IoU 0.320 against the real paste; the three-class split returned
# 0.265 of the frame with IoU 0.858, and the area test then passes instead of
# rejecting it.
SEGMENTATION_CLASS_COUNT = 3

# WHY A VALIDITY TEST ON THE SEGMENTED CANDIDATE IS NOT OPTIONAL.
#
# Eq. 10's first term, (1/2)ln[(s0^2+s1^2)/(2*s0*s1)], is a variance-MISMATCH
# term: it grows whenever the two classes have unequal spread, entirely
# independently of their means. Segmenting is therefore enough, on its own,
# to manufacture a large B out of a completely uniform noise map - cutting a
# unimodal distribution anywhere yields a narrow tail against a broad
# remainder, and Eq. 10 rewards that. Measured on this engine before the
# constraints below existed: an authentic single-compressed image scored
# D_max = 1.30 against a genuine splice's 1.24. The statistic was not merely
# weak, it was inverted, and every input saturated an order of magnitude
# above Th = 0.19.
#
# SE-MinCut does not have this failure because it is a spatially regularised
# cut, not a value split: on a map with no coherent structure it has nothing
# to cut along. The SKILL names exactly that property as the reason the
# authors chose it - "robustness against the fractal-noise-like texture
# typical of ghost/difference maps". Since no SE-MinCut implementation is
# available, the two properties it supplies are reimposed explicitly as
# validity conditions on the segmented candidate:
#
#   1. SPATIAL COHERENCE. A real ghost is one contiguous pasted region;
#      thresholded fractal noise is scattered confetti. The candidate must
#      hold most of its area in a single connected component.
#   2. MINORITY EXTENT. A splice is a region of the frame, not the frame.
#      A uniform map drifting wholesale across the split point produces a
#      "ghost" covering most of the image - measured at 50-86% on authentic
#      inputs, against 9-17% for a genuine splice.
#
# A candidate failing either is reported as a degenerate segmentation with
# B = 0 rather than a fabricated separation. Measured with both in force:
# spliced (q0=50 into q1=90) D_max = 1.241, firing at q2=51 against a true
# q0 of 50 with class-0 at 16.7% against a true splice area of 14%;
# authentic D_max = 0.000; reverse-direction splice (q0=95 into q1=70)
# D_max = 0.000, which is the correct answer per the SKILL's own statement
# that the method "does not resolve the reverse case".
# [ENGINEERING], restoring what the unavailable SE-MinCut would have given.
MINIMUM_GHOST_SPATIAL_COHERENCE = 0.5
MAXIMUM_GHOST_AREA_FRACTION = 0.4

# The extent condition is two-sided. A candidate covering a handful of pixels
# is no more a pasted region than one covering the whole frame, but it can
# still be perfectly contiguous and so satisfy the coherence test. Measured
# before this floor existed: at the most extreme candidate quality the
# normalised map is near-uniform, Otsu shaves off a 0.7%-of-frame sliver, and
# a spliced and an authentic image both returned an identical B = 1.4949 from
# it - the same structural artefact in both, carrying no content signal at
# all. The floor sits well under any realistic splice (Azarian-Pour's own
# test forgeries are 200x200 pastes into UCID frames, around 20% of the
# image; the splice used to validate this engine is 14%). [DERIVED]
MINIMUM_GHOST_AREA_FRACTION = 0.02

# ENHANCEMENT 2 (test-derived): the candidate must sit at an interior LOCAL
# MINIMUM of its own class-0 q2 profile.
#
# This implements a sentence of the SKILL the engine did not previously act
# on. SKILL B1, verbatim: "A genuinely double-quantized region shows a LOCAL
# MINIMUM in d at q2 equal to its true original quality q0." The engine
# segmented d and measured class separability at each q2 independently, never
# checking that the candidate was at a turning point at all.
#
# Without the check, an authentic-by-construction image still fired at
# q2 = 1, 6 and 11 with B up to 1.0287 after ENHANCEMENT 1 - higher than the
# 0.9684 a genuine q0=50 splice produced at its true ghost quality. At those
# extreme qualities the recompression error is dominated by image content and
# the profile is monotonically falling, so nothing there is a turning point.
# With the check in force both authentic ground-truth images return D_max
# exactly 0.0000 while the q0=50 and q0=70 splices are still found, at q2=51
# and q2=71 against true qualities of 50 and 70.
#
# The interior requirement also disposes of the sweep endpoints, where a
# local minimum is undefined for want of a neighbour on one side, and where
# both of the artefacts above happened to live.
REQUIRE_GHOST_LOCAL_MINIMUM = True
LOCAL_MINIMUM_NEIGHBOUR_OFFSET = 1

CONNECTED_COMPONENT_CONNECTIVITY = 8

# The grid shift of Eq. 8 zero-pads the image, and that padding is there to
# move content off the 8x8 DCT grid - it is not itself content to analyse.
# Left in, it is catastrophic: the black border recompresses quite unlike the
# photograph beside it, so the difference map acquires a bright frame that is
# contiguous, occupies a minority of the image, and therefore satisfies both
# validity conditions above. Measured before this crop existed, a spliced and
# an authentic image returned byte-identical D_max = 1.5699 at shift (4,4) -
# the statistic was reading the padding, not the picture.
#
# The analysis region is therefore the original content, inset by the
# smoothing window on every side so the w x w box filter cannot drag padding
# energy inward across the boundary either. [DERIVED]
ANALYSIS_BORDER_MARGIN = GHOST_SMOOTHING_WINDOW

# The difference map is already box-smoothed at w=16, so its usable detail
# sits well below full resolution; segmenting a reduced copy costs ~16x less
# and additionally suppresses the fractal noise the segmenter must resist.
# Labels are upsampled back before Eq. 10 is evaluated at full resolution.
# [ENGINEERING]
SEGMENTATION_DOWNSAMPLE_FACTOR = 4

# Guards against a degenerate class in Eq. 10 (a zero-variance class makes
# the log term undefined and the second term divide by zero).
MINIMUM_CLASS_PIXEL_COUNT = 2  # [STRUCTURAL]
MINIMUM_CLASS_STANDARD_DEVIATION = 1e-9  # [STRUCTURAL]

# ── Pipeline A: resampling periodicity (AUXILIARY, CANNOT SCORE) ───────────
# "K = 5 used throughout Kirchner & Bohme's main experiments, with K=3 and
# K=7 separately validated to give substantially the same results."
# Source: Kirchner & Bohme 2008, Eq. 7-8. [CORPUS]
PREDICTOR_NEIGHBOURHOOD_SIZE = 5
VALIDATED_NEIGHBOURHOOD_SIZES = (3, 5, 7)  # documented, not used in logic

# "center weight alpha_{floor(K^2/2)} := 0, i.e. a pixel does not predict
# itself". Source: Kirchner & Bohme 2008, Eq. 8. [CORPUS]
PREDICTOR_CENTRE_WEIGHT = 0.0

# "y_i ~ U(0, 2^l - 1)" for the M2 (genuinely acquired) class, with l the bit
# depth; Kirchner & Bohme work on 8-bit grayscale.
# Source: Kirchner & Bohme 2008, two-state model. [CORPUS]
UNIFORM_MODEL_BIT_DEPTH = 8
UNIFORM_MODEL_MAXIMUM_VALUE = 2 ** UNIFORM_MODEL_BIT_DEPTH - 1  # [DERIVED]

# "uniform prior Prob(y_i in M1) = Prob(y_i in M2)". Source: Eq. 10. [CORPUS]
CLASS_PRIOR_PROBABILITY = 0.5

# SKILL: "(exact convergence criterion not specified in the corpus for this
# paper - engineering recommendation: iterate until the change in
# log-likelihood or in alpha falls below a small tolerance, e.g. 10^-4,
# consistent with the EM stopping conventions used elsewhere in this
# engine's Benford and CFA modules)". The tolerance is the SKILL's own
# recommendation. [CORPUS-REC]
EM_CONVERGENCE_TOLERANCE = 1.0e-4
EM_MAXIMUM_ITERATIONS = 30  # no cap given anywhere. [ENGINEERING]

# Initial residual standard deviation for the M1 Gaussian, before the first
# M-step has run. No initialisation is specified in the SKILL. [ENGINEERING]
EM_INITIAL_RESIDUAL_SIGMA = 1.0
EM_MINIMUM_RESIDUAL_SIGMA = 1e-6  # [STRUCTURAL] guards a zero-sigma Gaussian

# "Kirchner & Bohme always crop to the center 256x256 block before detection
# to keep comparisons fair across parameter settings." [CORPUS]
ANALYSIS_WINDOW_SIZE = 256

# "downsampled x2 with nearest-neighbor from RAW specifically to remove
# CFA-interpolation periodicity that would otherwise confound the resampling
# signal ... 'found to be sufficient to reliably remove detectable traces of
# demosaicing'". Source: Kirchner & Bohme 2008, preprocessing. [CORPUS]
CFA_SUPPRESSION_DOWNSAMPLE_FACTOR = 2

# Search set: "|A| = 692 synthetic maps - 601 for scaling 0.5 <= S <= 2 in
# steps of dS=0.0025, 91 for rotation 0 <= Theta <= pi/4 in steps of
# dTheta = pi/360". Source: Kirchner & Bohme 2008, step 7. [CORPUS]
SCALING_SEARCH_MINIMUM = 0.5
SCALING_SEARCH_MAXIMUM = 2.0
SCALING_SEARCH_STEP = 0.0025
SCALING_SEARCH_COUNT = 601
ROTATION_SEARCH_MINIMUM = 0.0
ROTATION_SEARCH_MAXIMUM = np.pi / 4.0
ROTATION_SEARCH_STEP = np.pi / 360.0
ROTATION_SEARCH_COUNT = 91
FULL_SYNTHETIC_MAP_COUNT = 692

# Eq. 18's "+ 1/2 * 1^{2x1}" rounding offset. [CORPUS]
LATTICE_ROUNDING_OFFSET = 0.5

# Pipeline A's search set is 692 full-size DFTs. Exposed for pruning the same
# way Pipeline B's sweep is; 1 keeps the paper's full set. [ENGINEERING]
DEFAULT_SYNTHETIC_MAP_STEP = 1

# SKILL step 6 calls for "a radial high-pass weighting to suppress the
# dominant low-frequency/DC component, combined with a gamma contrast
# function gamma(.)", and states outright: "(exact gamma formula not printed
# in the extracted text of this paper - treat as a reference to the cited
# prior work)". The radial high-pass below is therefore this module's own
# construction, not the paper's gamma. Implementation Notes confirm the
# suppression itself is essential: "without the radial high-pass weighting,
# the DC term dominates and the periodic peaks ... are invisible".
# [ENGINEERING] / KNOWN_UNSOURCED_PARAMETER
RADIAL_HIGHPASS_CUTOFF_FRACTION = 0.05  # of the half-diagonal
RADIAL_HIGHPASS_EXPONENT = 1.0
RADIAL_HIGHPASS_DENOMINATOR_FLOOR = 1e-12  # [STRUCTURAL] divide-by-zero guard

# rho_T is "calibrated empirically per desired false-acceptance rate
# (Kirchner & Bohme calibrate rho_T on a held-out set of 400 known-original
# images)" - no numeric value is printed anywhere in this SKILL. rho
# therefore cannot be turned into a calibrated decision here, which is why
# Pipeline A is auxiliary-only. A caller who has measured its own rho_T may
# supply one via CalibrationSettings purely for reporting.
# [ENGINEERING] / KNOWN_UNSOURCED_PARAMETER
DEFAULT_RESAMPLING_THRESHOLD = None

# ── Condition-checker constants ─────────────────────────────────────────────
# SKILL: Pipeline A "fails outright after even moderate JPEG compression".
# No numeric quality boundary for "moderate" is given. [ENGINEERING]
RESAMPLING_MAXIMUM_COMPRESSION_LEVEL = 95.0

# Smallest image Pipeline B can smooth at all: one w x w window. [DERIVED]
MINIMUM_IMAGE_DIMENSION = GHOST_SMOOTHING_WINDOW

# ENHANCEMENT 4 (test-derived): a null ghost result does not deserve full
# confidence in the "authentic" reading it produces.
#
# When no candidate survives validation the engine reports raw_score 0.0,
# which the scorer maps to probability 0.0 - a confident REAL vote. Ground
# truth shows that vote is ambiguous by construction. An authentic image
# compressed once at q1=90 and a genuine splice of q0=95 content into a q1=70
# host returned byte-identical output: D_max exactly 0.0000, every candidate
# rejected. The second is a forgery, and the SKILL states plainly that the
# method "does not resolve the reverse case", so the two are indistinguishable
# here in principle rather than by accident.
#
# The weight is 0.5 because a null result is consistent with exactly two of
# the SKILL's own documented scenarios - a single-compression image and a
# reverse-direction splice - and the corpus supplies no basis for preferring
# either, so they are carried at equal weight. is_reliable is deliberately
# NOT set False: the measurement is sound and the fusion layer should still
# receive it, at a weight reflecting what it can actually distinguish.
# [DERIVED]
NULL_GHOST_RESULT_CONFIDENCE = 0.5

# ── Scorer constants ────────────────────────────────────────────────────────
# SKILL Output: "For fusion: (not specified in corpus) - engineering
# recommendation: score = clip(D_max / Th, 0, 1)". The ratio route is the
# SKILL's own recommendation. [CORPUS-REC]
MINIMUM_CALIBRATION_REFERENCE_COUNT = 10
PROVISIONAL_SIGMOID_SLOPE = 12.0
PROVISIONAL_SIGMOID_MIDPOINT = 0.5
SIGMOID_EXPONENT_LIMIT = 60.0

# ── Visualisation constants ─────────────────────────────────────────────────
EIGHT_BIT_DISPLAY_MAXIMUM = 255.0  # [STRUCTURAL]
EVIDENCE_MAP_MAX_DIMENSION = 2048  # [PRESENTATION]
MASK_OVERLAY_WEIGHT = 0.5  # [PRESENTATION]
MASK_OVERLAY_COLOUR = (0, 0, 255)  # [PRESENTATION] BGR red

# ── Audit-aid tuples ─────────────────────────────────────────────────────────
KNOWN_UNSOURCED_PARAMETERS = (
    "RADIAL_HIGHPASS_CUTOFF_FRACTION / RADIAL_HIGHPASS_EXPONENT - the "
    "SKILL states the gamma contrast function's 'exact formula [is] not "
    "printed in the extracted text of this paper'.",
    "DEFAULT_RESAMPLING_THRESHOLD (rho_T) - 'calibrated empirically per "
    "desired false-acceptance rate', no numeric value printed. This is why "
    "Pipeline A cannot contribute to raw_score.",
    "SEGMENTATION_SUPERPIXEL_COUNT / SEGMENTATION_COMPACTNESS / "
    "SEGMENTATION_DOWNSAMPLE_FACTOR - parameters of the SE-MinCut "
    "substitute, which the SKILL itself flags as 'an engineering "
    "substitution, not a like-for-like reproduction'.",
    "MINIMUM_GHOST_SPATIAL_COHERENCE / MINIMUM_GHOST_AREA_FRACTION / "
    "MAXIMUM_GHOST_AREA_FRACTION - the validity constraints standing in for "
    "SE-MinCut's spatial regularisation. Discovered as necessary by "
    "measurement, not read from the corpus; without them Eq. 10's "
    "variance-mismatch term saturates on every input and the statistic "
    "inverts. See the extended comment above.",
    "ANALYSIS_BORDER_MARGIN - the inset that keeps Eq. 8's zero-padding out "
    "of the statistics. Forced by the padding's own recompression behaviour, "
    "not specified in the corpus.",
    "DEFAULT_QUALITY_FACTOR_STEP / DEFAULT_GRID_SHIFT_STEP - the SKILL "
    "sanctions pruning the 6400-run sweep and states the q2 step should be "
    "'every 2-5 steps', but the exact values are ours.",
    "EM_MAXIMUM_ITERATIONS / EM_INITIAL_RESIDUAL_SIGMA - no iteration cap or "
    "initialisation is specified for the EM loop.",
    "RESAMPLING_MAXIMUM_COMPRESSION_LEVEL - the SKILL says Pipeline A fails "
    "after 'moderate' JPEG compression without quantifying moderate.",
    "PROVISIONAL_SIGMOID_SLOPE / MIDPOINT - fallback route only; the corpus "
    "route is the SKILL's own clip(D_max/Th, 0, 1).",
)

KNOWN_SKILL_AMBIGUITIES = (
    "The gamma contrast function of step 6 is cited to Popescu & Farid "
    "rather than re-derived, and is not printed. A radial high-pass weight "
    "is implemented in its place; the periodic-peak enhancement is "
    "therefore this module's construction, not the paper's.",
    "SE-MinCut has no standard Python implementation. The SKILL's own "
    "first-named substitute (SLIC superpixels) is used, with the two-class "
    "split taken over superpixel means but Eq. 10 evaluated on raw pixel "
    "values - keeping the segmenter's separation objective out of the "
    "statistic that then measures separation.",
    "Eq. 18's nu_{m_s}^{-1}(i) is the linear-index-to-2-D-coordinate map "
    "for the analysis window; it is implemented as the pixel coordinate "
    "grid of that window.",
)

TEST_DERIVED_ENHANCEMENTS = (
    "ENHANCEMENT 1 - SEGMENTATION_CLASS_COUNT = 3. The two-class Otsu cut "
    "spends itself separating the textured-background tail, leaving class-0 "
    "at a median 0.607-0.781 of the frame on all six supplied photographs, "
    "so 18-20 of every 20 candidates were discarded by the area test and "
    "only degenerate extreme-quality candidates ever survived. On ground "
    "truth at the true ghost quality the three-class cut moves class-0 from "
    "0.740 of the frame at IoU 0.320 to 0.265 at IoU 0.858.",
    "ENHANCEMENT 2 - REQUIRE_GHOST_LOCAL_MINIMUM. Implements the SKILL's own "
    "sentence 'a genuinely double-quantized region shows a local minimum in "
    "d at q2 equal to its true original quality q0', which the engine did "
    "not act on. Without it an authentic-by-construction image fired at "
    "q2=1/6/11 with B up to 1.0287, above the 0.9684 a real q0=50 splice "
    "produced at its true quality.",
    "ENHANCEMENT 3 - NULL_GHOST_RESULT_CONFIDENCE. An authentic image and a "
    "reverse-direction splice both return D_max exactly 0.0000, so the "
    "'authentic' reading of a null result is carried at half weight rather "
    "than full.",
)

REJECTED_ENHANCEMENTS = (
    "Segmenting a local-minimum DEPTH map, min(d_{q-1}, d_{q+1}) - d_q, in "
    "place of d itself. Tested across all nine ground-truth cases: every one "
    "still saturated, and the winning quality moved only from q2=6 to q2=16 "
    "with mask IoU falling to 0.000. The q2 profile oscillates strongly at "
    "low quality from quantization-table resonance, and those oscillations "
    "are far deeper than the ghost dip, so a depth map amplifies the "
    "artefact rather than removing it.",
    "Re-fitting BHATTACHARYYA_THRESHOLD away from the corpus value of 0.19. "
    "Th was fitted at FP=FN on 1000 original + 1000 tampered UCID images; "
    "nine self-built ground-truth cases are not a basis for replacing it, "
    "and doing so would be fitting the decision threshold to the test set.",
    "DEFAULT_QUALITY_FACTOR_STEP 5 -> 1 (a finer quality sweep). Implemented "
    "and reverted. The finer sweep genuinely finds more forgeries: with the "
    "corpus threshold held at 0.19 it detects 5 of 6 ground-truth splices at "
    "the correct quality, including the q0=60 and q0=80 pastes that a step of "
    "5 misses outright, and it locates them well - q2=60 exactly, mask IoU "
    "0.923. But it also fires on both authentic-by-construction images, at "
    "B=0.6683 (q2=12) and B=0.5928 (q2=21), three times the threshold. "
    "Scoring all nine cases with Th fixed: step 5 is right 6 times, step 3 "
    "four, step 1 five, step 2 three. Finer sweeps give the extreme-value "
    "maximisation more candidates to find a spurious minimum among, and a "
    "confident FAKE on an authentic photograph is the failure this engine "
    "already had. The recall cost is real and is reported as a limitation.",
    "DEFAULT_GRID_SHIFT_STEP 2 -> 4 (fewer grid shifts). Reverted, though the "
    "measurement is two-sided and worth keeping. The shift sweep never once "
    "improved a genuine detection: across 1, 4 and 16 shifts all five "
    "detectable ground-truth splices returned byte-identical D_max at an "
    "identical q2, every winner sitting at shift (0,0), a deliberately "
    "block-misaligned paste included. What extra shifts did change was the "
    "spurious maximum on images with no detectable ghost, which rose "
    "monotonically with the number of shifts - 0.7150 to 0.7150 to 1.1457 on "
    "the reverse-direction splice, 0.5977 to 0.7855 to 1.0227 at a quality "
    "gap of 2. More shifts are therefore a pure false-positive amplifier "
    "here. It is still reverted, for two reasons: in the shipped "
    "configuration (quality step 5) every one of those non-detections is "
    "already 0.0000 with the full 16 shifts, so there is no false-positive "
    "pressure to relieve; and the SKILL sanctions this pruning only when the "
    "paste 'is already known/suspected to be block-aligned', which an engine "
    "seeing a single image cannot know.",
    "Replacing raw_score's clip(D_max/Th, 0, 1) with an unsaturating map. "
    "Measured D_max values run 0.6-1.8, three to ten times Th, so every "
    "positive reports exactly 1.0 and the fusion layer sees no magnitude. "
    "The saturation is real and is reported as a limitation, but the "
    "normalisation is the SKILL's own stated recommendation and the corpus "
    "offers no alternative, so it is left alone rather than invented.",
)

KNOWN_UNIMPLEMENTED_MODULES = (
    "Pipeline C (Rao et al. 2025 six-feature PCA fusion) - the SKILL calls "
    "it a 'fusion-layer pattern reference, not a Fourier/ghost-specific "
    "technique', its final decision stage is [ML - excluded], its "
    "Sobel-gradient lighting cue is already implemented in this system's "
    "lighting module, and the SKILL's corpus-honesty note states the source "
    "'reports no quantitative accuracy/precision/F1/AUC numbers anywhere in "
    "the extracted text' and must not be cited as evidence of validated "
    "detection performance. It belongs to the bayes fusion module.",
    "The three counter-forensic attacks of Kirchner & Bohme (5x5 median "
    "filtering, edge-modulated geometric distortion, and the dual-path "
    "combination) are fully specified in the SKILL but are attacks on this "
    "engine, not detectors - out of scope for a detection engine. Their "
    "existence is carried in every reliability_note instead.",
    "The 'cheaper approximate variant' of Pipeline A (Laplacian second "
    "derivative in place of the EM p-map) - the SKILL marks it '(not in "
    "this specific paper - general technique referenced across the "
    "literature)', so it is outside this SKILL's sourced content.",
)
