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
