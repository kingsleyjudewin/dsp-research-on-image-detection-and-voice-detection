"""All parameters and thresholds for the JPEG-compression-artifact engine.

Provenance tags used throughout this file:
    [CORPUS]       - value/formula printed verbatim in the SKILL file.
    [CORPUS-RANGE] - SKILL gives a validated range but not one exact value;
                     the point picked within that range is documented below.
    [DERIVED]      - not printed in the SKILL, but forced by a constraint the
                     SKILL's own formulas operate under (shown in the comment).
    [ENGINEERING]  - the SKILL explicitly states no value is given ("not
                     specified in corpus", "determined ... in a training
                     process") and this module must supply a working default.
    [STRUCTURAL]   - shape/type constants with no forensic meaning of their own.
    [PRESENTATION] - display-only, never affects a score.

SCOPE DECISION (applies to every constant below). This SKILL assigns each of
its pipelines an explicit role, so no combination weight has to be invented:

  * A.1 (JPEG-history feature s) is a GATE - "run A.1 first as a cheap global
    gate (is this image JPEG-derived at all?)". An image being JPEG is not
    evidence of forgery, so s never contributes to raw_score.
  * A.2 (per-frequency quantization step) is explicitly "a nuisance/
    conditioning parameter, not itself a tampering score".
  * A.3 (quality factor + pixel-match ratio R) is "a conditioning parameter"
    whose R byproduct feeds CONFIDENCE. The SKILL calls its use as tampering
    evidence "an engineering extrapolation, not validated in the corpus", so
    it is not scored either.
  * B (double-quantization Fourier periodicity) is "the actual double-
    compression/splice-localization signal" - this alone drives raw_score.

Pipelines C (Wang & Zhang CNN) and D (Barni et al. CNN) are tagged
[ML - excluded from the no-ML engine] by the SKILL and are not implemented.
The capability that forfeits is recorded in KNOWN_UNIMPLEMENTED_MODULES and
surfaced in every reliability_note, per the SKILL's explicit instruction that
it "should be explicitly surfaced to the fusion layer as a known coverage
gap, not silently absorbed as 'no evidence found.'"
"""

from __future__ import annotations

import numpy as np

ENGINE_NAME = "jpeg_compression_artifact"  # [STRUCTURAL]
SKILL_VERSION = "1.0.0"  # [STRUCTURAL]

# ── Structural / contract constants ─────────────────────────────────────────
GRAYSCALE_IMAGE_DIMENSION_COUNT = 2  # [STRUCTURAL]
COLOUR_IMAGE_DIMENSION_COUNT = 3  # [STRUCTURAL]
FULL_CONFIDENCE = 1.0  # [STRUCTURAL]
ZERO_CONFIDENCE = 0.0  # [STRUCTURAL]
TRACE_DECIMAL_PLACES = 4  # [PRESENTATION]
MILLISECONDS_PER_SECOND = 1000.0  # [STRUCTURAL]

# ── Block-DCT structure ─────────────────────────────────────────────────────
# SKILL: "JPEG is a lossy, block-based (8x8 DCT) codec." [CORPUS]
DCT_BLOCK_SIZE = 8

# The level shift is part of the definition of the JPEG block-DCT the SKILL
# invokes throughout (pixels are shifted to a zero-centred range before the
# forward DCT), not a tunable forensic parameter. [STRUCTURAL]
JPEG_LEVEL_SHIFT = 128.0

# SKILL: truncation error arises where "pixel values exceeding [0,255] after
# inverse-DCT get clipped" and is "neglected in the rest of the analysis by
# restricting statistics to unsaturated 8x8 blocks". A clipped pixel lands
# exactly on a bound, so a block is treated as saturated when any pixel sits
# on either bound. [CORPUS] for the restriction, [ENGINEERING] for reading
# "unsaturated" as "no pixel exactly at a clipping bound".
PIXEL_VALUE_MINIMUM = 0
PIXEL_VALUE_MAXIMUM = 255

# ── Pipeline A.1: JPEG-history identification (GATE, never scored) ──────────
# Formula: R1 = (-1,+1),  R2 = (-2,-1) U (+1,+2)
# Source: Luo, Huang & Qiu 2010, Eq. 7 region definitions. [CORPUS]
REGION_R1_OUTER_BOUND = 1.0
REGION_R2_INNER_BOUND = 1.0
REGION_R2_OUTER_BOUND = 2.0

# SKILL: "Reported thresholds (256x256 down to 8x8 blocks) range t ~ 0.29-0.38",
# with false-positive rates 14.10% (256x256) to 21.06% (8x8). Transcribed for
# the record. [CORPUS-RANGE]
HISTORY_THRESHOLD_RANGE = (0.29, 0.38)

# MEASURED DEPARTURE - the printed range above cannot be used as an absolute
# cut on Eq. 7's output as this engine computes it. Evaluating Eq. 7 exactly
# as printed, over the JPEG-convention block DCT, on 1/f natural-statistics
# images gives s ~ 1.04-1.07 for never-compressed content and s ~ 26 (QF95)
# to ~2000 (QF85 and below) once compressed. The DIRECTION the SKILL states
# is confirmed emphatically - compressed images have far larger s - but every
# one of those values sits above 0.38, so thresholding there would pass every
# input, compressed or not. Only Eq. 7 is transcribed in this SKILL, not
# whatever normalisation the paper's threshold is expressed in, so the
# discrepancy cannot be reconciled from the corpus.
#
# The default below sits in the measured gap between the never-compressed
# cluster (~1.05, tight across seeds) and the lightest compression tested
# (~26 at QF95). It is an engineering value, NOT the paper's t, and the
# orchestrator should override it per deployment via
# CalibrationSettings.history_threshold.
# [ENGINEERING] / KNOWN_UNSOURCED_PARAMETER.
DEFAULT_HISTORY_THRESHOLD = 2.0

# ── Pipeline A.2: per-frequency quantization-step estimation (conditioning) ─
# Formula: eps ~ approximately Gaussian, mean 0, variance 1/12 (CLT over the
# uniform [-0.5,+0.5] rounding distribution of 64 summed spatial terms).
# Source: Luo et al. 2010. [CORPUS]
ROUNDING_ERROR_VARIANCE = 1.0 / 12.0

# Formula: P([d2(i,j)] = d1'(i,j)) = integral_{-0.5}^{+0.5} p_eps >= 91.50%
# Source: Luo et al. 2010, Eq. 11. Verified numerically against a N(0,1/12)
# CDF at build time: the exact value is 91.67%, consistent with the SKILL's
# stated ">= 91.50%" bound. [CORPUS]
ROUNDING_RECOVERY_PROBABILITY_BOUND = 0.9150
ROUNDING_HALF_WIDTH = 0.5

# Formula: if H(1)/H(0) > t AND H(1) > H_max -> q_hat = 1
# Source: Luo et al. 2010, Eq. 12. "t = 0.3 (empirically tuned in the paper
# across t in [0.10, 0.35], t=0.3 gave the best results and is used for all
# reported experiments)." [CORPUS]
GHOST_AT_ONE_RATIO_THRESHOLD = 0.3
GHOST_AT_ONE_SEARCH_RANGE = (0.10, 0.35)  # documented, not used in logic

# Formula: q_hat = argmin_k ( k | H(k) = H_max, k >= 2 )
# Source: Luo et al. 2010, Eq. 13. [CORPUS]
MINIMUM_QUANTIZATION_STEP = 2

# Eq. 12/13 pick a histogram bin by its COUNT, which presumes the
# coefficient population is peaked at zero so that H(q) is the tallest
# non-zero-bin spike. AC coefficients satisfy that; the DC coefficient does
# not - it carries each block's mean brightness and is spread across the
# whole dynamic range, so the argmax lands arbitrarily deep in the bulk
# (measured: est=240 against a true step of 5). A.1's own formulation
# already excludes DC ("for each DCT frequency position (i,j), 0 <= i,j <= 7,
# EXCEPT DC (0,0)"); the same reasoning is applied to A.2 here. [DERIVED]
SKIP_DC_IN_STEP_ESTIMATION = True

# ── Pipeline A.3: quantization-table / quality-factor detection ─────────────
# Formula: Q_hat_F = argmax_i R(J1, J2(i)), i = 1,...,100
# Source: Luo et al. 2010, Eq. 25. [CORPUS]
QUALITY_FACTOR_MINIMUM = 1
QUALITY_FACTOR_MAXIMUM = 100

# The IJG standard base luminance table t, transcribed exactly as printed in
# the SKILL's Implementation Notes (needed for Eq. 26). [CORPUS]
IJG_BASE_LUMINANCE_TABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.int64)

# Formula: Table_QF = floor(t * 50/QF + 0.5),        1 <= QF < 50
#                   = floor(t * (2 - QF/50) + 0.5),  50 <= QF <= 100
# Source: Luo et al. 2010, Eq. 26. [CORPUS]
QUALITY_FACTOR_PIVOT = 50
TABLE_SCALE_NUMERATOR = 50.0
TABLE_SCALE_OFFSET = 2.0
TABLE_ROUNDING_OFFSET = 0.5

# "(values less than 1 are floored up to 1)". [CORPUS]
QUANTIZATION_VALUE_MINIMUM = 1

# NOT printed in Eq. 26, but forced by the format Eq. 24-26 operate on: a
# baseline JPEG DQT marker stores 8-bit entries, so a table value above 255
# is unencodable. Eq. 26's real-arithmetic form exceeds 255 for every QF
# below ~24 (it reaches 6050 at QF=1), which no encoder can realise.
# [DERIVED]
QUANTIZATION_VALUE_MAXIMUM = 255

# Eq. 26 describes the scaling in exact real arithmetic. Real encoders
# (libjpeg, and therefore Pillow, and therefore essentially every JPEG this
# engine will ever be handed) compute it in INTEGER arithmetic, truncating
# the scale factor first:
#     scale_factor = 5000 // QF          (QF < 50)
#                  = 200 - 2*QF          (QF >= 50)
#     table        = clip((t*scale_factor + 50) // 100, 1, 255)
# Measured at build time: this integer model reproduces Pillow's own tables
# exactly for all QF in 1..100, while Eq. 26's real-arithmetic form differs
# by +-1 in at least one entry at 35 of those 100 quality factors.
#
# That difference is decisive for A.3 specifically, because Eq. 24 scores
# candidates by EXACT pixel identity - a table off by one anywhere collapses
# the match rate and returns the wrong Q_hat_F. The recompression sweep
# therefore drives the encoder at each candidate quality (which applies this
# integer model internally); Eq. 26 is implemented alongside it exactly as
# printed, and the two are reported together so the divergence stays visible
# rather than silently resolved. [DERIVED]
LIBJPEG_LOW_QUALITY_SCALE_NUMERATOR = 5000
LIBJPEG_HIGH_QUALITY_SCALE_BASE = 200
LIBJPEG_HIGH_QUALITY_SCALE_FACTOR = 2
LIBJPEG_TABLE_ROUNDING_OFFSET = 50
LIBJPEG_TABLE_DIVISOR = 100

# ── Pipeline B: double-quantization Fourier periodicity (SCORE-DRIVING) ─────
# Formula: F^{Q_beta}(u,v) = round(F^{Q_alpha}(u,v) * Q_alpha(u,v)/Q_beta(u,v))
# Source: Mahdian & Saic 2009, Eq. 4. [CORPUS] - the underlying model; it is
# descriptive (neither Q_alpha nor Q_beta is known for an image under test),
# not something this engine evaluates directly.

# "Select 10 low-frequency DCT positions (luminance channel only): (0,0),
# (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (0,2), (1,2), (0,3). Higher
# frequencies are excluded because they are frequently quantized entirely to
# zero, producing insufficient statistics." Source: Mahdian & Saic 2009,
# step 2. [CORPUS]
DOUBLE_QUANTIZATION_FREQUENCIES = (
    (0, 0), (1, 0), (2, 0), (3, 0), (0, 1),
    (1, 1), (2, 1), (0, 2), (1, 2), (0, 3),
)

# SKILL step 4: "i=1/DC is treated as a special case because it alone shows a
# clear peak under double compression rather than a decaying trend under
# single compression" - so the averaging filter and trend removal apply to
# i=2,...,10 only. i=1 is the first entry of the tuple above. [CORPUS]
DC_FREQUENCY_ORDINAL = 0

# SKILL Implementation Notes: "use integer-valued bins for DCT coefficient
# histograms ... do not apply continuous/KDE binning, which would blur
# exactly the periodic bin structure both methods depend on." [CORPUS]
HISTOGRAM_BIN_WIDTH = 1

# Pipeline B histograms the QUANTIZED integer coefficients, not the
# dequantized ones. Eq. 4 is written over F^{Q_alpha}(u,v) and
# F^{Q_beta}(u,v) - the integer bitstream values - and the SKILL's combining
# instruction is explicit that A's step estimator feeds B ("using A.2's
# per-frequency quantization-step estimator to help interpret which
# candidate secondary step q to test peaks against").
#
# This matters enormously and is easy to get wrong. A decoded-and-re-DCT'd
# coefficient is a multiple of its quantization step q, so its integer-binned
# histogram already has spikes every q bins with gaps between - a strong
# period-q structure present under SINGLE compression just as much as double.
# Feeding that straight to the FFT measures the quantization step, not the
# double-quantization artifact. Measured on 1/f natural-statistics images at
# QF1=70 -> QF2=85 against single QF85, the un-normalised score separates in
# the WRONG direction (single 0.443 vs double 0.322, a consistent -0.12
# across seeds); dividing by q first flips it to the correct direction
# (single 0.159 vs double 0.209, +0.037 to +0.067 across seeds).
#
# q is taken from A.3's recovered quantization table, which the paper
# benchmarks at 94-99% and which was verified end-to-end here, rather than
# from A.2 - A.2's Eq. 13 argmax is unreliable at exactly the low
# frequencies Pipeline B uses (see SKIP_DC_IN_STEP_ESTIMATION). A.2 remains
# what the SKILL calls it, a reported conditioning parameter, and is not
# load-bearing for the score. [DERIVED]
NORMALISE_HISTOGRAM_BY_QUANTIZATION_STEP = True

# Formula: M_i(f) = min{ |H_i|(f), ..., |H_i|(f-n) }, trailing window of
# length n. Source: Mahdian & Saic 2009, Eq. 5.
# n itself: "determined per quantization step in a training process - the
# paper does not give a single fixed default value; (engineering
# recommendation, not specified in corpus: calibrate n per deployment on a
# held-out set spanning the expected quantization-step range)".
# [ENGINEERING] / KNOWN_UNSOURCED_PARAMETER.
TREND_REMOVAL_WINDOW_LENGTH = 8

# "a denoising averaging filter is applied to |H_i|, i=2,...,10, before this
# step" - the filter's length is not given anywhere in the SKILL.
# [ENGINEERING] / KNOWN_UNSOURCED_PARAMETER.
AVERAGING_FILTER_LENGTH = 3

# SKILL step 6 replaces the paper's [ML-excluded] Gaussian-kernel SVM with:
# "threshold the peak prominence of |H_i~| directly". No numeric threshold is
# given for that substitute. [ENGINEERING] / KNOWN_UNSOURCED_PARAMETER.
# The unit-L2 normalisation the SKILL applies to |H_i| (step 3) bounds any
# single prominence by 1, so this sits on a naturally [0,1] scale and needs
# no invented rescaling.
PEAK_PROMINENCE_THRESHOLD = 0.05

# SKILL Implementation Notes: "detect this condition (fraction of a
# frequency's coefficients equal to zero exceeding some threshold) and
# exclude that frequency ... (exclusion rule not specified in corpus -
# engineering recommendation)". [ENGINEERING] / KNOWN_UNSOURCED_PARAMETER.
ZERO_COEFFICIENT_EXCLUSION_FRACTION = 0.95

# SKILL step 3: |H_i| is "normalized to unit length" before trend removal.
# [CORPUS]. Floor guards a degenerate all-zero spectrum.
SPECTRUM_NORM_FLOOR = 1e-12

# The FFT magnitude of a real histogram is symmetric, so only the first half
# carries independent information. [STRUCTURAL]
SPECTRUM_HALF_DIVISOR = 2

# ── ENHANCEMENT 1 (test-derived): container-supplied quantization table ────
# The SKILL's Implementation Notes make this the PREFERRED route and A.3 the
# fallback, in as many words: "standard JPEG files store their quantization
# tables directly in the file header (DQT marker) ... Use this for a fast QF
# estimate when the file is available as an actual .jpg; fall back to Luo's
# A.3 recompression-search method (Eq. 24-26) only when the image is a bitmap
# of unknown/stripped provenance." The engine previously always ran A.3.
#
# That is not a cosmetic preference. A.3 searches the 100-table IJG family
# generated by Eq. 26, so a file whose encoder used a NON-STANDARD table has
# no correct answer available in the candidate set, and Eq. 25's argmax then
# lands wherever the pixel-match rate happens to peak. Measured on all six
# supplied photographs: A.3 returned QF=100 on every one, at a match rate of
# 0.967-0.984 against a runner-up only 0.004-0.035 behind, while each file's
# own DQT marker carries a table that matches no IJG quality factor closely
# (best residual 11.05, DC step 6 where the IJG family gives 8 at QF74).
#
# A.3 itself is sound - on 20 ground-truth files built with the standard
# tables it recovered the exact true quality factor every time, at match
# rates of 0.9968 to 1.0000. The failure is one of candidate coverage, and
# reading the container's own table removes it. [CORPUS] for the preference.
PREFER_CONTAINER_QUANTIZATION_TABLE = True

# ── ENHANCEMENT 2 (test-derived): degenerate all-unit-step tables ──────────
# When the effective table is all ones, to_quantized_domain becomes a no-op
# and Pipeline B histograms the DEQUANTIZED coefficients. The build notes
# already record that this separates in the WRONG DIRECTION; that finding is
# now confirmed on never-compressed ground truth with the true tables known.
# Over three seeds, single versus easy-double separation measured:
#     normalisation ON  : +0.0209, +0.0373, +0.0273   (mean +0.0285)
#     normalisation OFF : -0.0710, -0.0691, -0.0755   (mean -0.0719)
# The sign flips, consistently.
#
# Every one of the six supplied photographs ran in exactly the OFF state,
# because A.3 returned QF=100 whose table is all ones - and the engine said
# nothing about it. The harness confirmed the consequence directly: the
# un-normalised Pipeline B score equalled the normalised one to four decimal
# places on all six.
#
# A score known to separate in the wrong direction is not a measurement, so
# this sets is_reliable=False rather than discounting confidence. It also
# correctly abstains on a genuine QF=100 image, where there is no
# quantization structure for Pipeline B to work with in the first place.
# [DERIVED]
MINIMUM_USABLE_QUANTIZATION_STEP = 2
MINIMUM_NON_UNIT_STEP_FRACTION = 0.5

# ── ENHANCEMENT 3 (test-derived): the A.1 gate's real operating range ──────
# The SKILL states the history feature is reliable when "background quality
# factor is moderate-to-high (QF >= ~85 ... to cleanly separate compressed/
# uncompressed)". Measurement contradicts the direction of that claim. Values
# of s over JPEGs of known quality, three never-compressed source families:
#     QF50-85 : 101 - 2913     (separates emphatically)
#     QF90    : 77 - 290       (still clear)
#     QF95    : 15 - 22        (marginal)
#     QF98    : 3.2 - 7.3      (failing)
#     QF100   : 1.0 - 4.5      (indistinguishable from never-compressed)
# against never-compressed content spanning 1.02 (uniform noise) to 45.93
# (a very smooth blob).
#
# So the classes OVERLAP and no threshold separates them: a real JPEG at
# QF98 scores below a never-compressed gradient. The gate is informative in
# the QF50-90 band and uninformative above it, the opposite way round from
# the SKILL's stated regime. This is carried as a note, NOT as a refitted
# threshold - eight images are no basis for replacing a decision boundary,
# and DEFAULT_HISTORY_THRESHOLD is already flagged for orchestrator
# override. [DERIVED]
HISTORY_FEATURE_UNRELIABLE_ABOVE_QUALITY = 95

# ── Condition-checker constants ─────────────────────────────────────────────
# SKILL: "Very low quality factor (QF~50): high-frequency coefficients
# quantize to zero, destroying the statistics both Luo's and Mahdian's
# methods rely on." [CORPUS] for the QF=50 danger point.
LOW_QUALITY_FACTOR_FLOOR = 50

# SKILL: "Luo et al. explicitly validate down to 8x8-pixel blocks, i.e. a
# single DCT block". Pipeline B, however, needs a populated per-frequency
# histogram across blocks, which one block cannot provide. No minimum block
# count is given for B. [ENGINEERING] / KNOWN_UNSOURCED_PARAMETER.
MINIMUM_BLOCKS_FOR_HISTOGRAM = 64

# Fraction of the 10 Pipeline-B frequencies that must survive the
# zero-coefficient exclusion rule for the aggregate score to be meaningful.
# [ENGINEERING] / KNOWN_UNSOURCED_PARAMETER.
MINIMUM_USABLE_FREQUENCY_FRACTION = 0.5

# ── Scorer constants ────────────────────────────────────────────────────────
# SKILL Output: Pipeline A.1's "(Calibration to [0,1] not specified in corpus
# - engineering recommendation)", and Pipeline B's aggregate scalar is
# likewise uncalibrated in the corpus. Same two-route pattern as this
# system's other engines. [ENGINEERING]
MINIMUM_CALIBRATION_REFERENCE_COUNT = 10
PROVISIONAL_SIGMOID_SLOPE = 25.0
PROVISIONAL_SIGMOID_MIDPOINT = 0.12
SIGMOID_EXPONENT_LIMIT = 60.0

# ── Visualisation constants ─────────────────────────────────────────────────
EVIDENCE_PANEL_HEIGHT_PIXELS = 90  # [PRESENTATION]
EVIDENCE_PANEL_WIDTH_PIXELS = 480  # [PRESENTATION]
EVIDENCE_PANEL_MARGIN_PIXELS = 6  # [PRESENTATION]
EIGHT_BIT_DISPLAY_MAXIMUM = 255.0  # [STRUCTURAL]
EVIDENCE_MAP_MAX_DIMENSION = 2048  # [PRESENTATION]
PEAK_MARKER_RADIUS_PIXELS = 3  # [PRESENTATION]
EVIDENCE_LABEL_FONT_SCALE = 0.35  # [PRESENTATION]
EVIDENCE_LINE_THICKNESS = 1  # [PRESENTATION]
EVIDENCE_LABEL_COLOUR = (220, 220, 220)  # [PRESENTATION] BGR
EVIDENCE_SPECTRUM_COLOUR = (255, 200, 90)  # [PRESENTATION] BGR
EVIDENCE_PEAK_COLOUR = (90, 90, 255)  # [PRESENTATION] BGR, above threshold
EVIDENCE_WEAK_PEAK_COLOUR = (140, 140, 140)  # [PRESENTATION] BGR, below

# ── Audit-aid tuples ─────────────────────────────────────────────────────────
KNOWN_UNSOURCED_PARAMETERS = (
    "TREND_REMOVAL_WINDOW_LENGTH (n, Eq. 5) - SKILL: 'determined per "
    "quantization step in a training process - the paper does not give a "
    "single fixed default value'.",
    "AVERAGING_FILTER_LENGTH - the SKILL states an averaging filter is "
    "applied before Eq. 5 but never gives its length.",
    "PEAK_PROMINENCE_THRESHOLD - the SKILL's training-free substitute for "
    "the [ML-excluded] SVM is 'threshold the peak prominence ... directly', "
    "with no numeric threshold given.",
    "ZERO_COEFFICIENT_EXCLUSION_FRACTION - SKILL: 'exclusion rule not "
    "specified in corpus - engineering recommendation'.",
    "DEFAULT_HISTORY_THRESHOLD - the SKILL gives the range 0.29-0.38 across "
    "image sizes but not the per-size mapping; the midpoint is used.",
    "MINIMUM_BLOCKS_FOR_HISTOGRAM / MINIMUM_USABLE_FREQUENCY_FRACTION - no "
    "minimum block count or usable-frequency count is given for Pipeline B.",
    "PROVISIONAL_SIGMOID_SLOPE / MIDPOINT - placeholder fallback route only; "
    "the SKILL states calibration to [0,1] is 'not specified in corpus' for "
    "both the A.1 feature and Pipeline B's aggregate.",
)

TEST_DERIVED_ENHANCEMENTS = (
    "ENHANCEMENT 1 - PREFER_CONTAINER_QUANTIZATION_TABLE. A.3 can only "
    "answer from the 100-table IJG family Eq. 26 generates, so a file "
    "encoded with a non-standard table has no correct candidate. Measured: "
    "A.3 returned QF=100 on all six supplied photographs, whose own DQT "
    "markers match no IJG quality factor closely. The SKILL already makes "
    "the header the preferred route and A.3 the fallback.",
    "ENHANCEMENT 2 - MINIMUM_USABLE_QUANTIZATION_STEP / "
    "MINIMUM_NON_UNIT_STEP_FRACTION. An all-unit-step table makes Pipeline "
    "B's normalisation a no-op, and that configuration was measured to "
    "separate single from double compression in the WRONG direction "
    "(-0.0719 against +0.0285) across three seeds. All six supplied images "
    "ran in that state silently; the run is now marked unreliable.",
    "ENHANCEMENT 3 - HISTORY_FEATURE_UNRELIABLE_ABOVE_QUALITY. The A.1 gate "
    "was measured to stop discriminating above QF95, the opposite of the "
    "SKILL's stated QF>=85 reliable regime. Carried as a note.",
)

REJECTED_ENHANCEMENTS = (
    "Changing Pipeline B's aggregation away from the mean over all 10 "
    "frequencies. Two alternatives were implemented and measured over three "
    "never-compressed source families, three seeds each. Mean excluding DC: "
    "average easy-case separation +0.0652 against the shipped +0.0691, and "
    "it INVERTS on one family (-0.0248). DC only: the best average by far "
    "(+0.1038 easy, +0.0716 hard) but it inverts catastrophically on the 1/f "
    "family (-0.2913), because DC prominence there saturates near 0.93 under "
    "single compression and falls under double, the reverse of its behaviour "
    "on Laplacian-statistics sources. The shipped mean is the only one of "
    "the three positive on every family tested, so it stays.",
    "Re-fitting DEFAULT_HISTORY_THRESHOLD to remove the A.1 false positives "
    "on smooth never-compressed content. No separating value exists: real "
    "JPEGs at QF98-100 score s=1.0-7.3, BELOW a never-compressed smooth "
    "gradient at 27.8 and a smooth blob at 45.9. The classes overlap, so any "
    "threshold trades one error for the other. Reported as a limitation.",
    "Gating A.1 on the R2 region's share of AC mass, or on AC energy, to "
    "catch the smooth-content false positives. Neither separates: a genuine "
    "JPEG has an R2 share of 0.0009-0.008 against 0.021-0.035 for the smooth "
    "never-compressed cases - the real JPEGs are MORE extreme on the very "
    "statistic that would have flagged the false positives - and the "
    "fraction of AC coefficients below 1 overlaps too (0.947-0.972 for real "
    "JPEGs, 0.950-0.956 for the smooth never-compressed ones).",
)

KNOWN_SKILL_AMBIGUITIES = (
    "H_max in Eq. 12/13 is read as max_{k>=2} H(k). Eq. 13 restricts its "
    "argmin to k>=2 against the same H_max, and only this reading leaves "
    "Eq. 12's two conditions both non-redundant - under a global max, "
    "'H(1) > H_max' would already imply 'H(1)/H(0) > 0.3'.",
    "Eq. 26's real-arithmetic scaling diverges from the integer arithmetic "
    "every real JPEG encoder uses, at 35 of 100 quality factors (measured), "
    "and omits the 255 ceiling a baseline 8-bit DQT marker requires. Eq. 26 "
    "is implemented as printed, but the A.3 sweep drives the encoder itself, "
    "because Eq. 24 scores candidates by exact pixel identity - see the "
    "comment above LIBJPEG_LOW_QUALITY_SCALE_NUMERATOR.",
    "'Zero-mean histogram' (Pipeline B step 3) is read as subtracting the "
    "histogram's own mean bin count before the FFT; this removes the f=0 "
    "term that would otherwise dominate the magnitude spectrum and makes "
    "the SKILL's own unit-length normalisation meaningful.",
)

KNOWN_UNIMPLEMENTED_MODULES = (
    "Pipeline C (Wang & Zhang 2016 histogram-feature CNN) - tagged "
    "[ML - excluded from the no-ML engine] by the SKILL.",
    "Pipeline D (Barni et al. 2017 pixel/noise-residual CNN) - tagged "
    "[ML - excluded]. Per the SKILL this is the ONLY documented technique "
    "that handles non-aligned (grid-shifted) double-JPEG and the QF1=QF2 "
    "case at all, so the no-ML engine is blind to both. The SKILL requires "
    "this be 'explicitly surfaced to the fusion layer as a known coverage "
    "gap, not silently absorbed as no evidence found' - it is carried in "
    "every reliability_note this engine emits.",
    "Mahdian & Saic's Gaussian-kernel SVM classification stage (step 6), "
    "[ML - excluded]; replaced by the SKILL's own recommended training-free "
    "peak-prominence threshold.",
    "Block-wise re-application of A.2/B for a spatial heatmap - the SKILL "
    "notes this is possible in principle but 'not itself validated "
    "end-to-end in the corpus'. evidence_map therefore renders the actual "
    "computed spectra rather than a claimed spatial localisation, and "
    "flagged_regions is always None.",
)
