"""All parameters, thresholds, and provenance tags for the noise-pattern engine.

Provenance tags used throughout this file:
    [CORPUS]      - value/formula printed verbatim in the SKILL file.
    [CORPUS-FINDING] - a value the SKILL reports as an empirical result
                    (e.g. "8x8 is the sweet spot"), reused here as a default.
    [DERIVED]     - not printed in the SKILL, but algebraically derivable from
                    a formula the SKILL DOES print.
    [ENGINEERING] - the SKILL explicitly states no value is given, or names a
                    technique without printing its formula, and this module
                    must supply a working default to run at all.
    [STRUCTURAL]  - shape/type constants with no forensic meaning of their own.
    [PRESENTATION]- display-only, never affects a score.

SCOPE DECISION (applies throughout): the SKILL explicitly names Pipeline A
"recommended PRIMARY for this engine's single-suspect-image, no-reference use
case" - the exact shape of this system's EngineInput (one image, no external
reference set). raw_score is therefore driven by Pipeline A alone. Pipeline C
is fully computable from one image with no reference camera and no
unsourced constants, so it runs every call as auxiliary (non-scoring)
evidence - the SKILL gives no A+C combination formula, so none is invented.
Pipeline B needs N reference images from the SAME physical camera - not part
of EngineInput - so it only runs if the orchestrator supplies them via
CalibrationSettings, and even then only its fully-specified pieces (the ML
estimator, Eq. 6, and the ZM+Wiener preprocessing, B.2) are implemented; the
weighted detector (Eq. 11, needing beta_b) and the correlation predictor
(B.4) require camera-specific constants (I_crit, tau, c) the SKILL gives
only as one specific Canon G2 example, explicitly stating they "do not
transfer directly across camera models" - implementing them with that
example's numbers for an arbitrary deployment camera would be using a value
not present in this SKILL file for that camera, so they are not implemented.
Pipeline D needs a training-set baseline (avg_mean_magnitude_train) never
given a number - it only runs if supplied via calibration, and its output
only modulates confidence, never raw_score, per the SKILL's own instruction
("down-weight the PRNU modules' confidence here").

A second gap: the SKILL names "the Mihcak wavelet-based denoising filter"
(and Lin et al.'s FDR enhancement) as the shared residual-extraction step for
Pipelines A/B/C, but never prints its formula anywhere in this file - only
its role (F, such that W=I-F(I)). It is implemented here as the simplest
concrete instantiation of that role: single-level 2-D DWT, reconstruct from
the LL subband only. [ENGINEERING]. FDR is not implemented (no formula given
at all in this SKILL for it).
"""

from __future__ import annotations

ENGINE_NAME = "noise_pattern_forgery"  # [STRUCTURAL]
SKILL_VERSION = "1.0.0"  # [STRUCTURAL]

# ── Structural / contract constants ─────────────────────────────────────────
GRAYSCALE_IMAGE_DIMENSION_COUNT = 2  # [STRUCTURAL]
COLOUR_IMAGE_DIMENSION_COUNT = 3  # [STRUCTURAL]
FULL_CONFIDENCE = 1.0  # [STRUCTURAL]
ZERO_CONFIDENCE = 0.0  # [STRUCTURAL]
TRACE_DECIMAL_PLACES = 4  # [PRESENTATION]
MILLISECONDS_PER_SECOND = 1000.0  # [STRUCTURAL]

# ── Shared residual-extraction filter F (role given, formula not) ──────────
# SKILL: "Extract the noise residual W = I - F(I) using the Mihcak
# wavelet-based denoising filter F" - no formula printed anywhere in this
# file. [ENGINEERING]: F is instantiated as a single-level 2-D DWT,
# reconstructed from the LL subband only (all detail subbands zeroed),
# giving the simplest possible wavelet-domain low-pass F(I).
RESIDUAL_FILTER_WAVELET_FAMILY = "haar"  # [ENGINEERING] - no family given.

# ── Pipeline A: blind local noise-level inconsistency (PRIMARY, SCORED) ────
# SKILL Implementation Notes: Chen et al. recommend 128x128 for >=1-megapixel
# images; Debiasi et al.'s cell-splitting sweet spot is 8x8 (10x10 is worse).
# "treat block size as a tunable parameter conditioned on input resolution
# ... start from Chen et al.'s 128x128 default for full-frame images and
# scale down proportionally for crops/small inputs". [CORPUS-FINDING]
MAXIMUM_BLOCK_SIZE_PIXELS = 128
TARGET_BLOCK_GRID_CELLS = 8  # Debiasi's validated sweet spot, reused as the
                             # minimum grid density for resolution scaling.
# SKILL: "sub-64x64 regions are explicitly flagged as difficult". [CORPUS]
MINIMUM_BLOCK_SIZE_PIXELS = 64

# SKILL step 3: "(a) the block's residual variance directly" - the simpler of
# the two offered statistics, used here to avoid pulling in Pipeline C's
# DFT-histogram machinery for what SKILL frames as an alternative choice.
# [CORPUS] (option a, explicitly offered).
LOCAL_STATISTIC_IS_RESIDUAL_VARIANCE = True

# SKILL step 4: "Compare each block's statistic against the local
# neighborhood median ... not a fixed global threshold". [CORPUS] for the
# principle; the neighbourhood's exact size is not numerically given.
# [ENGINEERING]: a 3x3 window of surrounding blocks (the smallest lattice
# neighbourhood with meaningful "local" context).
LOCAL_NEIGHBOURHOOD_WINDOW_BLOCKS = 3

# SKILL step 4: "Flag blocks whose statistic deviates significantly from
# their neighborhood" - "significantly" is not quantified. [ENGINEERING] /
# KNOWN_UNSOURCED_PARAMETER.
#
# ENHANCEMENT 2 (test-derived): the fixed factor-of-2 cutoff below is no
# longer the flag criterion. Diagnostic testing measured the median
# |log2(block variance / neighbourhood median)| of authentic photographs at
# 0.4192-0.7725, i.e. the TYPICAL authentic block already deviates by
# 1.34x-1.71x, so a factor-of-2 cutoff sits barely above the statistic's own
# noise floor: 24.1%-41.96% of every corpus image's cells clipped at the 1.0
# ceiling and raw_score came out exactly 1.000000 on all six. The flag
# threshold is now read off each image's own deviation spread
# (DEVIATION_ROBUST_SPREAD_MULTIPLE below), which is what SKILL step 4 asks
# for when it says the comparison must not use "a fixed global threshold".
# The constant is retained because the legacy diagnostic scalar still
# reports against it.
DEVIATION_FLAG_RATIO = 2.0

# ENHANCEMENT 1/2 parameters. [ENGINEERING] - the SKILL prints no numeric
# value for any of these; each is named in KNOWN_UNSOURCED_PARAMETERS.
TEXTURE_CONDITIONING_ENABLED = True
TEXTURE_FIT_POLYNOMIAL_DEGREE = 1  # [STRUCTURAL] straight line in log-log.
MINIMUM_BLOCKS_FOR_TEXTURE_FIT = 8  # [STRUCTURAL] below this the fit is
                                    # not meaningfully determined.
# Half the width of the central 50% of a normal distribution, i.e. the factor
# converting a median absolute deviation into a standard deviation. [DERIVED]
MEDIAN_ABSOLUTE_DEVIATION_TO_SIGMA = 0.6745
DEVIATION_ROBUST_SPREAD_MULTIPLE = 3.0  # [ENGINEERING]
MINIMUM_STATISTIC_FLOOR = 1e-12  # [STRUCTURAL] log-domain divide-by-zero guard.

# SKILL Output section: "[0,1]-normalized heatmap and a global scalar (max or
# top-k% mean)". [CORPUS] offers both; top-k% mean is picked as the more
# outlier-robust of the two explicitly-offered options. [ENGINEERING]
SCALAR_AGGREGATION_TOP_K_FRACTION = 0.10

# ── Pipeline C: blind cell-based PRNU spectral analysis (AUXILIARY) ────────
# SKILL: grid configurations "from the whole image as one 1x1 cell up to
# 10x10 = 100 cells" and Key findings: "8x8 cells is the sweet spot; 10x10 is
# *worse*". [CORPUS-FINDING] reused as the default grid.
PIPELINE_C_DEFAULT_GRID_CELLS = 8

# SKILL step 3: "magnitudes constrained to a universal range [0,8], divided
# into 100 bins". Implementation Notes: "dataset-specific ... recalibrate for
# a different image resolution/dataset rather than assuming these exact
# numbers transfer" - used here as the paper's own stated default, with that
# caveat carried into the reliability note. [CORPUS], flagged non-transferable.
PIPELINE_C_DFT_MAGNITUDE_RANGE = (0.0, 8.0)
PIPELINE_C_HISTOGRAM_BIN_COUNT = 100

# ── Pipeline B: reference-based PRNU (AUXILIARY, calibration-gated) ────────
# Formula: K_hat = sum(W_k * I_k) / sum(I_k^2). Source: Chen et al. 2008,
# Eq. 6. [CORPUS]
PRNU_ESTIMATOR_IS_ML = True

# B.2 step 6: "a 3x3 Wiener filter with variance obtained from the sample
# variance of the magnitude of F(ZM(K_hat))". [CORPUS] - fully specified,
# no unsourced constant.
WIENER_FILTER_KERNEL_SIZE = 3

MINIMUM_REFERENCE_IMAGES_FOR_PRNU = 2  # [STRUCTURAL] - Eq. 6 needs >=1 term
# to be non-degenerate; 2 is the practical minimum for a meaningful estimate.

# ── Pipeline D: FFT-spectrum noise-type triage (AUXILIARY, confidence-only) ─
# SKILL step 5: the printed pseudocode's single-threshold decision rule is
# self-contradictory (see the module docstring); the SKILL's own "most
# sensible resolution" is implemented, not the literal contradictory text.
# [CORPUS], resolved ambiguity per the SKILL's explicit instruction.
NOISE_TRIAGE_GAUSSIAN_THRESHOLD = -1.0
NOISE_TRIAGE_IMPULSE_THRESHOLD = 1.0
NOISE_TRIAGE_DENOISED_CONFIDENCE_PENALTY = 0.5  # [ENGINEERING] - SKILL
# instructs to "down-weight" confidence when denoising is suspected but
# gives no numeric penalty for this engine's confidence_weight scale.

# ── Condition-checker constants ─────────────────────────────────────────────
# SKILL: saturated pixels (I~255 for 8-bit) make the multiplicative PRNU term
# vanish (Eq. 19's attenuation). [CORPUS] for the principle; the exact
# saturation cutoff is this engine's own numeric choice. [ENGINEERING]
SATURATION_INTENSITY_FLOOR = 250.0
MAXIMUM_SATURATED_PIXEL_FRACTION = 0.5  # [ENGINEERING]

# Flat/textureless block floor - analogous principle to Eq. 19/20's texture
# dependence, no numeric floor given by the SKILL.
#
# ENHANCEMENT 3 (test-derived): this floor was 1.0, which is roughly TEN
# TIMES the median per-block residual variance of an ordinary JPEG
# photograph. Measured medians: campic 0.1358, campic2 0.1410, gen 0.1957,
# genratedimage 0.0954 - so 62.0%, 62.0%, 61.3% and 55.4% of their blocks
# fell under the old floor, tripping MAXIMUM_FLAT_BLOCK_FRACTION and making
# the engine abstain on four of six corpus images including BOTH authentic
# photographs. The floor is now the variance of uniform quantization error
# on a 1-LSB 8-bit quantizer, 1/12: a block whose residual varies by less
# than the rounding error of the container it arrived in genuinely carries
# no measurable noise. [DERIVED] from the 8-bit container, not from the
# SKILL. Verified: at 1/12 all eight corpus photographs pass (worst case
# genratedimage.jpeg at 49.1%, marginal) while a constant-grey frame still
# trips the gate at 100.0%.
BLOCK_VARIANCE_DEGENERACY_FLOOR = 1.0 / 12.0
MAXIMUM_FLAT_BLOCK_FRACTION = 0.5  # [ENGINEERING]

# ── Scorer constants (provisional-route placeholders, same pattern as the
# other engines in this system; SKILL: "Calibration not specified in corpus -
# engineering recommendation" for Pipeline A's scalar). [ENGINEERING]
MINIMUM_CALIBRATION_REFERENCE_COUNT = 10
# ENHANCEMENT 5 (test-derived): when no known-authentic reference scores are
# supplied the engine now declines to publish a probability at all instead of
# emitting the provisional sigmoid at full confidence. Evidence: on ground
# truth the aggregate scalar moves +0.0340 to +0.0772 for a real manipulation,
# while the spread between six untampered images is 0.1222 - the between-image
# spread is larger than the manipulation effect, so no fixed cutoff on this
# scalar can separate tampered from authentic. The SKILL's Output section
# states Pipeline A's calibration is "not specified in corpus - engineering
# recommendation". raw_score, the heatmap and flagged_regions are still
# published; only the unfounded probability is withheld. [ENGINEERING]
ABSTAIN_WHEN_UNCALIBRATED = True
PROVISIONAL_SIGMOID_SLOPE = 10.0
PROVISIONAL_SIGMOID_MIDPOINT = 0.3
SIGMOID_EXPONENT_LIMIT = 60.0

# ── Visualisation constants ─────────────────────────────────────────────────
EVIDENCE_DISPLAY_CLIP_PERCENTILE = 99.0  # [PRESENTATION]
EIGHT_BIT_DISPLAY_MAXIMUM = 255.0  # [STRUCTURAL]
EVIDENCE_MAP_MAX_DIMENSION = 2048  # [PRESENTATION]

# ── Audit-aid tuples ─────────────────────────────────────────────────────────
TEST_DERIVED_ENHANCEMENTS = (
    "ENHANCEMENT 1 - TEXTURE_CONDITIONING_ENABLED. Evidence: Spearman "
    "correlation between per-block residual variance and per-block Laplacian "
    "energy measured 0.9753 / 0.9768 / 0.9635 / 0.7726 / 0.9604 / 0.9912 on "
    "the six corpus images, so the block statistic ranked scene texture, not "
    "noise level. Consequence measured on ground truth: heatmap peak fell "
    "inside a known pasted region 0 times out of 12, and the top-10% of "
    "blocks overlapped the paste 0.083 of the time against a chance rate of "
    "0.083 - exactly chance. After conditioning: 6/18 peak hits and 0.268 "
    "overlap against 0.111 chance.",
    "ENHANCEMENT 2 - DEVIATION_ROBUST_SPREAD_MULTIPLE replaces "
    "DEVIATION_FLAG_RATIO as the flag criterion, and the aggregate scalar "
    "becomes the flagged-block fraction (SKILL step 5, 'Aggregate flagged "
    "blocks into a heatmap and a scalar summary') rather than the Output "
    "section's alternative top-k% mean. Evidence: the top-k% mean measured "
    "exactly 1.000000 on all six corpus images, on all six global nuisance "
    "transforms of each of them, and on all 18 ground-truth manipulations of "
    "a real photograph - a delta of +0.000000 in every single case. The "
    "flagged-block fraction moves in the correct direction on 17 of those 18.",
    "ENHANCEMENT 3 - BLOCK_VARIANCE_DEGENERACY_FLOOR 1.0 -> 1/12. Evidence: "
    "62.0%, 62.0%, 61.3% and 55.4% of the blocks of campic, campic2, gen and "
    "genratedimage fell below the old floor, so the engine abstained on four "
    "of six corpus images including both authentic photographs, while their "
    "median block variances were 0.1358, 0.1410, 0.1957 and 0.0954.",
    "ENHANCEMENT 4 - the saturation check's pass/fail boolean is now read. "
    "Evidence: fake .jpeg has 65% of its pixels at or above 250, the check "
    "returned failed, its result was discarded, and the engine published FAKE "
    "at probability 0.9991 on an image where Eq. 19's attenuation makes the "
    "multiplicative PRNU term vanish.",
    "ENHANCEMENT 5 - ABSTAIN_WHEN_UNCALIBRATED. Evidence: the aggregate "
    "scalar moves +0.0340 to +0.0772 under a real local noise manipulation "
    "while six untampered images span 0.1222, so the between-image spread "
    "exceeds the manipulation effect and no fixed cutoff separates them.",
)

REJECTED_ENHANCEMENTS = (
    "Robust MAD-sigma of the residual as the block statistic, replacing "
    "variance. REJECTED: raw_score stayed exactly 1.000000 on the authentic "
    "host and on all three real manipulations of it (cross-sensor splice, "
    "region denoised, region given foreign noise); delta +0.00000 in every "
    "case. Changing the statistic alone does not lift the saturation.",
    "MAD-sigma of the residual's HH wavelet subband. REJECTED as degenerate "
    "on JPEG input: 90 of campic's 108 blocks returned exactly 0.0 because "
    "the encoder had quantized that subband away. An apparent +0.45455 splice "
    "delta was traced to floor arithmetic on those zeros - np.where(x>0,...) "
    "and np.maximum(x,eps) disagree on values inside (0,eps) - and not to any "
    "detection.",
    "MAD-sigma of the INTENSITY's HH subband (Donoho's noise estimator). "
    "REJECTED: raw_score 1.000000 everywhere, delta +0.00000 on all three "
    "real manipulations.",
    "Trimmed (p10-p90) block variance. REJECTED: delta +0.00000 on all three "
    "real manipulations.",
    "Noise-floor estimator - the p5/p10/p25 percentile of the block's 5x5 "
    "local-variance map. REJECTED despite genuinely reducing texture coupling "
    "(Spearman 0.9753 -> 0.7475 at p5): localization stayed at chance "
    "(top-10%-inside 0.121 vs 0.111 chance) and the scalar still read 1.00000 "
    "on every authentic and every manipulated image.",
    "SKILL Eq. 20's f_T as the conditioning feature instead of intensity-plane "
    "Laplacian energy. REJECTED: f_T is computed from the residual, so "
    "denoising a region moves the feature in exactly the way that explains "
    "away the reduced variance. Measured top-10%-inside on denoised regions "
    "0.076, BELOW the 0.111 chance rate, versus 0.273 for the "
    "intensity-plane feature.",
    "Scalar = (top-k% mean - median) of the clipped heatmap. REJECTED as the "
    "weaker of the two working scalars: correct direction on 3/6 denoise "
    "trials against 5/6 for the flagged-block fraction.",
    "Scalar = (top-k% mean - median) / robust spread of the UNCLIPPED "
    "deviation field. REJECTED: saturates at 1.0000 on all seven images "
    "tested, delta +0.0000 on splice and foreign-noise trials.",
)

KNOWN_UNSOURCED_PARAMETERS = (
    "LOCAL_NEIGHBOURHOOD_WINDOW_BLOCKS - SKILL: 'the local neighborhood "
    "median', no numeric neighbourhood size given.",
    "DEVIATION_FLAG_RATIO - SKILL: 'deviates significantly', no numeric "
    "threshold given.",
    "SCALAR_AGGREGATION_TOP_K_FRACTION - SKILL offers 'max or top-k% mean' "
    "without a specific k.",
    "NOISE_TRIAGE_DENOISED_CONFIDENCE_PENALTY - SKILL instructs to "
    "'down-weight' confidence when denoising is detected, no numeric factor.",
    "SATURATION_INTENSITY_FLOOR / MAXIMUM_SATURATED_PIXEL_FRACTION - SKILL "
    "describes the saturation failure mode (Eq. 19) but gives no per-image "
    "gating cutoff for this engine's condition checker.",
    "BLOCK_VARIANCE_DEGENERACY_FLOOR / MAXIMUM_FLAT_BLOCK_FRACTION - no "
    "numeric floor given for detecting textureless blocks.",
    "PROVISIONAL_SIGMOID_SLOPE / MIDPOINT - placeholder fallback route only, "
    "SKILL gives no calibration at all for Pipeline A's aggregate scalar. "
    "Since ENHANCEMENT 5 this route no longer publishes a probability.",
    "TEXTURE_CONDITIONING_ENABLED / TEXTURE_FIT_POLYNOMIAL_DEGREE - the SKILL "
    "gives the rationale for conditioning on texture (step 4, citing Chen et "
    "al.'s B.4 predictor) and names the fitting tool, but prints no blind, "
    "camera-agnostic form of the predictor and no degree for the fit.",
    "DEVIATION_ROBUST_SPREAD_MULTIPLE - replaces DEVIATION_FLAG_RATIO as the "
    "quantification of step 4's 'deviates significantly'; equally unsourced, "
    "but measured to have dynamic range where the fixed ratio had none.",
    "MEDIAN_ABSOLUTE_DEVIATION_TO_SIGMA - a property of the normal "
    "distribution, not of this SKILL; it only rescales the multiple above.",
)

KNOWN_SKILL_AMBIGUITIES = (
    "The Mihcak wavelet-based denoising filter F (and Lin et al.'s FDR "
    "enhancement) is named but never mathematically specified anywhere in "
    "this SKILL file; implemented as the simplest concrete instantiation of "
    "its stated role (W=I-F(I)) - see RESIDUAL_FILTER_WAVELET_FAMILY. FDR "
    "is not implemented at all (no formula given).",
    "Pipeline D's printed decision-rule pseudocode is self-contradictory "
    "(a single 'threshold' variable used in two branches); the SKILL "
    "supplies its own 'most sensible resolution' "
    "(NOISE_TRIAGE_GAUSSIAN_THRESHOLD / NOISE_TRIAGE_IMPULSE_THRESHOLD), "
    "which is implemented here, not the literal contradictory text.",
)

KNOWN_UNIMPLEMENTED_MODULES = (
    "Pipeline B's weighted detector (Eq. 11, needing beta_b) and full "
    "correlation-predictor machinery (B.4, Eq. 18-22) - I_crit, tau, and c "
    "are given only as one specific Canon G2 example and the SKILL itself "
    "states they 'do not transfer directly across camera models'; only the "
    "ML estimator (Eq. 6), ZM+Wiener preprocessing (B.2), and raw unweighted "
    "per-block correlation are implemented.",
    "Pipeline B's Neyman-Pearson threshold / Generalized-Gaussian tail fit "
    "(B.3 steps 10-11) - requires a calibration dataset of known-camera "
    "images this engine's fixed contract has no slot for beyond the raw "
    "reference images already used for the ML estimator.",
    "Lin et al.'s FDR (Filtering Distortion Removal) enhancement - named in "
    "the SKILL as a second-stage step but given no formula.",
    "Pipeline C's cell aggregation uses only S_mean/S_rms (Eq. 5-6) as "
    "auxiliary evidence; the SKILL's D-EER-calibrated decision threshold is "
    "dataset-specific and not reproduced as a hard classification here.",
)
