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
# KNOWN_UNSOURCED_PARAMETER: a block is flagged when its statistic exceeds
# this multiple of its neighbourhood median.
DEVIATION_FLAG_RATIO = 2.0

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
# dependence, no numeric floor given. [ENGINEERING]
BLOCK_VARIANCE_DEGENERACY_FLOOR = 1.0
MAXIMUM_FLAT_BLOCK_FRACTION = 0.5  # [ENGINEERING]

# ── Scorer constants (provisional-route placeholders, same pattern as the
# other engines in this system; SKILL: "Calibration not specified in corpus -
# engineering recommendation" for Pipeline A's scalar). [ENGINEERING]
MINIMUM_CALIBRATION_REFERENCE_COUNT = 10
PROVISIONAL_SIGMOID_SLOPE = 10.0
PROVISIONAL_SIGMOID_MIDPOINT = 0.3
SIGMOID_EXPONENT_LIMIT = 60.0

# ── Visualisation constants ─────────────────────────────────────────────────
EVIDENCE_DISPLAY_CLIP_PERCENTILE = 99.0  # [PRESENTATION]
EIGHT_BIT_DISPLAY_MAXIMUM = 255.0  # [STRUCTURAL]
EVIDENCE_MAP_MAX_DIMENSION = 2048  # [PRESENTATION]

# ── Audit-aid tuples ─────────────────────────────────────────────────────────
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
    "SKILL gives no calibration at all for Pipeline A's aggregate scalar.",
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
