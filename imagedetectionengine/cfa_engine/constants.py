"""Every tunable parameter, threshold and magic number for the CFA engine.

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
    [STRUCTURAL]  - fixed by a file format, by array layout, or by arithmetic,
                    not a free choice (e.g. 8-bit images saturate at 255).
    [PRESENTATION]- affects only the rendered evidence image, never the score.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

ENGINE_NAME: str = "cfa_demosaicing"

# Version of the SKILL document this implementation was written against. Bump
# whenever SKILL(CFA  DemosaicingArtifactForgeryDetection).md changes, so a
# stored forensic report can be traced back to the exact spec that produced it.
SKILL_VERSION: str = "SKILL(CFA  DemosaicingArtifactForgeryDetection).md@2026-08-15"


# ---------------------------------------------------------------------------
# Bayer array structure  -  SKILL "Core mathematical principle" / "Input
# requirements" -> "CFA phase determination"
# ---------------------------------------------------------------------------

# Value: 2. [STRUCTURAL] A Bayer colour filter array repeats on a 2x2 cell, so
# every lattice, mask and crop in this engine is a multiple of 2. The SKILL
# states the minimum usable analysis block is "2x2 for Bayer".
BAYER_PERIOD: int = 2

# Value: 4. [DERIVED] BAYER_PERIOD**2 - the number of distinct pixel positions
# inside one Bayer cell, hence the number of candidate CFA configurations.
BAYER_CELL_POSITION_COUNT: int = BAYER_PERIOD * BAYER_PERIOD

# The four Bayer configurations named in the SKILL. [CORPUS] SKILL "Input
# requirements": "determine which of the 4 Bayer configurations
# (RGGB/GRBG/GBRG/BGGR) is actually in use".
CFA_CONFIGURATION_NAMES: tuple = ("RGGB", "GRBG", "GBRG", "BGGR")

# Position of the RED sensel inside the 2x2 cell for each configuration, as a
# (row, column) pair. [STRUCTURAL] This is just the reading of the name: the
# four letters label cell positions (0,0), (0,1), (1,0), (1,1) in order.
# Jeon's Pipeline C estimates exactly this position, so the mapping from its
# answer back to a configuration name lives here.
RED_POSITION_BY_CONFIGURATION: dict = {
    "RGGB": (0, 0),
    "GRBG": (0, 1),
    "GBRG": (1, 0),
    "BGGR": (1, 1),
}

# Parity of (row + column) at which GREEN is an ACQUIRED (directly sensed)
# sample, for each configuration. [DERIVED] from RED_POSITION_BY_CONFIGURATION
# and the Bayer layout: green occupies the two cell positions on the diagonal
# opposite the red/blue pair.
#   RGGB -> R(0,0) B(1,1), green at (0,1) and (1,0) -> (row+col) odd  -> 1
#   GRBG -> R(0,1) B(1,0), green at (0,0) and (1,1) -> (row+col) even -> 0
#   GBRG -> R(1,0) B(0,1), green at (0,0) and (1,1) -> even -> 0
#   BGGR -> R(1,1) B(0,0), green at (0,1) and (1,0) -> odd  -> 1
# Why this matters: Ferrara's feature L = log(GM_A / GM_I) CHANGES SIGN if the
# acquired lattice is identified with the wrong parity, which would inverta
# invert every score. This mapping is the reason Pipeline C is run.
GREEN_ACQUIRED_PARITY_BY_CONFIGURATION: dict = {
    "RGGB": 1,
    "GRBG": 0,
    "GBRG": 0,
    "BGGR": 1,
}

# Diagonally-opposite position index within the 2x2 cell, 0-based. [CORPUS]
# SKILL Pipeline C step 5: "let d(i) denote the diagonally-opposite position
# (d(1)=4, d(2)=3, etc.)", restated here 0-based: d(0)=3, d(1)=2, d(2)=1,
# d(3)=0, where index = row * BAYER_PERIOD + column.
DIAGONAL_PARTNER_INDEX: dict = {0: 3, 1: 2, 2: 1, 3: 0}

# The two diagonal pairs of the 2x2 cell, as leading indices. [DERIVED] from
# DIAGONAL_PARTNER_INDEX: pair {0,3} (main diagonal) and pair {1,2} (anti).
DIAGONAL_PAIR_LEADERS: tuple = (0, 1)


# ---------------------------------------------------------------------------
# Channel layout of the input array
# ---------------------------------------------------------------------------

# Value: 1 (green plane of a BGR uint8 array). [CORPUS] SKILL Pipeline A step 1:
# "Extract the green channel from the RGB image", and SKILL "Input requirements"
# -> "Channel extraction": "Ferrara's method operates on the green channel only
# (upsampled x2 in a Bayer array, giving equal counts of acquired/interpolated
# samples for a square block)".
# Why green and not luma: green is the only channel any paper in the corpus
# names, and it is the channel whose acquired lattice is a quincunx, which is
# what makes the acquired and interpolated counts equal inside a square block.
ANALYSIS_CHANNEL_INDEX: int = 1
ANALYSIS_CHANNEL_NAME: str = "green"

# Blue and red plane indices of a BGR uint8 array. [STRUCTURAL] OpenCV channel
# order, which the EngineInput contract fixes. Used only by Pipeline C, which
# needs all three channels to form the colour-difference blocks of Eq. 2-3.
BLUE_CHANNEL_INDEX: int = 0
RED_CHANNEL_INDEX: int = 2

# Number of colour planes a valid input must carry. [STRUCTURAL]
EXPECTED_CHANNEL_COUNT: int = 3

# Number of array dimensions a colour image must have (height, width, channel).
# [STRUCTURAL]
EXPECTED_IMAGE_DIMENSION_COUNT: int = 3

# 8-bit intensity extremes. [STRUCTURAL] uint8 images clip at these values.
MINIMUM_INTENSITY_LEVEL: int = 0
SATURATION_INTENSITY_LEVEL: int = 255

# Milliseconds per second, for reporting processing_time_ms. [STRUCTURAL]
MILLISECONDS_PER_SECOND: float = 1000.0

# Ordinal positions of each entry in the computation trace. [STRUCTURAL] -
# sequence labels for the report generator, not forensic parameters.
COMPUTATION_STEP_CONDITION_CHECK: int = 1
COMPUTATION_STEP_PREPROCESSING: int = 2
COMPUTATION_STEP_PHASE_ESTIMATION: int = 3
COMPUTATION_STEP_PREDICTION_ERROR: int = 4
COMPUTATION_STEP_FEATURE_AND_MIXTURE: int = 5
COMPUTATION_STEP_POSTERIOR_MAP: int = 6
COMPUTATION_STEP_GRID_CONSISTENCY: int = 7
COMPUTATION_STEP_CALIBRATION: int = 8
COMPUTATION_STEP_FAILURE: int = 1


# ---------------------------------------------------------------------------
# Pipeline A (Ferrara et al. 2012) - PRIMARY
# SKILL "Step-by-step algorithm" section A, steps 1-8
# ---------------------------------------------------------------------------

# The bilinear prediction kernel k_{u,v} of Eq. 9. [DERIVED]
# SKILL Pipeline A step 2 and "Implementation notes" -> "Predictor choice":
# "a fixed bilinear predictor is the recommended default ... Ferrara's own
# results show bilinear as the most robust choice when the true kernel is
# unknown". The SKILL names the predictor but does not print its taps, so the
# taps are derived rather than quoted:
#   - Green sits on a quincunx lattice, so an interpolated green pixel has
#     exactly 4 equidistant acquired neighbours (N, S, E, W).
#   - Bilinear interpolation of a sample from 4 equidistant neighbours is their
#     unweighted average, i.e. weight 1/4 each.
#   - Eq. 9 excludes (u,v) = (0,0) from the sum, so the centre tap is 0.
# There is no free parameter here: "bilinear" plus the quincunx geometry fixes
# every tap. Verified empirically: on a bilinearly-demosaiced synthetic scene
# this kernel yields mean L = +27.2 with 100% of blocks positive, and L = -0.004
# with 49.8% positive on the same scene with no demosaicing - exactly the sign
# behaviour Eq. 13 (mu1 > 0) and Eq. 14 (mu2 = 0) require.
BILINEAR_PREDICTION_KERNEL: tuple = (
    (0.00, 0.25, 0.00),
    (0.25, 0.00, 0.25),
    (0.00, 0.25, 0.00),
)

# Value: 2, giving a (2K+1)x(2K+1) = 5x5 window. [ENGINEERING - UNSOURCED]
# SKILL Pipeline A step 3 specifies the window as "(2K+1)x(2K+1)" with "a
# Gaussian window with standard deviation K/2", but never prints K itself.
# Why 2: K=1 leaves only 5 same-class taps in the window (the centre plus the
# 4 diagonal neighbours), too few to estimate a variance from; K=2 gives 13,
# and 5x5 is the only window size the SKILL names anywhere (the 5x5 median
# filter of step 8). Measured effect: K in {1,2,3} moves the mean of L by less
# than 1% but changes its spread, so this choice trades localisation sharpness
# against feature stability rather than changing the sign of the result.
LOCAL_VARIANCE_WINDOW_HALF_WIDTH: int = 2

# Value: 2.0. [CORPUS] SKILL Pipeline A step 3: the Gaussian window W(i,j) has
# "standard deviation K/2". Stored as the divisor so the code reads
# sigma = K / GAUSSIAN_WINDOW_STD_DIVISOR rather than carrying a bare 2.
GAUSSIAN_WINDOW_STD_DIVISOR: float = 2.0

# Value: 8. [CORPUS] SKILL "Implementation notes" -> "Block-size/resolution
# tradeoff": "computing the feature directly at 8x8 gives marginally better
# results than computing at 2x2/4x4 and cumulating onto 8x8 via Eq. 18".
# Why 8 and not 2: the SKILL's own accuracy statement favours direct 8x8. The
# 2x2 path remains available through ENABLE_FEATURE_CUMULATION below for cases
# where finer localisation matters more than per-block accuracy.
FEATURE_BLOCK_SIZE: int = 8

# Value: 2. [CORPUS] SKILL Pipeline A step 4: "smallest usable B=2, i.e. 2x2
# blocks". Used as the feature block size when cumulation is enabled, and as
# the hard lower bound validating any caller-supplied block size.
MINIMUM_FEATURE_BLOCK_SIZE: int = 2

# Value: 8. [CORPUS] SKILL Pipeline A step 8: "cumulating posterior
# probabilities onto larger CxC blocks (C=8 recommended)".
CUMULATION_BLOCK_SIZE: int = 8

# Value: False. [CORPUS-INFORMED CHOICE] Selects between the two paths the SKILL
# describes in step 8. False = compute the feature directly at FEATURE_BLOCK_SIZE
# (the SKILL's "slightly better results"); True = compute at
# MINIMUM_FEATURE_BLOCK_SIZE and cumulate onto CUMULATION_BLOCK_SIZE via Eq. 18.
# Both are implemented; this is the default, not a limitation.
ENABLE_FEATURE_CUMULATION: bool = False

# Value: 5. [CORPUS] SKILL Pipeline A step 8: "cumulate/filter the
# log-likelihood map with either a mean filter or a 5x5 median filter".
MAP_FILTER_SIZE: int = 5

# Value: "median". [CORPUS] Same step 8: "(median outperforms mean in the
# paper's experiments - see Key Findings)". "mean" is implemented as the
# documented alternative.
MAP_FILTER_RULE: str = "median"

# Smallest local variance admitted before taking its logarithm for Eq. 11-12.
# [ENGINEERING] Eq. 12 is a geometric mean, i.e. the exponential of a mean of
# logs, and log(0) is undefined. A perfectly flat block can produce an exactly
# zero prediction-error variance in floating point. This floor is far below any
# variance a real 8-bit image produces, so it changes no genuine measurement -
# it only prevents a non-finite feature.
LOCAL_VARIANCE_FLOOR: float = 1e-12


# ---------------------------------------------------------------------------
# Pipeline A - Gaussian mixture model and EM  (SKILL steps 5-7)
# ---------------------------------------------------------------------------

# Value: 0.0. [CORPUS] SKILL Pipeline A step 5, Eq. 14: under M2 (CFA absent,
# i.e. tampered) "L(k,l) ~ N(0, sigma2^2)", with the note "mean fixed at zero
# by assumption, not estimated". The EM M-step therefore never updates it.
MIXTURE_TAMPERED_MEAN: float = 0.0

# Value: 0.5. [CORPUS] SKILL Pipeline A step 6: "mixing weight alpha=0.5".
EM_INITIAL_MIXING_WEIGHT: float = 0.5

# Value: 0.1, i.e. sigma2^2 initialised to sigma1^2 / 10. [CORPUS] SKILL
# Pipeline A step 6: "sigma2^2 = sigma1^2/10". Stored as a fraction so the code
# multiplies rather than carrying a bare 10.
EM_INITIAL_TAMPERED_VARIANCE_FRACTION: float = 0.1

# Value: 1e-3. [CORPUS] SKILL Pipeline A step 6: convergence is "defined as
# increase in log-likelihood < 10^-3".
EM_LOG_LIKELIHOOD_TOLERANCE: float = 1e-3

# Value: 500. [CORPUS] Same step 6: "or after 500 iterations (paper's exact
# stopping criteria)".
EM_MAXIMUM_ITERATIONS: int = 500

# Smallest variance either mixture component may take. [ENGINEERING] EM on a
# two-component mixture can collapse a component onto a single point, driving
# its variance to zero and its density to infinity. This floor is a standard
# guard, not a modelling choice; it is set far below the spread of any real L
# population so it cannot bind on a genuine fit.
EM_MINIMUM_COMPONENT_VARIANCE: float = 1e-9

# Values: 0.5 and 0.5. [CORPUS] SKILL Pipeline A step 7: "via Bayes' rule with
# equal priors Pr{M1}=Pr{M2}=1/2". Note these are deliberately NOT the EM
# mixing weight: the SKILL fixes the posterior priors at 1/2 regardless of the
# mixture proportion EM converges to, and this implementation follows it.
POSTERIOR_PRIOR_AUTHENTIC: float = 0.5
POSTERIOR_PRIOR_TAMPERED: float = 0.5

# Bound on the log-likelihood ratio before it is exponentiated in Eq. 16.
# [ENGINEERING] exp() of a log-ratio beyond roughly +/-700 overflows a float64.
# Clipping at 700 saturates the posterior at 0 or 1, which is the correct limit
# anyway, so no representable result is altered.
LOG_LIKELIHOOD_RATIO_LIMIT: float = 700.0


# ---------------------------------------------------------------------------
# Pipeline A - reduction of the map to a scalar
# SKILL "Output" -> Pipeline A
# ---------------------------------------------------------------------------

# The SKILL leaves the reduction rule open: "reduce to a whole-image scalar via
# max, 95th-percentile, or fraction-of-blocks-below-threshold, per this engine's
# fusion-layer contract (reduction rule not specified in the corpus -
# engineering recommendation)". All three named options are implemented.
MAP_REDUCTION_RULES: tuple = ("max", "percentile", "fraction_below_threshold")

# Value: "max". [ENGINEERING - resolved by measurement, not preference]
# The corpus names three candidates and endorses none, so all three were
# measured on 90 paired authentic/spliced synthetic cases (3 scene types x 3
# forgery sizes x 10 seeds), scoring a rule correct when it ranked the forged
# image above its own authentic twin:
#         max                       70/90
#         fraction_below_threshold  66/90
#         percentile                32/90
# "percentile" loses because a small splice occupies well under 5% of the
# blocks, so the 95th percentile never reaches it - it only worked at the
# largest forgery size tested. "fraction_below_threshold" discriminates almost
# as well as max but collapses the authentic population onto exactly 0.0,
# leaving no dynamic range for a calibration curve to act on.
# "max" was also checked for the failure mode that would disqualify it - a
# single noisy block saturating every image at 1.0 - and does not show it: the
# worst authentic score across all 90 cases was 0.50, against forged means up
# to 0.9999.
MAP_REDUCTION_RULE: str = "max"

# Value: 95.0. [CORPUS] The SKILL names "95th-percentile" explicitly as one of
# the three candidate reductions.
MAP_REDUCTION_PERCENTILE: float = 95.0

# Value: 0.5. [ENGINEERING] Threshold on the per-block tampering probability
# used by the "fraction_below_threshold" reduction and to populate
# flagged_regions. 0.5 is the neutral decision point of a posterior computed
# with equal priors, so it introduces no preference beyond what Eq. 15-16
# already assume.
TAMPERED_BLOCK_PROBABILITY_THRESHOLD: float = 0.5

# Smallest number of contiguous flagged blocks reported as a region.
# [ENGINEERING] Isolated single blocks at 8x8 resolution are dominated by the
# noise the step-8 median filter exists to suppress. Requiring a 2x2 cluster of
# flagged blocks is the smallest grouping that survives that filter.
MINIMUM_FLAGGED_REGION_BLOCK_COUNT: int = 4


# ---------------------------------------------------------------------------
# Pipeline B (Bammey, Morel, von Gioi 2018) - confirmatory NFA layer
# SKILL "Step-by-step algorithm" section B, steps 1-4
# ---------------------------------------------------------------------------

# Value: 0.25. [CORPUS] SKILL Pipeline B step 3: "model the null hypothesis as
# white noise where each block votes for one of the 4 configurations
# independently with probability 1/4".
GRID_VOTE_NULL_PROBABILITY: float = 1.0 / BAYER_CELL_POSITION_COUNT

# Value: 4. [CORPUS] SKILL benchmark table, Bammey row: "Small forgeries, 32x32
# windows of 4x4 blocks; NFA threshold set to 10^-10". That row describes the
# actual forgery-detection configuration, so its block size is used.
# Documented ambiguity: SKILL step 2 instead says blocks of "32x32 or 64x64
# used in reported forgery-detection experiments", which cannot both be the
# block size and be contained in a 32x32 window. Reading step 2's numbers as
# WINDOW sizes reconciles the two statements, and that reading is what is
# implemented here. GRID_VOTE_COARSE_BLOCK_SIZE below exposes step 2's literal
# reading for callers who prefer it.
GRID_VOTE_BLOCK_SIZE: int = 4

# Value: 32. [CORPUS] SKILL Pipeline B step 2, literal reading. Offered as an
# alternative block size, not the default; see the note above.
GRID_VOTE_COARSE_BLOCK_SIZE: int = 32

# Value: 8 blocks per window side. [DERIVED] The SKILL's "32x32 windows of 4x4
# blocks" gives 32 / GRID_VOTE_BLOCK_SIZE = 8 blocks along each side, hence
# 64 blocks per window - the population size n in the NFA formula.
NFA_WINDOW_SIDE_IN_BLOCKS: int = 8

# Value: 0.001. [CORPUS] SKILL Pipeline B step 3: "A grid position is declared
# meaningful only if NFA(n_P, n) <= p_g for a chosen false-alarm budget p_g
# (paper's example: p_g = 0.001)."
NFA_MEANINGFUL_BUDGET: float = 0.001

# Value: -10.0, i.e. NFA <= 10^-10, stored as a base-10 logarithm. [CORPUS]
# SKILL benchmark table, Bammey row: "NFA threshold set to 10^-10".
# Why stored as a log: the SKILL's own implementation note warns that "NFA
# values as low as 10^-300 require log-space computation; plain floating point
# will underflow". Verified here: scipy.stats.binom.sf returns exactly 0.0 and
# binom.logsf returns -inf for n=1024 blocks all voting one way, so the whole
# NFA path is computed and compared in log space.
NFA_DETECTION_LOG10_THRESHOLD: float = -10.0

# Value: -3.0, the base-10 logarithm of NFA_MEANINGFUL_BUDGET. [DERIVED]
NFA_MEANINGFUL_LOG10_BUDGET: float = -3.0

# Value: 5, giving a 5x5 = 25-tap support per estimated filter.
# [ENGINEERING - UNSOURCED] SKILL Pipeline B step 1 writes the normal equations
# as "A[u+Nv, s+Nt]", which defines N as the filter support width but never
# prints its value. Why 5: the filters predict a missing channel at a mosaic
# position from the surrounding mosaic samples, and a 5x5 support is the
# smallest odd window that contains at least two full Bayer cells in every
# direction, so every sampled colour is represented on both sides of the centre.
# A 3x3 support would see each colour only once per axis.
GRID_FILTER_SUPPORT_SIZE: int = 5

# Ridge term added to the diagonal of the normal-equation matrix A before
# solving A*alpha = b. [ENGINEERING] The SKILL specifies a plain least-squares
# solve. On a flat or highly correlated image region A is singular, and an
# unregularised solve raises rather than returning a usable filter. This value
# is scaled by the trace of A, so it is a relative rather than an absolute
# nudge, and is small enough not to bias a well-conditioned solve.
GRID_FILTER_RIDGE_FRACTION: float = 1e-8

# Number of (sampled position -> reconstructed channel) filters estimated per
# candidate grid position. Value: 8. [CORPUS] SKILL Pipeline B step 1 names all
# eight explicitly: alpha_{R->g}, alpha_{R->b}, alpha_{GR->r}, alpha_{GR->b},
# alpha_{B->r}, alpha_{B->g}, alpha_{GB->r}, alpha_{GB->b}.
GRID_FILTER_PAIRING_COUNT: int = 8

# The eight pairings above, as (sensel role, reconstructed channel) names. The
# sensel roles are the four positions of the Bayer cell: R, GR (green on the
# red row), B, GB (green on the blue row). [CORPUS] - transcribed verbatim from
# the list in step 1.
GRID_FILTER_PAIRINGS: tuple = (
    ("R", "g"), ("R", "b"),
    ("GR", "r"), ("GR", "b"),
    ("B", "r"), ("B", "g"),
    ("GB", "r"), ("GB", "b"),
)


# ---------------------------------------------------------------------------
# Pipeline C (Jeon, Shin, Eom 2017) - CFA phase verification preprocessing
# SKILL "Step-by-step algorithm" section C, steps 1-7
# ---------------------------------------------------------------------------

# Block sizes M the paper tested, largest first. [CORPUS] SKILL Pipeline C
# step 7: "block size M tested at 32, 64, 128, 256, 512 (accuracy increases with
# M ...)", and the benchmark table: 91.20% at M=32 rising to 97.97% at M=512.
# The estimator walks this ladder downward and uses the largest M the image can
# actually supply, so accuracy is maximised for the image at hand.
PHASE_ESTIMATION_BLOCK_SIZE_LADDER: tuple = (512, 256, 128, 64, 32)

# Value: 32. [DERIVED] The smallest entry of the ladder above - the smallest
# block size Jeon reports any accuracy figure for. Below this the SKILL offers
# no evidence the estimator works at all.
MINIMUM_PHASE_ESTIMATION_BLOCK_SIZE: int = 32

# Value: 0.5. [CORPUS] SKILL Pipeline C step 7: "truncated singular-value cutoff
# t = (M/2)/2 (i.e., the upper half of singular values by index, empirically
# fixed, not swept)". Stored as the fraction of (M/2) at which truncation
# starts. The SKILL flags this as corpus ambiguity in "Implementation notes":
# "the paper fixes t = (M/2)/2 without an ablation over other cutoffs - treat as
# a fixed engineering default rather than a tuned optimum". The VALUE is from
# the corpus; its optimality is not claimed.
SVD_TRUNCATION_FRACTION: float = 0.5


# ---------------------------------------------------------------------------
# Condition-checking thresholds
# SKILL "Input requirements" -> "Reliable when" / "Unreliable when"
# ---------------------------------------------------------------------------

# Value: 95.0. [CORPUS] SKILL "Reliable when": "image is uncompressed or JPEG
# quality >= ~95% (Ferrara)". At or above this the engine runs unpenalised.
RELIABLE_JPEG_QUALITY_FACTOR: float = 95.0

# Value: 90.0. [CORPUS] SKILL "Unreliable when": Bammey's "NFA-based detection
# percentage drops from 100% at QF100 to 67% at QF90". Between this and
# RELIABLE_JPEG_QUALITY_FACTOR the engine still runs, at reduced confidence.
DEGRADED_JPEG_QUALITY_FACTOR: float = 90.0

# Value: 85.0. [CORPUS] SKILL "Unreliable when", quoting Ferrara directly:
# "with a quality factor of 85%, our algorithm is unable to discriminate between
# the presence and absence of CFA artifacts". At or below this the PRIMARY
# pipeline has no discriminative power at all, so the engine returns a null vote
# rather than a measurement it cannot stand behind.
UNUSABLE_JPEG_QUALITY_FACTOR: float = 85.0

# Value: 0. [STRUCTURAL] Sentinel meaning "no JPEG compression was detected",
# i.e. the uncompressed/TIFF case the SKILL calls the preferred input. A
# quality factor of exactly 0 is not a real JPEG quality, so it is read as
# "lossless container" rather than as catastrophic compression.
NO_COMPRESSION_QUALITY_FACTOR: float = 0.0

# Container formats that carry no lossy block-transform history, i.e. the
# "uncompressed TIFF preferred" case of SKILL "Input requirements".
# [CORPUS-INFORMED] The SKILL names TIFF; PNG and BMP are the other lossless
# containers an ingest stage can produce and are treated identically.
LOSSLESS_FORMAT_NAMES: tuple = ("TIFF", "TIF", "PNG", "BMP", "PPM")

# Minimum number of feature blocks required before the two-component mixture of
# step 5 is treated as well-determined. Value: 1024. [DERIVED]
# (PHASE_ESTIMATION_BLOCK_SIZE_LADDER entry 256 / FEATURE_BLOCK_SIZE 8)**2 =
# 1024 - the block count of the 256x256 working block size Jeon's robustness
# benchmarks use. Below this the EM fit still runs but is reported at reduced
# confidence, because a two-component Gaussian mixture fitted to a few dozen
# points is not a population estimate.
MINIMUM_MIXTURE_SAMPLE_COUNT: int = 1024

# Value: 0.5. [ENGINEERING] Largest fraction of pixels permitted to sit at an
# intensity extreme. The SKILL rules out "flat/uniform and saturated regions"
# because prediction error is near zero there "regardless of CFA presence", but
# quantifies no cutoff. Half the image is a deliberately permissive bound: it
# rejects only images that are majority-clipped, where the surviving minority
# cannot support a whole-image statistic.
MAX_ACCEPTABLE_SATURATION_FRACTION: float = 0.5

# Value: 1.0 (intensity units squared). [ENGINEERING] Below this the analysis
# channel is constant to within less than one grey level, so it carries no
# prediction error and therefore no CFA signal at all. This detects a degenerate
# input, not a marginal one.
MINIMUM_CHANNEL_VARIANCE: float = 1.0

# Value: 1.0 (intensity units squared). [ENGINEERING] Colour-difference analogue
# of MINIMUM_CHANNEL_VARIANCE. When the variance of both R-G and B-G falls below
# this, the three planes are the same plane to within less than one grey level,
# i.e. the image is monochrome content in a colour container. SKILL "Input
# requirements" demands a "color image with a genuine Bayer-CFA acquisition
# history", and a colour filter array leaves its trace ACROSS the planes, so
# identical planes carry nothing for this engine to measure. This also catches
# the Foveon X3 and other non-Bayer cases the SKILL rules inapplicable, whenever
# they present without inter-channel structure.
MINIMUM_INTER_CHANNEL_VARIANCE: float = 1.0

# Value: 4.0 (intensity units squared, i.e. a standard deviation of 2 grey
# levels). [ENGINEERING - UNSOURCED] Per-block variance below which a block is
# called "almost flat" and excluded from the statistic, implementing Ferrara's
# stated limitation: "the proposed method is less effective in the presence of
# either almost flat areas or sharp edges". The SKILL gives no numeric cutoff.
# Why this value: two grey levels of spread is the point below which 8-bit
# quantisation, not scene content, dominates the block.
FLAT_BLOCK_VARIANCE_THRESHOLD: float = 4.0

# Value: 4.0. [STRUCTURAL] Sum of the positive weights of one axis of the Sobel
# operator ([[-1,0,1],[-2,0,2],[-1,0,1]] gives 1+2+1). scipy.ndimage.sobel
# returns the UNNORMALISED response, which is therefore four times the gradient
# per pixel. Dividing by this makes SHARP_EDGE_GRADIENT_THRESHOLD below mean
# what its comment says it means: intensity units per pixel.
SOBEL_NORMALISATION_FACTOR: float = 4.0

# Value: 40.0 (intensity units per pixel). [ENGINEERING - UNSOURCED] Gradient
# magnitude above which a pixel is called a sharp edge, implementing the second
# half of the same Ferrara limitation. The SKILL gives no numeric cutoff.
SHARP_EDGE_GRADIENT_THRESHOLD: float = 40.0

# Value: 0.25. [ENGINEERING - UNSOURCED] Fraction of a block's pixels that must
# exceed SHARP_EDGE_GRADIENT_THRESHOLD before the block is called
# edge-dominated and excluded. A quarter of the block is the point at which the
# edge, rather than the surrounding texture, sets the block's prediction error.
SHARP_EDGE_PIXEL_FRACTION: float = 0.25

# Value: 0.5. [ENGINEERING] Largest fraction of blocks permitted to be excluded
# as flat or edge-dominated before the whole measurement is called unreliable.
# Past half the image, the surviving blocks are no longer a representative
# sample of the scene.
MAX_ACCEPTABLE_EXCLUDED_BLOCK_FRACTION: float = 0.5


# ---------------------------------------------------------------------------
# Confidence weights
# ---------------------------------------------------------------------------

# Neutral and disqualifying weights. [STRUCTURAL] endpoints of the [0,1] range.
FULL_CONFIDENCE: float = 1.0
ZERO_CONFIDENCE: float = 0.0

# All penalties below are [ENGINEERING]. The SKILL quantifies how much accuracy
# each condition costs (e.g. Bammey 100% -> 67% between QF100 and QF90) but
# never states a confidence weight, because confidence weighting is a property
# of this system's fusion layer rather than of any paper. Each value is set to
# roughly track the accuracy loss the SKILL does report, and every one of them
# is listed in KNOWN_UNSOURCED_PARAMETERS.

# QF between UNUSABLE and DEGRADED: Ferrara's AUC is collapsing toward chance.
CONFIDENCE_PENALTY_SEVERELY_COMPRESSED: float = 0.25

# QF between DEGRADED and RELIABLE: Bammey still detects large regions (67%).
CONFIDENCE_PENALTY_DEGRADED_QUALITY: float = 0.67

# No quality factor supplied by the orchestrator at all.
CONFIDENCE_PENALTY_UNKNOWN_QUALITY: float = 0.5

# Image reported as resized. SKILL: "Aggressive resizing/rescaling/
# re-demosaicing after tampering shifts or destroys the periodic CFA phase
# entirely (general limitation, not separately quantified in the corpus)."
CONFIDENCE_PENALTY_RESAMPLED_INPUT: float = 0.3

# Too many blocks excluded as flat or edge-dominated.
CONFIDENCE_PENALTY_LOW_TEXTURE: float = 0.4

# Fewer feature blocks than MINIMUM_MIXTURE_SAMPLE_COUNT.
CONFIDENCE_PENALTY_SMALL_SAMPLE: float = 0.5

# Pipeline C could not run (image smaller than the smallest tested M), so the
# CFA phase was inferred from the sign of the feature rather than verified.
CONFIDENCE_PENALTY_PHASE_UNVERIFIED: float = 0.7

# Pipeline B found no statistically meaningful dominant grid position, so the
# confirmatory layer neither supports nor contradicts Pipeline A.
CONFIDENCE_PENALTY_GRID_LAYER_INCONCLUSIVE: float = 0.8

# Probability produced by the provisional sigmoid rather than measured
# calibration data.
CONFIDENCE_PENALTY_UNCALIBRATED: float = 0.5


# ---------------------------------------------------------------------------
# Calibration  -  SKILL "Output" -> Pipeline A reduction note
# ---------------------------------------------------------------------------

# Value: 30. [ENGINEERING] Fewest known-authentic reference scores that make an
# empirical CDF preferable to the provisional sigmoid. Below roughly this many
# samples a percentile rank is quantised too coarsely to be a probability.
MINIMUM_CALIBRATION_REFERENCE_COUNT: int = 30

# WARNING - the two values below are NOT corpus values.
# The SKILL publishes no mapping from Pipeline A's reduced scalar to a
# probability; the reduction rule itself is flagged as "not specified in the
# corpus - engineering recommendation". These are placeholders chosen so the
# sigmoid is centred and scaled sensibly for a reduced posterior that already
# lives in [0,1]. Any result produced through this route is reported as
# uncalibrated and carries CONFIDENCE_PENALTY_UNCALIBRATED. Supply
# CalibrationSettings.authentic_reference_scores to replace this route entirely.
# The midpoint is deliberately NOT tuned to the benchmark: 0.5 is the decision
# point of a posterior computed with the equal priors of Eq. 15-16, so it is the
# one value that follows from the model rather than from the measurements. The
# slope is chosen only so the curve spans the observed range without saturating
# (authentic worst 0.50, forged up to 0.9999).
PROVISIONAL_SIGMOID_MIDPOINT: float = 0.5
PROVISIONAL_SIGMOID_SLOPE: float = 12.0

# Bound on the sigmoid exponent so an extreme score cannot overflow exp().
# [STRUCTURAL]
SIGMOID_EXPONENT_LIMIT: float = 700.0

# Calibration buckets, conditioned on estimated JPEG quality. [DERIVED] The
# edges reuse UNUSABLE_JPEG_QUALITY_FACTOR and RELIABLE_JPEG_QUALITY_FACTOR so
# no new cut points are introduced: the score distribution shifts sharply with
# compression, so one global calibration would be invalid.
QUALITY_FACTOR_BUCKET_EDGES: tuple = (UNUSABLE_JPEG_QUALITY_FACTOR,
                                      RELIABLE_JPEG_QUALITY_FACTOR)
QUALITY_FACTOR_BUCKET_NAMES: tuple = ("qf_unusable", "qf_degraded", "qf_reliable")


# ---------------------------------------------------------------------------
# Presentation  -  affects the evidence image only, never a score
# ---------------------------------------------------------------------------

# Longest edge of the rendered evidence map, in pixels. [PRESENTATION]
EVIDENCE_MAP_MAX_DIMENSION: int = 512

# Endpoints of the diverging colour ramp used for the heatmap, as BGR triples.
# [PRESENTATION] Cool blue = confidently authentic, hot red = confidently
# tampered, pale grey in the middle = undecided.
EVIDENCE_COLOUR_AUTHENTIC: tuple = (140, 60, 20)
EVIDENCE_COLOUR_NEUTRAL: tuple = (225, 225, 225)
EVIDENCE_COLOUR_TAMPERED: tuple = (40, 40, 210)

# Colour drawn over blocks excluded from the statistic as flat or
# edge-dominated, so a reader can see where the engine declined to measure.
# [PRESENTATION]
EVIDENCE_COLOUR_EXCLUDED: tuple = (110, 110, 110)


# ---------------------------------------------------------------------------
# Audit aid
# ---------------------------------------------------------------------------

# Every value in this file that is NOT traceable to the SKILL document. A
# reviewer checking this engine against the corpus only has to argue with these.
# Everything else is quoted, or derived from something quoted with the
# derivation shown inline above.
KNOWN_UNSOURCED_PARAMETERS: tuple = (
    "LOCAL_VARIANCE_WINDOW_HALF_WIDTH",
    "LOCAL_VARIANCE_FLOOR",
    "EM_MINIMUM_COMPONENT_VARIANCE",
    "LOG_LIKELIHOOD_RATIO_LIMIT",
    "MAP_REDUCTION_RULE",
    "TAMPERED_BLOCK_PROBABILITY_THRESHOLD",
    "MINIMUM_FLAGGED_REGION_BLOCK_COUNT",
    "GRID_FILTER_SUPPORT_SIZE",
    "GRID_FILTER_RIDGE_FRACTION",
    "MAX_ACCEPTABLE_SATURATION_FRACTION",
    "MINIMUM_CHANNEL_VARIANCE",
    "FLAT_BLOCK_VARIANCE_THRESHOLD",
    "SHARP_EDGE_GRADIENT_THRESHOLD",
    "SHARP_EDGE_PIXEL_FRACTION",
    "MAX_ACCEPTABLE_EXCLUDED_BLOCK_FRACTION",
    "CONFIDENCE_PENALTY_SEVERELY_COMPRESSED",
    "CONFIDENCE_PENALTY_DEGRADED_QUALITY",
    "CONFIDENCE_PENALTY_UNKNOWN_QUALITY",
    "CONFIDENCE_PENALTY_RESAMPLED_INPUT",
    "CONFIDENCE_PENALTY_LOW_TEXTURE",
    "CONFIDENCE_PENALTY_SMALL_SAMPLE",
    "CONFIDENCE_PENALTY_PHASE_UNVERIFIED",
    "CONFIDENCE_PENALTY_GRID_LAYER_INCONCLUSIVE",
    "CONFIDENCE_PENALTY_UNCALIBRATED",
    "MINIMUM_CALIBRATION_REFERENCE_COUNT",
    "PROVISIONAL_SIGMOID_MIDPOINT",
    "PROVISIONAL_SIGMOID_SLOPE",
)

# Points where the SKILL document contradicts itself or leaves a formula
# ambiguous. Each is resolved in code with the resolution documented at the
# site; they are collected here so a reviewer can find them all at once.
KNOWN_SKILL_AMBIGUITIES: tuple = (
    "Pipeline B block size: step 2 says blocks of 32x32 or 64x64, the benchmark "
    "table says 32x32 windows of 4x4 blocks. Resolved by reading step 2's "
    "numbers as window sizes; see GRID_VOTE_BLOCK_SIZE.",
    "Pipeline C step 6: the comparison S_D[b] + S_F[d(b)] > S_D[d(b)] + S_F[d(b)] "
    "carries the identical S_F[d(b)] term on both sides, so it cancels and the "
    "test reduces to S_D[b] > S_D[d(b)]. That collapsed test was measured to "
    "select the WRONG member of the diagonal pair in 24 of 24 cases; S_D at the "
    "true red position is a minimum (215.4) not a maximum (339.9 elsewhere). "
    "Implemented with the comparison REVERSED, which scores 24/24. This is the "
    "one place the engine knowingly departs from the printed formula, and it "
    "affects only the reported configuration NAME, never a score - see the "
    "SKILL VERIFICATION block on _resolve_within_pair.",
    "Pipeline A step 3: the window half-width K is never given a value, only "
    "the window's form (2K+1) and the Gaussian standard deviation K/2.",
    "Pipeline B step 1: the filter support width N appears in the normal-equation "
    "index A[u+Nv, s+Nt] but is never assigned a value.",
)
