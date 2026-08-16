"""Every tunable parameter, threshold and magic number for the geometry engine.

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
                    not a free choice (e.g. an angle folded into [0, pi/2]).
    [PRESENTATION]- affects only the rendered evidence image, never the score.

NAMING WARNING. Two different papers in this SKILL both use "alpha" and "beta"
for unrelated quantities:
    Yao et al. Eq. 7-8   alpha = EXPECTED height ratio, beta = MEASURED ratio.
    R-VPD     A4 step 5  alpha = 1.2 inlier weight gain, beta = 0.8 outlier decay.
Every name in this file spells out which is meant. Nothing here is called plain
"alpha" or "beta".
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

ENGINE_NAME: str = "geometric_consistency"

# Version of the SKILL document this implementation was written against. Bump
# whenever SKILL(Geometric Consistency Forgery Detection).md changes, so a
# stored forensic report can be traced back to the exact spec that produced it.
SKILL_VERSION: str = "SKILL(Geometric Consistency Forgery Detection).md@2026-08-15"


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

# Array dimension counts for colour and single-plane images. [STRUCTURAL] The
# SKILL admits "any RGB/grayscale image", so both are legal input.
COLOUR_IMAGE_DIMENSION_COUNT: int = 3
GRAYSCALE_IMAGE_DIMENSION_COUNT: int = 2

# 8-bit intensity extremes. [STRUCTURAL]
MINIMUM_INTENSITY_LEVEL: int = 0
MAXIMUM_INTENSITY_LEVEL: int = 255

# Milliseconds per second, for reporting processing_time_ms. [STRUCTURAL]
MILLISECONDS_PER_SECOND: float = 1000.0

# Homogeneous 2-D geometry: points and lines are 3-vectors, a plane point has
# 2 Cartesian coordinates. [STRUCTURAL] The SKILL's Implementation Notes require
# homogeneous coordinates throughout "so that a vanishing point genuinely at
# infinity ... does not require special-casing".
HOMOGENEOUS_VECTOR_LENGTH: int = 3
CARTESIAN_VECTOR_LENGTH: int = 2

# Index of the homogeneous scale component of a 3-vector. [STRUCTURAL]
HOMOGENEOUS_SCALE_INDEX: int = 2

# Smallest homogeneous scale component treated as finite. [ENGINEERING] Below
# this the point is taken to lie at infinity, which for a vanishing point means
# perfectly parallel image lines. Chosen far below any value a real intersection
# of two image-space lines produces.
HOMOGENEOUS_INFINITY_TOLERANCE: float = 1e-9

# Decimal places used when rounding values into the computation trace.
# [PRESENTATION] These affect only how numbers are displayed to the report
# generator and can never change a score.
TRACE_COARSE_DECIMAL_PLACES: int = 2
TRACE_DECIMAL_PLACES: int = 4
TRACE_FINE_DECIMAL_PLACES: int = 5
TRACE_SCORE_DECIMAL_PLACES: int = 6

# Ordinal positions of each entry in the computation trace. [STRUCTURAL] -
# sequence labels for the report generator, not forensic parameters.
COMPUTATION_STEP_CONDITION_CHECK: int = 1
COMPUTATION_STEP_PREPROCESSING: int = 2
COMPUTATION_STEP_VANISHING_POINT: int = 3
COMPUTATION_STEP_VANISHING_LINE: int = 4
COMPUTATION_STEP_HEIGHT_RATIO: int = 5
COMPUTATION_STEP_CALIBRATION: int = 6
COMPUTATION_STEP_FAILURE: int = 1


# ---------------------------------------------------------------------------
# Module A1 - parallel-line vanishing point (Yao et al. 2012)
# SKILL "Step-by-step algorithm" section A1, steps 1-4
# ---------------------------------------------------------------------------

# Value: 5.0 degrees. [CORPUS] SKILL A1 step 2: "discard line pairs with angle
# < 5 deg (near-parallel intersections are numerically ill-conditioned)".
# The SKILL's Implementation Notes escalate this: "Apply a minimum-angle filter
# before any line-intersection computation, not just within the RANSAC loop",
# which is why this same constant also gates module A4's RANSAC sampling.
MINIMUM_LINE_PAIR_ANGLE_DEGREES: float = 5.0

# Same threshold in radians. [DERIVED]
MINIMUM_LINE_PAIR_ANGLE_RADIANS: float = float(
    np.deg2rad(MINIMUM_LINE_PAIR_ANGLE_DEGREES))

# Upper bound of an acute angle between two undirected lines. [STRUCTURAL]
# R-VPD A4 step 5 defines theta_ij as the acute angle "folded into [0, pi/2]".
MAXIMUM_ACUTE_ANGLE_RADIANS: float = float(np.pi / 2.0)
HALF_TURN_RADIANS: float = float(np.pi)

# Fewest line segments from which a vanishing point may be estimated at all.
# Value: 2. [DERIVED] Two lines are the algebraic minimum for an intersection,
# and A1 step 2 operates on "line pairs", so a pair is the unit of evidence.
MINIMUM_LINES_FOR_VANISHING_POINT: int = 2

# Fewest line segments before the A1 estimate is considered trustworthy rather
# than merely computable. Value: 8. [ENGINEERING - UNSOURCED] The SKILL states
# A1 "produces good results for clear-cut man-made scenes but often fails when
# the scene lacks parallel lines" but never quantifies "lacks". Eight segments
# is four independent pairs, the point below which a single spurious edge can
# dominate the least-squares scatter matrix.
MINIMUM_LINES_FOR_CONFIDENT_ESTIMATE: int = 8

# Canny edge-detector hysteresis thresholds feeding the Hough transform.
# [ENGINEERING - UNSOURCED] SKILL A1 step 1 says "Extract straight edges via
# Hough transform on detected edges" without naming an edge detector or its
# parameters. These are OpenCV's conventional 1:3 low:high ratio on 8-bit input.
CANNY_LOW_THRESHOLD: int = 50
CANNY_HIGH_THRESHOLD: int = 150
CANNY_APERTURE_SIZE: int = 3

# Probabilistic Hough transform parameters. [ENGINEERING - UNSOURCED] SKILL A1
# step 1 names the Hough transform but gives no parameter values. Distance
# resolution of 1 pixel and angular resolution of 1 degree are the standard
# full-resolution settings; the vote threshold and length limits are set so a
# segment must span a meaningful fraction of the image to be admitted.
HOUGH_DISTANCE_RESOLUTION_PIXELS: float = 1.0
HOUGH_ANGLE_RESOLUTION_RADIANS: float = float(np.deg2rad(1.0))
HOUGH_VOTE_THRESHOLD: int = 60
HOUGH_MINIMUM_LINE_LENGTH_PIXELS: int = 40
HOUGH_MAXIMUM_LINE_GAP_PIXELS: int = 8

# Number of values describing one Hough segment: x1, y1, x2, y2. [STRUCTURAL]
# Used to normalise the detector's output shape, which differs between OpenCV
# major versions - 4.x returns (N, 1, 4) and 5.x returns (N, 4).
HOUGH_ENDPOINT_COUNT: int = 4

# Cap on the number of line segments carried into the estimator. Value: 400.
# [ENGINEERING] The 5-degree pairwise filter of A1 step 2 is O(n^2), so an
# unbounded segment list makes runtime quadratic in scene clutter. Segments are
# kept longest-first, so the cap discards the least reliable evidence.
MAXIMUM_LINE_SEGMENT_COUNT: int = 400

# Levenberg-Marquardt refinement limits for A1 step 3. [ENGINEERING] The SKILL
# names the algorithm and its objective but gives no iteration or tolerance
# budget. These are scipy's own conventional stopping values, tightened only so
# the refinement terminates on a fixed budget for a real-time engine.
LEVENBERG_MARQUARDT_MAX_EVALUATIONS: int = 2000
LEVENBERG_MARQUARDT_TOLERANCE: float = 1e-8


# ---------------------------------------------------------------------------
# Module A4 - recurrence-based vanishing point (R-VPD, Bharadwaj et al. 2025)
# SKILL "Step-by-step algorithm" section A4, steps 1-7
# ---------------------------------------------------------------------------

# SIFT descriptor dimensionality. Value: 128. [CORPUS] SKILL A4 step 1: "SIFT
# feature extraction over the whole image (128-dimensional descriptor per
# keypoint, standard DoG-pyramid SIFT)".
SIFT_DESCRIPTOR_LENGTH: int = 128

# Cap on SIFT keypoints retained. Value: 1200. [ENGINEERING] SKILL A4 step 7
# states the method is "O(n^2) in the number of detected SIFT features
# (dominated by the pairwise distance-matrix computation in clustering)" with a
# median runtime of 10.73 s on real images. Capping the feature count is the
# only lever that bounds that quadratic term; 1200 features gives a 1200x1200
# distance matrix, which is the largest that stays inside a second.
MAXIMUM_SIFT_KEYPOINT_COUNT: int = 1200

# Linkage method and metric for the visual-word clustering of A4 step 2.
# [CORPUS] SKILL A4 Eq. 1: "Dist_AB = min{dist_ij}" with "dist_ij =
# ||fd_i - fd_j||_2", and Eq. 2: "Dist_CP = min{Dist_AP, Dist_BP}". Taking the
# minimum pairwise distance between groups, and updating a merged group's
# distance as the minimum of its parents', IS the definition of single-linkage
# agglomerative clustering under the Euclidean metric. The two equations are
# therefore implemented exactly by this pair of strings, not approximated.
HIERARCHICAL_LINKAGE_METHOD: str = "single"
HIERARCHICAL_LINKAGE_METRIC: str = "euclidean"

# Distance at which the single-linkage dendrogram is cut into visual words.
# Value: 200.0 in SIFT descriptor units. [ENGINEERING - UNSOURCED] SKILL A4
# step 2 describes merging "iteratively" but never states the stopping rule or
# cut height. SIFT descriptors are 128-dimensional vectors of magnitude up to
# 512 per element after the conventional scaling, so this cut is well inside the
# descriptor space; it is exposed as a constructor argument precisely because
# the corpus does not fix it.
VISUAL_WORD_CUT_DISTANCE: float = 200.0

# Fewest features a visual word must hold to be a usable recurring pattern.
# Value: 3. [DERIVED] The angle score S_A and scale score S_S of A4 step 3 are
# both defined over "ordered triplets (A,B,C)", so three features is the
# arithmetic minimum at which either score exists.
MINIMUM_FEATURES_PER_VISUAL_WORD: int = 3

# Largest visual word admitted to forward feature selection. Value: 40.
# [ENGINEERING] Forward selection over a visual word is quadratic in its size;
# clusters larger than this are repetitive texture rather than a countable
# recurring structure, and dominate runtime without adding line evidence.
MAXIMUM_FEATURES_PER_VISUAL_WORD: int = 40

# Number of features forward selection grows a subset to. Value: 6.
# [ENGINEERING - UNSOURCED] SKILL A4 step 3 describes "forward feature
# selection within each visual word, scoring candidate feature subsets" but
# never states the target subset size. Six points is the smallest subset that
# supports four ordered triplets, enough for S_A and S_S to average over more
# than a single measurement.
FORWARD_SELECTION_TARGET_SUBSET_SIZE: int = 6

# Weighted-RANSAC weight updates for A4 step 5. Values: 1.2 and 0.8. [CORPUS]
# SKILL A4 step 5, verbatim: "inliers l_i in I: w'_i = w_i * alpha; outliers
# l_j in O: w'_j = w_j * beta, with alpha=1.2, beta=0.8 in the paper's
# implementation".
# NAMED EXPLICITLY, not "alpha"/"beta", because Yao et al.'s Eq. 7-8 use those
# same two letters for the expected and measured height ratios - see the module
# docstring warning.
RANSAC_INLIER_WEIGHT_GAIN: float = 1.2
RANSAC_OUTLIER_WEIGHT_DECAY: float = 0.8

# RANSAC iteration and restart budget. [ENGINEERING - UNSOURCED] SKILL A4 step 5
# says the procedure is "Repeated with re-initialized weights across multiple
# runs to mitigate the ill-conditioning of near-parallel intersections" but
# gives neither an iteration count nor a number of runs. These are set so the
# total sampling budget stays inside the corpus's reported per-image runtime.
RANSAC_ITERATIONS_PER_RUN: int = 200
RANSAC_RESTART_COUNT: int = 5

# Distance in pixels within which a line is counted an inlier to a candidate
# vanishing point. Value: 12.0. [ENGINEERING - UNSOURCED] SKILL A4 step 5 says
# to "classify all other lines as inliers/outliers by distance to that point"
# without stating the distance or its threshold. Measured as the perpendicular
# distance from the candidate point to the infinite line.
RANSAC_INLIER_DISTANCE_PIXELS: float = 12.0

# Seed for the RANSAC sampler. [ENGINEERING] Fixed so a forensic result is
# reproducible: the same image must yield the same vanishing point and therefore
# the same score every time the report is regenerated.
RANSAC_RANDOM_SEED: int = 0

# Fewest inliers a RANSAC consensus must reach to be reported. Value: 3.
# [DERIVED] Two lines define the candidate point by construction, so a third
# agreeing line is the first genuine corroboration.
MINIMUM_RANSAC_INLIER_COUNT: int = 3


# ---------------------------------------------------------------------------
# Vanishing-point confidence gating
# SKILL "Output" - vanishing-point estimation
# ---------------------------------------------------------------------------

# Smallest RANSAC inlier FRACTION accepted from module A4. Value: 0.5.
# [ENGINEERING - UNSOURCED] The SKILL is emphatic that confidence must gate the
# engine - "This confidence must gate whether the module proceeds at all ... a
# low-confidence VP estimate should cause the module to abstain rather than emit
# a possibly-spurious height-ratio score" - and names "RANSAC inlier
# count/fraction for A4, or line-fit residual for A1" as the indicator, but
# publishes no cutoff for either. A simple majority is used: below half the
# lines agreeing, the scene has no single dominant vanishing point, which is
# also the Manhattan-world condition the SKILL rules inapplicable.
MINIMUM_VANISHING_POINT_INLIER_FRACTION: float = 0.5

# Largest A1 line-fit residual accepted, in pixels. Value: 25.0.
# [ENGINEERING - UNSOURCED] The other confidence indicator the SKILL names.
# Expressed as the root-mean-square orthogonal distance from measured line
# endpoints to the refined lines through the estimated vanishing point, so it is
# in the same units as the image and comparable across resolutions.
MAXIMUM_LINE_FIT_RESIDUAL_PIXELS: float = 25.0


# ---------------------------------------------------------------------------
# Module A2 - reference-object vanishing line (Yao et al. 2012, Eq. 5-6)
# ---------------------------------------------------------------------------

# Smallest denominator magnitude at which Eq. 6 is considered solvable.
# [ENGINEERING] Eq. 6's denominator (v'_R2 - v'_R1 + eta*v_R1 - eta*v_R2) goes to
# zero when the two reference objects subtend proportionally identical image
# heights, at which point the vanishing line is unconstrained by them. This
# floor detects that degeneracy instead of returning an arbitrarily large v0.
MINIMUM_EQUATION_SIX_DENOMINATOR: float = 1e-6


# ---------------------------------------------------------------------------
# Module B - height-ratio consistency test (Yao et al. 2012, Eq. 7-8)
# THE CORE FORGERY DECISION
# ---------------------------------------------------------------------------

# Value: 0.05. [CORPUS] SKILL B step 4, verbatim: "for an authentic image, the
# probability of kappa falling outside the interval [0.8*alpha, 1.2*alpha] is
# set to a constant 0.05".
AUTHENTIC_RATIO_OUTLIER_PROBABILITY: float = 0.05

# Value: 0.2, the half-width of [0.8*alpha, 1.2*alpha] as a fraction of alpha.
# [CORPUS] Read directly off the same interval in SKILL B step 4.
AUTHENTIC_RATIO_INTERVAL_HALF_WIDTH_FRACTION: float = 0.2

# Value: 0.1, i.e. sigma = 0.1 * alpha. [CORPUS] SKILL B step 4: the 0.05
# design choice above, "solved against the Gaussian CDF - fixes sigma = 0.1*alpha".
# VERIFIED, not merely copied: P(|kappa - alpha| > 0.2*alpha) = 0.05 requires
# 0.2*alpha/sigma = Phi^-1(0.975) = 1.95996, giving sigma = 0.10204*alpha. The
# paper's rounded 0.1 is therefore self-consistent to within 2%. The engine uses
# the published 0.1 rather than the exact 0.10204 so that its numbers reproduce
# the paper's Table I.
# The SKILL's Implementation Notes caution that this is "a fixed, non-adaptive
# constant ... not re-derived or validated against a large dataset ... treat
# sigma as a starting point for this engine's own calibration, not a universally
# correct value."
RATIO_SIGMA_FRACTION_OF_EXPECTED: float = 0.1

# Value: 0.5. [CORPUS] SKILL B step 4: "Decision threshold T = 0.5: C < T
# implies at least one object considered forged." Also quoted in SKILL "Output":
# "The paper's own threshold T=0.5 can seed (not replace) this engine's
# calibration."
CONSISTENCY_DECISION_THRESHOLD: float = 0.5

# Factor 2 in Eq. 8, C = 2 * F(-|alpha - beta|; 0, sigma^2). [STRUCTURAL]
# It is what normalises C to reach exactly 1.0 when beta equals alpha, since
# F(0) = 0.5 for a zero-mean distribution. Not a free parameter.
CONSISTENCY_NORMALISATION_FACTOR: float = 2.0

# Expected height ratio assumed for an automatically proposed object pair.
# Value: 1.0. [ENGINEERING - UNSOURCED, AND THE MOST CONSEQUENTIAL ASSUMPTION
# IN THIS ENGINE]
# SKILL B step 2 requires alpha "either from general prior knowledge about the
# object classes (e.g. typical relative human heights) or from a trusted
# reference pair with the same depth". Neither is available to an engine whose
# input contract is an image and nothing else: there is no object classifier
# upstream and EngineInput carries no annotation field.
# 1.0 encodes "these two regions depict the same kind of thing, so they should
# be the same real-world height". That holds for the recurring structures the
# corpus's own A4 module is built to find - fence posts, railings, matched
# facade elements - and fails for any pair of genuinely different objects.
# CONSEQUENCE: every automatically proposed pair carries
# CONFIDENCE_PENALTY_ASSUMED_RATIO_PRIOR, and the reliability note says the
# prior was assumed. Supply real pairs through the constructor to remove it.
DEFAULT_EXPECTED_HEIGHT_RATIO: float = 1.0

# Smallest expected ratio accepted from a caller. [STRUCTURAL] sigma is defined
# as a fraction OF alpha, so a non-positive alpha makes Eq. 8's distribution
# degenerate and its consistency score meaningless.
MINIMUM_EXPECTED_HEIGHT_RATIO: float = 1e-6

# Perturbation used to measure how far a pair's measured ratio moves per pixel
# of vanishing-line error. Value: 1.0 pixel. [STRUCTURAL] A one-pixel probe
# makes the reported sensitivity read directly as "ratio units per pixel". This
# is a DIAGNOSTIC derived from Eq. 7 by perturbation; it introduces no model
# beyond Eq. 7 and never influences a score.
SENSITIVITY_PROBE_PIXELS: float = 1.0

# Value: 0.6745. [DERIVED] The standard-normal quantile at which Eq. 8's
# consistency C falls to the paper's threshold T = 0.5. Solving
# 2*Phi(-d/sigma) = 0.5 gives d/sigma = -Phi^-1(0.25) = 0.67449. Combined with
# sigma = 0.1*alpha this means the paper flags a pair once its measured height
# ratio departs from expectation by about 6.7% - verified numerically: Eq. 8
# returns C = 0.5003 at |alpha - beta| = 0.0674*alpha. Used only to report how
# much vanishing-line error a pair could absorb before flipping verdict.
CONSISTENCY_THRESHOLD_SIGMA_MULTIPLE: float = 0.6745

# Fewest object pairs needed for module B to return a score. Value: 1.
# [DERIVED] Eq. 7 is defined on a pair, so one valid pair is the minimum.
MINIMUM_OBJECT_PAIR_COUNT: int = 1

# Fewest pairs before the result is treated as corroborated rather than single
# shot. Value: 3. [ENGINEERING - UNSOURCED] SKILL B step 5 recommends that
# "several measurements of beta are taken ... and averaged to improve accuracy -
# the paper does not give a specific minimum count, just the general
# recommendation". Three is the smallest count that can outvote one bad pair.
MINIMUM_PAIRS_FOR_CORROBORATION: int = 3

# Smallest denominator magnitude at which Eq. 7 is considered solvable.
# [ENGINEERING] Eq. 7's denominator (v_B1 - v_B2)(v0 - v'_B2) vanishes when an
# object has zero image height, or when its base sits exactly on the vanishing
# line (infinitely far away). Both are degenerate rather than informative.
MINIMUM_EQUATION_SEVEN_DENOMINATOR: float = 1e-6


# ---------------------------------------------------------------------------
# Module D - SLIC superpixel region proposal (Sasmal & Dhal 2023 survey)
# SKILL "Step-by-step algorithm" section D
# ---------------------------------------------------------------------------

# SLIC parameters. [ENGINEERING - UNSOURCED] SKILL D names SLIC as the
# "Recommended default ... for its speed, memory efficiency, and strong boundary
# adherence" but publishes no parameter values. 300 segments gives regions of
# roughly object scale on a typical photograph; compactness 10 is scikit-image's
# own documented balance between colour proximity and spatial proximity.
SLIC_SEGMENT_COUNT: int = 300
SLIC_COMPACTNESS: float = 10.0
SLIC_SIGMA: float = 1.0
SLIC_START_LABEL: int = 1

# Smallest side, as a fraction of the image's shorter side, for a proposed
# region to be treated as a candidate object. Value: 0.04. [ENGINEERING -
# UNSOURCED] Superpixels smaller than this are texture fragments rather than
# objects, and their top/bottom coordinates are dominated by segmentation noise
# rather than by object extent, which Eq. 7 differences directly.
MINIMUM_REGION_SIDE_FRACTION: float = 0.04

# Largest side, as a fraction of the image's shorter side, for a proposed
# region. Value: 0.6. [ENGINEERING - UNSOURCED] A region spanning most of the
# frame is background, not an object resting on the reference plane.
MAXIMUM_REGION_SIDE_FRACTION: float = 0.6

# Cap on proposed regions carried into the pairwise test. Value: 40.
# [ENGINEERING] Module B is O(n^2) in region count.
MAXIMUM_PROPOSED_REGION_COUNT: int = 40

# Number of bins per colour channel in the appearance signature used to group
# regions into same-class sets. Value: 4, giving 4^3 = 64 bins. [ENGINEERING -
# UNSOURCED] The SKILL offers no automated way to decide two regions depict the
# same class of object, which DEFAULT_EXPECTED_HEIGHT_RATIO above assumes. A
# coarse colour histogram is the weakest possible surrogate and is deliberately
# coarse: it is a filter against pairing obviously unlike regions, not a
# classifier, and it is one reason the automatic path is confidence-penalised.
APPEARANCE_HISTOGRAM_BINS_PER_CHANNEL: int = 4

# Largest chi-square distance between two regions' colour histograms for them to
# be paired. Value: 0.5. [ENGINEERING - UNSOURCED] See the note above.
MAXIMUM_APPEARANCE_DISTANCE: float = 0.5

# Floor added to a histogram bin before the chi-square distance divides by it.
# [STRUCTURAL] Prevents a division by zero on empty bins.
HISTOGRAM_DISTANCE_EPSILON: float = 1e-10

# Fewest pixels a region must have before its colour signature is computed.
# Value: 1. [STRUCTURAL] A histogram of an empty region is undefined.
MINIMUM_REGION_PIXEL_COUNT: int = 1


# ---------------------------------------------------------------------------
# Ground-plane sanity check for candidate object pairs
# SKILL "Implementation notes" - degenerate object pairs
# ---------------------------------------------------------------------------

# Fewest pixels a region's base must sit below the vanishing line. Value: 2.0.
# [ENGINEERING] SKILL "Implementation notes" recommends exactly this check:
# "cross-check candidate pairs' bounding-box bottoms against a coarse
# ground-plane/horizon-line estimate before accepting them as valid pairs". An
# object resting on the reference plane must have its base BELOW the plane's
# vanishing line in image coordinates; a base at or above it is either floating,
# on a raised surface, or infinitely distant. Two pixels of margin keeps Eq. 7's
# (v0 - v_B2) factor away from zero.
MINIMUM_BASE_BELOW_VANISHING_LINE_PIXELS: float = 2.0

# Fewest pixels of image height a region must have to give Eq. 7 a usable
# (v_B1 - v_B2) difference. Value: 8.0. [ENGINEERING - UNSOURCED] Below this,
# one pixel of segmentation error is more than 12% of the measured height, which
# already exceeds the 6.7% divergence at which Eq. 8 crosses T = 0.5.
MINIMUM_REGION_IMAGE_HEIGHT_PIXELS: float = 8.0


# ---------------------------------------------------------------------------
# Condition-checking thresholds
# SKILL "Input requirements"
# ---------------------------------------------------------------------------

# Smallest image side the engine will analyse. Value: 64. [ENGINEERING -
# UNSOURCED] The SKILL sets no resolution floor. Below this, the Hough vote
# threshold and minimum segment length cannot both be met, so the engine would
# always abstain anyway; failing early gives a clearer note.
MINIMUM_IMAGE_SIDE_PIXELS: int = 64

# JPEG quality below which the engine notes, but does not penalise, compression.
# Value: 50.0. [CORPUS-INFORMED] Unlike every other engine in this system, the
# SKILL records compression as a STRENGTH: Yao et al.'s method is "explicitly
# validated as robust to down-sampling and low-quality JPEG recompression, a
# regime where most trace-based methods ... fail - this module is complementary
# to, not redundant with, the DSP-artifact-based modules". No confidence penalty
# is applied for compression; this threshold only drives an explanatory note
# telling the report generator why this engine still voted when others abstained.
NOTABLE_JPEG_QUALITY_FACTOR: float = 50.0

# Sentinel meaning "no JPEG compression reported". [STRUCTURAL]
NO_COMPRESSION_QUALITY_FACTOR: float = 0.0


# ---------------------------------------------------------------------------
# Confidence weights
# ---------------------------------------------------------------------------

# Neutral and disqualifying weights. [STRUCTURAL] endpoints of the [0,1] range.
FULL_CONFIDENCE: float = 1.0
ZERO_CONFIDENCE: float = 0.0

# All penalties below are [ENGINEERING]. The SKILL describes each condition
# qualitatively but never states a confidence weight, because confidence
# weighting is a property of this system's fusion layer rather than of any paper.

# The expected height ratio was assumed rather than supplied. This is the
# heaviest penalty in the engine because DEFAULT_EXPECTED_HEIGHT_RATIO is the
# single assumption most likely to be wrong on an arbitrary photograph.
CONFIDENCE_PENALTY_ASSUMED_RATIO_PRIOR: float = 0.35

# The vanishing point came from module A4 (recurrence) rather than A1 (explicit
# lines). A1 is the SKILL's primary method when lines exist.
CONFIDENCE_PENALTY_RECURRENCE_FALLBACK: float = 0.8

# Fewer object pairs than MINIMUM_PAIRS_FOR_CORROBORATION contributed, so SKILL
# B step 5's averaging recommendation could not be honoured.
CONFIDENCE_PENALTY_UNCORROBORATED: float = 0.6

# Fewer line segments than MINIMUM_LINES_FOR_CONFIDENT_ESTIMATE were available.
CONFIDENCE_PENALTY_SPARSE_LINE_EVIDENCE: float = 0.6

# Image reported as resized. Downsampling is a documented STRENGTH of this
# method, so this is a mild note-level penalty rather than a disqualification.
CONFIDENCE_PENALTY_RESAMPLED_INPUT: float = 0.9

# Probability produced by the paper's own calibration rather than measured
# reference data. Lighter than the other engines' uncalibrated penalty because
# SKILL "Output" states C is "already a calibrated, paper-defined [0,1] score".
CONFIDENCE_PENALTY_PAPER_CALIBRATION: float = 0.8


# ---------------------------------------------------------------------------
# Calibration  -  SKILL "Output" - height-ratio consistency
# ---------------------------------------------------------------------------

# Value: 30. [ENGINEERING] Fewest known-authentic reference scores that make an
# empirical CDF preferable to the paper's own calibration.
MINIMUM_CALIBRATION_REFERENCE_COUNT: int = 30

# Corpus-reported score ranges, retained so the report generator can state where
# a measurement sits relative to the paper's own results. [CORPUS] SKILL
# benchmark table, Yao 2012 Table I: authentic images scored C = 0.880-0.979,
# forged images C = 0.020-0.208, with T = 0.5 "correctly separating every case
# shown". Expressed here as tampering scores, 1 - C.
YAO_AUTHENTIC_TAMPERING_SCORE_MINIMUM: float = 1.0 - 0.979
YAO_AUTHENTIC_TAMPERING_SCORE_MAXIMUM: float = 1.0 - 0.880
YAO_FORGED_TAMPERING_SCORE_MINIMUM: float = 1.0 - 0.208
YAO_FORGED_TAMPERING_SCORE_MAXIMUM: float = 1.0 - 0.020

# Bound on the sigmoid exponent so an extreme score cannot overflow exp().
# [STRUCTURAL]
SIGMOID_EXPONENT_LIMIT: float = 700.0

# Slope of the optional logistic recalibration, used only when a caller
# overrides the paper route. [ENGINEERING - UNSOURCED]
PROVISIONAL_SIGMOID_SLOPE: float = 12.0

# Calibration buckets. [DERIVED] The edge reuses NOTABLE_JPEG_QUALITY_FACTOR so
# no new cut point is introduced. Two buckets only, because this method's
# performance is documented as compression-robust, so a fine-grained
# quality-conditioned calibration would be modelling noise.
QUALITY_FACTOR_BUCKET_EDGES: tuple = (NOTABLE_JPEG_QUALITY_FACTOR,)
QUALITY_FACTOR_BUCKET_NAMES: tuple = ("qf_low", "qf_normal")


# ---------------------------------------------------------------------------
# Presentation  -  affects the evidence image only, never a score
# ---------------------------------------------------------------------------

# Longest edge of the rendered evidence map, in pixels. [PRESENTATION]
EVIDENCE_MAP_MAX_DIMENSION: int = 900

# BGR colours for the annotated evidence overlay. [PRESENTATION]
EVIDENCE_COLOUR_VANISHING_LINE: tuple = (0, 215, 255)
EVIDENCE_COLOUR_SUPPORT_LINE: tuple = (200, 160, 60)
EVIDENCE_COLOUR_CONSISTENT_REGION: tuple = (110, 200, 90)
EVIDENCE_COLOUR_INCONSISTENT_REGION: tuple = (60, 60, 230)
EVIDENCE_COLOUR_VANISHING_POINT: tuple = (255, 255, 255)

# Line weights and marker sizes for the overlay, in pixels. [PRESENTATION]
EVIDENCE_LINE_THICKNESS: int = 2
EVIDENCE_BOX_THICKNESS: int = 2
EVIDENCE_VANISHING_POINT_RADIUS: int = 7

# Weight of the original image beneath the overlay. [PRESENTATION]
EVIDENCE_BASE_IMAGE_WEIGHT: float = 0.65
EVIDENCE_OVERLAY_WEIGHT: float = 0.35


# ---------------------------------------------------------------------------
# Audit aid
# ---------------------------------------------------------------------------

# Every value in this file that is NOT traceable to the SKILL document. A
# reviewer checking this engine against the corpus only has to argue with these.
# Everything else is quoted, or derived from something quoted with the
# derivation shown inline above.
KNOWN_UNSOURCED_PARAMETERS: tuple = (
    "MINIMUM_LINES_FOR_CONFIDENT_ESTIMATE",
    "CANNY_LOW_THRESHOLD",
    "CANNY_HIGH_THRESHOLD",
    "CANNY_APERTURE_SIZE",
    "HOUGH_DISTANCE_RESOLUTION_PIXELS",
    "HOUGH_ANGLE_RESOLUTION_RADIANS",
    "HOUGH_VOTE_THRESHOLD",
    "HOUGH_MINIMUM_LINE_LENGTH_PIXELS",
    "HOUGH_MAXIMUM_LINE_GAP_PIXELS",
    "MAXIMUM_LINE_SEGMENT_COUNT",
    "LEVENBERG_MARQUARDT_MAX_EVALUATIONS",
    "LEVENBERG_MARQUARDT_TOLERANCE",
    "MAXIMUM_SIFT_KEYPOINT_COUNT",
    "VISUAL_WORD_CUT_DISTANCE",
    "MAXIMUM_FEATURES_PER_VISUAL_WORD",
    "FORWARD_SELECTION_TARGET_SUBSET_SIZE",
    "RANSAC_ITERATIONS_PER_RUN",
    "RANSAC_RESTART_COUNT",
    "RANSAC_INLIER_DISTANCE_PIXELS",
    "MINIMUM_VANISHING_POINT_INLIER_FRACTION",
    "MAXIMUM_LINE_FIT_RESIDUAL_PIXELS",
    "DEFAULT_EXPECTED_HEIGHT_RATIO",
    "MINIMUM_PAIRS_FOR_CORROBORATION",
    "MINIMUM_REGION_IMAGE_HEIGHT_PIXELS",
    "SLIC_SEGMENT_COUNT",
    "SLIC_COMPACTNESS",
    "SLIC_SIGMA",
    "MINIMUM_REGION_SIDE_FRACTION",
    "MAXIMUM_REGION_SIDE_FRACTION",
    "APPEARANCE_HISTOGRAM_BINS_PER_CHANNEL",
    "MAXIMUM_APPEARANCE_DISTANCE",
    "MINIMUM_IMAGE_SIDE_PIXELS",
    "CONFIDENCE_PENALTY_ASSUMED_RATIO_PRIOR",
    "CONFIDENCE_PENALTY_RECURRENCE_FALLBACK",
    "CONFIDENCE_PENALTY_UNCORROBORATED",
    "CONFIDENCE_PENALTY_SPARSE_LINE_EVIDENCE",
    "CONFIDENCE_PENALTY_RESAMPLED_INPUT",
    "CONFIDENCE_PENALTY_PAPER_CALIBRATION",
    "MINIMUM_CALIBRATION_REFERENCE_COUNT",
    "PROVISIONAL_SIGMOID_SLOPE",
)

# Parts of the SKILL document this engine deliberately does NOT implement, with
# the reason. Collected here so a reviewer can see the scope boundary at a
# glance rather than inferring it from absence.
KNOWN_UNIMPLEMENTED_MODULES: tuple = (
    "A3 texture-orientation vanishing point: the SKILL describes it only in "
    "prose ('a locally-adaptive soft-voting algorithm ... low-confidence pixels "
    "are discarded') and attributes the method to prior work [13] without "
    "reproducing a single formula. There is nothing to implement without "
    "inventing the algorithm.",
    "Module C contour/shape consistency: the SKILL itself brackets this module "
    "as '(This module is a general shape-recognition method repurposed for this "
    "engine - the repurposing itself is engineering synthesis, not validated "
    "for forgery detection in the source paper.)' and marks its classification "
    "stage '[ML - excluded]'. Fisher Vector encoding additionally requires a "
    "GMM codebook whose component count and training corpus the SKILL never "
    "specifies, so the training-free substitute cannot be built either.",
    "Eq. 3 and Eq. 4 (absolute object height from image coordinates) are "
    "REFERENCED by the SKILL but never printed in it. They are not needed: "
    "Eq. 7 is the ratio form, and module B is defined entirely on ratios.",
)

# Points where the SKILL document is internally inconsistent or leaves a formula
# under-specified. Each is resolved in code with the resolution documented at
# the site; they are collected here so a reviewer can find them all at once.
KNOWN_SKILL_AMBIGUITIES: tuple = (
    "Eq. 1 is printed as 'zeta*[u,v,1]^T = (1/zeta)*K*[R|t]*X', carrying zeta on "
    "both sides, and Eq. 2's extrinsic matrix is printed 3x3 while multiplying "
    "the 4-vector [x,y,z,1]. Both are transcription slips in the projection "
    "DERIVATION only; neither equation is evaluated numerically anywhere in this "
    "engine, which uses the ratio forms Eq. 5-7 exclusively.",
    "A4 step 6 states the final vanishing point comes from 'eigenvalue "
    "decomposition of the largest inlier set' but adds that 'the specific "
    "eigen-formulation is referenced to prior work, not re-derived in this "
    "paper's extracted text'. The canonical formulation is used: the vanishing "
    "point is the unit eigenvector of SUM l_i l_i^T with the smallest "
    "eigenvalue, which is the least-squares point satisfying l_i^T v = 0 for "
    "every inlier line.",
    "A4 step 3's scale score S_S is defined as the difference in 'relative "
    "size-change ratio' between segments A->B and B->C without stating how that "
    "ratio is formed. It is implemented as the ratio of successive SIFT "
    "keypoint scales, sigma_B/sigma_A against sigma_C/sigma_B, which is the "
    "only reading under which the quantity measures projective foreshortening "
    "as the SKILL says it does.",
    "A4 step 2 gives the merge rule for hierarchical clustering (Eq. 1-2) but "
    "never states the stopping height, so VISUAL_WORD_CUT_DISTANCE is unsourced.",
    "B step 2 requires an expected ratio alpha from 'general prior knowledge "
    "about the object classes' or 'a trusted reference pair'. Neither is "
    "obtainable from the fixed EngineInput contract, which is why "
    "DEFAULT_EXPECTED_HEIGHT_RATIO exists and why the automatic path is the "
    "most heavily confidence-penalised route in this engine.",
)
