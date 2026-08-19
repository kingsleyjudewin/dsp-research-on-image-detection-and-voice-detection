"""Core mathematical computation for the perspective / geometry engine.

Two classes:

    VanishingPointEstimator  Module A. Estimates the reference plane's
                             vanishing point and vanishing line by three of the
                             SKILL's four documented routes:
                               A1 parallel lines (Yao et al. 2012) - PRIMARY,
                               A4 recurrence     (R-VPD 2025)      - fallback,
                               A2 reference objects (Eq. 5-6)      - on demand.
                             A3 (texture-orientation soft voting) is NOT
                             implemented: the SKILL describes it in prose only
                             and attributes it to prior work [13] without
                             reproducing a formula. See
                             constants.KNOWN_UNIMPLEMENTED_MODULES.

    HeightRatioAnalyser      Module B. The core forgery decision: Eq. 7 recovers
                             the real-world height ratio of an object pair from
                             image coordinates plus the vanishing line, and
                             Eq. 8 turns its divergence from the expected ratio
                             into a consistency score C in [0, 1].

Every formula carries a SKILL VERIFICATION block naming the equation, its
variables, its source paper and the range its output should occupy. Those blocks
are the mechanism by which this file is checked against the SKILL document.

NAMING WARNING, repeated from constants.py: Yao et al.'s alpha/beta (expected
and measured height ratio) and R-VPD's alpha/beta (1.2 inlier gain, 0.8 outlier
decay) are unrelated quantities that share two letters. Nothing in this file is
called plain "alpha" or "beta".
"""

from __future__ import annotations

import itertools
import logging
from typing import Optional

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import least_squares
from scipy.stats import norm

from . import constants
from .contracts import (HeightRatioAnalysis, HeightRatioMeasurement,
                        LineSegment, ObjectRegion, PreparedScene,
                        VanishingPointEstimate)
from .utils import (acute_angle_between_directions, chi_square_distance,
                    fit_line_least_squares, homogeneous_to_cartesian,
                    intersect_lines, line_through_points,
                    mean_perpendicular_distance, point_to_line_distance,
                    signed_direction_angle, smallest_eigenvalue_of_two_by_two,
                    smallest_eigenvector, wrap_angle_difference)

logger = logging.getLogger(__name__)


class VanishingPointEstimator:
    """Estimates the reference plane's vanishing point and horizon row."""

    def __init__(self, visual_word_cut_distance: Optional[float] = None) -> None:
        """Configure the recurrence module's clustering cut height.

        Args:
            visual_word_cut_distance: Height at which the single-linkage
                dendrogram of A4 step 2 is cut into visual words. Exposed
                because the SKILL never states a stopping rule for the merge.
        """
        self.visual_word_cut_distance = float(
            visual_word_cut_distance
            if visual_word_cut_distance is not None
            else constants.VISUAL_WORD_CUT_DISTANCE)

    def estimate(self, scene: PreparedScene) -> VanishingPointEstimate:
        """Estimate the vanishing point, preferring explicit lines.

        SKILL A1 is labelled "primary when explicit lines are available" and A4
        the "fallback when no explicit lines exist", so A1 is tried first and A4
        only runs when A1 cannot produce a confident estimate.

        Args:
            scene: Prepared scene carrying line segments and SIFT keypoints.

        Returns:
            VanishingPointEstimate, possibly with method "none" when neither
            route succeeded.
        """
        from_lines = self._estimate_from_lines(scene.line_segments)
        if from_lines is not None and self._is_confident(from_lines):
            return from_lines

        from_recurrence = self._estimate_from_recurrence(scene)
        if from_recurrence is not None:
            return from_recurrence
        if from_lines is not None:
            return from_lines
        return self._empty_estimate("No vanishing point could be estimated: "
                                    "the scene yielded neither usable parallel "
                                    "lines nor a recurring pattern.")

    @staticmethod
    def _is_confident(estimate: VanishingPointEstimate) -> bool:
        """Judge whether an estimate clears both confidence indicators.

        SKILL "Output" names "RANSAC inlier count/fraction for A4, or line-fit
        residual for A1" as the confidence indicators, and requires that
        confidence gate whether the module proceeds at all.

        Args:
            estimate: Candidate vanishing-point estimate.

        Returns:
            True when the estimate is trustworthy enough to build on.
        """
        return (estimate.homogeneous_point is not None
                and estimate.inlier_fraction
                >= constants.MINIMUM_VANISHING_POINT_INLIER_FRACTION
                and estimate.line_fit_residual_pixels
                <= constants.MAXIMUM_LINE_FIT_RESIDUAL_PIXELS)

    @staticmethod
    def _empty_estimate(note: str) -> VanishingPointEstimate:
        """Build the placeholder returned when no estimate was possible.

        Args:
            note: Explanation of why estimation failed.

        Returns:
            VanishingPointEstimate carrying no point.
        """
        return VanishingPointEstimate(
            homogeneous_point=None, vanishing_line_row=None, method="none",
            inlier_count=0, total_line_count=0, inlier_fraction=0.0,
            line_fit_residual_pixels=float("inf"), is_at_infinity=False,
            inlier_segments=[], note=note)

    # -- Module A1: parallel-line vanishing point ---------------------------

    def _estimate_from_lines(self,
                             segments: list) -> Optional[VanishingPointEstimate]:
        """Estimate the vanishing point from explicit straight edges.

        Args:
            segments: LineSegment objects from the Hough transform.

        Returns:
            VanishingPointEstimate, or None when too few usable lines exist.
        """
        usable = self._select_well_conditioned_lines(segments)
        if len(usable) < constants.MINIMUM_LINES_FOR_VANISHING_POINT:
            return None

        initial = self._line_scatter_vanishing_point(usable)
        cartesian = homogeneous_to_cartesian(
            initial, constants.HOMOGENEOUS_INFINITY_TOLERANCE)

        if cartesian is None:
            # A point at infinity means the image lines are exactly parallel;
            # there is nothing for the optimiser to move, so refinement is
            # skipped and the estimate is reported as being at infinity.
            return self._assemble_line_estimate(initial, usable,
                                                float("inf"), True)

        refined, residual = self._refine_levenberg_marquardt(usable, cartesian)
        homogeneous = np.array([refined[0], refined[1], 1.0], dtype=np.float64)
        return self._assemble_line_estimate(homogeneous, usable, residual, False)

    def _assemble_line_estimate(self,
                                homogeneous_point: np.ndarray,
                                segments: list,
                                residual: float,
                                is_at_infinity: bool) -> VanishingPointEstimate:
        """Package an A1 result together with its inlier support.

        Args:
            homogeneous_point: Estimated vanishing point.
            segments: Lines the estimate was built from.
            residual: Root-mean-square endpoint residual in pixels.
            is_at_infinity: Whether the point lies at infinity.

        Returns:
            Fully populated VanishingPointEstimate.
        """
        inlier_count, inlier_segments = self._count_inliers(homogeneous_point,
                                                            segments)
        return VanishingPointEstimate(
            homogeneous_point=homogeneous_point,
            vanishing_line_row=self._vanishing_line_row(homogeneous_point),
            method="A1_parallel_lines",
            inlier_count=inlier_count,
            total_line_count=len(segments),
            inlier_fraction=float(inlier_count) / float(len(segments)),
            line_fit_residual_pixels=residual,
            is_at_infinity=is_at_infinity,
            inlier_segments=inlier_segments,
            note=(f"Vanishing point estimated from {len(segments)} Hough line "
                  f"segments surviving the {constants.MINIMUM_LINE_PAIR_ANGLE_DEGREES:.0f}"
                  f"-degree minimum-pair-angle filter, refined by "
                  f"Levenberg-Marquardt to a {residual:.2f}-pixel endpoint "
                  f"residual, with {inlier_count} lines in consensus."))

    @staticmethod
    def _select_well_conditioned_lines(segments: list) -> list:
        """Keep only lines that form a large enough angle with some other line.

        Args:
            segments: Candidate LineSegment objects.

        Returns:
            Segments having at least one partner at or above the minimum angle.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: discard line pairs with angle < 5 degrees.
        # Variables: the angle is the acute angle between the two lines'
        #            direction vectors, folded into [0, pi/2].
        # Source: Yao, Wang, Zhao & Zhang 2012 (IEEE SPL) - SKILL A1 step 2,
        #            "discard line pairs with angle < 5 deg (near-parallel
        #            intersections are numerically ill-conditioned)". The
        #            SKILL's Implementation Notes broaden it: "Apply a minimum-
        #            angle filter before any line-intersection computation, not
        #            just within the RANSAC loop."
        # Expected range: a subset of the input. Lines with no partner above the
        #            threshold contribute only ill-conditioned intersections and
        #            are dropped entirely.
        # ────────────────────────────────────────────────────
        if len(segments) < constants.MINIMUM_LINES_FOR_VANISHING_POINT:
            return []

        keep: list = []
        for index, segment in enumerate(segments):
            has_partner = any(
                acute_angle_between_directions(segment.direction, other.direction)
                >= constants.MINIMUM_LINE_PAIR_ANGLE_RADIANS
                for position, other in enumerate(segments) if position != index)
            if has_partner:
                keep.append(segment)
        return keep

    @staticmethod
    def _line_scatter_vanishing_point(segments: list) -> np.ndarray:
        """Least-squares vanishing point of a set of homogeneous lines.

        Args:
            segments: LineSegment objects.

        Returns:
            Length-3 homogeneous vanishing point, unit norm.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: v = argmin_{|v|=1} SUM_i (l_i^T v)^2
        #            = the unit eigenvector of SUM_i l_i l_i^T belonging to the
        #              smallest eigenvalue.
        # Variables: l_i = homogeneous coefficients of the i-th detected line;
        #            v = the homogeneous vanishing point every line should pass
        #            through, so that l_i^T v = 0 exactly for a perfect fit.
        # Source: R-VPD (Bharadwaj, Collins & Liu 2025) - SKILL A4 step 6,
        #            "Final VP estimate: eigenvalue decomposition of the largest
        #            inlier set". The SKILL adds that "the specific eigen-
        #            formulation is referenced to prior work, not re-derived in
        #            this paper's extracted text", so the canonical homogeneous
        #            least-squares formulation above is used and recorded in
        #            constants.KNOWN_SKILL_AMBIGUITIES.
        # Expected range: a unit 3-vector. Its third component is near zero when
        #            the lines are near-parallel, i.e. the vanishing point is
        #            near infinity - which homogeneous coordinates represent
        #            without special-casing, as the SKILL's Implementation Notes
        #            require.
        # ────────────────────────────────────────────────────
        coefficients = np.array([segment.homogeneous for segment in segments],
                                dtype=np.float64)
        scatter_matrix = coefficients.T @ coefficients
        return smallest_eigenvector(scatter_matrix)

    def _refine_levenberg_marquardt(self,
                                    segments: list,
                                    initial_point: np.ndarray) -> tuple:
        """Refine a vanishing point by minimising endpoint-to-line distances.

        Args:
            segments: LineSegment objects supporting the estimate.
            initial_point: Cartesian (x, y) starting estimate.

        Returns:
            Tuple of (refined Cartesian point, root-mean-square residual).
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: "lines are modified to pass a single point such that sum of
        #           squared orthogonal distances from the endpoints of the
        #           measured lines to the modified lines is minimized."
        # Variables: for candidate vanishing point v and a segment with
        #            endpoints p1, p2, the modified line through v with unit
        #            normal n has d(p,L) = |n . (p - v)|, so that segment's
        #            objective is min_{|n|=1} n^T M n with
        #            M = (p1-v)(p1-v)^T + (p2-v)(p2-v)^T - the smaller
        #            eigenvalue of the 2x2 matrix M.
        # Source: Yao et al. 2012 - SKILL A1 step 3, "Refine the vanishing-point
        #            estimate via Levenberg-Marquardt maximum-likelihood
        #            estimation".
        # Expected range: residuals in pixels, non-negative. The optimiser
        #            minimises their sum of squares, exactly the stated sum of
        #            squared orthogonal distances.
        # ────────────────────────────────────────────────────
        solution = self._solve_least_squares(segments, initial_point)
        if solution is None:
            return np.asarray(initial_point, dtype=np.float64), float("inf")
        residuals = np.asarray(solution.fun, dtype=np.float64)
        return solution.x, float(np.sqrt(np.mean(residuals ** 2)))

    def _solve_least_squares(self, segments: list, initial_point: np.ndarray):
        """Run the Levenberg-Marquardt solve, returning None on failure.

        Args:
            segments: LineSegment objects supporting the estimate.
            initial_point: Cartesian (x, y) starting estimate.

        Returns:
            The scipy OptimizeResult, or None when the solve failed.
        """
        try:
            return least_squares(
                self._endpoint_residuals,
                np.asarray(initial_point, dtype=np.float64),
                method="lm", args=(segments,),
                max_nfev=constants.LEVENBERG_MARQUARDT_MAX_EVALUATIONS,
                xtol=constants.LEVENBERG_MARQUARDT_TOLERANCE,
                ftol=constants.LEVENBERG_MARQUARDT_TOLERANCE)
        except (ValueError, np.linalg.LinAlgError) as error:
            logger.warning("Levenberg-Marquardt refinement failed on %d "
                           "segments: %s", len(segments), error)
            return None

    def _endpoint_residuals(self,
                            candidate_point: np.ndarray,
                            segments: list) -> np.ndarray:
        """Per-segment orthogonal residual for a candidate vanishing point.

        Args:
            candidate_point: Cartesian (x, y) vanishing point under test.
            segments: LineSegment objects.

        Returns:
            Float array holding one residual per segment, in pixels.
        """
        return np.array([self._single_segment_residual(candidate_point, segment)
                         for segment in segments], dtype=np.float64)

    @staticmethod
    def _single_segment_residual(candidate_point: np.ndarray,
                                 segment: LineSegment) -> float:
        """Orthogonal distance of one segment's endpoints to its modified line.

        Args:
            candidate_point: Cartesian (x, y) vanishing point under test.
            segment: The measured segment.

        Returns:
            Root of the smaller eigenvalue of the endpoint scatter matrix,
            i.e. the residual whose square is the segment's contribution to the
            objective.
        """
        offsets = np.array([
            [segment.start[0] - candidate_point[0],
             segment.start[1] - candidate_point[1]],
            [segment.end[0] - candidate_point[0],
             segment.end[1] - candidate_point[1]]], dtype=np.float64)
        scatter_matrix = offsets.T @ offsets
        return float(np.sqrt(smallest_eigenvalue_of_two_by_two(scatter_matrix)))

    @staticmethod
    def _count_inliers(homogeneous_point: np.ndarray, segments: list) -> tuple:
        """Count lines passing within the inlier distance of a point.

        Args:
            homogeneous_point: Candidate vanishing point.
            segments: LineSegment objects to classify.

        Returns:
            Tuple of (inlier count, list of inlier segments).
        """
        cartesian = homogeneous_to_cartesian(
            homogeneous_point, constants.HOMOGENEOUS_INFINITY_TOLERANCE)
        if cartesian is None:
            # Every line is trivially consistent with a point at infinity in
            # its own direction, so consensus is not measurable there.
            return 0, []

        inliers = [segment for segment in segments
                   if point_to_line_distance(cartesian, segment.homogeneous)
                   <= constants.RANSAC_INLIER_DISTANCE_PIXELS]
        return len(inliers), inliers

    @staticmethod
    def _vanishing_line_row(homogeneous_point: np.ndarray) -> Optional[float]:
        """Image row v0 of the reference plane's vanishing line.

        Args:
            homogeneous_point: Estimated vanishing point.

        Returns:
            The row v0, or None when the vanishing point lies at infinity and
            therefore fixes no row.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: zeta*[u,v,1]^T = [[f,0,c_u],[0,f,v0],[0,0,1]] *
        #                           [[1,0,0],[0,0,-H],[0,1,0]] * [x,y,z,1]^T
        # Variables: for a LEVELED, UNTILTED camera (r11=r23=r32=1, all other
        #            off-diagonal rotation terms 0, t_y=-H, t_x=t_z=0), the
        #            intrinsic matrix's vertical principal coordinate IS the
        #            vanishing-line ordinate v0.
        # Source: Yao et al. 2012 - SKILL "Core mathematical principle",
        #            Eq. 2: "the vanishing line of the reference plane (z=0)
        #            sits at image row v0".
        # Expected range: an image row. Because the camera is assumed level, the
        #            vanishing line is horizontal, so the row of ANY vanishing
        #            point of a direction lying in the reference plane is v0 -
        #            which is what makes a single vanishing point sufficient
        #            here. The SKILL's own tilt criterion, "the vanishing line
        #            is inside the image", is checked against this value by the
        #            condition module.
        # ────────────────────────────────────────────────────
        cartesian = homogeneous_to_cartesian(
            homogeneous_point, constants.HOMOGENEOUS_INFINITY_TOLERANCE)
        if cartesian is None:
            return None
        return float(cartesian[1])

    # -- Module A2: reference-object vanishing line -------------------------

    @staticmethod
    def vanishing_line_from_reference_objects(
            first_region: ObjectRegion,
            second_region: ObjectRegion,
            known_height_ratio: float) -> Optional[float]:
        """Solve for the vanishing-line row from two objects of known ratio.

        Args:
            first_region: Reference object R, supplying v_R1 and v_R2.
            second_region: Reference object R', supplying v'_R1 and v'_R2.
            known_height_ratio: Yao's eta, the known ratio z'_R / z_R.

        Returns:
            The vanishing-line row v0, or None when unconstrained.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: eta = z'_R/z_R
        #              ~ (v'_R1 - v'_R2)(v0 - v_R2)
        #              / [(v_R1 - v_R2)(v0 - v'_R2)]                    [Eq. 5]
        #          v0 ~ (v_R2 v'_R2 - v_R2 v'_R1 + eta v_R1 v'_R2
        #               - eta v_R2 v'_R2)
        #             / (v'_R2 - v'_R1 + eta v_R1 - eta v_R2)           [Eq. 6]
        # Variables: v_R1, v_R2 = image rows of R's top and bottom; v'_R1,
        #            v'_R2 = the same for R'; eta = their KNOWN real height
        #            ratio; v0 = the vanishing-line ordinate being solved for.
        # Source: Yao et al. 2012 - SKILL A2, Eq. 5 and Eq. 6.
        # Expected range: an image row, normally above both objects' bases.
        # VERIFIED ALGEBRAICALLY: rearranging Eq. 5 to
        #            v0[eta(v_R1-v_R2) - (v'_R1-v'_R2)]
        #              = eta(v_R1-v_R2)v'_R2 - (v'_R1-v'_R2)v_R2
        #            expands to exactly Eq. 6's printed numerator and
        #            denominator, term for term.
        # ────────────────────────────────────────────────────
        numerator, denominator = VanishingPointEstimator._equation_six_terms(
            first_region, second_region, known_height_ratio)
        if abs(denominator) < constants.MINIMUM_EQUATION_SIX_DENOMINATOR:
            logger.info("Eq. 6 denominator vanished; the reference pair leaves "
                        "the vanishing line unconstrained")
            return None
        return float(numerator / denominator)

    @staticmethod
    def _equation_six_terms(first_region: ObjectRegion,
                            second_region: ObjectRegion,
                            known_height_ratio: float) -> tuple:
        """Evaluate Eq. 6's numerator and denominator, term for term.

        Args:
            first_region: Reference object R.
            second_region: Reference object R'.
            known_height_ratio: Yao's eta.

        Returns:
            Tuple of (numerator, denominator) exactly as Eq. 6 prints them.
        """
        first_top, first_bottom = first_region.top_row, first_region.bottom_row
        second_top = second_region.top_row
        second_bottom = second_region.bottom_row

        numerator = (first_bottom * second_bottom
                     - first_bottom * second_top
                     + known_height_ratio * first_top * second_bottom
                     - known_height_ratio * first_bottom * second_bottom)
        denominator = (second_bottom - second_top
                       + known_height_ratio * first_top
                       - known_height_ratio * first_bottom)
        return numerator, denominator

    # -- Module A4: recurrence-based vanishing point ------------------------

    def _estimate_from_recurrence(
            self, scene: PreparedScene) -> Optional[VanishingPointEstimate]:
        """Estimate the vanishing point from recurring-pattern structure.

        Args:
            scene: Prepared scene carrying SIFT keypoints and descriptors.

        Returns:
            VanishingPointEstimate, or None when no consensus was reached.
        """
        labels = self._cluster_visual_words(scene.keypoint_descriptors)
        if labels is None:
            return None

        implicit_lines = self._build_implicit_lines(scene, labels)
        combined = self._select_well_conditioned_lines(
            implicit_lines + list(scene.line_segments))
        if len(combined) < constants.MINIMUM_LINES_FOR_VANISHING_POINT:
            return None

        point, inliers = self._weighted_ransac(combined)
        if point is None:
            return None
        return self._assemble_recurrence_estimate(point, inliers, combined,
                                                  len(implicit_lines))

    def _assemble_recurrence_estimate(self,
                                      homogeneous_point: np.ndarray,
                                      inliers: list,
                                      combined: list,
                                      implicit_count: int) -> VanishingPointEstimate:
        """Package an A4 result together with its inlier support.

        Args:
            homogeneous_point: Estimated vanishing point.
            inliers: Lines in the winning consensus set.
            combined: All lines considered.
            implicit_count: How many lines came from recurring patterns.

        Returns:
            Fully populated VanishingPointEstimate.
        """
        cartesian = homogeneous_to_cartesian(
            homogeneous_point, constants.HOMOGENEOUS_INFINITY_TOLERANCE)
        residual = (float(np.sqrt(np.mean(
            self._endpoint_residuals(cartesian, inliers) ** 2)))
            if cartesian is not None and inliers else float("inf"))

        return VanishingPointEstimate(
            homogeneous_point=homogeneous_point,
            vanishing_line_row=self._vanishing_line_row(homogeneous_point),
            method="A4_recurrence",
            inlier_count=len(inliers),
            total_line_count=len(combined),
            inlier_fraction=float(len(inliers)) / float(len(combined)),
            line_fit_residual_pixels=residual,
            is_at_infinity=cartesian is None,
            inlier_segments=inliers,
            note=(f"Vanishing point estimated by the recurrence route: "
                  f"{implicit_count} implicit lines fitted through "
                  f"recurring-pattern feature groups were merged with "
                  f"{len(combined) - implicit_count} explicit edge lines, and "
                  f"weighted RANSAC found {len(inliers)} of {len(combined)} in "
                  f"consensus."))

    def _cluster_visual_words(self,
                              descriptors: np.ndarray) -> Optional[np.ndarray]:
        """Group SIFT descriptors into visual words by single-linkage merging.

        Args:
            descriptors: Array of shape (n, 128) of SIFT descriptors.

        Returns:
            Integer cluster label per descriptor, or None when there are too
            few descriptors to cluster.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: Dist_AB = min{dist_ij}, for all i in [1,N_A], j in [1,N_B],
        #                    dist_ij = ||fd_i - fd_j||_2                 [Eq. 1]
        #          Dist_CP = min{Dist_AP, Dist_BP}, for all P in S\{A,B} [Eq. 2]
        # Variables: A, B = two feature groups being merged into C; fd_i = the
        #            128-dimensional SIFT descriptor of feature i; N_A, N_B =
        #            group sizes; S = the set of all groups.
        # Source: Bharadwaj, Collins & Liu 2025 (WACV) - SKILL A4 step 2.
        # Expected range: cluster labels. Taking the MINIMUM pairwise distance
        #            between groups (Eq. 1) and updating a merged group's as the
        #            MINIMUM of its parents' (Eq. 2) IS single-linkage
        #            agglomerative clustering under the Euclidean metric, so
        #            both equations are implemented exactly, not approximated.
        # ────────────────────────────────────────────────────
        if descriptors is None or descriptors.shape[0] < \
                constants.MINIMUM_FEATURES_PER_VISUAL_WORD:
            return None

        try:
            merge_tree = linkage(np.asarray(descriptors, dtype=np.float64),
                                 method=constants.HIERARCHICAL_LINKAGE_METHOD,
                                 metric=constants.HIERARCHICAL_LINKAGE_METRIC)
        except (ValueError, MemoryError) as error:
            logger.warning("hierarchical clustering failed on %d descriptors: "
                           "%s", descriptors.shape[0], error)
            return None
        return fcluster(merge_tree, t=self.visual_word_cut_distance,
                        criterion="distance")

    def _build_implicit_lines(self,
                              scene: PreparedScene,
                              labels: np.ndarray) -> list:
        """Fit one implicit line through each usable visual word.

        Args:
            scene: Prepared scene carrying keypoint positions and scales.
            labels: Cluster label per keypoint.

        Returns:
            List of LineSegment objects, one per usable visual word.
        """
        lines: list = []
        for label in np.unique(labels):
            member_indices = np.flatnonzero(labels == label)
            if not self._is_usable_visual_word(member_indices):
                continue

            positions = scene.keypoint_positions[member_indices]
            scales = scene.keypoint_scales[member_indices]
            selected = self._select_features_forward(positions, scales)
            segment = self._fit_implicit_line(positions[selected],
                                              scales[selected])
            if segment is not None:
                lines.append(segment)
        return lines

    @staticmethod
    def _is_usable_visual_word(member_indices: np.ndarray) -> bool:
        """Judge whether a visual word is large enough but not too large.

        Args:
            member_indices: Indices of the features in the visual word.

        Returns:
            True when the word can support the triplet-based scores without
            dominating runtime.
        """
        return (constants.MINIMUM_FEATURES_PER_VISUAL_WORD
                <= member_indices.size
                <= constants.MAXIMUM_FEATURES_PER_VISUAL_WORD)

    def _select_features_forward(self,
                                 positions: np.ndarray,
                                 scales: np.ndarray) -> np.ndarray:
        """Greedily grow the most geometrically consistent feature subset.

        SKILL A4 step 3: "Forward feature selection within each visual word,
        scoring candidate feature subsets by three geometric-consistency
        measures". The SKILL does not state a target subset size, so
        FORWARD_SELECTION_TARGET_SUBSET_SIZE carries that choice.

        Args:
            positions: Array of shape (n, 2) of keypoint (x, y) coordinates.
            scales: Array of shape (n,) of keypoint scales.

        Returns:
            Integer index array of the selected features.
        """
        feature_count = positions.shape[0]
        target = min(constants.FORWARD_SELECTION_TARGET_SUBSET_SIZE,
                     feature_count)
        if feature_count <= constants.MINIMUM_FEATURES_PER_VISUAL_WORD:
            return np.arange(feature_count)

        selected = list(self._best_seed_triplet(positions, scales))
        while len(selected) < target:
            remaining = [index for index in range(feature_count)
                         if index not in selected]
            best = min(remaining,
                       key=lambda index: self._composite_score(
                           positions[selected + [index]],
                           scales[selected + [index]]))
            selected.append(best)
        return np.array(sorted(selected), dtype=np.int64)

    def _best_seed_triplet(self,
                           positions: np.ndarray,
                           scales: np.ndarray) -> tuple:
        """Find the three features with the lowest composite score.

        Args:
            positions: Array of shape (n, 2) of keypoint coordinates.
            scales: Array of shape (n,) of keypoint scales.

        Returns:
            Tuple of three feature indices.
        """
        indices = range(positions.shape[0])
        return min(
            itertools.combinations(indices,
                                   constants.MINIMUM_FEATURES_PER_VISUAL_WORD),
            key=lambda triplet: self._composite_score(
                positions[list(triplet)], scales[list(triplet)]))

    def _composite_score(self,
                         positions: np.ndarray,
                         scales: np.ndarray) -> float:
        """Combined geometric-consistency score of a candidate feature subset.

        Args:
            positions: Array of shape (n, 2) of keypoint coordinates.
            scales: Array of shape (n,) of keypoint scales.

        Returns:
            The composite score S_C; lower means more geometrically consistent.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: S_C = S_L * exp(S_A + S_S) / N^2
        # Variables: S_L = linearity score, S_A = angle score, S_S = scale
        #            score, N = number of features in the visual word.
        # Source: Bharadwaj, Collins & Liu 2025 - SKILL A4 step 3, "Composite
        #            score ... lower S_C = more geometrically consistent, hence
        #            more suitable for line-fitting".
        # Expected range: non-negative. The 1/N^2 factor rewards larger subsets,
        #            so a subset must earn its size by staying collinear and
        #            uniformly foreshortened; the exponential makes the angle
        #            and scale penalties multiplicative against the linearity
        #            term rather than merely additive.
        # ────────────────────────────────────────────────────
        feature_count = positions.shape[0]
        if feature_count < constants.MINIMUM_FEATURES_PER_VISUAL_WORD:
            return float("inf")

        ordered = self._order_along_principal_direction(positions)
        linearity = self._linearity_score(positions)
        angle = self._angle_score(positions[ordered])
        scale = self._scale_score(scales[ordered])
        return float(linearity * np.exp(angle + scale)
                     / float(feature_count * feature_count))

    @staticmethod
    def _order_along_principal_direction(positions: np.ndarray) -> np.ndarray:
        """Sort feature indices along the subset's dominant direction.

        The SKILL's angle and scale scores are defined over "ordered triplets",
        and the ordering that makes them measure recession toward a vanishing
        point is the sequence along the recurring pattern itself.

        Args:
            positions: Array of shape (n, 2) of keypoint coordinates.

        Returns:
            Index array ordering the features along their principal axis.
        """
        centred = positions - positions.mean(axis=0)
        # The principal direction is the largest-variance axis, obtained as the
        # eigenvector the total-least-squares line fit does NOT use as normal.
        line = fit_line_least_squares(positions)
        direction = np.array([-line[1], line[0]], dtype=np.float64)
        return np.argsort(centred @ direction)

    @staticmethod
    def _linearity_score(positions: np.ndarray) -> float:
        """Average perpendicular distance of features from their fitted line.

        Args:
            positions: Array of shape (n, 2) of keypoint coordinates.

        Returns:
            The linearity score S_L in pixels; lower means more collinear.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: S_L = average perpendicular distance of keypoints from a
        #                fitted line.
        # Variables: the line is fitted through the candidate subset's keypoint
        #            positions; the distance is measured perpendicular to it.
        # Source: Bharadwaj, Collins & Liu 2025 - SKILL A4 step 3, "Linearity
        #            score S_L: average perpendicular distance of keypoints from
        #            a fitted line - lower is better (stronger collinearity)".
        # Expected range: non-negative, in pixels. Zero for perfectly collinear
        #            features.
        # ────────────────────────────────────────────────────
        return mean_perpendicular_distance(positions,
                                           fit_line_least_squares(positions))

    @staticmethod
    def _angle_score(ordered_positions: np.ndarray) -> float:
        """Average change in direction across consecutive feature triplets.

        Args:
            ordered_positions: Array of shape (n, 2), sorted along the pattern.

        Returns:
            The angle score S_A in radians; lower means more uniform direction.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: S_A = mean over ordered triplets (A,B,C) of
        #                | direction_angle(A->B) - direction_angle(B->C) |
        # Variables: A, B, C = three features taken in order along the recurring
        #            pattern; direction_angle = the segment's bearing.
        # Source: Bharadwaj, Collins & Liu 2025 - SKILL A4 step 3, "Angle score
        #            S_A: for ordered triplets (A,B,C), the absolute difference
        #            in directional angle between segments A->B and B->C,
        #            averaged across all triplets - lower means more uniform
        #            direction (consistent with points receding toward one
        #            vanishing point)".
        # Expected range: [0, pi] radians, in practice near 0 for a genuine
        #            recurring pattern.
        # INTERPRETATION: the SKILL says "ordered triplets" without defining the
        #            order, so consecutive triplets along the subset's principal
        #            axis are used - the only ordering under which the score
        #            measures recession, as the SKILL says it does. Taking all
        #            n(n-1)(n-2) permutations instead would average the quantity
        #            to a constant and destroy its meaning.
        # ────────────────────────────────────────────────────
        differences = [
            wrap_angle_difference(
                signed_direction_angle(ordered_positions[index],
                                       ordered_positions[index + 1]),
                signed_direction_angle(ordered_positions[index + 1],
                                       ordered_positions[index + 2]))
            for index in range(ordered_positions.shape[0] - 2)]
        return float(np.mean(differences)) if differences else 0.0

    @staticmethod
    def _scale_score(ordered_scales: np.ndarray) -> float:
        """Average change in size-change ratio across consecutive triplets.

        Args:
            ordered_scales: Array of shape (n,), sorted along the pattern.

        Returns:
            The scale score S_S; lower means more uniform foreshortening.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: S_S = mean over ordered triplets (A,B,C) of
        #                | ratio(A->B) - ratio(B->C) |
        # Variables: ratio(A->B) = sigma_B / sigma_A, the relative size change
        #            between consecutive SIFT keypoint scales.
        # Source: Bharadwaj, Collins & Liu 2025 - SKILL A4 step 3, "Scale score
        #            S_S: for the same triplets, the absolute difference in
        #            relative size-change ratio between A->B and B->C, averaged
        #            - lower means more uniform projective foreshortening".
        # Expected range: non-negative, near 0 for a pattern receding uniformly.
        # INTERPRETATION: the SKILL does not state how the "relative size-change
        #            ratio" is formed. The ratio of successive keypoint scales
        #            is used, which is the only reading under which the quantity
        #            measures projective foreshortening as the SKILL says it
        #            does. Recorded in constants.KNOWN_SKILL_AMBIGUITIES.
        # ────────────────────────────────────────────────────
        safe_scales = np.maximum(np.asarray(ordered_scales, dtype=np.float64),
                                 constants.HOMOGENEOUS_INFINITY_TOLERANCE)
        ratios = safe_scales[1:] / safe_scales[:-1]
        if ratios.size < constants.MINIMUM_LINES_FOR_VANISHING_POINT:
            return 0.0
        return float(np.mean(np.abs(np.diff(ratios))))

    @staticmethod
    def _fit_implicit_line(positions: np.ndarray,
                           scales: np.ndarray) -> Optional[LineSegment]:
        """Fit one line through a selected feature subset.

        SKILL A4 step 4: "fit an orientation vector through corresponding
        feature pairs across recurring-pattern instances via least squares,
        oriented from larger-scale to smaller-scale features (pointing toward
        the vanishing point)".

        Args:
            positions: Array of shape (n, 2) of selected keypoint coordinates.
            scales: Array of shape (n,) of their scales.

        Returns:
            LineSegment spanning the subset, or None when it is degenerate.
        """
        if positions.shape[0] < constants.MINIMUM_LINES_FOR_VANISHING_POINT:
            return None

        # Order from larger to smaller scale so the segment points toward the
        # vanishing point, as the SKILL specifies.
        by_descending_scale = np.argsort(-np.asarray(scales, dtype=np.float64))
        start = positions[by_descending_scale[0]]
        end = positions[by_descending_scale[-1]]

        offset = np.asarray(end, dtype=np.float64) - np.asarray(start,
                                                                dtype=np.float64)
        length = float(np.linalg.norm(offset))
        if length == 0.0:
            return None
        return LineSegment(start=(float(start[0]), float(start[1])),
                           end=(float(end[0]), float(end[1])),
                           homogeneous=line_through_points(start, end),
                           direction=offset / length,
                           length=length)

    def _weighted_ransac(self, lines: list) -> tuple:
        """Find the vanishing point supported by the largest set of lines.

        Args:
            lines: LineSegment objects to reach consensus over.

        Returns:
            Tuple of (homogeneous vanishing point or None, inlier segments).
        """
        generator = np.random.default_rng(constants.RANSAC_RANDOM_SEED)
        best_inliers: list = []

        # SKILL A4 step 5: "Repeated with re-initialized weights across multiple
        # runs to mitigate the ill-conditioning of near-parallel intersections."
        for _ in range(constants.RANSAC_RESTART_COUNT):
            inliers = self._run_single_ransac(lines, generator)
            if len(inliers) > len(best_inliers):
                best_inliers = inliers

        if len(best_inliers) < constants.MINIMUM_RANSAC_INLIER_COUNT:
            return None, []
        return self._line_scatter_vanishing_point(best_inliers), best_inliers

    def _run_single_ransac(self, lines: list, generator) -> list:
        """One RANSAC run with freshly initialised line weights.

        Args:
            lines: LineSegment objects to reach consensus over.
            generator: Seeded numpy random generator.

        Returns:
            The largest inlier set found in this run.
        """
        weights = self._initial_line_weights(lines)
        best_inliers: list = []

        for _ in range(constants.RANSAC_ITERATIONS_PER_RUN):
            pair = self._sample_line_pair(lines, weights, generator)
            if pair is None:
                continue

            candidate = intersect_lines(lines[pair[0]].homogeneous,
                                        lines[pair[1]].homogeneous)
            inlier_flags = self._classify_inliers(candidate, lines)
            weights = self._update_weights(weights, inlier_flags)

            inliers = [line for line, is_inlier in zip(lines, inlier_flags)
                       if is_inlier]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
        return best_inliers

    @staticmethod
    def _initial_line_weights(lines: list) -> np.ndarray:
        """Initial RANSAC sampling weight of each line.

        Args:
            lines: LineSegment objects.

        Returns:
            Float array of non-negative weights, one per line.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: w_i = SUM_{j=1, j!=i}^{n} e^{-theta_ij}
        # Variables: theta_ij = the acute angle between lines l_i and l_j,
        #            "derived from arccos of the direction-vector dot product,
        #            folded into [0, pi/2]"; n = number of lines.
        # Source: Bharadwaj, Collins & Liu 2025 - SKILL A4 step 5, "giving
        #            higher initial weight to more nearly-parallel line pairs".
        # Expected range: (0, n-1]. A line parallel to every other scores near
        #            n-1; a line perpendicular to every other scores near
        #            (n-1)*e^{-pi/2} = 0.208*(n-1).
        # ────────────────────────────────────────────────────
        directions = np.array([line.direction for line in lines],
                              dtype=np.float64)
        cosines = np.abs(directions @ directions.T)
        angles = np.arccos(np.clip(cosines, -1.0, 1.0))

        contributions = np.exp(-angles)
        # Exclude the self term j == i, which would otherwise add e^0 = 1.
        np.fill_diagonal(contributions, 0.0)
        return contributions.sum(axis=1)

    @staticmethod
    def _update_weights(weights: np.ndarray,
                        inlier_flags: np.ndarray) -> np.ndarray:
        """Reward lines agreeing with the candidate point and penalise the rest.

        Args:
            weights: Current sampling weights.
            inlier_flags: Boolean array marking inliers.

        Returns:
            Updated weight array.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: inliers  l_i in I: w'_i = w_i * alpha
        #          outliers l_j in O: w'_j = w_j * beta
        #          with alpha = 1.2, beta = 0.8.
        # Variables: I, O = the inlier and outlier sets of the current candidate
        #            intersection; w = the sampling weight.
        # Source: Bharadwaj, Collins & Liu 2025 - SKILL A4 step 5, verbatim
        #            including both constants.
        # Expected range: strictly positive. NOTE the deliberate renaming:
        #            these are RANSAC_INLIER_WEIGHT_GAIN and
        #            RANSAC_OUTLIER_WEIGHT_DECAY, never "alpha"/"beta", because
        #            Yao et al.'s Eq. 7-8 use those same letters for the
        #            expected and measured height ratios in module B below.
        # ────────────────────────────────────────────────────
        multipliers = np.where(inlier_flags,
                               constants.RANSAC_INLIER_WEIGHT_GAIN,
                               constants.RANSAC_OUTLIER_WEIGHT_DECAY)
        return weights * multipliers

    @staticmethod
    def _classify_inliers(candidate_point: np.ndarray,
                          lines: list) -> np.ndarray:
        """Mark which lines pass close enough to a candidate vanishing point.

        Args:
            candidate_point: Homogeneous candidate point.
            lines: LineSegment objects to classify.

        Returns:
            Boolean array, True where the line is an inlier.
        """
        cartesian = homogeneous_to_cartesian(
            candidate_point, constants.HOMOGENEOUS_INFINITY_TOLERANCE)
        if cartesian is None:
            return np.zeros(len(lines), dtype=bool)

        distances = np.array([point_to_line_distance(cartesian, line.homogeneous)
                              for line in lines], dtype=np.float64)
        return distances <= constants.RANSAC_INLIER_DISTANCE_PIXELS

    @staticmethod
    def _sample_line_pair(lines: list,
                          weights: np.ndarray,
                          generator) -> Optional[tuple]:
        """Draw two lines by weight that are not near-parallel to each other.

        The minimum-angle filter is applied here as well as at selection time,
        per the SKILL's Implementation Note that it belongs "before any
        line-intersection computation, not just within the RANSAC loop".

        Args:
            lines: LineSegment objects.
            weights: Current sampling weights.
            generator: Seeded numpy random generator.

        Returns:
            Tuple of two distinct line indices, or None when the draw was
            ill-conditioned.
        """
        positive = np.maximum(weights, 0.0)
        total = float(positive.sum())
        if total <= 0.0:
            return None

        first, second = generator.choice(len(lines), size=2, replace=False,
                                         p=positive / total)
        angle = acute_angle_between_directions(lines[first].direction,
                                               lines[second].direction)
        if angle < constants.MINIMUM_LINE_PAIR_ANGLE_RADIANS:
            return None
        return int(first), int(second)


class HeightRatioAnalyser:
    """Tests object pairs for consistency with the reference plane's geometry."""

    def analyse(self,
                vanishing_line_row: float,
                regions: list,
                supplied_expected_ratios: Optional[dict] = None
                ) -> HeightRatioAnalysis:
        """Evaluate every admissible object pair under Eq. 7 and Eq. 8.

        Args:
            vanishing_line_row: Yao's v0, the reference plane's horizon row.
            regions: Candidate ObjectRegion objects resting on that plane.
            supplied_expected_ratios: Expected ratios keyed by the
                (first identifier, second identifier) tuple. Pairs absent from
                this mapping fall back to the engine's assumed ratio.

        Returns:
            HeightRatioAnalysis summarising every evaluated pair.
        """
        ratios = supplied_expected_ratios or {}
        measurements: list = []
        rejected = 0

        for first, second in itertools.combinations(regions, 2):
            if not self._is_admissible_pair(vanishing_line_row, first, second,
                                            ratios):
                rejected += 1
                continue
            measurement = self._measure_pair(vanishing_line_row, first, second,
                                             ratios)
            if measurement is None:
                rejected += 1
                continue
            measurements.append(measurement)

        return self._summarise(measurements, rejected)

    def _measure_pair(self,
                      vanishing_line_row: float,
                      first: ObjectRegion,
                      second: ObjectRegion,
                      supplied_ratios: dict) -> Optional[HeightRatioMeasurement]:
        """Run Eq. 7 and Eq. 8 on one admissible object pair.

        Args:
            vanishing_line_row: Yao's v0.
            first: The first object of the pair.
            second: The second object.
            supplied_ratios: Caller-supplied expected ratios.

        Returns:
            HeightRatioMeasurement, or None when Eq. 7 was degenerate.
        """
        measured_ratio = self._height_ratio(vanishing_line_row, first, second)
        if measured_ratio is None:
            return None

        expected_ratio, was_assumed = self._expected_ratio_for(
            first, second, supplied_ratios)
        if expected_ratio < constants.MINIMUM_EXPECTED_HEIGHT_RATIO:
            return None

        sigma = self._sigma_for(expected_ratio)
        consistency = self._consistency(expected_ratio, measured_ratio, sigma)
        sensitivity = self._ratio_sensitivity(vanishing_line_row, first, second,
                                              measured_ratio)
        return HeightRatioMeasurement(
            first_region_id=first.identifier,
            second_region_id=second.identifier,
            measured_ratio=measured_ratio, expected_ratio=expected_ratio,
            sigma=sigma, consistency=consistency,
            is_consistent=consistency >= constants.CONSISTENCY_DECISION_THRESHOLD,
            expected_ratio_was_assumed=was_assumed,
            ratio_sensitivity_per_pixel=sensitivity,
            tolerable_vanishing_line_error_pixels=self._tolerable_error(
                expected_ratio, sensitivity))

    @staticmethod
    def _expected_ratio_for(first: ObjectRegion,
                            second: ObjectRegion,
                            supplied_ratios: dict) -> tuple:
        """Resolve the expected height ratio for a pair and its provenance.

        Args:
            first: The first object of the pair.
            second: The second object.
            supplied_ratios: Caller-supplied expected ratios.

        Returns:
            Tuple of (expected ratio, True when it was assumed not supplied).
        """
        key = (first.identifier, second.identifier)
        if key in supplied_ratios:
            return float(supplied_ratios[key]), False
        return float(constants.DEFAULT_EXPECTED_HEIGHT_RATIO), True

    def _ratio_sensitivity(self,
                           vanishing_line_row: float,
                           first: ObjectRegion,
                           second: ObjectRegion,
                           measured_ratio: float) -> float:
        """How far the measured ratio moves per pixel of vanishing-line error.

        Estimated by perturbing v0 by one pixel in Eq. 7 and differencing, so it
        introduces no model beyond Eq. 7 itself. It exists because Eq. 7 is
        markedly more sensitive to v0 for pairs straddling very different depths
        than for pairs at similar depths, and a forensic reader needs to know
        which kind of pair a verdict rests on.

        Args:
            vanishing_line_row: Yao's v0.
            first: The first object of the pair.
            second: The second object.
            measured_ratio: Yao's beta at the unperturbed v0.

        Returns:
            |d beta / d v0| in ratio units per pixel; 0.0 if unmeasurable.
        """
        perturbed = self._height_ratio(
            vanishing_line_row + constants.SENSITIVITY_PROBE_PIXELS,
            first, second)
        if perturbed is None:
            return 0.0
        return float(abs(perturbed - measured_ratio)
                     / constants.SENSITIVITY_PROBE_PIXELS)

    @staticmethod
    def _tolerable_error(expected_ratio: float, sensitivity: float) -> float:
        """Vanishing-line error at which this pair would cross the threshold.

        Substituting sigma = 0.1*alpha into Eq. 8 shows C falls to T = 0.5 when
        |alpha - beta| reaches 0.6745*sigma. Dividing that budget by the pair's
        measured sensitivity gives the error in pixels it can absorb.

        Args:
            expected_ratio: Yao's alpha.
            sensitivity: |d beta / d v0| from _ratio_sensitivity.

        Returns:
            Tolerable v0 error in pixels; infinite when the pair is insensitive.
        """
        if sensitivity <= 0.0:
            return float("inf")
        ratio_budget = (constants.CONSISTENCY_THRESHOLD_SIGMA_MULTIPLE
                        * constants.RATIO_SIGMA_FRACTION_OF_EXPECTED
                        * expected_ratio)
        return float(ratio_budget / sensitivity)

    @staticmethod
    def _height_ratio(vanishing_line_row: float,
                      first: ObjectRegion,
                      second: ObjectRegion) -> Optional[float]:
        """Real-world height ratio of two objects from their image coordinates.

        Args:
            vanishing_line_row: Yao's v0.
            first: Object B, supplying v_B1 and v_B2.
            second: Object B', supplying v'_B1 and v'_B2.

        Returns:
            Yao's beta, or None when the pair makes Eq. 7 degenerate.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: beta = z'_B/z_B
        #               ~ (v'_B1 - v'_B2)(v0 - v_B2)
        #               / [(v_B1 - v_B2)(v0 - v'_B2)]                    [Eq. 7]
        # Variables: v_B1, v_B2 = image rows of object B's top and bottom;
        #            v'_B1, v'_B2 = the same for object B'; v0 = the
        #            vanishing-line ordinate, KNOWN here; beta = the ratio of
        #            the two objects' real-world heights z'_B and z_B, the
        #            unknown being solved for.
        # Source: Yao, Wang, Zhao & Zhang 2012 - SKILL B step 1. Structurally
        #            identical to Eq. 5, but with v0 known and the ratio
        #            unknown - "the mirror-image use of the same underlying
        #            relationship".
        # Expected range: strictly positive - (v_B1 - v_B2) and (v0 - v_B2) are
        #            both negative in image coordinates, and likewise for the
        #            primed pair, so the sign cancels. Corpus values span
        #            0.784-1.369 across Yao et al.'s Table I.
        # ────────────────────────────────────────────────────
        first_extent = first.top_row - first.bottom_row
        second_extent = second.top_row - second.bottom_row
        first_depth = vanishing_line_row - first.bottom_row
        second_depth = vanishing_line_row - second.bottom_row

        denominator = first_extent * second_depth
        if abs(denominator) < constants.MINIMUM_EQUATION_SEVEN_DENOMINATOR:
            return None
        return float((second_extent * first_depth) / denominator)

    @staticmethod
    def _sigma_for(expected_ratio: float) -> float:
        """Standard deviation of the authentic height-ratio distribution.

        Args:
            expected_ratio: Yao's alpha.

        Returns:
            Sigma, a fixed fraction of the expected ratio.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: sigma = 0.1 * alpha
        # Variables: alpha = the expected height ratio; sigma = the standard
        #            deviation of (kappa - alpha), where kappa is an authentic
        #            image's true height ratio.
        # Source: Yao et al. 2012 - SKILL B step 4: "for an authentic image, the
        #            probability of kappa falling outside the interval
        #            [0.8*alpha, 1.2*alpha] is set to a constant 0.05, which -
        #            solved against the Gaussian CDF - fixes sigma = 0.1*alpha."
        # Expected range: strictly positive, one tenth of alpha.
        # VERIFIED, not merely copied: P(|kappa - alpha| > 0.2*alpha) = 0.05
        #            requires 0.2*alpha/sigma = Phi^-1(0.975) = 1.95996, giving
        #            sigma = 0.10204*alpha. The paper's rounded 0.1 is therefore
        #            self-consistent with its own stated design choice to within
        #            2%. The published 0.1 is used so this engine reproduces the
        #            paper's Table I exactly.
        # CAUTION carried from the SKILL's Implementation Notes: this is "a
        #            fixed, non-adaptive constant ... treat sigma as a starting
        #            point for this engine's own calibration, not a universally
        #            correct value."
        # ────────────────────────────────────────────────────
        return float(constants.RATIO_SIGMA_FRACTION_OF_EXPECTED * expected_ratio)

    @staticmethod
    def _consistency(expected_ratio: float,
                     measured_ratio: float,
                     sigma: float) -> float:
        """Consistency of a measured height ratio with its expected value.

        Args:
            expected_ratio: Yao's alpha.
            measured_ratio: Yao's beta, from Eq. 7.
            sigma: Standard deviation from _sigma_for.

        Returns:
            Yao's C in [0, 1]; 1 means the ratios agree exactly.
        """
        # ── SKILL VERIFICATION ──────────────────────────────
        # Formula: C = 2 * F(-|alpha - beta|; 0, sigma^2)                [Eq. 8]
        # Variables: F = the cumulative distribution function of (kappa - alpha),
        #            assumed N(0, sigma^2) for authentic images; alpha =
        #            expected ratio; beta = measured ratio; C = the consistency
        #            measure.
        # Source: Yao et al. 2012 - SKILL B step 3.
        # Expected range: [0, 1], "decreases monotonically as |alpha - beta|
        #            increases". C = 1 exactly when beta equals alpha, since
        #            F(0) = 0.5 and the leading factor 2 normalises it. The
        #            paper's threshold is T = 0.5: C < T implies at least one
        #            object is considered forged.
        # DERIVED CONSEQUENCE worth recording: substituting sigma = 0.1*alpha,
        #            C falls to exactly T = 0.5 when |alpha - beta| = 0.6745*sigma
        #            = 0.0674*alpha. The paper's decision rule therefore flags a
        #            pair once its measured height ratio departs from expectation
        #            by about 6.7% - a strict criterion, consistent with Table I
        #            where authentic images score C = 0.880-0.979 and forged ones
        #            C = 0.020-0.208.
        # ────────────────────────────────────────────────────
        divergence = abs(float(expected_ratio) - float(measured_ratio))
        cumulative = float(norm.cdf(-divergence, loc=0.0, scale=sigma))
        return float(np.clip(
            constants.CONSISTENCY_NORMALISATION_FACTOR * cumulative, 0.0, 1.0))

    @staticmethod
    def _is_admissible_pair(vanishing_line_row: float,
                            first: ObjectRegion,
                            second: ObjectRegion,
                            supplied_ratios: dict) -> bool:
        """Screen out object pairs that would make Eq. 7 meaningless.

        SKILL "Implementation notes" - degenerate object pairs: "both objects
        must have distinct, unambiguous top/bottom image coordinates on the
        same reference plane; objects that are partially occluded, overlapping,
        or not actually resting on the assumed plane ... will silently produce a
        meaningless beta", with the engineering recommendation to "cross-check
        candidate pairs' bounding-box bottoms against a coarse
        ground-plane/horizon-line estimate before accepting them as valid pairs."

        Args:
            vanishing_line_row: Yao's v0.
            first: The first object of the pair.
            second: The second object.
            supplied_ratios: Caller-supplied expected ratios; a pair named here
                is trusted and skips the appearance screen.

        Returns:
            True when the pair may be measured.
        """
        for region in (first, second):
            if region.image_height < constants.MINIMUM_REGION_IMAGE_HEIGHT_PIXELS:
                return False
            # An object resting on the plane must have its base BELOW the
            # plane's vanishing line in image coordinates.
            if (region.bottom_row - vanishing_line_row
                    < constants.MINIMUM_BASE_BELOW_VANISHING_LINE_PIXELS):
                return False

        if (first.identifier, second.identifier) in supplied_ratios:
            return True
        return HeightRatioAnalyser._appearance_compatible(first, second)

    @staticmethod
    def _appearance_compatible(first: ObjectRegion,
                               second: ObjectRegion) -> bool:
        """Judge whether two regions plausibly depict the same kind of thing.

        This is a surrogate, not a classifier. SKILL B step 2 wants the expected
        ratio to come from "general prior knowledge about the object classes",
        which an image-only contract cannot supply; comparing coarse colour
        histograms at least avoids pairing obviously unlike regions before the
        engine assumes they are the same height. It is one reason the automatic
        path carries CONFIDENCE_PENALTY_ASSUMED_RATIO_PRIOR.

        Args:
            first: The first object of the pair.
            second: The second object.

        Returns:
            True when the two appearance signatures are close enough to pair.
        """
        if first.appearance_signature is None or \
                second.appearance_signature is None:
            return True

        distance = chi_square_distance(first.appearance_signature,
                                       second.appearance_signature,
                                       constants.HISTOGRAM_DISTANCE_EPSILON)
        return distance <= constants.MAXIMUM_APPEARANCE_DISTANCE

    @staticmethod
    def _corroborated_consistency(measurements: list) -> tuple:
        """Worst object's median consistency across the pairs it belongs to.

        ENHANCEMENT 1. The SKILL's own reason for the pair minimum - "one
        spliced object is typically inconsistent with several others" - is the
        fix: a splice disagrees with EVERY partner, a mislocalised box with one,
        and a median tells them apart where a minimum cannot. Measurements in
        constants.TEST_DERIVED_ENHANCEMENTS.

        Args:
            measurements: HeightRatioMeasurement objects, at least one.

        Returns:
            Tuple of (corroborated consistency, worst object id, partner count).
        """
        by_object: dict = {}
        for item in measurements:
            for identifier in (item.first_region_id, item.second_region_id):
                by_object.setdefault(identifier, []).append(item.consistency)

        corroborated, worst_id, partners = 1.0, -1, 0
        for identifier, values in by_object.items():
            if len(values) < constants.MINIMUM_PARTNERS_FOR_CORROBORATION:
                continue
            median = float(np.median(values))
            if median < corroborated:
                corroborated, worst_id, partners = median, identifier, len(values)

        if worst_id == -1:
            # Too few objects for any of them to have several partners; the
            # SKILL's pair minimum is the only statistic available.
            worst = min(measurements, key=lambda item: item.consistency)
            return (float(worst.consistency), int(worst.first_region_id),
                    len(by_object.get(worst.first_region_id, [])))
        return corroborated, int(worst_id), int(partners)

    @staticmethod
    def _summarise(measurements: list, rejected: int) -> HeightRatioAnalysis:
        """Reduce the per-pair measurements to the analysis record.

        SKILL B step 5 wants "several measurements of beta ... averaged", so
        mean and minimum are both carried; ENHANCEMENT 1's corroborated
        statistic drives the score.

        Args:
            measurements: HeightRatioMeasurement objects, possibly empty.
            rejected: Pairs discarded before measurement.

        Returns:
            HeightRatioAnalysis.
        """
        if not measurements:
            return HeightRatioAnalysis(measurements=[], minimum_consistency=1.0,
                                       corroborated_consistency=1.0,
                                       mean_consistency=1.0,
                                       mean_measured_ratio=0.0,
                                       evaluated_pair_count=0,
                                       rejected_pair_count=rejected,
                                       any_ratio_assumed=False)

        consistencies = np.array([item.consistency for item in measurements],
                                 dtype=np.float64)
        ratios = np.array([item.measured_ratio for item in measurements],
                          dtype=np.float64)
        return HeightRatioAnalyser._build_analysis(
            measurements, rejected, consistencies, ratios)

    @staticmethod
    def _build_analysis(measurements: list,
                        rejected: int,
                        consistencies: np.ndarray,
                        ratios: np.ndarray) -> HeightRatioAnalysis:
        """Assemble the analysis record from the per-pair measurements.

        Args:
            measurements: HeightRatioMeasurement objects, at least one.
            rejected: Pairs discarded before measurement.
            consistencies: Per-pair C values.
            ratios: Per-pair beta values.

        Returns:
            HeightRatioAnalysis.
        """
        corroborated, worst_id, partners = \
            HeightRatioAnalyser._corroborated_consistency(measurements)
        return HeightRatioAnalysis(
            measurements=measurements,
            minimum_consistency=float(consistencies.min()),
            corroborated_consistency=corroborated,
            worst_object_id=worst_id,
            worst_object_partner_count=partners,
            mean_consistency=float(consistencies.mean()),
            mean_measured_ratio=float(ratios.mean()),
            evaluated_pair_count=len(measurements),
            rejected_pair_count=rejected,
            any_ratio_assumed=any(item.expected_ratio_was_assumed
                                  for item in measurements))
