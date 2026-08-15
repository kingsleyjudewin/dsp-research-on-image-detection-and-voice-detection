# Perspective / Geometric Consistency Forgery Detection

## Core mathematical principle

**Underlying concept.** Under a distortion-free pinhole camera model, a 3-D world point `(x,y,z)` maps to a 2-D image point `(u,v)` via (Yao et al., Eq. 1):
```
ζ·[u,v,1]ᵀ = (1/ζ)·[[f,0,c_u],[0,f,c_v],[0,0,1]] · [[r11,r12,r13,t_x],[r21,r22,r23,t_y],[r31,r32,r33,t_z]] · [x,y,z,1]ᵀ
```
`ζ` = depth factor (`= r31·x+r32·y+r33·z+t_z`), the first matrix is the camera intrinsic matrix (`f` = focal length in pixels, `(c_u,c_v)` = principal point), the second is the extrinsic matrix (rotation `r_ij`, translation `t`). This linear projective relationship forces every point on a shared reference plane to obey a fixed cross-ratio/vanishing-line relationship, and forces parallel 3-D lines to converge to a common **vanishing point** in the image.

**Property exploited.** For a **leveled, untilted camera** (`r11=r23=r32=1`, all other off-diagonal rotation terms `0`; translation `t_y=-H, t_x=t_z=0`), the projection simplifies to (Eq. 2):
```
ζ·[u,v,1]ᵀ = [[f,0,c_u],[0,f,v0],[0,0,1]] · [[1,0,0],[0,0,-H],[0,1,0]] · [x,y,z,1]ᵀ
```
and the **vanishing line** of the reference plane (`z=0`) sits at image row `v0` (the line through the vanishing point, parallel to the plane, at the intersection of the image plane and the plane through the camera center parallel to the reference plane). Given `v0` and camera height `H`, **the height of any object resting on the reference plane can be recovered from its image-space top/bottom coordinates alone** (Eq. 3–4) — **without any camera intrinsics or EXIF data**, which is exactly the metadata a forger is most likely to have stripped or corrupted. A spliced object from a different photo (different camera position, focal length, or scene depth) is almost never resized to match the host image's vanishing-line/height-ratio constraints, even under careful manual retouching — this geometric inconsistency is the exploitable signal.

## Input requirements

- **Format**: any RGB/grayscale image containing at least two objects presumed to rest on a common reference plane (e.g. the ground), or sufficient man-made/recurring structure to estimate a vanishing point even without explicit object pairs.
- **Preprocessing**: edge/line extraction (Hough transform) for the parallel-line vanishing-point method; SIFT feature extraction + hierarchical clustering for the recurrence-based fallback; superpixel segmentation (SLIC recommended) for candidate object/region proposal when bounding boxes aren't manually available.
- **Reliable when**: camera roll and tilt are small or zero (Yao et al.'s simplified formula assumes this explicitly); the scene contains either (a) usable parallel lines at angle ≥5° from each other, or (b) recurring patterns (evenly-spaced structures — railings, fence posts, tiled floors) even without explicit long lines; at least one pair of same-reference-plane objects is identifiable.
- **Unreliable / inapplicable when**:
  - **Significant camera tilt/roll**: Yao et al.'s core formula (Eq. 4) assumes negligible tilt; a full tilt-compensated version needs a **second (vertical) vanishing point**, which the paper states is unreliable to estimate in practice ("accuracy of the estimation cannot be guaranteed because the angle between two vertical lines is too small... even worse, the vertical vanishing point is unknown if the scenes do not contain enough vertical lines"). The paper's own experiments show Eq. 4 "still holds approximately as long as the tilt is small enough that the vanishing line is inside the image" — i.e., a soft degradation, not a hard cutoff, but unquantified beyond that qualitative statement.
  - **No usable geometric structure**: texture-only, close-up, or single-object/no-ground-plane images. R-VPD explicitly states it is "designed primarily to deal with images with RPs [recurring patterns]" and has "limitations on RP images with strong perspective and on non-RP images."
  - **Multiple vanishing points (Manhattan-world scenes)**: both Yao et al.'s method and R-VPD assume a **single dominant vanishing point**; R-VPD's own conclusion explicitly flags this as future work ("not Manhattan-world scenes with multiple vanishing points such as those found in indoor/urban environments").
  - **No suitable reference object with known height ratio** — the reference-object-based vanishing-line estimation sub-method (Eq. 5–6) fails outright without one.

## Step-by-step algorithm

### A. Vanishing point / vanishing line estimation

**A1 — Parallel-line based (Yao et al. 2012), primary when explicit lines are available:**
1. Extract straight edges via **Hough transform** on detected edges.
2. Group edges into parallel-line families; **discard line pairs with angle < 5°** (near-parallel intersections are numerically ill-conditioned).
3. Refine the vanishing-point estimate via **Levenberg-Marquardt maximum-likelihood estimation**: "lines are modified to pass a single point such that sum of squared orthogonal distances from the endpoints of the measured lines to the modified lines is minimized."
4. Limitation stated directly in the source: "produces good results for clear-cut man-made scenes but often fails when the scene lacks parallel lines."

**A2 — Reference-object based (Yao et al. 2012), when two objects `R`, `R'` with known height ratio `η` are visible on the plane:**
```
η = z'_R/z_R ≈ (v'_{R1}−v'_{R2})(v0−v_{R2}) / (v_{R1}−v_{R2})(v0−v'_{R2})          [Eq. 5]
```
Solving for the vanishing-line ordinate `v0` directly:
```
v0 ≈ (v_{R2}v'_{R2} − v_{R2}v'_{R1} + η·v_{R1}v'_{R2} − η·v_{R2}v'_{R2}) / (v'_{R2} − v'_{R1} + η·v_{R1} − η·v_{R2})          [Eq. 6]
```
Fails without a suitable reference object with a *known* height ratio.

**A3 — Texture-orientation based (Yao et al. 2012, cited to a prior soft-voting method [13], used when neither A1 nor A2 apply):** a locally-adaptive soft-voting algorithm estimates texture orientation per pixel with a confidence score; low-confidence pixels are discarded; remaining pixels vote for vanishing-point candidates (VPCs) within local voting regions; the VPC with the most votes is the initial vanishing point, updated for scenes with two dominant edges. Explicitly noted to **degrade in accuracy as objects on the ground increase** (more competing texture/clutter).

**A4 — Recurrence-based (R-VPD, Bharadwaj, Collins & Liu 2025), fallback when no explicit lines exist:**
1. **SIFT feature extraction** over the whole image (128-dimensional descriptor per keypoint, standard DoG-pyramid SIFT).
2. **Hierarchical clustering** to group features into "visual words": treat each feature as its own group initially; iteratively merge the closest groups using
   ```
   Dist_AB = min{dist_ij}, ∀i∈[1,N_A], ∀j∈[1,N_B],   dist_ij = ‖fd_i − fd_j‖_2          [Eq. 1]
   Dist_CP = min{Dist_AP, Dist_BP}, ∀P ∈ S\{A,B}          [Eq. 2, Lance-Williams-style update after merging A,B → C]
   ```
3. **Forward feature selection** within each visual word, scoring candidate feature subsets by three geometric-consistency measures:
   - **Linearity score `S_L`**: average perpendicular distance of keypoints from a fitted line — lower is better (stronger collinearity).
   - **Angle score `S_A`**: for ordered triplets `(A,B,C)`, the absolute difference in directional angle between segments `A→B` and `B→C`, averaged across all triplets — lower means more uniform direction (consistent with points receding toward one vanishing point).
   - **Scale score `S_S`**: for the same triplets, the absolute difference in relative size-change ratio between `A→B` and `B→C`, averaged — lower means more uniform projective foreshortening.
   - **Composite score**:
     ```
     S_C = S_L × exp(S_A + S_S) / N²          [N = number of features in the visual word]
     ```
     lower `S_C` = more geometrically consistent, hence more suitable for line-fitting.
4. **Implicit line fitting**: fit an orientation vector through corresponding feature pairs across recurring-pattern instances via least squares, oriented from larger-scale to smaller-scale features (pointing toward the vanishing point); merge with any explicit edge lines found by a standard line-segment detector.
5. **Weighted RANSAC**: initial per-line weight
   ```
   w_i = Σ_{j=1,j≠i}^n e^{−θ_ij}          [θ_ij = acute angle between lines l_i, l_j, derived from arccos of the direction-vector dot product, folded into [0,π/2]]
   ```
   giving higher initial weight to more nearly-parallel line pairs. Each iteration: randomly sample two lines weighted by `w`, compute their intersection, classify all other lines as inliers/outliers by distance to that point, then update weights — inliers `l_i∈I`: `w'_i = w_i·α`; outliers `l_j∈O`: `w'_j = w_j·β`, with **`α=1.2, β=0.8`** in the paper's implementation. Repeated with re-initialized weights across multiple runs to mitigate the ill-conditioning of near-parallel intersections.
6. **Final VP estimate**: eigenvalue decomposition of the largest inlier set (the specific eigen-formulation is referenced to prior work, not re-derived in this paper's extracted text).
7. **Complexity**: `O(n²)` in the number of detected SIFT features (dominated by the pairwise distance-matrix computation in clustering); median runtime **10.73s (RPVP-Real) / 1.31s (RPVP-Synthetic) on CPU**, vs. 0.86s/0.52s for the GPU-based NeurVPS baseline and 0.52s/0.52s for GPVPD (also GPU) — R-VPD is CPU-only and needs no training data, at the cost of being slower per-image than the GPU deep baselines.

### B. Cross-object / cross-region height-ratio consistency test (Yao et al. 2012) — the core forgery decision

1. Given the vanishing line `v0` (from A1/A2/A3/A4) and two manually- or superpixel-selected object bounding boxes `B`, `B'` with top/bottom image coordinates `(u_{B1},v_{B1}), (u_{B2},v_{B2})` and `(u'_{B1},v'_{B1}), (u'_{B2},v'_{B2})`, compute the **height ratio**:
   ```
   β = z'_B/z_B ≈ (v'_{B1}−v'_{B2})(v0−v_{B2}) / (v_{B1}−v_{B2})(v0−v'_{B2})          [Eq. 7]
   ```
   (structurally identical to Eq. 5, but here `v0` is *known* and `β` is the *unknown* being solved for — the mirror-image use of the same underlying relationship.)
2. Obtain the **expected/ground-truth ratio `α`** either from general prior knowledge about the object classes (e.g. typical relative human heights) or from a trusted reference pair with the same depth.
3. **Consistency measure**:
   ```
   C = 2·F(−|α−β|; 0, σ²)          [Eq. 8]
   ```
   `F(·)` = the **cumulative distribution function** of `(κ−α)`, where authentic images are assumed to have their true height ratio `κ` obey `(κ−α) ~ N(0,σ²)`. `C ∈ [0,1]` and **decreases monotonically as `|α−β|` increases**.
4. **Parameter settings used in the paper's own experiments** (exact, not illustrative): for an authentic image, the probability of `κ` falling outside the interval `[0.8α, 1.2α]` is set to a constant **`0.05`**, which — solved against the Gaussian CDF — fixes **`σ = 0.1α`**. Decision threshold **`T = 0.5`**: `C < T` ⟹ at least one object considered forged.
5. **Averaging**: "several measurements of `β`" are taken (i.e., across multiple object pairs or multiple estimates of the same pair) and averaged to improve accuracy — the paper does not give a specific minimum count, just the general recommendation.

### C. Contour/shape-consistency secondary check (Yang, Fang, Fei, Yu & Wei 2023 — *not a forgery paper, repurposed*)

`(This module is a general shape-recognition method repurposed for this engine — the repurposing itself is engineering synthesis, not validated for forgery detection in the source paper.)`

1. Segment a candidate object's silhouette (e.g. via the superpixel/clustering pipeline below).
2. Extract a **two-level contour feature**: dense sampling points encoded via local triangular-curvature/tangential-angle/Fourier-descriptor features, plus larger contour fragments encoding global shape.
3. Encode both levels via **Fisher Vector encoding** with power/L2 normalization, concatenate, and reduce dimensionality via **PCA**.
4. `[ML — excluded]`: the source paper's own classification/matching stage uses a **linear SVM**. **Training-free substitute (engineering recommendation)**: compare the Fisher-vector-encoded shape signature of a candidate region **directly** (Euclidean or cosine distance) against the shape statistics of same-class objects elsewhere in the same image, rather than against a trained classifier — a large distance flags shape/scale implausibility as supporting evidence alongside the height-ratio check (step B).
5. Reported robustness (source paper, general shape-recognition task, not forgery): accuracy drop **< 1% up to Gaussian boundary noise σ=1.0**.

### D. Superpixel segmentation (Sasmal & Dhal 2023 survey) — preprocessing only

Recommended default: **SLIC** (Simple Linear Iterative Clustering), cited across the survey as the most widely-used method for its speed, memory efficiency, and strong boundary adherence, combined with a lightweight partitional clustering step (K-means or Fuzzy C-Means on color/texture features) for region-merging. Used to generate candidate object/region masks automatically, avoiding the manual bounding-box selection Yao et al.'s original method relies on.

## Output

- **Vanishing-point estimation (A1–A4)**: a 2-D point (or line, for the reference-plane vanishing line) plus a **confidence indicator** — RANSAC inlier count/fraction for A4, or line-fit residual for A1. **This confidence must gate whether the module proceeds at all** — see Input Requirements; a low-confidence VP estimate should cause the module to **abstain** rather than emit a possibly-spurious height-ratio score.
- **Height-ratio consistency (B)**: `C ∈ [0,1]` directly from Eq. 8 — **this is already a calibrated, paper-defined `[0,1]` score**, unlike most other modules in this engine where a `[0,1]` mapping had to be recommended as an engineering addition. `C` near `1` = consistent/authentic; `C` near `0` = inconsistent/forged. For fusion-layer use, the natural tampering score is `1−C` (or `1−C_min` across all evaluated object pairs, since one spliced object is typically inconsistent with several others). The paper's own threshold `T=0.5` can seed (not replace) this engine's calibration.
- **Contour/shape check (C)**: unbounded Fisher-vector distance; `(not specified in corpus — engineering recommendation)` normalize via percentile rank against same-class objects in the image, or a fixed calibration set.

## Key findings from papers

**Manipulation types detected best**: scale/perspective-inconsistent splicing — an object pasted at the wrong apparent size for its position on the reference plane. Yao et al.'s method is explicitly validated as **robust to down-sampling and low-quality JPEG recompression**, a regime where most trace-based methods (resampling detectors, JPEG-ghost detectors) fail — this module is *complementary* to, not redundant with, the DSP-artifact-based modules elsewhere in this engine specifically because it survives exactly the post-processing that defeats them.

**Documented failure cases / limitations**: see Input Requirements — tilt/roll sensitivity, single-VP assumption, dependence on usable geometric structure. Additionally: the height-ratio check is blind to same-scale, same-plane, geometrically-consistent composites (it only catches scale/perspective inconsistency, not all forgery types) — explicitly acknowledged by Yao et al. as a reason to combine with other cues ("resampling trace and illumination-constraint based methods").

**Benchmark tables**:

| Paper | Dataset | Metric | Value | Conditions |
|---|---|---|---|---|
| Yao 2012 | Own test images (real-scene, downloaded + author-shot), 4 authentic + 4 forged sets | Estimated ratio β / Ideal ratio α / Consistency C / True-or-fake | Authentic: β≈0.784–1.129, α≈0.796–1.126, **C=0.880–0.979** (all correctly True); Forged: β≈0.860–1.369, α≈0.984–1.126, **C=0.020–0.208** (all correctly False) | Table I, exact per-image values; threshold T=0.5 correctly separates every case shown |
| Yao 2012 | same | Robustness | Method remains effective under down-sampling and low-quality JPEG recompression | Explicit claim, qualitative — no separate quantitative degradation curve given in the extracted text |
| R-VPD 2025 | RPVP-Synthetic (3200 images) | AUC of angle accuracy `AA^θ°` at θ=2°/5°/10°, median angle error | R-VPD: **0.97 / 3.49 / 7.44, median 0.60°** — best of all 8 methods compared (LSD, J-Linkage, GPVPD×3, NeurVPS×3) | p-values all <0.001 vs. R-VPD, i.e. statistically significant win |
| R-VPD 2025 | RPVP-Real (1400 images) | Same metrics | R-VPD: **1.03 / 3.75 / 8.63, median 0.74°** — best of all methods compared (NeurVPS-tmm17 excluded for data-leakage reasons) | |
| R-VPD 2025 | RPVP-Real-Exclusive (416 images, no TMM17 overlap) | Same metrics, vs. NeurVPS-tmm17 specifically | R-VPD: 3.71/8.54, median 0.71° vs. NeurVPS-tmm17: 3.64/8.21, median 0.72° | p=0.58 — **not statistically significant; "on par" with the supervised deep-learning SOTA**, despite R-VPD needing no training data |
| R-VPD 2025 | TMM17-Test (275 natural-scene images, not all containing recurring patterns) | Same metrics | R-VPD: 2.42/5.95, median 1.85° — **second-best**, behind NeurVPS-tmm17 (3.49/8.14, median 0.86°) but p=0.003 (>0.001 threshold), so not decisively different | Tests robustness beyond R-VPD's designed use case (images without RPs) |
| R-VPD 2025 | RPVP-Real / RPVP-Synth | Processing time (median) | R-VPD: 10.73s / 1.31s (CPU) vs. NeurVPS 0.86s/0.52s and GPVPD 0.52s/0.52s (both GPU, up to 400MB memory) | J-Linkage: 18.67s/0.75s (CPU) — R-VPD faster than the other CPU baseline, slower than GPU deep baselines |
| Yang et al. 2023 (contour, general shape recognition — not forgery) | Animal / MPEG-7 / ETH-80 | Accuracy | 92.70% / 99.26% / 98.32% | State-of-the-art among boundary/region shape descriptors at publication; robust to boundary noise up to σ=1.0 (<1% accuracy drop) |

## Implementation notes

- **Homogeneous coordinates / points at infinity**: implement the vanishing-point/line machinery in homogeneous coordinates throughout (standard for projective geometry) so that a vanishing point genuinely at infinity (perfectly parallel lines in the image, camera pointed exactly along the plane) does not require special-casing.
- **Ill-conditioned near-parallel intersections**: both A1 (Yao et al.'s explicit `<5°` discard rule) and A4 (R-VPD's multi-run re-initialization strategy) independently arrived at the same underlying problem — line-pair intersection is numerically unstable as the angle between lines shrinks. Apply a minimum-angle filter before any line-intersection computation, not just within the RANSAC loop.
- **Degenerate object pairs** for the height-ratio check (B): both objects must have distinct, unambiguous top/bottom image coordinates on the *same* reference plane; objects that are partially occluded, overlapping, or not actually resting on the assumed plane (e.g. one flying/floating, one on a raised surface) will silently produce a meaningless `β` — the source paper relies on manual/careful object-pair selection and does not give an automated sanity check for this; `(engineering recommendation: cross-check candidate pairs' bounding-box bottoms against a coarse ground-plane/horizon-line estimate before accepting them as valid pairs.)`
- **Eq. 8's σ=0.1α is a fixed, non-adaptive constant** derived from a single assumed 5%-outside-`[0.8α,1.2α]` design choice — this was not re-derived or validated against a large dataset in the extracted text (Table I's 8 examples are the paper's entire quantitative validation); treat `σ` as a starting point for this engine's own calibration, not a universally correct value.
- **No public reference code found in the extracted text** for Yao et al. R-VPD **does** provide public code and data: `http://vision.cse.psu.edu/data/data.shtml`.
- **Recommended Python libraries**:
  - `opencv-python` (`cv2.HoughLinesP`) for parallel-line extraction (A1); `cv2.goodFeaturesToTrack`/`cv2.SIFT_create` for keypoint/SIFT extraction (A4).
  - `scipy.optimize.least_squares` for the Levenberg-Marquardt vanishing-point refinement (A1).
  - `scipy.cluster.hierarchy` for hierarchical clustering (A4, Eq. 1–2).
  - `skimage.segmentation.slic` for superpixel generation (Pipeline D).
  - `scipy.stats.norm.cdf` directly implements `F(·)` in Eq. 8.
  - `numpy.linalg.eig` for the final eigenvalue-decomposition VP estimate (A4 step 6).

## Key references

- **yao2012.pdf** — H. Yao, S. Wang, Y. Zhao, X. Zhang, "Detecting Image Forgery Using Perspective Constraints," IEEE Signal Processing Letters, vol. 19, no. 3, pp. 123–126, March 2012. Source of: the full pinhole camera model and vanishing-line derivation (Eq. 1–4), the three vanishing-point estimation sub-methods, the height-ratio consistency test and its exact parameters (Eq. 7–8, σ=0.1α, T=0.5), and the complete Table I benchmark.
- **Recurrence-Based-Vanishing-Point-Detection.pdf** — S. Bharadwaj, R.T. Collins, Y. Liu, "Recurrence-based Vanishing Point Detection," WACV 2025. Source of: the full SIFT-clustering + Linearity/Angle/Scale scoring + weighted-RANSAC pipeline (Eq. 1–2 plus the scoring/RANSAC formulas), the RPVP-Synthetic/RPVP-Real/RPVP-Real-Exclusive/TMM17-Test benchmark tables, and the runtime/complexity analysis.
- **Multi-level-contour-combination-features-for-shape-recognition.pdf** — C. Yang, L. Fang, B. Fei, Q. Yu, H. Wei, "Multi-level contour combination features for shape recognition," Computer Vision and Image Understanding, 229 (2023) 103650. Source of: the two-level contour/Fisher-Vector shape-consistency method, repurposed (not validated in the source for this use) as a secondary forgery-detection signal. `[ML — excluded]` for its own SVM classification stage; training-free distance-based substitute recommended above.
- **A-survey-on-the-utilization-of-Superpixel-image-for-clustering-based-image-segmentation.pdf** — B. Sasmal, K.G. Dhal, Multimedia Tools and Applications, 82 (2023) 35493–35555. Source of: the SLIC-based superpixel preprocessing recommendation. Not re-read in full in this pass beyond confirming its recommended-method guidance (already extracted in the prior version of this file).
- **A-Novel-Approach-to-Image-Forgery-Detection-Techniques-in-Real-World-Applications.pdf** (and duplicate) — D. Patil, K. Patil, V. Narawade, Springer LNEE 925, 2022. `[ML]`, marginal — generic CNN copy-move classifier, no geometric reasoning; not re-read in this pass.
- **Clustering-based-Image-Text-Graph-Matching-for-Domain-Generalization.pdf**, **Digital-image-watermarking-using-deep-learning-A-survey.pdf**, **Machine-Learning-and-Visual-Perception.pdf** — off-topic/background only (vision-language domain generalization, active watermarking forensics, general ML textbook); not re-read in this pass.
