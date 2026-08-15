# Perspective / Geometric Consistency Forgery Detection

## Purpose

A camera performing a perspective projection of a 3-D scene imposes hard geometric
constraints on everything it images: parallel lines in the world (building edges,
road curbs, window mullions, shadows cast by a common light source) converge to
shared vanishing points in the 2-D image, and any two objects resting on the same
reference plane (e.g. the ground) have a height ratio that is fixed by that plane's
vanishing line, independent of where they sit in the frame. These constraints hold
for every object that was actually present in the scene when the photo was taken.
When an attacker splices an object or region from a different photo (a different
camera position, focal length, or scene) into a target image, the pasted content is
almost never resized/warped to obey the host image's vanishing points and reference-plane
height ratios — even careful manual retouching rarely gets the perspective, scale, and
occluding-contour shape of the spliced region exactly consistent with its surroundings.
Perspective/geometric-consistency analysis exploits this: it estimates the scene's
vanishing points/lines and the expected size and shape relationships between objects,
then flags regions whose apparent geometry statistically deviates from what the rest
of the image implies.

## Techniques found in the literature

### 1. Detecting Image Forgery Using Perspective Constraints
Heng Yao, Shuozhong Wang, Yan Zhao, Xinpeng Zhang. *IEEE Signal Processing Letters*,
vol. 19, no. 3, March 2012. File: `yao2012.pdf`. **(Core paper for this module.)**

- **Technique:** Constraint-based (not trace-based) forensics using a pinhole camera
  model. For a leveled, untilted camera, if two objects `B`, `B'` rest on the same
  reference plane, their real-world height ratio `β = z'_B / z_B` can be computed purely
  from the image coordinates of their top/bottom points and the vanishing line `v0` of
  the reference plane (Eq. 7 in the paper):
  `β ≈ (v'_B1 − v'_B2)(v0 − vB2) / (vB1 − vB2)(v0 − v'_B2)`.
  This requires *no camera intrinsics/EXIF* — only the vanishing line, which survives
  re-compression/resampling.
- **Vanishing-line/point estimation — three sub-methods:**
  1. **Parallel-line based:** extract straight edges (Hough transform / edge detection),
     group into parallel-line families, refine the intersection point with
     Levenberg–Marquardt maximum-likelihood estimation (discard line pairs with angle
     < 5°, since near-parallel intersections are ill-conditioned).
  2. **Reference-object based:** if two objects with a *known* height ratio η are visible
     on the plane, the vanishing line ordinate `v0` can be solved directly (Eq. 6).
  3. **Texture-orientation based:** locally-adaptive soft-voting on textured planes
     (e.g. road/floor texture) when no clean parallel lines exist.
- **Tampering score:** estimate ratio `β` from the (possibly forged) image, compare to
  an expected/reference ratio `α` (from general knowledge or a trusted reference pair
  with the same depth). Authentic images obey `(κ − α) ~ N(0, σ²)`. Consistency measure:
  `C = 2F(−|α − β|; 0, σ²)` where `F` is the Gaussian CDF, `C ∈ [0,1]`. Threshold
  `T = 0.5`: `C < T` ⇒ flagged as forged. Averaging several `β` measurements improves
  accuracy.
- **Results/tradeoffs:** correctly separated genuine vs. spliced object pairs in test
  images (Table I: C ≈ 0.88–0.98 for authentic pairs vs. C ≈ 0.02–0.21 for forged
  pairs), and remained effective under down-sampling and low-quality JPEG
  re-compression — a regime where most trace-based methods (resampling/JPEG-ghost
  artifacts) fail. Limitations: needs camera roll/tilt to be small (or a vertical
  vanishing point too, which is harder to estimate); needs at least one usable
  vanishing line, i.e. enough man-made structure or textured ground plane; detects
  *scale/height inconsistency*, not all forgery types (complementary to other cues).

### 2. Recurrence-based Vanishing Point Detection (R-VPD)
Skanda Bharadwaj, Robert T. Collins, Yanxi Liu. *WACV 2025*.
File: `Recurrence-Based-Vanishing-Point-Detection.pdf`.

- **Technique:** Unsupervised VP estimation that does **not** require explicit straight
  line segments — the classical failure mode of Hough/LSD-based VP methods and the
  weak spot of Yao et al.'s parallel-line method. Pipeline: SIFT feature extraction →
  hierarchical clustering of features into "visual words" (Eq. 1–2, Euclidean distance
  on 128-D SIFT descriptors) → discovery of *Recurring Patterns* ("things that recur,"
  e.g. evenly spaced fence posts, railings, tiled floors) → forward feature selection
  using three geometric-consistency scores per candidate feature set: **Linearity
  score** (S_L, perpendicular deviation from a fitted line), **Angle score** (S_A,
  uniformity of directional change across ordered triplets), **Scale score** (S_S,
  uniformity of relative size change, reflecting the projective scale-foreshortening
  law) — combined into `S_C = S_L × exp(S_A + S_S) / N²` → fits *implicit* lines
  through corresponding feature pairs across RP instances, merges with any *explicit*
  edge lines → **weighted RANSAC** (line weights `w_i = Σ e^{−θij}` favoring near-parallel
  line pairs, adaptively boosted/decayed for inliers/outliers) finds the VP as the
  intersection with the most inlier support; final VP via eigenvalue decomposition of
  the inlier line set.
- **Results:** Outperforms classical LSD and J-Linkage, and matches/exceeds supervised
  deep methods (NeurVPS, GPVPD) on real-world images containing recurring patterns but
  few/no explicit lines (median angle error 0.71–0.74° vs. NeurVPS-tmm17's 0.72° on
  their RPVP-Real-Exclusive benchmark). Runs on CPU (~1–11s/image; O(n²) in number of
  SIFT features), no GPU/training data needed, unlike the deep baselines (~0.5–0.9s but
  need a GPU and up to 400MB memory). **Limitation acknowledged by the authors:**
  designed for scenes with a single dominant VP (not Manhattan-world multi-VP scenes),
  and still degrades on images with neither explicit lines nor recurring patterns
  (e.g. tight close-ups, texture-only crops).

### 3. Multi-level contour combination features for shape recognition
Chengzhuan Yang, Lincong Fang, Benjie Fei, Qian Yu, Hui Wei. *Computer Vision and
Image Understanding*, 229 (2023) 103650. File:
`Multi-level-contour-combination-features-for-shape-recognition.pdf`.

- **Technique (contour/shape-consistency, supporting role):** splits an object's
  silhouette contour into two levels — dense sampling points (local, triangular
  curvature/tangential-angle/Fourier-descriptor features) and larger contour
  fragments (global shape) — encodes both with Fisher Vector encoding + power/L2
  normalization, concatenates, reduces dimensionality with PCA, classifies/matches
  with a linear SVM. Robust to noise (accuracy drops <1% up to Gaussian boundary noise
  σ=1.0) and to large intra-class/deformation variance.
- **Reported results:** 92.70% (Animal), 99.26% (MPEG-7), 98.32% (ETH-80) accuracy;
  state-of-the-art among boundary/region shape descriptors at time of publication.
- **Relevance:** not a forgery paper — a general shape-recognition/matching method.
  Useful in this engine as the **contour/shape-consistency sub-check**: once a
  candidate spliced object's silhouette is segmented (e.g. via superpixels, see below),
  its multi-level contour signature can be compared against the shape statistics of
  same-class objects elsewhere in the image or a reference shape library; a large
  Fisher-vector distance flags shape/scale implausibility that complements the
  vanishing-point height-ratio check. Marginally relevant on its own, useful as a
  reusable feature-encoding recipe.

### 4. A survey on the utilization of Superpixel image for clustering based image segmentation
Buddhadev Sasmal, Krishna Gopal Dhal. *Multimedia Tools and Applications* (2023)
82:35493–35555. File:
`A-survey-on-the-utilization-of-Superpixel-image-for-clustering-based-image-segmentation.pdf`.

- **Contribution (supporting/preprocessing role, not forgery-specific):** broad survey
  of superpixel generation algorithms (SLIC most widely used/recommended for its speed,
  memory efficiency, and strong boundary adherence; also SEEDS, watershed, Normalized
  Cuts, turbopixels, etc.) combined with partitional clustering (K-means, Fuzzy
  C-Means, GA/PSO-optimized clustering) for region-merging/segmentation. Extensive
  comparison table (63 pages, Table 1) of superpixel+clustering combinations across
  medical/plant/SAR imaging domains, with documented pros/cons per method (e.g. SLIC+FCM
  is fast but sensitive to cluster-count choice; watershed-based methods handle
  irregular boundaries better but are noise-sensitive).
- **Relevance:** purely a segmentation-methodology reference, no forgery content.
  Directly useful for this engine's **preprocessing stage** — generating superpixel
  regions that are then merged/clustered to isolate candidate spliced objects/regions
  before contour extraction (feeds paper #3) and before selecting object pairs for the
  vanishing-point height-ratio check (feeds paper #1).

### 5. A Novel Approach to Image Forgery Detection Techniques in Real World Applications
Dhanishtha Patil, Kajal Patil, Vaibhav Narawade. *Applications of AI and ML*, Springer
LNEE 925 (2022). File:
`A-Novel-Approach-to-Image-Forgery-Detection-Techniques-in-Real-World-Applications.pdf`
(a duplicate copy also exists as `... (1).pdf`).

- **Contribution:** end-to-end CNN classifiers (VGG-16, VGG-19, EfficientNetB0) trained
  on MICC-F2000/F220/CoMoFoD copy-move forgery datasets; EfficientNetB0 achieved best
  results (Accuracy 0.9825, F1 0.9861, MCC 0.9631).
- **Relevance:** **marginal** — this is a generic deep-learning image classifier for
  copy-move forgery, with no geometric/perspective/vanishing-point reasoning at all.
  It is only useful as a reminder that a learned end-to-end classifier is a plausible
  *complementary* branch to fuse with the geometric-consistency score in the overall
  engine's fusion layer, not as a technique for this specific module.

### 6. Clustering-based Image-Text Graph Matching for Domain Generalization
Nokyung Park et al. *ICPR 2024*, LNCS 15310. File:
`Clustering-based-Image-Text-Graph-Matching-for-Domain-Generalization.pdf`.

- **Relevance: marginal/off-topic.** This is a vision-language domain-generalization
  paper (aligning image-region graphs with text-description graphs for
  classification robustness across visual domains). It has no forgery-detection or
  geometry content; only tangential value is its graph-based clustering-and-matching
  formulation, which is a generic pattern (cluster local features → match structured
  graphs) but not directly reusable here. Not recommended as an implementation
  reference for this module.

### 7. Digital image watermarking using deep learning: A survey
Khalid M. Hosny, Amal Magdi, Osama ElKomy, Hanaa M. Hamza. *Computer Science Review*
53 (2024) 100662. File: `Digital-image-watermarking-using-deep-learning-A-survey.pdf`.

- **Relevance: marginal/off-topic.** Covers *active* forensics (embedding/extracting
  watermarks with GANs/DNNs for copyright and tamper-evidence), which is a
  fundamentally different paradigm from the *passive/blind* perspective-consistency
  analysis this module performs (passive methods assume no watermark or reference
  signal was embedded ahead of time). Included in the folder only as general forensics
  background; no techniques from it are used here.

### 8. Machine Learning and Visual Perception
Baochang Zhang, Ce Li, Nana Lin (book, De Gruyter). File:
`Machine-Learning-and-Visual-Perception.pdf`.

- **Relevance: marginal/background only.** A general ML textbook (decision trees,
  Bayesian learning, SVM, AdaBoost, PCA/subspace learning, compressed sensing, deep
  learning, reinforcement learning). No forgery or perspective-geometry content.
  Useful only as a math/algorithm reference for generic building blocks used elsewhere
  in this pipeline (e.g. SVM classification as used in paper #3, PCA dimensionality
  reduction), not as a source of forgery-detection technique.

## Recommended approach for this engine

1. **Preprocessing / region proposal (superpixel segmentation).** Run SLIC (or
   comparable) superpixel generation on the input image, then merge superpixels with a
   lightweight clustering step (K-means or Fuzzy C-Means on color/texture features, per
   the superpixel survey's recommendations) to obtain candidate object/region masks.
   This both (a) supplies clean silhouettes for contour-consistency comparison and
   (b) lets an operator/heuristic select pairs of "same reference plane" objects (e.g.
   people, vehicles, boxes standing on the ground) for the height-ratio check without
   requiring manual bounding boxes as in the original Yao et al. paper.

2. **Vanishing point / vanishing line estimation.** Primary path: parallel-line based
   estimation (Hough-transform edge/line extraction + Levenberg–Marquardt refinement)
   whenever the scene has enough man-made straight structure (building edges, road
   markings, furniture). Fallback path for scenes with little explicit line structure
   (close-ups, natural/organic scenes, heavily cropped images): use an R-VPD-style
   recurrence-based estimator — SIFT feature clustering into recurring visual-word
   groups, implicit line fitting scored by Linearity/Angle/Scale, weighted RANSAC over
   the union of implicit and explicit lines. This second path is the single most
   valuable addition from the literature reviewed here, since it directly plugs the
   "no explicit lines" gap that defeats the classical Yao et al. method and most
   Hough-based competitors.

3. **Cross-object / cross-region perspective consistency check.** For each candidate
   pair of same-plane objects (from step 1), compute the height ratio `β` from their
   image-space top/bottom coordinates and the estimated vanishing line (Yao et al. Eq.
   4/7). Compare against an expected ratio `α` — derived from a trusted reference
   object pair in the same image when available, or from class-prior height statistics
   (e.g. typical human height ranges) otherwise. Compute the consistency score
   `C = 2·Φ(−|α − β|; 0, σ²)` and flag pairs with `C` below a tuned threshold
   (paper used `T = 0.5`, `σ = 0.1α`). This should be run on *every* plausible
   same-plane object pair, not just one, and results aggregated (e.g. minimum C across
   pairs, or majority vote) since a single spliced object will be inconsistent with
   several others in the scene.

4. **Contour/shape-consistency check (secondary signal).** For each segmented
   candidate region, extract a multi-level contour feature (sampling-point +
   contour-fragment, Fisher-vector encoded, per paper #3) and compare against
   class-conditional shape priors or against other same-class objects detected
   elsewhere in the image. Large deviation adds supporting evidence of
   splicing/resizing, especially useful when the vanishing-point check is ambiguous
   (e.g. object partially occluded, ambiguous reference plane).

5. **Tampering score for the fusion layer.** Emit a single scalar (or small feature
   vector) per image: primarily `1 − C_min` (worst-case inconsistency across evaluated
   object pairs) from step 3, optionally combined with the normalized contour-feature
   distance from step 4 (e.g. weighted sum, or feed both as separate features into the
   fusion classifier/DSP that combines this module with the engine's other forgery
   cues — resampling, JPEG-ghost, illumination-consistency, CNN-based classifiers such
   as paper #5). Keep the raw `(α, β, C)` values and VP confidence as auxiliary
   metadata for explainability (this module can point to *which* object pair and *which*
   vanishing line drove the flag — a useful property for a forensic report).

6. **Known limitations to document/handle explicitly:**
   - Requires sufficient geometric structure in the scene: either explicit parallel
     lines (parallel-line method) or repeating/recurring structure (R-VPD fallback).
     Texture-only, close-up, or single-object-with-no-ground-plane images will produce
     low-confidence or unusable VP estimates — the module should detect this (e.g. few
     RANSAC inliers, low S_C scores) and abstain / down-weight its contribution to the
     fusion score rather than emit a false positive/negative.
   - Assumes small camera roll/tilt for the simplified height-ratio formula; a full
     tilt-compensated version needs a second (vertical) vanishing point, which is
     harder to estimate reliably — flag high-tilt images as reduced-confidence.
   - Single dominant-VP assumption (both Yao et al. and R-VPD): indoor/urban
     Manhattan-world scenes with multiple orthogonal VPs need per-plane VP estimation,
     not a single global VP.
   - The height-ratio check only catches scale/perspective-inconsistent splices; it is
     blind to same-scale, same-plane copy-move forgeries or geometrically-consistent
     composites — hence its role as one signal among several in the fusion layer, not
     a standalone detector.

## References

- `yao2012.pdf` — H. Yao, S. Wang, Y. Zhao, X. Zhang, "Detecting Image Forgery Using
  Perspective Constraints," *IEEE Signal Processing Letters*, vol. 19, no. 3, pp.
  123–126, March 2012.
- `Recurrence-Based-Vanishing-Point-Detection.pdf` — S. Bharadwaj, R. T. Collins, Y.
  Liu, "Recurrence-based Vanishing Point Detection," *WACV 2025*.
- `Multi-level-contour-combination-features-for-shape-recognition.pdf` — C. Yang, L.
  Fang, B. Fei, Q. Yu, H. Wei, "Multi-level contour combination features for shape
  recognition," *Computer Vision and Image Understanding*, 229 (2023) 103650.
- `A-survey-on-the-utilization-of-Superpixel-image-for-clustering-based-image-segmentation.pdf`
  — B. Sasmal, K. G. Dhal, "A survey on the utilization of Superpixel image for
  clustering based image segmentation," *Multimedia Tools and Applications*, 82
  (2023) 35493–35555.
- `A-Novel-Approach-to-Image-Forgery-Detection-Techniques-in-Real-World-Applications.pdf`
  (and duplicate `... (1).pdf`) — D. Patil, K. Patil, V. Narawade, "A Novel Approach
  to Image Forgery Detection Techniques in Real World Applications," in *Applications
  of Artificial Intelligence and Machine Learning*, Springer LNEE vol. 925, 2022.
- `Clustering-based-Image-Text-Graph-Matching-for-Domain-Generalization.pdf` — N.
  Park, D. Chae, J. Shim, S. Kim, E.-S. Kim, J. Kim, "Clustering-based Image-Text
  Graph Matching for Domain Generalization," *ICPR 2024*, LNCS 15310.
- `Digital-image-watermarking-using-deep-learning-A-survey.pdf` — K. M. Hosny, A.
  Magdi, O. ElKomy, H. M. Hamza, "Digital image watermarking using deep learning: A
  survey," *Computer Science Review*, 53 (2024) 100662.
- `Machine-Learning-and-Visual-Perception.pdf` — B. Zhang, C. Li, N. Lin, *Machine
  Learning and Visual Perception* (book), De Gruyter.
