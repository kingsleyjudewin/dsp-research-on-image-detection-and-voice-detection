# Lighting / Illumination Inconsistency Forgery Detection

## Core mathematical principle

**Underlying concept.** A camera image is formed by light transport: for a Lambertian surface point with normal `n` and albedo `ρ`, illuminated by a scene light field `L`, observed intensity is (standard image-formation model, not itself derived in either source paper in this folder) `I(n) = ρ · L · Φ(n)` for some shading function `Φ` of the surface normal — commonly approximated via low-order Spherical Harmonics. Every genuinely-photographed region of a single image shares **one** consistent light transport environment (subject to multiple light sources, but a *fixed*, physically-coherent one); composited or GAN-synthesized content frequently fails to reproduce this consistency because the generator/editor does not solve the inverse-lighting problem for the inserted region.

**Property exploited.** Inconsistency in local shading, gradient structure, or (in principle) light-direction estimates between the manipulated region and its surroundings is evidence of compositing. **This corpus does not contain a paper that directly measures or models this in a physically-interpretable way** — see Corpus Gap below. What the corpus *does* contain is (a) one **general, training-free gradient-magnitude heuristic** (extracted from a different module's paper) that is only loosely connected to "lighting," and (b) one **learned, non-physical, non-lighting-specific artifact-localization network** that merely *motivates itself* by observing that one particular DeepFake generator (FaceSwap) leaves an "unnatural light around the eyes and nose" artifact, without ever modeling light transport at all.

## Input requirements

- **Format**: any RGB image; the one deep-learning paper in this module operates specifically on **face crops** (its backbones are face-forgery classifiers), but the Sobel-gradient cue is domain-agnostic and needs no face detection.
- **Preprocessing**: grayscale conversion for the Sobel-gradient cue (`double(gray_img)` per the source's own MATLAB code); for a genuine photometric approach (Pipeline B, design-only), face/landmark detection and a 3D face-model fit would be required — `(not available in this corpus; see Corpus Gap)`.
- **Reliable / unreliable conditions**: cannot be meaningfully characterized for this module from the current corpus — neither implementable technique here (the gradient heuristic or the attention network) has a stated, quantified operating envelope specific to *lighting* consistency (as opposed to general image-forgery detection). Treat the entire module as **low-confidence** until better source material is added (see Corpus Gap).

## Step-by-step algorithm

### A. Sobel-gradient magnitude heuristic (Rao, Ghanekar, Chitnis, Dawkhar & Mishra 2025 — extracted from the `fourier transform ghost transform` module's PCA-fusion paper) — the only training-free, implementable technique currently in this corpus

1. Convert the image to grayscale: `gray_img`.
2. Compute the image gradient (the paper's own MATLAB code, transcribed exactly):
   ```matlab
   [Gx, Gy] = gradient(double(gray_img));
   gradient_mag = sqrt(Gx.^2 + Gy.^2);
   max_grad = max(gradient_mag(:));
   ```
3. **Decision logic as stated in the source**: "if large gradients or multiple light directions are detected, this may suggest manipulation." **No threshold value, no formal multi-direction-detection procedure, and no quantitative validation of this specific module is given anywhere in the source paper** — this is a one-paragraph qualitative description accompanying a code snippet, not a validated detector. There is no experiment isolating this module's contribution to the paper's overall (also unvalidated — see that module's file) PCA fusion score.
4. **What "multiple light directions" would even mean operationally is left completely unspecified** — the paper does not describe how a single scalar `max_grad` (a magnitude, with no directional/angular component) could detect "multiple light directions." This is an internal inconsistency in the source: the stated decision rule requires directional information the given code never computes. `(Corpus ambiguity — flagged explicitly, not resolved.)`

**Honest assessment**: this is at most a generic **edge/gradient-strength** feature (high gradient magnitude correlates with sharp edges, splice boundaries, or strong shading discontinuities in general — not specifically lighting-direction inconsistency). It could be repurposed as a weak, generic "structural discontinuity" signal, but should **not** be presented to the fusion layer as validated lighting-consistency evidence.

### B. Photometric consistency check via Spherical-Harmonics shading — design specification only, NOT corpus-backed

This pipeline is **entirely engineering synthesis** — no paper in this repository provides these formulas, parameters, or any validation. It is included because it is the technically correct approach to this problem, so that the module has a target to build toward, but every step below is marked accordingly.

1. `(not specified in the corpus — engineering recommendation)` Detect face region(s)/landmarks (e.g. a standard facial-landmark detector) and fit a coarse 3D morphable face model to obtain per-pixel surface normals `n(x,y)` within each detected face/object region.
2. `(not specified in the corpus)` Estimate low-order Spherical Harmonics illumination coefficients `L_{lm}` per region by least-squares fitting the standard Lambertian-SH shading model `I(n) ≈ ρ · Σ_{l,m} L_{lm}·Y_{lm}(n)` against observed pixel intensities in diffuse (non-specular, non-shadowed) mask regions.
3. `(not specified in the corpus)` For images/frames containing multiple regions expected to share one illumination environment (e.g. two faces, or a face and a background object), compare the recovered SH coefficient vectors via cosine similarity or L2 distance in SH-coefficient space; a single physical scene should yield near-identical low-frequency lighting across regions.
4. `(not specified in the corpus)` Secondary photometric sub-checks in the same spirit: specular-highlight position consistency (highlight in each eye should lie along the reflection direction of one common light source) and cast-shadow direction/softness consistency at silhouette boundaries.
5. **Corpus gap**: none of this pipeline's formulas, thresholds, or empirical validation exist in the current research folder. The classic prior-art this design is based on (Johnson & Farid's illuminant-direction inconsistency work; O'Brien & Farid's specular-highlight consistency work) is **not present in this corpus** and should be sourced before this pipeline is treated as anything more than a target design.

### C. Dual-attention DeepFake artifact localization (Li, Wang, Wang, Zhao, Wang 2021) `[ML — excluded from the no-ML engine]`

Documented in full because it is the only substantial technical content in this module's source PDF, but it is **not a lighting-modeling technique at all** — see the explicit relevance caveat below.

1. **Channel attention**: for a convolutional feature map `F ∈ ℝ^{C×H×W}`, global-average-pool and global-max-pool to two length-`C` vectors, pass each through a **shared** bottleneck MLP (compression ratio **8**), sum, and apply sigmoid:
   ```
   M_c(F) = σ( MLP(GAP(F)) + MLP(GMP(F)) )          [Eq. 1]
   F' = M_c(F) ⊗ F
   ```
2. **Spatial attention**: channel-pool `F'` via max and average across the channel dimension into two `H×W` maps, concatenate, apply a **7×7 convolution** and sigmoid:
   ```
   M_s(F) = σ( f([F_avg^S, F_max^S]) )          [Eq. 2]
   F'' = M_s(F') ⊗ F'
   ```
3. **Placement**: channel attention is computed first (so it does not interfere with the subsequent spatial-attention computation), then spatial attention is applied to the channel-attended map. Inserted **after the first convolution of block 1** (finest-grained, used for localization read-out) and **after the pooling layers of the remaining blocks** for MesoNet; analogous insertion points specified for Meso-Inception (after blocks 1 and 4), Xception (between blocks 1 and 2), and VGG-19/EfficientNet-B0 (last block, with an added batch-norm layer per block to control overfitting).
4. **Training**: end-to-end with only real/fake **image-level** labels — no manipulation ground-truth masks required (the paper's stated advantage over the supervised-mask baseline it compares against, Dang et al. 2020's "FFD"). Optimizer: Adam, `β1=0.9, β2=0.999`, fixed learning rate `0.001`.
5. **Localization read-out**: the spatial-attention map `M_s` from the **first** block, upsampled to image resolution, serves as the pixel-level artifact-localization heatmap.

**What this network does and does not claim about lighting** (precise, to prevent downstream misreading): the paper's *only* connection to illumination is a single motivating observation in its introduction (Fig. 1) that FaceSwap-synthesized images show "unnatural light around the eyes and nose" as one example of a visible artifact type among several (others: blending boundaries for Deepfakes, mouth-region artifacts for Face2Face, no observable artifacts at all for StyleGAN). **No shading model, light-direction estimate, shadow geometry, or reflectance computation appears anywhere in the method.** The dual-attention mechanism is a generic, class-agnostic artifact-localizer trained end-to-end on whatever visual cues best separate real from fake in the training set — it will pick up compression artifacts, blending seams, and texture irregularities just as readily as anything lighting-related, and there is no mechanism to isolate a "lighting-specific" signal from its output.

## Output

- **Pipeline A**: raw `max_grad` scalar (unbounded, image- and content-dependent — no normalization or calibration given). `(Not specified in corpus.)` Given the internal inconsistency noted above (a magnitude-only feature cannot detect "multiple directions"), do not treat this as more than a generic edge-strength auxiliary signal; if used, normalize per-image (e.g. divide by the image's own median gradient magnitude) rather than using an absolute cutoff, since the source gives no calibration at all.
- **Pipeline B**: `(not specified in corpus)` — design target only; would output a `[0,1]` cross-region SH-inconsistency score, e.g. `1 − cosine_similarity(SH_A, SH_B)`.
- **Pipeline C** (`[ML]`, documented not implemented): binary real/fake classification with an associated spatial-attention heatmap; PBCA and IINC are the paper's stated localization-quality metrics (Pixel-wise Binary Classification Accuracy — higher is better; Inverse Intersection Non-Containment — lower is better), but their formal formulas are **not present in the extracted pages of this paper** — treat their exact definitions as `(not verified in this pass — likely defined by citation to prior localization-metric literature)`.

## Key findings from papers

**Manipulation types detected best**: Pipeline C detects a broad range of face-forgery types (entire-face GAN synthesis and partial manipulation: expression swap, identity swap, attribute editing) at the *whole-image classification* level well; its *localization* quality is stated to be best on datasets with clear, spatially-compact artifacts and notably worse on heavily-compressed content (see below). Neither Pipeline A nor Pipeline B has any validated "detects X best" claim in this corpus.

**Documented failure cases / limitations**:
- Pipeline C's own results (Table 1) show **AUC and accuracy drop substantially on the FaceForensics++ Low-Quality (LQ) subset** (Accuracy 0.8401, AUC 0.9217 — the two lowest of all six datasets tested) and on **Celeb-DF (v2)** (Accuracy 0.8975, AUC 0.9673) — the paper attributes the FF-LQ drop to heavy compression causing image distortion severe enough that "it is even hard for humans to distinguish," which also drives up the false-negative rate specifically.
- Pipeline C's cross-dataset/cross-generator generalization is **not evaluated to the level of rigor needed to trust it as a general "lighting artifact" detector** — accuracy gains are backbone- and dataset-specific (3.50% MesoNet down to 0.89% EfficientNet average gain), and the paper's own introduction notes advanced GANs (StyleGAN) leave **no identifiable artifacts at all**, meaning this entire technique family — lighting-related or not — has a fundamental blind spot against sufficiently advanced generators.
- Pipeline A has no documented failure cases because it has no documented successes either — it was never evaluated as a standalone detector in its source paper.

**Benchmark table** (Pipeline C, Table 1, MesoNet + dual attention, exact values from source):

| Dataset | TPR | TNR | FPR | FNR | Accuracy | AUC | PBCA | IINC |
|---|---|---|---|---|---|---|---|---|
| FF-RAW | 0.9839 | 1.0 | 0.0 | 0.0164 | 0.9918 | 0.9997 | 0.7168 | 0.6006 |
| FF-HQ | 0.9493 | 0.9518 | 0.0482 | 0.0507 | 0.9505 | 0.9925 | 0.7082 | 0.6404 |
| FF-LQ | 0.8492 | 0.7307 | 0.1693 | 0.1508 | 0.8401 | 0.9217 | 0.7113 | 0.6319 |
| Celeb-DF (v2) | 0.9469 | 0.8658 | 0.1811 | 0.0378 | 0.8975 | 0.9673 | — | — |
| UADFV | 0.9528 | 0.9960 | 0.0042 | 0.0451 | 0.9744 | 0.9792 | 0.8404 | 0.6869 |
| DT-HQ | 1.0 | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | — | — |
| DT-LQ | 1.0 | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | — | — |
| DeeperF | 0.9875 | 0.9815 | 0.0184 | 0.0126 | 0.9845 | 0.9982 | 0.7094 | 0.6211 |
| StyleGAN | 0.9060 | 0.9385 | 0.0636 | 0.0910 | 0.9223 | 0.9748 | — | — |

(PBCA/IINC only reported where ground-truth manipulation masks exist — FF and DeeperF variants and UADFV.) Average accuracy improvement over unmodified backbones: **3.50% (MesoNet), 2.56% (Meso-Inception), 1.64% (VGG-19), 1.36% (Xception), 0.89% (EfficientNet)**, averaged across all six datasets.

## Corpus gap

This module has the **thinnest evidentiary base of all nine** in this engine:

- **Zero papers** in this folder (or elsewhere in the corpus, as far as this review found) perform actual **photometric lighting analysis** — no light-direction estimation, no shading model, no shadow-consistency geometry, no specular-highlight analysis.
- The **one implementable no-ML technique** (Sobel gradient magnitude) is a two-sentence, unvalidated heuristic borrowed from a different module's fusion-pattern paper, with an internally inconsistent decision rule (see Pipeline A step 3).
- The **one substantial paper** in this module's own folder is a general-purpose learned artifact localizer that is explicitly *not* a lighting technique and says so only in passing.
- **Recommendation**: source at least 2–3 dedicated illumination-forensics papers (e.g. the classic Johnson & Farid illuminant-direction-inconsistency line of work, and specular-highlight consistency work such as O'Brien & Farid) before treating this module's output as more than a placeholder. Until then, **this module should carry the lowest reliability weight of the nine detectors in the fusion layer** — its output is not backed by validated technique, and a near-zero or explicitly "abstain" weight is more honest than presenting an unvalidated gradient heuristic as equivalent evidence to, e.g., the CFA or Benford modules' well-validated statistics.

## Implementation notes

- If Pipeline A is implemented anyway (as a cheap, generic auxiliary signal — not as "lighting evidence"), normalize `max_grad` relative to the image's own gradient-magnitude distribution (e.g., a percentile rank) rather than using any absolute threshold, since none is given in the source.
- Pipeline C's attention mechanism, if ever implemented despite the no-ML constraint, is architecture-agnostic and cheap to bolt onto any existing CNN classifier already present elsewhere in a full (non-restricted) version of this engine — but per this engine's constraints, it is excluded.
- **Recommended Python libraries** (Pipeline A only, given B is unspecified and C is excluded): `numpy.gradient` is the direct Python equivalent of MATLAB's `gradient()` function used in the source code.

## Key references

- **Exposing-DeepFakes-via-Localizing-the-Manipulated-Artifacts.pdf** — W. Li, Q. Wang, R. Wang, L. Zhao, L. Wang, "Exposing DeepFakes via Localizing the Manipulated Artifacts," ICICS 2021, LNCS 12919, pp. 3–20, Springer. DOI: 10.1007/978-3-030-88052-1_1. Source of: the complete dual-attention (channel + spatial) formulas (Eq. 1–2), the training/insertion configuration, and the full Table 1 benchmark. `[ML — excluded from the no-ML engine]`. **Not a lighting-modeling paper** — see explicit relevance caveat in Pipeline C.
- **Image-Tampering-Detection-Using-Multi-Feature-Scoring-and-PCA-Based-Classification.pdf** (folder: `fourier transform ghost transform`) — A. Rao, A. Ghanekar, D. Chitnis, M. Dawkhar, D. Mishra, CISCON 2025. Source of: the Sobel-gradient MATLAB code (Pipeline A) — the module's only training-free, implementable element. Also carries **no quantitative validation** for this specific sub-module (see that module's `SKILL.md` for the same caveat in full).
