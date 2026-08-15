# CFA / Demosaicing Artifact Forgery Detection

## Purpose

Almost every digital camera sensor captures only one color value per pixel, through
a Color Filter Array (CFA) — most commonly the Bayer pattern (RGGB / GRBG / GBRG /
BGGR). The missing two color channels at every pixel are reconstructed by an
in-camera demosaicing (interpolation) algorithm. This interpolation is a *linear or
near-linear predictive process applied periodically* over the sensor lattice, so it
leaves a statistically regular, spatially periodic correlation pattern between
neighboring pixels (interpolated pixels are locally predictable from their
neighbors; captured/acquired pixels are not, or are predictable in a different,
out-of-phase way). Any local image manipulation — splicing in content from another
image, copy-moving a region, smoothing, resampling/rescaling, or re-rendering
pixels — either destroys this periodic CFA correlation (native captured pixels get
overwritten/blended, destroying the pattern) or introduces a *foreign, misaligned*
periodicity (the pasted content carries its own, differently-phased or absent CFA
signature). Because the CFA/demosaicing fingerprint is normally uniform and
invisible across an untouched photo, any region where it is locally absent,
weakened, or inconsistent with the rest of the image is strong evidence of
tampering — and, unlike many global statistical fingerprints, this artifact can be
measured on very small blocks (down to 2x2), enabling fine-grained pixel/block-level
forgery localization rather than only whole-image authenticity classification.

## Techniques found in the literature

### 1. Ferrara, Bianchi, De Rosa, Piva — "Image Forgery Localization via Fine-Grained
Analysis of CFA Artifacts" (IEEE TIFS, 2012)
File: `ferrara2012.pdf` (duplicate copy: `ferrara2012 (1).pdf` — identical content,
can be deleted/ignored)

- **Core idea**: Model the CFA demosaicing residual (prediction error) as having
  different local variance depending on whether a pixel was directly acquired by
  the sensor or interpolated. For a Bayer pattern, applying a linear predictor
  to the green channel gives a *quincunx* lattice of acquired vs. interpolated
  green samples; when CFA interpolation is genuinely present, the variance of the
  prediction error is much lower at the positions that match the interpolation
  kernel (interpolated positions) than at acquired positions.
- **Feature**: `L(k,l) = log[GM_A(k,l) / GM_I(k,l)]`, the log-ratio of the
  geometric mean of the locally-weighted prediction-error variance at acquired (A)
  vs. interpolated (I) pixel positions, computed on blocks as small as 2x2.
- **Statistical model**: Under hypothesis M1 (CFA present) L is Gaussian with
  positive mean; under M2 (CFA absent/destroyed, i.e. tampered) L is Gaussian with
  zero mean. Parameters are estimated per-image via Expectation-Maximization (EM)
  on a 2-component Gaussian Mixture Model.
- **Output**: Per-block posterior probability `Pr{M1 | L(k,l)}` via Bayes' rule,
  producing a fine-grained (2x2 or larger, cumulated to e.g. 8x8) **tampering
  probability / likelihood map**. Log-likelihood maps can be smoothed with a mean
  or median filter to reduce noise/false positives.
- **Results**: AUC ~0.97–0.99 on uncompressed TIFF images (ideal and 4-camera
  realistic scenarios); clearly outperforms prior Fourier-based (Gallagher & Chen)
  and sensor-noise-based (Dirik & Memon) detectors at 8x8 block resolution.
  Performance degrades sharply under JPEG compression below ~95% quality (low-pass
  filtering removes the high-frequency CFA artifact); the method struggles on
  flat/saturated regions and very sharp edges (prediction error too small either
  way to be diagnostic); not directly applicable to non-Bayer sensors (e.g. super
  CCD). This is the seminal, most directly applicable paper for this engine.

### 2. Bammey, Morel, Grompone von Gioi — "Automatic Detection of Demosaicing Image
Artifacts and its Use in Tampering Detection" (IEEE MIPR, 2018)
File: `bammey2018.pdf`

- **Core idea**: Rather than assuming a fixed demosaicing kernel, estimate a
  *linear filter per CFA grid hypothesis* (8 filters: one per missing-channel /
  sampled-color pairing) via least-squares, independently for each of the 4
  possible Bayer grid phase positions. The grid position whose estimated filters
  best reconstruct the image (lowest residual) is the correct CFA configuration.
- **Statistical rigor — *a contrario* / NFA framework**: Instead of a bare
  threshold, the method computes a Number of False Alarms (NFA) for each detected
  CFA grid position using a binomial model of "how many blocks would vote for this
  grid position by pure chance." A detection is only declared meaningful if its
  NFA is below a chosen false-alarm rate (e.g. 0.001), giving statistically
  guaranteed low false-alarm forgery detections — a stronger guarantee than most
  other CFA methods, which only give a raw likelihood/heatmap that must be
  visually judged.
- **Forgery detection**: Image is divided into blocks; each block votes for the
  CFA grid position with smallest residual. A window is flagged forged if it
  contains a significant (low-NFA) grid vote that disagrees with the image's
  globally dominant CFA position.
- **Results**: Detects small forged regions (32x32 blocks) with NFA as low as
  10^-27; robust up to JPEG quality ~90 (67% detection at Q90, NFA 10^-4);
  correctly identifies forged regions in real tampered images with NFA < 10^-10.
  Explicitly warns that with only 1/4 probability, a copy-paste could by chance
  preserve the same CFA grid phase and be missed.

### 3. Jeon, Shin, Eom — "Estimation of Bayer CFA Pattern Configuration Based on
Singular Value Decomposition" (EURASIP J. Image and Video Processing, 2017)
File: `jeon2017.pdf`

- **Core idea**: A *precursor* step, not itself a tampering detector — accurately
  identifying which of the 4 Bayer CFA phase configurations (RGGB/GRBG/GBRG/BGGR)
  a given image actually used. Most CFA-tampering detectors (including #1 and #2
  above) assume the CFA pattern is already known; this paper shows that assumption
  is often wrong for anything beyond simple bilinear demosaicing.
- **Method**: Build "color difference" blocks (R−G, B−G) which are nearly constant
  in flat regions. Apply Singular Value Decomposition (SVD) to each candidate
  color-difference block; large singular values encode low-frequency background
  content (irrelevant), small (truncated, kept) singular values encode
  high-frequency texture/edge content where the acquired-vs-interpolated variance
  imbalance actually shows up. Sum the truncated singular values for diagonal
  pairs of the 2x2 CFA cell; the pairing/position with the largest asymmetry
  identifies the true R/G/B layout.
- **Results**: 91–98% identification accuracy across 8 demosaicing algorithms
  (bilinear through complex AMaZE/LMMSE/HPHD), consistently beating a prior
  max/min-counting method, and ~5x faster (0.035s vs 0.176s per 256x256 block).
  Degrades under JPEG compression (both methods drop to ~20–30% accuracy) — same
  weakness as other CFA techniques.
- **Relevance**: Useful as a pre-processing module — determine or verify the CFA
  phase before running the Ferrara-style block variance detector, especially if
  source camera metadata is missing or the image may have been re-saved/re-sampled
  in a way that shifted the grid.

### 4. Singh, Singh — "Digital Image Forensic Approach Based on the Second-Order
Statistical Analysis of CFA Artifacts" (Forensic Science Int'l: Digital
Investigation, 2020)
File: `singh2020.pdf`

- **Core idea**: Instead of directly modeling variance of the prediction error
  (first/second-order per-pixel statistic as in Ferrara), re-interpolate the image
  under all 4 candidate Bayer CFA configurations, take the pixel-wise squared
  difference between the original and each re-interpolated version, and select the
  difference image with the *maximum sum* as the "target difference image"
  (the one that best reveals the true, disturbed CFA correlation). This puts the
  CFA-inconsistency evidence into a **difference domain**, which is claimed to be
  less dependent on raw image content than the spatial domain.
- **Higher-order statistical feature**: Compute block DCT (8x8) of the target
  difference image, then build **Markov Transition Probability Matrices (MTPM)**
  — both intra-block (horizontal/vertical/diagonal adjacency of DCT coefficients
  within a block) and inter-block (same coefficient position across neighboring
  blocks) — to capture how CFA artifacts disturb the correlation structure of
  neighboring DCT coefficients. Concatenating rows of all intra/inter MTPMs yields
  a 648-dimensional feature vector.
- **Classifier**: An SVM is trained on these MTPM feature vectors to discriminate
  forged vs. authentic blocks/images (a learned discriminator on top of a
  hand-engineered CFA-sensitive feature, unlike the purely statistical/Bayesian
  approaches in #1–#2).
  Additionally supports a scalar/localization mode: threshold-based, morphology-
  cleaned forged-region localization directly from the selected difference image.
- **Results**: Outperforms Ferrara, Dirik & Memon, Gallagher & Chen, and other
  baselines in ROC/AUC and minimum decision error (Pe = 0.0751 vs. 0.10–0.35 for
  prior methods); lower average computation time (157s vs 160–205s per test in
  their benchmark, still fairly slow — this is a heavier, learning-based pipeline,
  not real-time); degrades under JPEG re-compression like all CFA methods; noted
  to fail on Foveon X3 sensors (no CFA) and cameras with super CCD.

### Marginally relevant / background-only papers

- **`mayer2018.pdf`** — Mayer & Stamm, "Accurate and Efficient Image Forgery
  Detection Using Lateral Chromatic Aberration" (IEEE TIFS, 2018). Not a CFA/
  demosaicing method — it exploits a *lens optical* artifact (wavelength-dependent
  focal shift causing R/G/B channel misalignment near image edges), not the sensor
  CFA interpolation pattern. Still directly relevant to this engine's broader
  "chromatic/optical artifact" forgery family: models local vs. global lateral
  chromatic aberration (LCA) displacement inconsistency as a hypothesis test
  (Gaussian noise vs. Gaussian-with-forgery-offset), derives an optimal
  Mahalanobis-distance detection statistic, and proposes an efficient "diamond
  search" block-matching algorithm (2 orders of magnitude faster than exhaustive
  search) for estimating local LCA. Complementary signal to CFA analysis — could
  be fused as an additional detector in the same engine, but is a distinct
  physical mechanism (lens optics, not sensor demosaicing) and requires visible
  color fringing near high-contrast edges, so it is weak/unusable in image centers
  or on cameras with strong chromatic-aberration correction.

- **`islam2020.pdf`** — Islam, Karmakar, Kamruzzaman, Murshed, "A Robust Forgery
  Detection Method for Copy-Move and Splicing Attacks in Images" (Electronics/MDPI,
  2020). General DCT + Local Binary Pattern (LBP) + SVM forgery detector, not
  CFA-specific. Relevant only as a general splicing/copy-move baseline and as an
  example of combining block-DCT with a texture descriptor (LBP) and an SVM
  classifier — architecturally similar in spirit to Singh & Singh's MTPM+SVM
  pipeline (#4) but does not touch demosaicing artifacts at all. Marginal
  relevance; useful mainly for fusion-layer inspiration (a second, independent
  spatial-domain detector to combine with CFA evidence).

- **`An-Unpaired-Learning-Based-Method-for-Image-Despeckling.pdf`** — Zafari &
  Jalali, despeckling for coherent imaging (SAR / digital holography), 2025.
  Off-topic for CFA forensics: it addresses multiplicative *speckle noise* removal
  via a Bayesian MAP framework (BD-QMAP) and is evaluated purely on PSNR/SSIM
  reconstruction quality, with no forgery/tampering angle and no CFA/demosaicing
  content. Only tangential value: illustrates a Bayesian "reconstruct and compare
  to observation" methodology structurally similar to CFA re-interpolation
  residual analysis, but not usable in this module directly.

- **`Image-Interpolation-Using-Non-adaptive-Scaling-Algorithms-for-Multimedia-
  Applications-A-Survey.pdf`** — Neetha, Moses, Selvathi (Springer, 2021). A
  broad survey of *image up-scaling/resampling* interpolation algorithms (linear,
  cubic-spline, convolution-based), compared by PSNR/SSIM. Not about CFA
  demosaicing or forgery — this is about post-capture geometric resampling
  (zoom/resize), a related but distinct interpolation artifact (resampling
  periodicity, exploited by different forensic techniques such as Popescu &
  Farid's resampling detector, referenced in several of the papers above).
  Useful only as background on interpolation-kernel behavior (e.g., which kernels
  are more/less linear, hence easier/harder to model with an EM/least-squares
  predictor as in papers #1–#3) and as a reminder that resizing/rescaling a
  spliced region is itself a known anti-forensic step that destroys CFA evidence.

- **`A-new-robust-training-free-proactive-deepfake-detection-scheme-using-
  watermarking-and-identity-aware-hashing.pdf`** — Lai et al. (Expert Systems
  With Applications, 2026). Proactive face-deepfake detection via embedded
  watermarking and identity-aware hashing. Entirely unrelated to CFA/demosaicing
  or passive image forensics — it is a *proactive* (watermark-before-publish)
  defense against GAN/diffusion-based face swaps, not a passive artifact analysis
  technique. No reusable technique for this module; included in the folder likely
  because it is filed under the broader "image/video forgery detection" research
  umbrella rather than CFA specifically. Skip for this engine's CFA module.

## Recommended approach for this engine

**Primary algorithm**: Implement the Ferrara et al. (2012) block-level CFA
likelihood-map method as the core detector — it is the most directly applicable,
well-validated, and computationally tractable (2x2 to 8x8 block resolution, EM
parameter estimation converges in a few hundred iterations per image):

1. **CFA pattern verification** (optional but recommended): before running the
   detector, use the SVD-based method of Jeon et al. (#3) to confirm/estimate the
   Bayer phase (RGGB/GRBG/GBRG/BGGR) actually used, rather than assuming a fixed
   pattern from EXIF/camera model — this avoids silent failure on images whose CFA
   metadata is stripped, wrong, or from an unfamiliar sensor.
2. **Prediction-error feature extraction**: Extract the green channel (upsampled
   2x in a Bayer array, so it has equal counts of acquired/interpolated samples).
   Apply a fixed linear predictor (bilinear is a robust default per Ferrara's
   results — it is more stable than higher-order predictors when the true
   in-camera demosaicing algorithm is unknown, even though matching the true
   kernel gives a small further gain). Compute the locally-weighted (Gaussian
   window, e.g. 5x5) prediction-error variance separately for acquired-lattice and
   interpolated-lattice pixel positions.
3. **Feature and probability map**: Compute `L(k,l) = log[GM_A / GM_I]` per
   B x B block (start at 2x2, cumulate to 8x8 for stability — 8x8 gave the best
   AUC/localization tradeoff in the literature). Fit a 2-component Gaussian
   mixture (`mu1>0, sigma1` for CFA-present; `mu2=0, sigma2` for CFA-absent) via
   EM, then convert to a per-block posterior probability of tampering via Bayes'
   rule. This produces a **tampering probability heatmap** directly, at native
   image resolution down to a few pixels per cell.
4. **Denoising the map**: Apply a 5x5 median filter to the log-likelihood map
   before thresholding — median filtering outperformed mean filtering in the
   Ferrara experiments and preserves edges of the tampered region better.
5. **Fusion-layer score**: Reduce the heatmap to a single tampering score for this
   detector's contribution to the overall multi-module fusion (e.g. max or
   95th-percentile of the per-block posterior over the image, or the fraction of
   blocks below a probability threshold), while still passing the full heatmap
   downstream for optional visual/region overlay output.
6. **Optional secondary/complementary signal**: For images where CFA artifacts are
   weak or the image has been resized (destroying CFA phase), fall back to or
   fuse with a second-order/MTPM+SVM detector (Singh & Singh, #4) trained on the
   engine's target image distribution — it showed the best raw AUC/decision-error
   numbers in these papers and is more robust to some post-processing, at the cost
   of needing labeled training data and being significantly slower (not real-time
   friendly). If lens/optical artifacts are also in scope for this engine, lateral
   chromatic aberration analysis (Mayer & Stamm, diamond-search LCA, from the
   marginal papers above) is a cheap, largely independent signal that can be
   fused in as well, since it fails/succeeds on different image regions (LCA needs
   high-contrast edges away from the optical center; CFA analysis needs textured,
   non-flat, non-saturated regions).

**Known limitations to document/handle in the fusion layer**:
- **JPEG compression** is the dominant failure mode for every CFA-based method
  surveyed: detection AUC collapses once quality drops below ~90–95%, because
  quantization destroys the fine high-frequency correlation the predictor relies
  on. The engine should either (a) skip/down-weight this detector's score when
  input JPEG quality is below ~90 (estimable from quantization tables), or (b)
  route heavily-compressed images to non-CFA detectors instead.
- **Resizing/rescaling and re-demosaicing** (e.g. an image saved, resized, and
  re-saved, or a spliced region that was itself scaled to fit) shifts or destroys
  the periodic CFA phase entirely; this is a known anti-forensic step. Cross-check
  with a resampling/periodicity detector if available.
- **Flat/uniform and saturated regions** give near-zero prediction error
  regardless of CFA presence, causing false negatives (undetectable tampering) in
  sky, walls, overexposed highlights, etc. — the heatmap will be unreliable there
  and consumers of the score should be aware of low-texture regions as blind
  spots.
- **Sharp, strong edges** can also mimic the CFA-absent statistical signature
  (false positives) — combine with edge-density masking or morphological
  filtering of the likelihood map to reduce spurious block flags.
- **Non-Bayer sensors** (Foveon X3, some super-CCD arrays) do not produce CFA
  demosaicing artifacts at all; the whole family of techniques is inapplicable and
  should be disabled/skipped when source camera metadata indicates such a sensor.
- **Copy-move forgeries with matching CFA phase** (about 1-in-4 chance for a
  same-camera copy-move) can be invisible to CFA-phase-mismatch-based logic (Jeon,
  Bammey), though Ferrara's variance-imbalance feature can still catch these in
  many cases since it does not rely purely on phase matching.

## References

1. P. Ferrara, T. Bianchi, A. De Rosa, A. Piva, "Image Forgery Localization via
   Fine-Grained Analysis of CFA Artifacts," IEEE Trans. Information Forensics and
   Security, vol. 7, no. 5, pp. 1566–1577, Oct. 2012.
   — `ferrara2012.pdf`, `ferrara2012 (1).pdf` (duplicate)
2. Q. Bammey, J.-M. Morel, R. Grompone von Gioi, "Automatic Detection of
   Demosaicing Image Artifacts and its Use in Tampering Detection," 2018 IEEE
   Conference on Multimedia Information Processing and Retrieval (MIPR).
   — `bammey2018.pdf`
3. J. J. Jeon, H. J. Shin, I. K. Eom, "Estimation of Bayer CFA Pattern
   Configuration Based on Singular Value Decomposition," EURASIP Journal on Image
   and Video Processing, 2017:47.
   — `jeon2017.pdf`
4. G. Singh, K. Singh, "Digital Image Forensic Approach Based on the Second-Order
   Statistical Analysis of CFA Artifacts," Forensic Science International: Digital
   Investigation, vol. 32, 200899, 2020.
   — `singh2020.pdf`
5. O. Mayer, M. C. Stamm, "Accurate and Efficient Image Forgery Detection Using
   Lateral Chromatic Aberration," IEEE Trans. Information Forensics and Security,
   2018 (early access). — `mayer2018.pdf` (marginal — lens optics, not CFA)
6. M. M. Islam, G. Karmakar, J. Kamruzzaman, M. Murshed, "A Robust Forgery
   Detection Method for Copy-Move and Splicing Attacks in Images," Electronics,
   9(9):1500, 2020. — `islam2020.pdf` (marginal — general DCT+LBP+SVM, non-CFA)
7. A. Zafari, S. Jalali, "An Unpaired Learning-Based Method for Image
   Despeckling," 2025 IEEE CISA. — `An-Unpaired-Learning-Based-Method-for-Image-
   Despeckling.pdf` (off-topic — SAR speckle noise removal)
8. C. H. Neetha, C. J. Moses, D. Selvathi, "Image Interpolation Using
   Non-adaptive Scaling Algorithms for Multimedia Applications—A Survey," Advances
   in Automation, Signal Processing, Instrumentation, and Control (Springer LNEE
   700), 2021. — `Image-Interpolation-Using-Non-adaptive-Scaling-Algorithms-for-
   Multimedia-Applications-A-Survey.pdf` (background — resampling, not demosaicing)
9. Z. Lai, Y. Zhang, D. Li, Z. Yao, C. Wang, C. Qin, "A New Robust Training-Free
   Proactive Deepfake Detection Scheme Using Watermarking and Identity-Aware
   Hashing," Expert Systems With Applications, vol. 308, 131126, 2026.
   — `A-new-robust-training-free-proactive-deepfake-detection-scheme-using-
   watermarking-and-identity.pdf` (off-topic — proactive face-deepfake watermarking)
