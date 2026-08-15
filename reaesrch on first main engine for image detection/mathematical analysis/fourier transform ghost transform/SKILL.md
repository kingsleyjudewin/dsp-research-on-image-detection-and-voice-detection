# Fourier-Domain & JPEG Ghost Forgery Detection

## Purpose

Two independent physical processes leave detectable, periodic or quasi-periodic
signatures that a spatial-domain inspection of a tampered image will not reveal:

1. **Resampling / interpolation periodicity.** Any geometric manipulation of part of an
   image — scaling, rotating, stretching a spliced or copy-moved region to fit its new
   context — requires the editor to resample the pixel grid. Interpolation (nearest
   neighbour, bilinear, bicubic, spline) synthesizes new pixel values as a locally linear
   (or near-linear) combination of their neighbours. This manufactures a periodic pattern
   of linear dependence between pixels, with a period set by the resampling factor. That
   periodicity is invisible in the spatial domain but produces sharp, distinct peaks in the
   2-D Fourier spectrum of a suitable derived signal (the "p-map" of local linear-predictor
   residuals, or higher-order derivatives/error signal of the image). Untouched, natively
   captured regions of a photo do not exhibit this periodicity, so its presence — and its
   spatial localization — is strong evidence that a region was geometrically transformed
   after capture, i.e., spliced or resized.

2. **JPEG "ghosts" from double compression at mismatched quality.** When a forger
   splices content from a second JPEG-compressed source into a JPEG image, or edits a
   region and re-saves, the composite is typically saved once more as JPEG. Untouched
   background pixels have then been JPEG-quantized twice at the *same* quality factor
   (their original save, invisibly re-encoded at the same setting), while the tampered
   region was quantized once at a *different* quality factor before being embedded and
   re-saved with the rest of the image. If you resave (recompress) the whole dubious image
   at a sweep of candidate quality factors and difference it against the original each
   time, blocks that were already quantized at that particular quality factor lose very
   little additional information (their difference energy is at a local minimum — this is
   the "ghost"), while blocks quantized at other qualities lose more. Because JPEG
   quantization operates in the 8x8 block DCT (frequency) domain, this is fundamentally a
   frequency-domain artifact of double quantization, distinct from but complementary to
   the resampling-periodicity cue above.

Both cues are undone by "no-op" spatial filtering that preserves visual appearance —
that is precisely why they are valuable: they detect processing history the eye cannot
see.

## Techniques found in the literature

**Kirchner & Böhme, "Hiding Traces of Resampling in Digital Images"** (IEEE TIFS,
2008) — *kirchner2008.pdf*. Directly relevant, the strongest resampling-detection
reference in this set.
- Recaps and relies on Popescu & Farid's resampling detector as the state of the art:
  each pixel `y_i` is modeled as a locally linear combination of its `K x K`
  neighbourhood, `y_i = P^(alpha,i)·y + eps_i`. An Expectation-Maximization procedure
  assigns each pixel a probability `p_i` of belonging to the "high linear dependence" set
  M1 vs. the "low dependence" set M2, producing a per-pixel probability map (the **p-map**).
  EM alternates: E-step computes `p_i` via Bayes' rule from the current predictor weights
  and residual variance; M-step re-estimates the weight vector `alpha` by weighted least
  squares, `alpha = (Y'WY)^-1 Y'Wy`, and the residual variance `sigma_M1` as a weighted
  RMS of residuals. Iterate E/M to convergence.
  - **Resampling leaves a conspicuous periodic pattern in the p-map.** Taking the 2-D
    DFT of the p-map and applying a radial high-pass weighting + gamma contrast function
    `gamma(·)` makes the periodicity show up as sharp, localized peaks in the spectrum
    that are absent for un-resampled image regions.
  - Detection/decision criterion: correlate the observed spectrum against a bank of
    synthetic periodic patterns predicted for each candidate transform matrix A (scaling
    factor / rotation angle) via `rho = max_A || (|gamma(DFT(p))| ⊙ |DFT(s^A)|)^(1/2) ||`;
    flag as resampled if `rho` exceeds an empirically calibrated threshold.
  - The paper's main contribution is actually *counter-forensic*: it shows resampling
    periodicity is fragile — a 5x5 median filter, geometric jitter distortion (perturbing
    each resampled pixel's source position by Gaussian noise, optionally "edge-modulated"
    via Sobel-detected image structure so distortion concentrates where least visible),
    or a dual-path combination of both can suppress the p-map peaks almost entirely while
    keeping PSNR/wPSNR degradation small (detection rates for FAR<=1% dropped from ~100%
    to well below 20-25% in benchmarks on 500 raw images). It also confirms detectors fail
    outright once an image has been through even moderate prior JPEG compression, because
    JPEG blocking artifacts mask the periodic residual.
  - **Engineering takeaway:** treat naive p-map/DFT resampling detection as a useful but
    gameable and JPEG-fragile signal — best applied to never-recompressed regions/crops
    and fused with other cues rather than trusted alone; expect degraded performance on
    JPEG images and design the fusion layer to discount this signal when JPEG artifacts
    dominate.

**Azarian-Pour, Babaie-Zadeh & Sadri, "An Automatic JPEG Ghost Detection Approach for
Digital Image Forensics"** (ICEE 2016) — *azarian-pour2016.pdf*. Directly relevant,
the core JPEG-ghost algorithm reference.
- Reviews and extends Farid's original JPEG ghost method [Farid 2009]. Core formula:
  given dubious image `I`, recompress it at a sweep of candidate quality factors `q2 =
  1..100` to get `I_q2`, then compute a smoothed block-difference "energy" image
  `delta(x,y,q2) = 1/(3w^2) * sum_c sum_{i,j in w x w window} (I(x+i,y+j,c) -
  I_q2(x+i,y+j,c))^2` (summed over R,G,B channels, window size `w=16` typical), then
  min-max normalize per pixel across the q2 sweep to `d(x,y,q2) in [0,1]`. A genuinely
  double-quantized region shows a **local minimum in `d` at `q2` equal to its original,
  lower quality factor `q0`** — the "ghost" — because recompressing at the quality it was
  already quantized at changes it least.
  - Key limitation of the original method: it required manual visual inspection of the
    difference stack to find the ghost, needed the tampered/background quality
    relationship `q1 > q0` (background higher quality than the splice), and assumed
    aligned 8x8 DCT grids between the tampered patch and the background.
  - **Proposed automation:** (1) zero-pad/shift the dubious image over `d_x, d_y = 0..7`
    to search all 64 possible DCT grid alignments (handles the forger having shifted the
    inserted object); (2) run the ghost-energy computation for every `(q2, d_x, d_y)`
    combination (6400 runs for 100 quality factors x 64 shifts); (3) segment each
    resulting ghost/difference map into "ghost region" vs. "rest of image" using
    SE-MinCut image segmentation (chosen for robustness to the fractal-noise-like texture
    of ghost maps); (4) score each segmentation by the 1-D Bhattacharyya distance between
    the two classes' pixel-value mean/variance, `B = 1/4 * ln(...) + (mu0-mu1)^2 /
    (4(sigma0^2+sigma1^2))`; (5) take the `(q2, d_x, d_y)` triple that maximizes B as
    `D_max`, and classify the image as forged if `D_max > threshold` (empirically Th =
    0.19 on the UCID dataset, 1000 originals / 1000 tampered).
  - Reported results (UCID, splices of size 200x200, background/foreground quality
    factors drawn from disjoint ranges): mean segmentation accuracy 97.73%, precision
    91.01%. Sensitivity depends strongly on the quality-factor gap `delta_q = q1 - q0`:
    averaged sensitivity exceeds 95% once `delta_q > 22`; near q1=q0 the ghost is very
    faint (hard forensic case). The method inherits the constraint that it only detects
    the case where the *outer* re-save used a higher quality factor than the spliced
    region's original quality — a splice pasted at a *higher* effective quality than the
    surrounding image will not show a clean ghost by this technique.

**Detecting Periodicities with Fourier Analysis** (Environmental Data Analysis with
MATLAB/Python textbook, Ch. 6) — *Detecting-periodicities-with-Fourier-analysis.pdf*.
General-purpose reference, not forgery-specific.
- Rigorous refresher on 1-D discrete Fourier analysis fundamentals directly reusable when
  implementing the DFT/peak-detection step of a resampling detector: amplitude/period/
  frequency/angular-frequency definitions, the Fourier series representation of a signal
  as a sum of paired sine/cosine terms, the Nyquist frequency limit `f_nyq = 1/(2*dt)`
  and the aliasing of any content above it back into the detectable band, and the
  bookkeeping of frequency bins (`N/2+1` unique frequencies for `N` samples). Useful as a
  clean reference for correctly implementing/validating the FFT/peak-picking step and for
  avoiding aliasing bugs when computing the 2-D DFT of the p-map/residual signal, but
  contains no image-forensics content.

**Rao, Ghanekar, Chitnis, Dawkhar & Mishra, "Image Tampering Detection Using
Multi-Feature Scoring and PCA-Based Classification"** (CISCON 2025) —
*Image-Tampering-Detection-Using-Multi-Feature-Scoring-and-PCA-Based-Classification.pdf*
(duplicated as the "(1)" file — identical content). Directly relevant as a fusion-layer
architecture reference.
- MATLAB pipeline combining six independently scored forensic cues — metadata/EXIF
  inspection, Error Level Analysis (ELA: recompress at 80% quality, difference against
  original), JPEG quantization-table analysis (irregular/inconsistent tables flag
  possible re-quantization), wavelet (Haar) noise-inconsistency in high-frequency
  subbands cH/cV, **DFT/DCT frequency-energy analysis** (2-D DCT of the grayscale image;
  anomalous high-frequency energy flagged as possible splicing/manipulation), and Sobel-
  gradient lighting-inconsistency detection.
  - Each module outputs a per-image scalar score; scores are stacked into a feature
    vector and reduced via PCA to a single first-principal-component "unified suspicion
    score," normalized to [0,1]; images above a 0.33 threshold are classified
    forged/authentic, and the pattern across the six module scores is used to guess the
    forgery type (splicing, recompression, lighting mismatch, copy-move).
  - Reported accuracy/precision figures are qualitative/small-scale (ten test images);
    the paper's real contribution for this engine is architectural: a concrete example of
    fusing a frequency-domain score with several independent forensic scores via PCA into
    one calibrated tampering score, which is exactly the fusion-layer pattern this engine
    needs.

**Kaur, Jindal & Singh, "Passive Image Forgery Detection Techniques: A Review,
Challenges, and Future Directions"** (Wireless Personal Communications, 2024) —
*Passive-Image-Forgery-Detection-Techniques-A-Review-Challenges-and-Future-Directions.pdf*
(duplicated as the "(1)" file). Marginally relevant — broad survey, not a ghost/
periodicity paper.
- Comprehensive taxonomy of passive image forgery detection (copy-move vs. splicing vs.
  retouching), covering block-based, keypoint-based, and deep-learning detectors. Its
  DCT-domain content is centered on copy-move forgery detection (Fridrich et al.'s
  original block-DCT feature matching, and numerous DCT/DWT/Markov-feature hybrids for
  splicing detection) rather than JPEG ghost/double-compression or resampling-periodicity
  detection specifically — no dedicated treatment of either technique in this set. Useful
  primarily as a reference list / literature map and for terminology and general
  forgery-detection framing (Fig. 1 forensics taxonomy: analog vs. digital -> multimedia
  forensics -> image/video/audio forensics).

**Marginally relevant / tangential papers** (included in the folder but outside the
domain — extracted only what generalizes):
- *An-extremum-guided-interpolation-for-sparsely-sampled-photoacoustic-imaging.pdf*
  (Wang, Yan, Ma & Han, Photoacoustics 2023) — biomedical photoacoustic signal
  reconstruction. Proposes an "extremum-guided interpolation" (EGI) that picks local
  extrema within a search window as interpolation anchors instead of assuming pure
  linear/spline interpolation, for sparsely sampled 1-D time-domain PA sensor signals.
  Not about images or forensics; only tangential relevance is the general reminder that
  interpolation choice affects the character of introduced artifacts — irrelevant to the
  forgery-detection module beyond that observation.
- *Review-of-imaging-buffers-used-in-stochastic-optical-reconstruction-microscopy.pdf*
  (Wang, Sun & Ma, Chinese Chemical Letters 2025) — chemistry/biology review of
  photo-switching buffer chemistry for STORM super-resolution microscopy. No image
  forensics, Fourier, or compression content whatsoever; not usable for this engine.
- *Image-Processing-and-Pattern-Recognition.pdf* (Frank Y. Shih, Wiley/IEEE Press) —
  552-page general image processing/pattern recognition textbook. Chapter 2 is a
  standard treatment of the (continuous, discrete, fast) Fourier Transform — useful only
  as generic DFT/FFT theory background, not forgery-specific; no ghost or resampling
  detection content found.

## Recommended approach for this engine

Implement two independent detectors and fuse their outputs into one heatmap + scalar
score, following the pattern demonstrated by the multi-feature/PCA paper.

**1. Resampling/periodicity detector (spliced-and-resized region localization)**
- For each channel (or luminance), compute a local linear-predictor residual per pixel
  using a `K x K` (K=5 is a good default per Kirchner & Böhme) neighbourhood predictor
  fit by EM (weighted least squares, iterate E-step probability/M-step weights to
  convergence) to obtain a p-map of "how strongly linearly predictable" each pixel is
  from its neighbours.
  - A cheaper approximate variant, if EM is too costly for real-time use: take the
    second-derivative (Laplacian) of the image, then compute its 2-D DFT directly — the
    resampling periodicity still surfaces as spectral peaks, at the cost of being noisier
    than the full p-map/EM approach.
- Take the 2-D DFT of the p-map, apply a radial high-pass weighting + gamma contrast
  boost to suppress the DC/low-frequency dominant component and expose the periodic
  peaks.
- Peak-detect in the resulting spectrum (local maxima above a noise floor, e.g. after a
  max-filter as in Kirchner & Böhme). Presence of strong, structured, non-DC peaks
  indicates resampling; peak *position* encodes the resampling scale/rotation and can be
  matched against a bank of synthetic candidate patterns (`s^A` for hypothesized scale/
  rotation matrices A) to both detect and estimate the transform parameters.
- Run this in a sliding-window/blockwise fashion over the image (not just globally) to
  localize *which region* shows anomalous periodicity relative to the rest of the frame
  — a genuine, un-doctored photo should have fairly uniform (near-absent) periodicity
  everywhere; a spliced-and-rescaled patch stands out locally.
- **Known limitation to design around:** this detector degrades sharply after JPEG
  recompression and is defeatable by median filtering or geometric-jitter counter-
  forensics (see Kirchner & Böhme results — detection can drop from ~100% to <20-25%).
  Treat its output as a corroborating signal, weighted down when the image shows heavy
  JPEG blocking or when confidence is otherwise low, never as sole evidence.

**2. JPEG ghost detector (double-compression / mismatched-quality region localization)**
- Only applicable to JPEG-derived images (check for JPEG markers / re-saved-as-JPEG
  input); skip or down-weight for lossless-source images.
- Recompress the full dubious image at every quality factor `q2 = 1..100` (or a coarser
  practical sweep, e.g. every 2-5 steps for speed).
- For each `q2`, compute the smoothed per-block squared-difference energy `delta(x,y,q2)`
  against the recompressed version over a `w x w` window (w=16 is the literature
  default), then min-max normalize per pixel across the q2 sweep to get `d(x,y,q2) in
  [0,1]`.
- For each spatial block, find the `q2` that minimizes `d` — this is the block's implied
  original quality factor. Blocks whose minimizing `q2` differs materially from the
  image-wide majority/background quality factor are flagged as the "ghost" (double-
  quantized at a different quality than the rest of the image).
- To remove the need for manual inspection and handle non-aligned DCT grids from
  shifted/pasted content, sweep the 64 possible 8x8 block-grid alignments `(d_x, d_y) in
  {0..7}^2` as well as q2 (Azarian-Pour et al.), segment each resulting ghost map into
  "ghost region" vs. "background" (graph-cut/MinCut style segmentation is robust to the
  fractal-noise-like texture of ghost maps), score each segmentation's separation via the
  Bhattacharyya distance between the two classes, and keep the `(q2, d_x, d_y)` that
  maximizes that distance. Flag the image as tampered if the max distance exceeds a
  calibrated threshold (~0.19 as a starting point, recalibrate on your own dataset).
- **Known limitation:** classic JPEG ghost detection only reliably surfaces the case
  where the spliced region's original quality is *lower* than the final re-save quality
  (`q1 > q0`); a region pasted in at equal or higher quality than the surroundings will
  not show a clean ghost by this method, and detection sensitivity collapses as `q1 - q0
  -> 0` (near-equal quality factors). Don't over-promise detection for subtle
  quality-factor mismatches.

**3. Fusing into a tampering score/heatmap**
- Compute both detectors at the block or region level and produce two per-pixel/per-
  block maps: a resampling-periodicity confidence map and a JPEG-ghost confidence map
  (each normalized to [0,1]).
- Stack these two maps alongside any other DSP-based forensic scores available elsewhere
  in the engine (ELA, noise-inconsistency, lighting/Sobel-gradient inconsistency,
  quantization-table irregularity) into a per-region feature vector, following the six-
  feature PCA fusion pattern from the Multi-Feature Scoring paper: PCA-reduce to a single
  principal component as the unified suspicion score, normalize to [0,1], and threshold
  for a binary forged/authentic call (empirically calibrate the threshold, e.g. starting
  near 0.3-0.35, on a labeled validation set rather than trusting literature defaults
  as-is).
- For visualization/localization, overlay the higher of the two normalized confidence
  maps (or their max/weighted-sum) as a heatmap on the original image, and report the
  region(s) with peak fused score as the likely tampered area.
- Weight the two detectors adaptively: down-weight the resampling/periodicity score when
  the image shows strong pre-existing JPEG blocking (since periodicity detection is
  known to fail there), and down-weight the JPEG-ghost score when no JPEG structure is
  present at all or when the quality-factor gap between candidate regions is small
  (ghost is faint/absent).

**Known limitations overall**
- Both techniques are format/processing-history dependent (resampling detection needs
  never- or lightly-compressed data; ghost detection needs JPEG structure) — verify
  applicability before trusting either score, and route accordingly in the fusion layer.
- Both are known to be defeatable by a deliberate counter-forensic actor (median
  filtering + geometric jitter defeats resampling detection; careful global
  recompression at a single uniform quality defeats ghost detection) — treat their output
  as probabilistic evidence to combine with other independent forensic cues, not as
  standalone proof of tampering.
- Threshold values quoted in the literature (Th=0.19 for Bhattacharyya distance,
  0.33 for the PCA suspicion score) were calibrated on specific, relatively small
  datasets (UCID, CASIA v1, ad hoc test sets) and should be recalibrated against this
  engine's own validation data before deployment.

## References

- Kirchner, M. & Böhme, R. "Hiding Traces of Resampling in Digital Images." *IEEE
  Transactions on Information Forensics and Security*, vol. 3, no. 4, Dec. 2008,
  pp. 582-592. — `kirchner2008.pdf`
- Azarian-Pour, S., Babaie-Zadeh, M. & Sadri, A. R. "An Automatic JPEG Ghost Detection
  Approach for Digital Image Forensics." 2016 24th Iranian Conference on Electrical
  Engineering (ICEE), pp. 1645-1649. — `azarian-pour2016.pdf`
- (Unnamed textbook chapter) "Detecting periodicities with Fourier analysis," Chapter 6
  of *Environmental Data Analysis with MATLAB/Python*, Elsevier, 2022. —
  `Detecting-periodicities-with-Fourier-analysis.pdf`
- Rao, A., Ghanekar, A., Chitnis, D., Dawkhar, M. & Mishra, D. "Image Tampering
  Detection Using Multi-Feature Scoring and PCA-Based Classification." 2025 Control
  Instrumentation System Conference (CISCON), IEEE, 2025. —
  `Image-Tampering-Detection-Using-Multi-Feature-Scoring-and-PCA-Based-Classification.pdf`
  and duplicate `Image-Tampering-Detection-Using-Multi-Feature-Scoring-and-PCA-Based-Classification (1).pdf`
- Kaur, N., Jindal, N. & Singh, K. "Passive Image Forgery Detection Techniques: A
  Review, Challenges, and Future Directions." *Wireless Personal Communications*,
  vol. 134, 2024, pp. 1491-1529. —
  `Passive-Image-Forgery-Detection-Techniques-A-Review-Challenges-and-Future-Directions.pdf`
  and duplicate
  `Passive-Image-Forgery-Detection-Techniques-A-Review-Challenges-and-Future-Directions (1).pdf`
- Wang, H., Yan, L., Ma, C. & Han, Y. "An extremum-guided interpolation for sparsely
  sampled photoacoustic imaging." *Photoacoustics*, vol. 32, 2023, 100535. (Tangential —
  biomedical signal interpolation, not image forensics.) —
  `An-extremum-guided-interpolation-for-sparsely-sampled-photoacoustic-imaging.pdf`
- Wang, C., Sun, Z. & Ma, D. "Review of imaging buffers used in stochastic optical
  reconstruction microscopy." *Chinese Chemical Letters*, vol. 36, 2025, 110677.
  (Tangential — microscopy chemistry, not usable for this engine.) —
  `Review-of-imaging-buffers-used-in-stochastic-optical-reconstruction-microscopy.pdf`
- Shih, Frank Y. *Image Processing and Pattern Recognition: Fundamentals and
  Techniques.* Wiley/IEEE Press. (Tangential — general textbook, Ch. 2 covers standard
  Fourier Transform theory only.) — `Image-Processing-and-Pattern-Recognition.pdf`
