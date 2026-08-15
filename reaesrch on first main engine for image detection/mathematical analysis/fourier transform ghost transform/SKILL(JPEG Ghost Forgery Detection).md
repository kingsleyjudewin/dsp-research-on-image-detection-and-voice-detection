# Fourier-Domain & JPEG Ghost Forgery Detection

## Core mathematical principle

Two physically distinct mechanisms, both exposed by transforming a derived signal into the frequency domain:

**1. Resampling/interpolation periodicity.** Any geometric transformation of part of an image (scaling, rotation, shearing a spliced/copy-moved region to fit its new context) requires resampling the pixel grid, and interpolation synthesizes new samples as a locally linear combination of neighbors. Kirchner & Böhme formalize this: each pixel `y_i` is modeled as `y_i = P^{α,i}·y + ε_i` (Eq. 7), a weighted sum of its `K×K` neighborhood plus residual. Interpolated pixels are *strongly* linearly predictable from their neighbors (high `p_i` in a two-state model); genuinely acquired pixels are not. This predictability varies **periodically** across the resampled lattice with a period set by the resampling factor, and the periodicity shows up as sharp, localized peaks in the 2-D DFT of the resulting probability map. Untouched regions of a photo exhibit no such periodicity, so its local presence is evidence of post-capture geometric transformation (splicing or resizing).

**2. JPEG "ghosts" from double compression at mismatched quality.** When a forger splices JPEG-sourced content into a host image and re-saves as JPEG, the background is quantized twice at the *same* quality (its original save, invisibly repeated), while the spliced region was quantized once at a *different* quality before being pasted and re-saved with the rest. Recompressing the whole dubious image at a sweep of candidate quality factors and differencing against the original: blocks already quantized at a given candidate quality lose very little further information (their difference energy hits a **local minimum** — the "ghost"); blocks quantized at other qualities lose more. This is a double-quantization artifact in the 8×8 block-DCT domain, distinct from but complementary to resampling periodicity.

Both cues are invisible to the eye and survive many kinds of "no-op" visual retouching — which is precisely what makes them valuable, and also precisely why both are documented in this corpus to be **actively defeatable by a knowledgeable adversary** (see Key Findings).

## Input requirements

- **Format**: resampling detection works on any bitmap (ideally never-compressed or lightly-compressed); JPEG ghost detection requires the dubious image to already be **JPEG-derived** (has 8×8 block-DCT structure to exploit).
- **Preprocessing**:
  - Resampling detector: operates per-channel or on grayscale; **no color-space conversion required** by the corpus (Kirchner & Böhme use 8-bit grayscale directly, downsampled ×2 with nearest-neighbor from RAW specifically **to remove CFA-interpolation periodicity** that would otherwise confound the resampling signal — an explicit, important preprocessing step: "found to be sufficient to reliably remove detectable traces of demosaicing").
  - JPEG ghost detector: operates on all 3 channels (R,G,B), summed (Azarian-Pour Eq. 5–6); no color-space conversion needed beyond having RGB available.
  - Both: crop to a **fixed analysis window** if comparing across different resampling parameters — Kirchner & Böhme always crop to the **center 256×256 block** before detection to keep comparisons fair across parameter settings.
- **Reliable when**: resampling detector — image is never-compressed or very lightly compressed, standard linear/bicubic interpolation was used (not sinc-like kernels), and the transformed region is reasonably large; ghost detector — background quality factor `q1` exceeds the spliced region's original quality `q0` by a wide margin (`Δq = q1 − q0 > 22` for reliable detection, Azarian-Pour's own sensitivity curve).
- **Unreliable / inapplicable when**:
  - Resampling detector fails **outright after even moderate JPEG compression** — periodic blocking artifacts interfere with the periodic resampling residual (Kirchner & Böhme, explicit, both as a stated assumption and as an empirical finding: "virtually all resampling detectors fail after moderate JPEG compression").
  - Resampling detector is defeated by **sinc-kernel interpolation** even without any deliberate attack — downscaling and moderate upscaling with sinc/spline interpolation are "virtually undetectable" by construction (no adversarial post-processing needed).
  - Ghost detector fails when the quality-factor gap is small (`Δq → 0`, the ghost signal is very faint — hard forensic case, explicit in Azarian-Pour) or when the spliced region's original quality is **higher** than the surrounding re-save quality (`q0 > q1` — the classical ghost method's fundamental directional constraint, present in both the original Farid method and Azarian-Pour's automated extension).
  - Ghost detector is not meaningful on images with no JPEG compression history at all.

## Step-by-step algorithm

### A. Resampling / periodicity detector (Popescu & Farid's EM method, as formalized and extended by Kirchner & Böhme 2008) — PRIMARY for geometric-splice localization

1. **Local linear predictor (per pixel).** For pixel `y_i` in an `m_y × n_y` image, define the predictor over a `K×K` neighborhood (`K = 2L+1`, `L` integer; **`K=5` used throughout Kirchner & Böhme's main experiments**, with `K=3` and `K=7` separately validated to give substantially the same results):
   ```
   y_i = P^{α,i} · y + ε_i                                    [Eq. 7]
   P^{α,i} = 1^{1×K²} · ( (1^{1×dim(y)} ⊗ α) ⊙ N^i )          [Eq. 8]
   ```
   `α` = vector of `K²` unobservable predictor weights (center weight `α_{⌊K²/2⌋} := 0`, i.e. a pixel does not predict itself); `N^i` = `K² × dim(y)` neighborhood indicator matrix selecting pixel `i`'s `K²` local neighbors; `⊙` = elementwise product; `⊗` = Kronecker product.

2. **Two-state EM model.** Every pixel `y_i` is assumed to belong to one of two sets: `M1` (high linear dependence — the interpolated/resampled set) with `y_i ~ N(P^{α,i}·y, σ_{M1})`, or `M2` (low dependence, i.e. genuinely acquired) with `y_i ~ U(0, 2^ℓ − 1)` (uniform).

3. **E-step**: compute per-pixel posterior probability of belonging to `M1`, using Bayes' theorem with a uniform prior `Prob(y_i∈M1) = Prob(y_i∈M2)`:
   ```
   p_i = Prob(y_i ∈ M1 | y_i)
       = Prob(y_i|y_i∈M1)·Prob(y_i∈M1) / Σ_{k=1}^{2} Prob(y_i|y_i∈M_k)·Prob(y_i∈M_k)          [Eq. 10]
   ```

4. **M-step**: re-estimate the predictor weights via **weighted least squares**, and the `M1`-class residual standard deviation as a **weighted RMS**:
   ```
   α = (Y'WY)⁻¹ Y'Wy                                          [Eq. 11]
   σ_{M1} = sqrt( Σ_i p_i·ε_i² / Σ_i p_i )                     [Eq. 12]
   ```
   `Y` stacks all `K²`-length local-neighborhood row vectors for every pixel; `W = diag(p)` holds the E-step posteriors. Iterate E-step/M-step to convergence (Kirchner & Böhme do not print an explicit numeric stopping tolerance in the extracted methodology — `(exact convergence criterion not specified in the corpus for this paper — engineering recommendation: iterate until the change in log-likelihood or in α falls below a small tolerance, e.g. 10⁻⁴, consistent with the EM stopping conventions used elsewhere in this engine's Benford and CFA modules)`).

5. **Form the p-map**: `p = diag(W) ∈ [0,1]^{(m_x×n_x)}`, i.e. the vector of converged per-pixel `M1`-membership probabilities, reshaped to image dimensions.

6. **Spectral analysis**: take the 2-D DFT of the p-map. Apply a **radial high-pass weighting** to suppress the dominant low-frequency/DC component, combined with a **gamma contrast function `γ(·)`** to boost visibility of the periodic peaks (the exact functional form of `γ` is cited by Kirchner & Böhme to Popescu & Farid's original method rather than re-derived in this paper — `(exact γ formula not printed in the extracted text of this paper — treat as a reference to the cited prior work)`).

7. **Detection statistic**: correlate the observed (contrast-enhanced, high-pass-weighted) spectrum against a bank of **synthetic periodic patterns**, one per candidate transformation matrix `A` (scaling factor and/or rotation angle) in a search set `𝒜`:
   ```
   s_t^A = || A·ν_{m_s}⁻¹(i) − ⌊ A·ν_{m_s}⁻¹(i) + ½·1^{2×1} ⌋ ||          [Eq. 18, synthetic map]
   ρ = max_{A∈𝒜} || ( |γ(DFT(p))| ⊙ |DFT(s^A)| )^{1/2} ||                [Eq. 19, decision statistic]
   ```
   Flag as resampled if `ρ` exceeds a threshold `ρ_T`, calibrated empirically per desired false-acceptance rate (Kirchner & Böhme calibrate `ρ_T` on a held-out set of 400 known-original images, separate from the 3900 test images per parameter setting).
   - Search set used in experiments: `|𝒜| = 692` synthetic maps — 601 for scaling `0.5 ≤ S ≤ 2` in steps of `ΔS=0.0025`, 91 for rotation `0 ≤ Θ ≤ π/4` in steps of `ΔΘ = π/360 (≈5°... actually 0.5°)`.
   - **Cheaper approximate variant** `(not in this specific paper — general technique referenced across the literature)`: take the image's second derivative (Laplacian) directly and 2-D DFT it, skipping the EM p-map step entirely, at the cost of a noisier spectrum.

8. **Run block-wise / sliding-window**, not just globally, to *localize* which region shows anomalous periodicity relative to the rest of the frame — an authentic photo should show near-uniform (near-absent) periodicity everywhere; a spliced-and-resized patch stands out locally.

**Known adversarial defeats (documented, not hypothetical — quantified in the same paper):**
- **5×5 median filter** applied post-resampling: destroys the p-map's periodic structure almost entirely (visually confirmed, Fig. 3) at the cost of noticeable blur; larger windows (7×7) degrade quality further but only marginally improve undetectability beyond what 5×5 already achieves; detection rates for 3×3-filtered images remain ~90%+ for scaling, but 5×5 drops rotation-hiding detection below 25%.
- **Geometric distortion with edge modulation**: perturb each resampled pixel's source position by a Gaussian-distributed disturbance vector `e = (e_h, e_v) ~ N(0,σ)` (Eq. 13), modulated by Sobel-edge strength so distortion concentrates where visually least perceptible (Eq. 14–15, exact Sobel kernels `h = (2^{-ℓ}-1)⊗(1,0,-1,2,0,-2,1,0,-1)` and `v` the transpose-like analog). At `σ=0.4`, this "eliminates the characteristic traces from the p-map and its spectrum" while an image-quality cost of only ~4dB PSNR improvement over naive (unmodulated) distortion.
- **Dual-path approach** (Eq. 16): decompose the resampled image into low-frequency (median-filtered) and high-frequency (geometric-distortion-with-edge-modulation) components and recombine — reported as the **most effective single countermeasure** in the paper, yielding detection rates below 20% at FAR≤1% for most tested scaling/rotation parameters, while preserving image quality far better than either technique alone (Fig. 12–15).
- **Quantitative summary (Table I, upsampling S=1.4)**: of images successfully defeating detection via geometric distortion, **not a single "detection success" also found the correct synthetic transformation map** — meaning the few images that *do* trigger a `ρ > ρ_T` alarm under attack are triggering on spurious/incorrect matches, not genuine detections, which is a subtler failure mode than simple non-detection.

### B. JPEG ghost detector — original (Farid, as reviewed) and fully-automated (Azarian-Pour, Babaie-Zadeh & Sadri 2016) — PRIMARY for splice localization on JPEG-derived images

**B1. Manual/semi-manual ghost energy (foundational formulation reviewed in Azarian-Pour et al.):**
```
d(x,y,q2) = (1/3w²) · Σ_{c∈{R,G,B}} Σ_{i=0}^{w-1} Σ_{j=0}^{w-1} ( I(x+i,y+j,c) − I_{q2}(x+i,y+j,c) )²      [Eq. 6, smoothed difference energy]
```
`I_{q2}` = the dubious image recompressed at candidate quality factor `q2`; `w` = smoothing window size, **`w=16`** (paper's stated default, following the original method). Min-max normalize per pixel across the `q2` sweep:
```
d(x,y,q2) = ( δ(x,y,q2) − min_q[δ(x,y,q2)] ) / ( max_q[δ(x,y,q2)] − min_q[δ(x,y,q2)] )      [Eq. 7]
```
A genuinely double-quantized region shows a **local minimum in `d` at `q2` equal to its true original quality `q0`**.

**B2. Full automation (Azarian-Pour et al.'s contribution — removes manual inspection and the aligned-grid assumption):**

1. **Grid-shift sweep**: zero-pad/shift the dubious `m×n` image by `(d_x, d_y) ∈ {0,...,7}²` to produce a padded `(m+d_x)×(n+d_y)` image `I'`, handling the case where the forger did not paste content aligned to the original 8×8 DCT grid. Compute the same difference-energy statistic on the shifted image:
   ```
   δ_{(q2,dx,dy)}(x,y) ≜ (1/3w²) Σ_{c∈{R,G,B}} Σ_{i,j=0}^{w-1} ( I'(x+i,y+j,c) − I'_{q2}(x+i,y+j,c) )²      [Eq. 8]
   d_{(q2,dx,dy)}(x,y) ≜ ( δ_{(q2,dx,dy)}(x,y) − min_q[δ_{(q2,dx,dy)}(x,y)] ) / ( max_q[δ_{(q2,dx,dy)}(x,y)] − min_q[δ_{(q2,dx,dy)}(x,y)] )      [Eq. 9]
   ```
   Full sweep: `q2 = 1,...,100` × `(d_x,d_y) ∈ {0,...,7}²` = **6400 total (q2, dx, dy) combinations** for a full 100-quality-factor, 64-shift search (paper uses `w=16` throughout, matching B1).

2. **Segmentation (Step 2)**: for each `(q2, d_x, d_y)` combination, segment the resulting difference map into 2 classes — **class-0 (ghost/tampered region candidate)** and **class-1 (rest of image)** — using the **SE-MinCut** segmentation algorithm (Estrada & Jepson), chosen specifically for its robustness against the fractal-noise-like texture typical of ghost/difference maps. Output: a binary indexed image `Y(x,y)`.

3. **Classification via Bhattacharyya distance (Step 3)**: measure how distinguishable the two segmented classes are using the 1-D Bhattacharyya distance:
   ```
   B = (1/2)·ln[ (σ0² + σ1²) / (2σ0σ1) ] + (μ0 − μ1)² / (4(σ0² + σ1²))      [Eq. 10]
   ```
   `μ0, σ0²` = mean/variance of class-0 (ghost) pixel values; `μ1, σ1²` = mean/variance of class-1 pixels.

4. **Automatic decision**: after running Steps 1–3 across all 6400 `(q2, d_x, d_y)` combinations, take the parameter triple `(q2,m, dx,m, dy,m)` that **maximizes** `B`, call the resulting maximum distance `D_max`. Classify the image as **forged if `D_max > Th`**. Threshold `Th = 0.19`, determined by minimizing the classification error rate (FP = FN) on a training set of 1000 original + 1000 tampered UCID images. `q2,m` is reported as the tampered region's estimated quality factor; `dx,m, dy,m` report the detected DCT grid misalignment.

**Known limitations (paper's own stated constraint, inherited from the underlying method):** works only when the spliced region's original quality `q0` is **lower** than the surrounding re-save quality `q1` (`q1 > q0`); does not resolve the reverse case. Sensitivity depends strongly on `Δq = q1 − q0`: exceeds 95% average sensitivity once `Δq > 22`; the minimum `Δq` needed for ≥90% sensitivity varies with `q1` itself (paper's Fig. 6 — this minimum-Δq curve is **non-monotonic**, peaking around mid-range `q1` values, not simply "bigger gap always easier").

### C. Six-feature PCA fusion architecture (Rao, Ghanekar, Chitnis, Dawkhar & Mishra 2025) — fusion-layer pattern reference, not a Fourier/ghost-specific technique

This paper is included in this module's folder for its **DFT/DCT frequency-energy score** (one of its six modules) and because it is the only paper in the entire corpus providing a **fully worked, zero-labeled-calibration score-fusion example** — directly useful as the cold-start fallback for this engine's Bayesian fusion layer (see the `bayes fusion` module). Its six per-image forensic scores:

1. **Metadata inspection**: checks EXIF tags (e.g. `DateTimeOriginal`) via MATLAB's `imfinfo()`; missing/unusual tags raise a suspicion score. `(No formula given — a presence/absence heuristic.)`
2. **Error Level Analysis (ELA)**: recompress at a fixed quality (paper resizes to 80% as its stated "simulating compression" step — note this is **resizing**, not JPEG-requantizing at a lower quality factor, which is a nonstandard variant of classical ELA and worth flagging as such) and computes the absolute difference against the original; higher difference = higher suspicion.
3. **JPEG quantization table analysis**: inspects the quantization table for irregularities suggesting double/inconsistent compression.
4. **Wavelet noise inconsistency**: single-level **Haar** DWT; inspects high-frequency subbands `cH`, `cV` for "unexpected noise behavior." `(No explicit formula given beyond naming the subbands — thin methodological description in this source.)`
5. **DFT/DCT frequency-energy analysis**: 2-D DCT of the grayscale image; energy in the higher frequencies is inspected, with anomalously high high-frequency energy flagged as possible splicing/manipulation. `(No explicit energy formula, threshold, or frequency-band cutoff given — this is the paper's stated frequency-domain module but it is described only qualitatively in the extracted text.)`
6. **Lighting inconsistency (Sobel-gradient)**: the **only training-free lighting cue in this entire research corpus**. Exact MATLAB implementation as printed in the paper:
   ```matlab
   [Gx, Gy] = gradient(double(gray_img));
   gradient_mag = sqrt(Gx.^2 + Gy.^2);
   max_grad = max(gradient_mag(:));
   ```
   The paper's stated decision logic is qualitative: "if large gradients or multiple light directions are detected, this may suggest manipulation" — **no explicit threshold, no formal multi-direction detection procedure, and no quantitative validation of this specific module is given anywhere in the paper.** Treat this as a minimally-specified heuristic, not a validated detector — see the `lighting` module's `SKILL.md` for the fuller treatment of this gap.

7. **Fusion**: stack the six per-image scores into a feature vector; apply **PCA**, take the **first principal component** as the unified suspicion score, normalize to `[0,1]`; **threshold at 0.33** for a binary forged/authentic call (threshold chosen empirically, not derived). Forgery *type* (splicing, recompression, lighting mismatch, copy-move) is inferred from "the scoring patterns observed in individual forensic methods" via an unspecified decision rule `(rule not printed in the extracted text — corpus ambiguity)`. `[ML — excluded]`: the paper's final decision stage is a trained **SVM** on top of the PCA score; for a no-ML engine, threshold the normalized first-principal-component score directly at 0.33 as the paper itself demonstrates this score alone already separates real/fake images with "obvious clustering" (their Fig. 3) before the SVM is even applied.

**Important corpus-honesty note**: this paper reports **no quantitative accuracy/precision/F1/AUC numbers anywhere in the extracted text** — its "Evaluation and Results" section consists of one qualitative ELA heatmap example and one qualitative PCA scatter-plot description ("obvious clustering") over "ten diverse image samples." Do not cite this paper as evidence of validated detection performance; use it only for its architectural pattern (score-level PCA fusion) and its Sobel-gradient formula.

## Output

- **Pipeline A (resampling)**: primary output is the **p-map** (per-pixel `[0,1]` probability of high linear dependence) and its derived scalar `ρ` (unbounded, compared to a calibrated threshold `ρ_T`). For fusion: `(not specified in corpus)` — engineering recommendation: `score = clip(ρ/ρ_T, 0, 1)` per region, run block-wise for a heatmap.
- **Pipeline B (ghost)**: primary output is `D_max` (the maximized Bhattacharyya distance, unbounded above, compared to `Th=0.19`) plus the estimated tampered-region quality factor `q2,m` and grid shift `(dx,m, dy,m)`, and the segmentation mask `Y(x,y)` itself as a **direct localization heatmap** (this is the strongest localization output of any technique in this module — it is literally a segmented binary tamper mask, not just a score). For fusion: `(not specified in corpus)` — engineering recommendation: `score = clip(D_max / Th, 0, 1)`.
- **Pipeline C (PCA fusion pattern)**: normalized `[0,1]` unified suspicion score, `0` = authentic-leaning, `1` = forged-leaning, threshold `0.33` used in the source paper (unvalidated quantitatively, see caution above).

## Key findings from papers

**Manipulation types detected best**: Pipeline A — geometric splicing/resizing of pasted content (scaling, rotation) on never/lightly-compressed images. Pipeline B — splicing from a lower-quality JPEG source into a higher-quality host, re-saved as JPEG, with `Δq > 22`. Pipeline C — general multi-cue triage (splicing, recompression, lighting mismatch, copy-move) as a coarse first-pass classifier, though its own validation is thin (see above).

**Documented failure cases / limitations**: see Input Requirements. In addition — Pipeline A's resampling detector is **actively, quantitatively defeated** by three documented counter-forensic techniques from within the *same* source paper (median filtering, edge-modulated geometric distortion, and the dual-path combination), which is a stronger and more specific caution than a generic "may be evadable" — the paper is itself a demonstration of an attack against a well-established prior detector (Popescu & Farid's). Pipeline B is fundamentally directional (`q1 > q0` only) and fails near `Δq=0`.

**Benchmark tables**:

| Paper | Dataset | Metric | Value | Conditions |
|---|---|---|---|---|
| Kirchner & Böhme 2008 | 500 never-compressed 8-bit grayscale, Canon PowerShot S70, downsampled ×2 | Detection rate | ~100% | Baseline (no attack), scaling/rotation, FAR≤1% |
| Kirchner & Böhme 2008 | same | Detection rate | <20–25% | 5×5 median filter attack |
| Kirchner & Böhme 2008 | same | Detection rate | <20% (most parameters) | Dual-path attack (median + edge-modulated geometric distortion, σ=0.4), FAR≤1% |
| Kirchner & Böhme 2008 | same | Image quality cost | ~4dB PSNR gain with edge modulation vs. without, at comparable detection-hiding | Geometric distortion attack |
| Kirchner & Böhme 2008 | Mahdian & Saic derivative-based detector (re-implemented for comparison) | False acceptance | up to 30% at 100% "correct" detection for plain upsampling | Confirms Popescu & Farid's EM method as the stronger baseline of the two |
| Azarian-Pour 2016 | UCID, 1000 orig + 1000 tampered (200×200 splices) | Accuracy / Precision | 97.73% / 91.01% | Automated ghost detection, Th=0.19 |
| Azarian-Pour 2016 | same | Sensitivity | >95% | Δq = q1−q0 > 22 |
| Azarian-Pour 2016 | same | Search space | 6400 (q2,dx,dy) combinations | 100 quality factors × 64 grid shifts |
| Rao et al. 2025 | 10 diverse images (CASIA v1 + team-sourced) | — | **No quantitative metrics reported** | Qualitative ELA heatmap + PCA clustering only |

## Implementation notes

- **FFT/DFT normalization and aliasing**: when implementing the p-map spectral analysis, be careful with 2-D DFT conventions (`numpy.fft.fft2` is unnormalized by default) and apply `numpy.fft.fftshift` before radial high-pass weighting so the DC component is centered as the gamma/high-pass function expects.
- **The 6400-run ghost sweep is the dominant cost** in Pipeline B — for a practical engine, prune it: (a) skip grid shifts if the paste region is already known/suspected to be block-aligned (e.g. from a prior CFA/JPEG-compression-module localization result), reducing 64→1; (b) coarsen the `q2` sweep to every 2–5 steps rather than every integer quality factor, since ghost minima are broad, not needle-sharp (not explicitly validated in the corpus, but consistent with the smoothing window `w=16` already blurring fine `q2` resolution).
- **DC/low-frequency suppression** in the p-map spectrum is essential — without the radial high-pass weighting, the DC term dominates and the periodic peaks (which the whole method depends on) are invisible at normal display/threshold scales.
- **CFA-periodicity confound**: Kirchner & Böhme's explicit preprocessing step (downsample ×2 via nearest-neighbor before analysis) to strip CFA-interpolation periodicity is directly relevant to this engine's cross-module design — resampling-periodicity detection (this module) and CFA-artifact detection (the `color filter array` module) exploit *related* periodic-correlation phenomena and can produce false positives against each other if not carefully sequenced; run CFA-phase verification first, or explicitly downsample before resampling-detection, to avoid this confound.
- **No public reference code found in the extracted text** for any of the three primary papers in this module.
- **Recommended Python libraries**:
  - `numpy.fft.fft2` / `numpy.fft.fftshift` for the p-map spectral analysis (Pipeline A).
  - `scipy.optimize.lsq_linear` or direct normal-equation solve (`numpy.linalg.lstsq`) for the EM M-step weighted least squares (Eq. 11).
  - `PIL.Image.save(..., quality=q2)` or `cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q2])` for the recompression sweep in Pipeline B.
  - `skimage.segmentation` (no direct SE-MinCut implementation; `skimage.segmentation.slic` + graph-cut via `skimage.future.graph.cut_normalized`, or `maxflow`/`PyMaxflow` for a direct min-cut formulation) as the closest available substitute for the paper's SE-MinCut step — `(the exact SE-MinCut algorithm is not available as a standard Python package; this is an engineering substitution, not a like-for-like reproduction)`.
  - `scipy.spatial.distance` does not include Bhattacharyya distance directly; implement Eq. 10 manually (a 4-line closed-form given the class means/variances).
  - `scipy.ndimage.sobel` for the lighting-gradient cue (Pipeline C item 6).
  - `sklearn.decomposition.PCA` for the score-fusion pattern (Pipeline C) — noting the final classification step in the source paper is `[ML]` and should be replaced by direct thresholding of the first-principal-component score per this engine's no-ML constraint.

## Key references

- **kirchner2008.pdf** — M. Kirchner, R. Böhme, "Hiding Traces of Resampling in Digital Images," IEEE TIFS, vol. 3, no. 4, pp. 582–592, Dec. 2008. Source of: the complete EM-based p-map derivation (Eq. 7–12), the spectral detection statistic (Eq. 18–19), and all three counter-forensic attack techniques with their full quantitative detection-rate/image-quality tradeoff tables.
- **azarian-pour2016.pdf** — S. Azarian-Pour, M. Babaie-Zadeh, A.R. Sadri, "An Automatic JPEG Ghost Detection Approach for Digital Image Forensics," ICEE 2016. Source of: the automated grid-shift ghost-energy sweep (Eq. 8–9), the SE-MinCut segmentation + Bhattacharyya-distance classification pipeline (Eq. 10), and the accuracy/precision/sensitivity benchmarks.
- **Image-Tampering-Detection-Using-Multi-Feature-Scoring-and-PCA-Based-Classification.pdf** (and duplicate `... (1).pdf`) — A. Rao, A. Ghanekar, D. Chitnis, M. Dawkhar, D. Mishra, "Image Tampering Detection Using Multi-Feature Scoring and PCA-Based Classification," CISCON 2025. Source of: the six-module score-fusion architecture, the exact Sobel-gradient lighting-cue MATLAB code, and the PCA-to-first-principal-component fusion pattern — reused as the cold-start fusion fallback in the `bayes fusion` module. **No quantitative benchmark values are available from this source** (see explicit caveat above).
- **Detecting-periodicities-with-Fourier-analysis.pdf** — Textbook chapter (*Environmental Data Analysis with MATLAB/Python*, Elsevier 2022, Ch. 6). Background only: general 1-D DFT/Nyquist/aliasing reference, no image-forensics content. Not re-read in this pass.
- **Passive-Image-Forgery-Detection-Techniques-A-Review-Challenges-and-Future-Directions.pdf** (and duplicate) — Kaur, Jindal, Singh, *Wireless Personal Communications*, 2024. Marginal survey/taxonomy reference; not re-read in this pass.
- **An-extremum-guided-interpolation-for-sparsely-sampled-photoacoustic-imaging.pdf**, **Review-of-imaging-buffers-used-in-stochastic-optical-reconstruction-microscopy.pdf**, **Image-Processing-and-Pattern-Recognition.pdf** — tangential/off-topic (biomedical signal processing, microscopy chemistry, general DFT textbook background). Not re-read in this pass; see prior version of this file for what little generalizes.
