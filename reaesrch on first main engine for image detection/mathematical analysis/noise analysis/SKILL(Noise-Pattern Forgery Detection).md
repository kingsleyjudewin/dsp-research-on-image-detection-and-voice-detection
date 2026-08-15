# Noise-Pattern Forgery Detection

## Core mathematical principle

**Underlying concept — PRNU.** Every imaging sensor imprints a unique multiplicative noise fingerprint, **Photo-Response Non-Uniformity (PRNU)**, caused by pixel-to-pixel variation in photon-to-electron conversion sensitivity. Chen, Fridrich, Goljan & Lukáš (2008) derive the full sensor output model from first principles:
```
I = g^γ · [(1+K)Y + Λ]^γ + Θ_q          [Eq. 1]
```
`g` = color-channel gain, `γ ≈ 0.45` = gamma-correction factor, `Y` = incident light intensity, `K` = the zero-mean, noise-like PRNU factor (the sensor fingerprint), `Λ` = combination of dark current/shot/read noise, `Θ_q` = quantization noise. Taylor-expanding the gamma nonlinearity `(1+x)^γ ≈ 1+γx` at `x=0` and keeping only the dominant terms yields the working **linear model**:
```
I ≈ I^(0) + I^(0)·K + Θ          [Eq. 2, with K now absorbing γ, and Θ a complex combination of remaining noise terms]
```
`I^(0) = (gY)^γ` is the noise-free sensor output. This is a **multiplicative** signal (`I^(0)·K`), not additive, and is present in every image the sensor ever captures, independent of scene content.

**Property exploited.** Because `K` is fixed per physical sensor and content-independent, (a) it can be used as a "camera fingerprint" for source-camera matching (reference-based), and (b) any region whose noise residual fails to correlate with the expected PRNU — because it was pasted from a different sensor, denoised/smoothed, or synthetically generated with no sensor noise at all — is evidence of tampering. Separately, **denoising-residual-based comparison** (Chen et al.'s underlying signal-flattening filter, or wavelet HH-subband statistics) allows a **blind, no-reference** version of this idea: comparing *local* noise-level statistics *within a single image* rather than against an external camera-fingerprint database.

## Input requirements

- **Format**: RGB image; PRNU extraction is typically performed **per color channel then combined**, or on a grayscale/luminance conversion for lighter-weight variants.
- **Preprocessing**:
  - **Reference-based PRNU (Chen et al.)**: requires `N` images from the **same physical camera** (not necessarily the specific image under test) to estimate `K̂`. Best source images are **bright, unsaturated, smooth-content** — the paper's explicit recommendation is **out-of-focus images of a cloudy sky**; natural scenes work but need roughly **twice as many images** (≈ double `N`) for the same estimate quality, because edges/texture increase the residual noise variance.
  - **Blind/no-reference (Debiasi et al.)**: no external reference needed; extract PRNU directly from the single suspect image via denoising-residual subtraction.
  - **PRNU factor preprocessing** (Chen et al., critical and often-omitted step): the raw ML-estimated `K̂` contains **systematic artifacts shared across cameras of the same brand/sensor design** (CFA color-interpolation periodicity, row/column sensor-readout bias) that must be suppressed before use, or false camera-matches and degraded forgery-detection sensitivity result (see Step-by-step algorithm, Pipeline B).
- **Reliable when**: sufficiently textured or at least non-flat, non-saturated image regions (PRNU correlation is genuinely a function of local **intensity and texture**, formalized below); block size matched to resolution (Chen et al. recommend **128×128** for ≥1-megapixel images); minimal denoising/smoothing has been applied post-capture.
- **Unreliable / inapplicable when**:
  - **Saturated pixels** (`I[i] ≈ 255` for 8-bit): the multiplicative PRNU term `I·K` vanishes as `I` approaches the sensor's dynamic-range ceiling — Chen et al. model this explicitly via an attenuation function (Eq. 19, see below).
  - **Flat/low-texture regions**: correlation is measurably lower in textured areas because the denoising filter's residual variance `σ²_Ξ` is larger and the attenuation factor is reduced there — counterintuitively, this means *smooth* regions give the *strongest* raw signal but the *predictor* must still account for texture's effect on other cameras/content (see the texture feature below).
  - **Heavy JPEG recompression**: Chen et al.'s own FRR tables show orders-of-magnitude degradation from RAW → JPEG90 → JPEG75 (see benchmark table).
  - **Denoising/smoothing attacks**: the single largest documented failure mode across this whole module — an attacker who denoises the tampered region (or the whole image) after tampering removes the very signal being measured.
  - **Very small tampered regions**: reliable correlation estimation needs enough samples per block; sub-64×64 regions are explicitly flagged as difficult without careful block-size tuning.

## Step-by-step algorithm

Three pipelines: **(A)** blind local noise-level inconsistency — recommended **PRIMARY** for this engine's single-suspect-image, no-reference use case; **(B)** reference-based PRNU correlation (Chen et al.) — used when a source camera or multiple same-camera images are available; **(C)** blind cell-based PRNU spectral analysis (Debiasi et al.) — a second no-reference option, validated specifically for face-morph detection but directly transferable to general local self-consistency checking. A fourth, thin (**D**) noise-type triage classifier is included for completeness.

### A. Blind local noise-level inconsistency — engineering synthesis built from B/C's shared machinery, PRIMARY for single-image use

`(This specific combination — comparing PRNU/noise-residual statistics across regions of the SAME image without a reference camera — is not validated end-to-end as one integrated pipeline in either source paper; each source paper validates a different piece: Chen et al. validate the residual-extraction and correlation machinery reference-based, Debiasi et al. validate cell-splitting and spectral-feature aggregation blind but for a different task (morph detection, not generic tampering). The combination below is an engineering recommendation assembled from validated sub-components.)`

1. Extract the **noise residual** `W = I − F(I)` using the **Mihcak wavelet-based denoising filter** `F` (Chen et al. cite this as their Filter 1; the same filter, further refined with the **FDR (Filtering Distortion Removal) enhancement** of Lin et al., is used by Debiasi et al.).
2. Tile the image into overlapping or non-overlapping blocks. Block-size guidance is genuinely split across the two source papers — see the explicit parameter-disagreement note below.
3. Within each block, compute a **local noise-level statistic** — either (a) the block's residual variance directly, or (b) reuse Debiasi et al.'s **cell-wise DFT-magnitude-histogram peak features** (`P_val`, `P_pos`, `P_pv` — full definitions in Pipeline C below), which were explicitly designed to expose *spectral* alterations in the PRNU/residual signal without a reference.
4. Compare each block's statistic against the **local neighborhood median** (not a fixed global threshold — natural images have spatially-varying baseline noise levels, and Chen et al.'s own correlation-predictor work, Pipeline B step 6, demonstrates the same value of a raw statistic means different things in different texture/intensity contexts). Flag blocks whose statistic deviates significantly from their neighborhood.
5. Aggregate flagged blocks into a heatmap and a scalar summary (see Output section).

### B. Reference-based PRNU correlation (Chen, Fridrich, Goljan & Lukáš 2008) — used when a source camera is available

**B.1 — PRNU estimation (maximum-likelihood, from N reference images):**

1. For each reference image `I_k`, `k=1,...,N`, compute the noise residual `W_k = I_k − Î_k^(0)` where `Î_k^(0) = F(I_k)` is the Mihcak-wavelet-denoised image.
2. Model `W_k/I_k = K + Ξ_k/I_k` where `Ξ_k[i]` is treated as **white Gaussian noise (WGN)** with variance `σ²` for smooth reference content.
3. **Maximum-likelihood estimator** (derived by maximizing the log-likelihood `L(K)` in Eq. 5 and solving `∂L(K)/∂K=0`):
   ```
   K̂ = Σ_{k=1}^N (W_k·I_k) / Σ_{k=1}^N (I_k)²          [Eq. 6]
   ```
4. **Cramér-Rao lower bound** on estimator variance, confirming the ML estimator is minimum-variance unbiased:
   ```
   var(K̂) ≥ σ² / Σ_{k=1}^N (I_k)²          [Eq. 7]
   ```
   `var(K̂) ~ 1/N` — doubling reference images halves the estimator variance; luminance `I_k` should be as high as possible without saturating (unsaturated pixels near 255 carry the most information per Eq. 7's denominator).

**B.2 — PRNU preprocessing (critical, often-omitted step):**

5. Suppress artifacts shared across same-brand/same-sensor cameras: for **column/row-wise bias** (from CFA color interpolation and row/column-wise sensor readout), subtract the column average from each pixel (per color channel, per column), then subtract the row average from each pixel (per row):
   ```
   ZM(K̂) = K̂ with zero mean enforced in every row and every column
   ```
   The discarded component `LP(K̂) = K̂ − ZM(K̂)` (the "linear pattern") is itself weak (SNR below −10dB for compact/SLR cameras, stronger for cheap/cellphone cameras) but can measurably inflate false camera-match rates if not removed.
6. For **polygonal/structured sensor-design artifacts** visible as peaks in the Fourier transform of `ZM(K̂)`, apply a **Wiener filter in the Fourier domain**, keeping only the **noise component**:
   ```
   WF(ZM(K̂)) = ℱ⁻¹{ ℱ(ZM(K̂)) − W(ℱ(ZM(K̂))) }
   ```
   `W` = a 3×3 Wiener filter with variance obtained from the sample variance of the magnitude of `ℱ(ZM(K̂))`.
7. **Empirical validation of preprocessing effectiveness** (Chen et al.'s own cross-camera-correlation check, Tables I–II): raw `K̂` cross-correlation between two different-model cameras sharing the same sensor (Canon G2 vs. S40) dropped from **0.0134–0.0251** (raw) to **0.0007–0.0013** (after `ZM` + Wiener) — i.e., roughly a **10–20× reduction** in spurious cross-camera correlation, directly validating the necessity of this preprocessing stage.

**B.3 — Detection (hypothesis testing):**

8. Model the region under test as either `H0: W = Ξ` (noise only, no PRNU — tampered/foreign) or `H1: W = TX + Ξ` (PRNU present — authentic), where `X = IK̂` is the nonattenuated PRNU term, `T[i]` is a pixel-wise multiplicative attenuation factor, and `Ξ` is colored Gaussian noise.
9. Divide the image into `M` disjoint blocks; within block `b`, the **optimal detector is the normalized generalized matched filter**:
   ```
   ρ = Σ_{b=1}^M β_b·ρ_b          [Eq. 11]
   ρ_b = corr(X_b, W_b)          [normalized correlation within block b]
   β_b = (T̂_b/σ̂²_b)·‖X_b‖ / sqrt( Σ_i (T̂²_i/σ̂²_i)‖X_i‖² ) · sqrt( Σ_i (1/σ̂²_i)‖W_i‖² )
   ```
10. **Shaping factor and variance estimation** — because `T_b` and `σ²_b` are unknown per block, they are obtained from a **trained predictor** (Pipeline B.4 below) applied to `ρ̂_b`, then:
    ```
    σ̂²_b = (1/(c²|B_b|))·‖W_b‖²·(1−ρ̂²_b)          [Eq. 15]
    T̂_b = (ρ̂_b/a)·‖W_b‖ / ‖X_b‖
    ```
11. **Neyman-Pearson decision**: set a threshold on `ρ` (or per-block `ρ_b`) achieving a chosen false-alarm rate (FAR); the distribution of `ρ_b` under `H0` (no PRNU) is estimated empirically from many images against a **different** camera's `K̂`; the distribution under `H1` uses the correlation predictor (below) to obtain `p(x|H1)`, since it is not tractable to fit one simple parametric model (the paper fits a **Generalized Gaussian (GG)** distribution to the empirical test statistic as a practical approximation, validated visually against log-tail plots in Fig. 4–6).

**B.4 — Correlation predictor** (needed because `ρ_b` is only known empirically under `H1`, not analytically):

12. The predictor maps three block-level features to a predicted correlation `ρ̂_b ∈ [0,1]`:
    - **Intensity feature**:
      ```
      f_I = (1/|B_b|)·Σ_{i∈B_b} att(I[i])          [Eq. 18]
      att(x) = { e^{-(x−I_crit)²/τ},  x > I_crit
               { x/I_crit,            x ≤ I_crit          [Eq. 19]
      ```
      `I_crit` and `τ` are camera-dependent constants found by brute-force search (paper's example for a Canon G2: `I_crit=250, τ=6`, searched over `I_crit ∈ [230,255], τ ∈ [3,8]`).
    - **Texture feature**:
      ```
      f_T = (1/|B_b|)·Σ_{i∈B_b} 1/(1+var_5(F[i]))          [Eq. 20]
      ```
      `var_5(F[i])` = variance of the high-pass-filtered image `F` (conveniently, the intermediate wavelet LH/HL/HH data from the denoising filter itself, summed over the two outermost subbands, 6 subbands total) in a 5×5 neighborhood of pixel `i`.
    - **Signal-flattening feature**:
      ```
      f_S = (1/|B_b|)·|{ i ∈ B_b : σ_I[i] < c·I[i] }|          [Eq. 21]
      ```
      fraction of pixels in the block whose local intensity variance `σ²_I[i]` (5×5 neighborhood) falls below a threshold scaled by pixel intensity; `c` is a constant depending on the PRNU's variance (paper's example: `c=0.03` for Canon G2). Captures regions attenuated by low-pass processing (e.g. JPEG compression) that would otherwise cause the predictor to *overestimate* correlation.
    - **Combined feature** (used in the paper's actual predictor, via polynomial multivariate least-squares fitting of `ρ` against `f_I, f_T, f_S`, and their combination):
      ```
      f_TI = (1/|B_b|)·Σ_{i∈B_b} att(I[i]) / (1+var_5(F[i]))          [Eq. 22]
      ```

### C. Blind cell-based PRNU spectral analysis (Debiasi, Scherhag, Rathgeb, Uhl & Busch 2018) — no reference camera needed

1. **PRNU extraction**: as in Pipeline B (Mihcak filter), **plus FDR (Filtering Distortion Removal) enhancement** (Lin et al.) as a second-stage SNR improvement step that discards components severely contaminated by filtering errors introduced during denoising.
2. **PRNU splitting**: divide the extracted PRNU into `N` equisized rectangular cells (grid configurations tested from the whole image as one `1×1` cell up to **`10×10` = 100 cells**).
3. **Cell-wise DFT feature extraction**: for each cell `C_n`, compute the 2-D DFT magnitude spectrum, then its histogram `H` (paper's implementation: magnitudes constrained to a universal range `[0,8]`, divided into **100 bins**, range established empirically from the DFT-of-all-extracted-PRNUs distribution). Extract one scalar feature per cell using any of:
   ```
   P_val = max_{n=1..b} H(n)                              [Eq. 2, peak height/relative frequency]
   P_pos = argmax_{n=1..b} H(n)                            [Eq. 3, peak bin position]
   P_pv  = max_{n=1..b} H(n) · argmax_{n=1..b} H(n)        [Eq. 4, combined]
   ```
4. **Cell aggregation** into one global scalar `S` per image — two best-performing strategies (of several tried):
   ```
   S_mean = (1/N)·Σ_{n=1}^N P_n          [Eq. 5]
   S_rms  = sqrt( (1/N)·Σ_{n=1}^N P_n² )          [Eq. 6]
   ```
5. **Decision**: simple threshold on `S` — this is the entire classification step, no trained classifier beyond an empirically-chosen scalar cutoff (D-EER, the equal-error-rate operating point, used to report performance).

### D. Coarse FFT-spectrum noise-type triage (Jain & Arolkar 2024) — cheap gate, not a localization tool

1. Convert to grayscale (mean across channels).
2. Compute the FFT magnitude spectrum: `F[u,v] = ΣΣ f(x,y)·exp(−j2π(ux/N+vy/M))`.
3. Compute the **mean magnitude** and **standard deviation** of the spectrum over the whole image.
4. `z_score = (mean_magnitude − avg_mean_magnitude_train) / std_dev`, where `avg_mean_magnitude_train` is computed once from a labeled training set per noise category.
5. **Decision rule — as literally printed in the paper is self-contradictory**: the pseudocode states `if z_score < threshold: Gaussian; elif z_score > threshold: Impulse; else: No Noise` (a single `threshold` variable used in two branches), while the surrounding prose states the rule as "`z < −1.0` ⟹ Gaussian; `z > −1.0` ⟹ Impulse; `−1.0 ≤ z ≤ 1.0` ⟹ No noise" — the second and third clauses **overlap** (`z=0`, for instance, satisfies both "`z > −1.0`" ⟹ Impulse and "`−1.0 ≤ z ≤ 1.0`" ⟹ No noise) as literally written. `(Corpus ambiguity, not resolved by the source paper itself.)` **Most sensible resolution** (engineering interpretation, treating the pseudocode's two branches as using two distinct thresholds `−1.0` and `+1.0`): `z < −1.0 → Gaussian blur; z > +1.0 → Impulse; −1.0 ≤ z ≤ +1.0 → No noise`. Implement this resolved version, not the literal contradictory text.

## Output

- **Pipeline A**: per-block anomaly score (e.g. z-scored deviation of local noise variance from neighborhood median), aggregable into a `[0,1]`-normalized heatmap and a global scalar (max or top-k% mean). `(Calibration not specified in corpus — engineering recommendation.)`
- **Pipeline B**: `ρ_b ∈ [-1,1]` (correlation, in practice concentrated near 0 for `H0` and positive for `H1`) per block, or the aggregate `ρ`; final binary decision via Neyman-Pearson threshold at a chosen FAR. For a `[0,1]` fusion-layer score: `(not specified in corpus)` — engineering recommendation: `1 − ρ_b` normalized via the empirical `H0`/`H1` distributions the paper already constructs (GG-fit CDF), giving an actual calibrated tampering probability rather than an ad hoc rescaling.
- **Pipeline C**: raw `S_mean`/`S_rms` (unbounded scalar, dataset/camera-dependent scale) compared to a D-EER-calibrated threshold. Same "not specified in corpus" caveat applies for a general `[0,1]` fusion score.
- **Pipeline D**: discrete label (`Gaussian blur` / `Impulse` / `No noise`) — not a continuous score; use only as a categorical gate (e.g. "this region was likely denoised — down-weight the PRNU modules' confidence here") rather than as fusion-layer evidence in its own right, especially given its low measured accuracy (see below).

## Key findings from papers

**Manipulation types detected best**: Pipeline B — splicing from a different camera/sensor into a host image, localized down to 128×128-ish blocks under favorable (RAW/high-quality) conditions. Pipeline C — face morphing specifically (its validated task), and by direct extension, any manipulation that perturbs the *spectral* character of the PRNU (non-linear warping, averaging/blending operations) even without a reference image.

**Documented failure cases / limitations**:
- Chen et al.: explicitly state some **content-preserving malicious edits are undetectable by this method entirely** — their example: changing the color of a stain to look like a blood stain, which alters no noise statistics at all. PRNU-based detection is fundamentally blind to edits that don't disturb the sensor-noise layer.
- Chen et al.: FRR degrades by **orders of magnitude** as processing intensifies (RAW → JPEG90 → JPEG75 → Wiener-filtered+JPEG90 → gamma-corrected+JPEG90 → scaled+JPEG90) — see benchmark table; the Wiener-filtered condition specifically simulates a denoising counter-attack and shows some of the worst degradation.
- Debiasi et al.: the method is **robust to scaling and sharpening** but **fails against histogram equalization (EQU)** — D-EER only improvable to 11.9% even with 8×8 cell fragmentation, vs. 0.7–2.2% for other post-processing types; the paper states explicitly "further improvement of the detection algorithms is needed to counter this type of post-processing."
- Debiasi et al.: performance is **non-monotonic in cell count** — 8×8 cells is the sweet spot; 10×10 is *worse* than 8×8 for several feature/aggregation combinations (over-fragmentation apparently degrades the per-cell DFT statistic's reliability).
- Jain & Arolkar: low absolute accuracy even on its own narrow 3-class task (see table) — should be treated as a weak triage signal only, on a very small test set (40 images).

**Benchmark tables**:

| Paper | Dataset | Metric | Value | Conditions |
|---|---|---|---|---|
| Chen 2008 | Canon G2 vs. Canon S40 (same sensor, different camera model) | Cross-camera correlation | 0.0134–0.0251 (raw) → 0.0007–0.0013 (after ZM+Wiener) | Validates the necessity of PRNU preprocessing |
| Chen 2008 | 8 test cameras (Canon S40/G2, Olympus C765-1/C765-2/C3030, Sigma SD9) | FRR at FAR=10⁻⁵ | e.g. Canon S40, block 1: 3.5e-2 (RAW) → 9.6e-2 (Wiener+JPG90) → 1.7e-3 (γ=0.5+JPG90) | Full FRR tables V–X, 8 blocks × 6 processing conditions × 6 cameras |
| Chen 2008 | 345 forgeries, Canon G2 | Localization success | 85% of forgeries had ≥2/3 of tampered region correctly localized | JPEG-90, qualitative example in Fig. 9 |
| Debiasi 2018 | FRGCv2, 961 bona fide + 2,414 morphed faces | D-EER | 2.1% (1×1, whole-image PRNU) down to **1.4%** (8×8 cells, best config) | `P_pos|S_mean` and `P_pos|S_rms`, 8×8 cells — best overall |
| Debiasi 2018 | same | D-EER under post-processing | 0.7–2.2% (scaling), 10.8% (sharpening), **11.9%** (histogram equalization, best achievable) | 8×8 cells |
| Debiasi 2018 | same | BPCER10 / BPCER20 | as low as 0.0% / 0.0% (scaled images, 8×8 cells) up to 78.4% / 85.9% (EQU, 1×1 cell) | Full Table II |
| Jain & Arolkar 2024 | Kadid10k-derived, 40 test images | Accuracy per class | Gaussian 81.25% (13/16), No-noise 60.00% (9/15), Impulse 55.56% (5/9) | Small test set; training set >300 images |

## Implementation notes

- **Block-size disagreement across sources** (explicit, not resolved by either paper — they validate different tasks at different scales): Chen et al. recommend **128×128** for typical ≥1-megapixel camera images (a tradeoff between the assumption of local stationarity of `T` and `σ²` holding, and having enough samples for statistical significance); Debiasi et al. work with much smaller **face-crop cells down to 1/64th of a 320×320 crop** (i.e., 40×40-ish) because their target images and task (face morph detection) are inherently smaller-scale. For this engine, treat block size as a **tunable parameter conditioned on input resolution**, not a fixed constant — start from Chen et al.'s 128×128 default for full-frame images and scale down proportionally for crops/small inputs, consistent with Debiasi et al.'s finding that finer fragmentation (up to a point — 8×8, not 10×10) improves sensitivity to *localized* alterations.
- **The zero-mean + Wiener-filter PRNU preprocessing step (Pipeline B.2) is easy to accidentally skip** since it doesn't change the PRNU's *appearance* dramatically — but Chen et al.'s own Table I/II numbers show it is responsible for a 10–20× reduction in spurious correlation. Always apply it before any cross-image or cross-region comparison, including the blind Pipeline A/C variants (both papers' methods build on the same underlying Mihcak-filter extraction).
- **`I_crit`, `τ`, `c` (Pipeline B.4) are camera-specific constants** found by brute-force search in the source paper — they do not transfer directly across camera models; recalibrate per deployment camera/dataset if reference-based PRNU (Pipeline B) is used, or avoid the correlation-predictor machinery entirely by using the blind Pipelines A/C instead, which need no camera-specific tuning.
- **The DFT magnitude range `[0,8]` and 100-bin histogram (Pipeline C step 3) are dataset-specific constants** (Debiasi et al. state the range was established from their own dataset's extracted PRNUs) — recalibrate the bin range for a different image resolution/dataset rather than assuming these exact numbers transfer.
- **No public reference code found in the extracted text** for any of the four papers, though Chen et al.'s method is widely known in the forensics community as the basis for later public PRNU toolboxes (not named in this paper's own text).
- **Recommended Python libraries**:
  - `pywt` for the wavelet-based denoising filter (Mihcak-style) — `pywt.wavedec2`/`pywt.waverec2` with soft/hard thresholding per subband, consistent with this engine's `wavelet analysis` module's own noise-estimation machinery (shared infrastructure opportunity).
  - `numpy.fft.fft2` for the cell-wise DFT magnitude spectrum (Pipeline C).
  - `scipy.ndimage.generic_filter` or a manual sliding-window implementation for the 5×5 local-variance features (Eq. 20–21).
  - `scipy.optimize.curve_fit` for fitting the Generalized Gaussian distribution to the test-statistic tails (Pipeline B.3's `p(x|H0)`/`p(x|H1)` density estimation).
  - `numpy.polynomial` or `sklearn.preprocessing.PolynomialFeatures` + `numpy.linalg.lstsq` for the polynomial multivariate least-squares correlation predictor (Pipeline B.4) — note this is a **regression fit**, not a trained classifier in the ML sense used elsewhere in this engine's no-ML constraint discussion, but it does require a calibration dataset of known-camera images to fit against; treat it as camera-specific calibration, analogous to the calibration steps used in other modules, not as an excluded ML technique.

## Key references

- **chen2008.pdf** — M. Chen, J. Fridrich, M. Goljan, J. Lukáš, "Determining Image Origin and Integrity Using Sensor Noise," IEEE TIFS, vol. 3, no. 1, March 2008. Source of: the complete sensor-output model (Eq. 1–2), the ML PRNU estimator and CRLB (Eq. 6–7), the zero-mean/Wiener preprocessing steps and their validated effectiveness (Tables I–II), the hypothesis-testing detection framework (Eq. 8–17), the correlation predictor and all three block-level features (Eq. 18–22), and the full FRR benchmark tables across 6 cameras × 6 processing conditions.
- **debiasi2018.pdf** — L. Debiasi, U. Scherhag, C. Rathgeb, A. Uhl, C. Busch, "PRNU-based Detection of Morphed Face Images," WaveLab/Hochschule Darmstadt, 2018. Source of: the FDR-enhanced blind PRNU extraction, the cell-splitting scheme, the DFT-magnitude-histogram peak features (Eq. 2–4), the mean/RMS aggregation (Eq. 5–6), and the full D-EER/BPCER benchmark tables including post-processing robustness.
- **Gaussian-and-Impulse-Noise-Identification-from-Image-Using-Frequency-Domain-Analysis.pdf** — A. Jain, H. Arolkar, GLS University, in *Smart Trends in Computing and Communications* (LNNS 946), Springer, 2024. Source of: the FFT-spectrum mean/std-dev noise-type triage and its (internally ambiguous) z-score decision rule, resolved above; the low-accuracy benchmark on a 40-image test set.
- **Fusing-Multi-scale-Attention-and-Transformer-for-Detection-and-Localization-of-Image-Splicing.pdf** — Y. Xu, J. Zheng, C. Shao, BICS 2023/2024. `[ML]`, marginal — not noise-based; not re-read in this pass.
- **Keypoint-Based-Tampered-Image-Identification.pdf** — G.G. Rajput et al., Smart Trends LNNS 946, 2024. `[ML]`, marginal — geometric/keypoint copy-move detection, not noise analysis; not re-read in this pass.
- **Policy-Gradient-Driven-Noise-Mask.pdf** — M.C. Yavuz, Y. Yang, ICPR 2024/2025. `[ML]`, marginal — noise used as a training-time regularizer for medical classifiers, not forensic analysis; not re-read in this pass.
- **Digital-Image-Forgery-Detection.pdf** — V. Tyagi, SpringerBriefs, Springer Nature Singapore, 2026. Background survey only; not re-read in this pass.
