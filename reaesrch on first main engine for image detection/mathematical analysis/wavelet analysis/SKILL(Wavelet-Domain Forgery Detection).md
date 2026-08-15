# Wavelet-Domain Forgery Detection

## Core mathematical principle

**Underlying concept.** A 2-D discrete wavelet transform decomposes an image into a low-frequency approximation subband (`LL`) and three high-frequency detail subbands (`LH`, `HL`, `HH` — horizontal, vertical, diagonal), at each of several decomposition levels, giving a **multi-resolution, spatially-localized** split of image content. Genuine, unprocessed high-frequency wavelet coefficients (sensor noise, fine texture) follow a sharply-peaked **Laplacian distribution** (Stamm & Liu, Eq. 2): `P(X=x) = (λ/2)·e^{−λ|x|}`. This is the same class of statistical model (Laplacian/generalized-Gaussian) underpinning this engine's Benford and JPEG-compression modules — wavelet coefficients and block-DCT coefficients share the same underlying natural-image statistics story.

**Property exploited.** Lossy wavelet-based compression (SPIHT, EZW, JPEG2000) achieves its compression by **scalar quantization** of wavelet coefficients — clustering the continuous Laplacian population onto a discrete set of reconstruction values `{q_k}`. This is directly analogous to JPEG's DCT-domain quantization and produces the same kind of forensically-detectable clustering artifact in the coefficient histogram (Stamm & Liu's Fig. 1 shows an unmistakable visual contrast: a smooth unimodal peak for an uncompressed image's `HL` coefficients vs. a comb of sharp, evenly-spaced spikes after SPIHT compression). Separately, the **high-pass detail subbands carry the forensically-useful noise residual** of an image — extracting and thresholding them is the standard way to isolate sensor-noise/PRNU-like signal from image content, feeding this engine's `noise analysis` module. And the **coarse (LL) subband**, being a compressed, denoised summary of image structure, is a computationally cheap domain in which to search for duplicated (copy-moved) regions.

## Input requirements

- **Format**: grayscale or single-channel (all three forensically-relevant papers here — Kashyap & Joshi, Stamm & Liu, and the threshold-formula papers — operate on grayscale, though Kashyap & Joshi note their moment-invariant features extend to RGB by concatenating per-channel feature vectors, `72 = 24×3`-dimensional).
- **Preprocessing**: 2-D DWT (or SWT) decomposition to a chosen level and wavelet family (see Step-by-step algorithm for the genuine, sourced disagreement across papers on both parameters); block tiling for copy-move search (Kashyap & Joshi tile the **coarse LL subband**, not the full-resolution image, specifically to reduce search space).
- **Reliable when**: sufficient image resolution for stable per-subband/per-block statistics; content has not been aggressively smoothed/denoised after tampering (defeats the noise-residual path); copy-move regions are **not** scaled or rotated (Kashyap & Joshi's explicit, stated limitation of their specific block-matching pipeline).
- **Unreliable / inapplicable when**:
  - **Anti-forensic dithering has been applied** — Stamm & Liu's own paper is a complete, quantitatively-validated demonstration that the wavelet-compression-history detector this module would otherwise rely on can be **100% defeated** (see Key Findings) by an attacker with knowledge of the detection method.
  - **Scaled or rotated copy-move regions** — the Kashyap & Joshi block-matching pipeline is explicitly stated to not handle these (left as future work in the source).
  - **Speckle-noise-dominated content** (e.g. certain latent-fingerprint/SD-OCT sensor modalities) — the median-based noise estimator assumes additive, roughly-Gaussian noise and mgaga2019's own review notes speckle-specific techniques (NAWT) are needed for that noise type specifically.

## Step-by-step algorithm

Three distinct forensic pipelines: **(A)** wavelet noise-residual extraction (feeds this engine's `noise analysis` module — training-free, purely statistical); **(B)** wavelet-compression-history detection (training-free, but with a fully-worked, quantitatively-validated adversarial defeat documented in the same source paper); **(C)** Haar-DWT copy-move block-matching (training-free, PCA-based, fully self-contained).

### A. Wavelet noise-residual extraction — feeds the `noise analysis` module

`(This pipeline synthesizes formulas from a REVIEW paper — mgaga2019.pdf — which itself surveys four distinct prior studies, each with its own dataset/wavelet-filter/decomposition-level choice. The formulas below are accurately sourced, but note explicitly where "best" choices are drawn from different, non-directly-comparable studies within that review, rather than from one single controlled experiment.)`

1. Decompose the image via **DWT or SWT** to `N` levels (see wavelet-family/level guidance below).
2. **Robust noise-standard-deviation estimator** from the diagonal (`HH`) detail subband at level `i` (this exact formula recurs across multiple studies reviewed in mgaga2019 and is also implicit in Stamm & Liu's underlying model):
   ```
   σ_{W_Φ} = median(|HH_i|) / 0.6745,   i = 1,...,k          [mgaga2019, Eq. 7]
   ```
   The constant `0.6745` is the 75th-percentile point of the standard half-normal distribution — dividing by it converts the median absolute value of (assumed zero-mean Gaussian) noise coefficients into an unbiased estimate of `σ`, the standard median-absolute-deviation (MAD) noise estimator.
3. **Signal-vs-noise variance separation**, needed to compute an adaptive (not fixed) threshold — from the same review's synthesis of Yinping et al.'s adaptive-threshold method:
   ```
   G(x,y) = F(x,y) + Φ(x,y)                              [Eq. 5, noisy = clean + noise]
   W_G = W_F + W_Φ                                        [Eq. 6, in the wavelet domain]
   σ²_{W_G}(D,i) = (1/N(i)²)·Σ_{x=1}^{N(i)}Σ_{y=1}^{N(i)} W_G(D,i)²          [Eq. 8, noisy-image variance per direction D∈{H,V,diag}, per level i]
   σ²_{W_G} = σ²_{W_F} + σ²_{W_Φ}                          [Eq. 9]
   σ_{W_F}(D,i) = sqrt( max(σ²_{W_G}(D,i) − σ²_{W_Φ}(i), 0) )          [Eq. 10, clean-signal std. dev., clipped at 0]
   ```
4. **Threshold selection** — apply to the detail-subband coefficients before extracting the residual. Three distinct formulas are given across the sources reviewed (genuine disagreement — use per the stated tradeoff, not a single "best" value):
   - **VisuShrink / universal threshold**: `T = σ√(2·log(N))` — `N` = number of high-frequency coefficients from the DWT (mgaga2019, Eq. 3, itself citing Donoho's classical 1995 result).
   - **BayesShrink**: `T = σ²_noise / σ_signal` (mgaga2019, Eq. 4).
   - **Golden-ratio-modified universal threshold**: `T = σ√(1.618·log(N))` (mgaga2019, Eq. 13, reviewing Sasirekha et al. — replaces VisuShrink's constant `2` with the golden ratio `1.618`, combined with a **weighted-median** noise estimator using weight function `W(x,y) = 1/exp(HH(x,y))` applied to the diagonal subband before the median-based `σ` computation, Eq. 14–16).
   - **Streamlined/piecewise shrinkage** (Zhu & Wang 2021 — a genuinely different, smoother formula, not part of the mgaga2019 review):
     ```
     ŵ_λ = { w,                     |w| ≥ λ1
           { w·(1 − |λ2/w|^n),      λ2 ≤ |w| < λ1
           { 0,                     |w| < λ2                    [Eq. 3]
     λ1 = σ√(2ln(N) + 2ln(ln(N))),   λ2 = α·σ√(2ln(N) + 2ln(ln(N)))
     ```
     `α ∈ [0,1]` and `n` (positive integer) are tunable; paper's own experiment uses **`α=0.5, n=2`**, 3-level **sym4** decomposition. This form blends smoothly between hard-threshold behavior (above `λ1`) and zero (below `λ2`), avoiding both the hard threshold's discontinuity (Gibbs-effect risk) and the soft threshold's constant bias at large coefficient values.
5. Apply hard or soft thresholding using the selected `T`/`λ` (classical forms, mgaga2019 Eq. 1–2 / Zhu & Wang Eq. 1–2, identical definitions across both sources):
   ```
   hard:  f(x) = { x, |x|≥T; 0, |x|<T }
   soft:  f(x) = { x−T, x>T; 0, |x|≤T; x+T, x<−T }
   ```
6. The **denoising residual** — original coefficients minus thresholded coefficients — is the forensic signal passed to the `noise analysis` module for local noise-level comparison across image regions.

### B. Wavelet-compression-history detection and its documented anti-forensic defeat (Stamm & Liu 2010)

**B1 — Detection side (what a detector looks for; the paper reviews this as the target it attacks, citing Lin et al. 2009's classifier achieving 99.6% baseline accuracy):**
1. Compute the `N`-level 2-D DWT of the image, producing `LL`, `LH`, `HL`, `HH` subbands per level (tree-based schemes like SPIHT/EZW additionally organize coefficients into significance-map bitplanes, but the resulting coefficient-histogram artifact is the same).
2. Model uncompressed coefficients within a subband as Laplacian: `P(X=x) = (λ/2)e^{−λ|x|}` (Eq. 2).
3. Model quantization as mapping each coefficient to the nearest of a set of quantized values `{...,q_{-1},q_0,q_1,...}` with boundaries `{...,b_{-1},b_0,b_1,...}`: `Y=q_k` if `b_k ≤ X < b_{k+1}` (Eq. 1). Integrating the Laplacian pdf over each quantization bin gives the **compressed-coefficient probability mass function**:
   ```
   P(Y=q_k) = { ½(e^{−λb_k} − e^{−λb_{k+1}}),   k≥1
              { 1 − ½(e^{λb0}+e^{−λb1}),         k=0
              { ½(e^{λb_{k+1}} − e^{λb_k}),       k≤−1          [Eq. 3]
   ```
   This produces the sharp, evenly-spaced comb-histogram signature (visually confirmed in Fig. 1) that a forensic detector keys off.

**B2 — The anti-forensic attack (this paper's actual contribution — reverses the detector):**
1. **Estimate the pre-compression Laplacian parameter `λ̂`** for each subband by fitting the compressed coefficient histogram `h_k` (observed count at quantized value `q_k`) to the model `h_k = c·e^{−λ̂|q_k|}` (Eq. 4) via **weighted least squares** on the log-linearized form:
   ```
   min_{λ̂,c} Σ_k h_k·(log(h_k) − log(c) + λ̂|q_k|)²          [Eq. 5]
   ```
   yielding a closed-form 2×2 linear system (Eq. 6) solvable directly for `log(c)` and `λ̂`.
2. **Iteratively correct for tree-based-compression bias**: because significance-map bitplane truncation biases the histogram toward zero-valued coefficients (successive bitplanes are truncated at boundaries that don't align with quantization-bin boundaries), the naive `λ̂` estimate is systematically off. Correct via the iterative update:
   ```
   ĥ_k^{(i)} = { c^{(i)},              k=0
               { h_k + ½(h_0 − c^{(i)}),  k=±1
               { h_k,                   otherwise          [Eq. 7]
   ```
   re-fit `λ̂^{(i)}`, `c^{(i)}` from `ĥ_k^{(i)}` each iteration, terminate when `|λ̂^{(i)}−λ̂^{(i-1)}|/λ̂^{(i)} < τ` (`τ` a user-defined tolerance).
3. **Construct the anti-forensic dither distribution** — additive noise `D`, conditionally dependent on the coefficient value `Y` it's added to, chosen so the resulting distribution of `Z=Y+D` (Eq. 9) matches the *estimated pre-compression* Laplacian exactly (provided `λ̂=λ`):
   ```
   P(D=d | Y=q_k, k≠0) = { (1/α_k)·e^{−sgn(q_k)·λ̂·d},  (b_k−q_k) ≤ d < (b_{k+1}−q_k)
                          { 0,  otherwise                    [Eq. 10]
   P(D=d | Y=0) = { (1/α_0)·e^{−λ̂|d|},  b0 > d > b1
                  { 0,  otherwise                             [Eq. 11]
   ```
   with normalizing constants `α_k`, `α_0` given in closed form (Eq. 10–11 companion terms). This is proven (Eq. 12, combining Eq. 3/10/11) to yield exactly `P(Z=z) = (λ/2)e^{−λ|z|}` — i.e., **the dithered coefficients are provably indistinguishable from a genuine uncompressed Laplacian population**, when `λ̂=λ`.
4. **Zero-coefficient correction**: separately compute the number of "excess" zero-valued coefficients `N_e = h_0 − N_s(1 − ½(e^{λ̂b0}+e^{λ̂b1}))` (Eq. 8, `N_s` = total coefficients in the subband); randomly select `N_e` zero-valued coefficients and change half to `q_1`-range values, half to `q_{-1}`-range values, before applying dither, to correct the histogram's zero-bin over-representation from bitplane truncation.
5. Add the constructed dither `D` to every coefficient: `Z = Y + D` (Eq. 9).
6. **Bounded distortion guarantee**: because the dither distribution's support is restricted to the interval `(b_k, b_{k+1})` around each quantized value, the paper derives an explicit bound on how far the anti-forensically-modified coefficient can differ from the *original pre-compression* coefficient `X` (Eq. 13) — the attack is provably visually gentle, not just empirically so.

### C. Haar-DWT copy-move block matching via blur-moment invariants (Kashyap & Joshi 2013)

1. **Single-level Haar DWT**, keeping only the **coarse `LL` subband** to cut data volume while preserving structure:
   ```
   ψ(x) = Σ_{k} (−1)^k a_{N−1−k}√2·φ(2x−k)          [Eq. 1, Haar wavelet function]
   f(x,y) = Σ_j Σ_k Σ_l d_{j,k,l}·ψ_{j,k}(x)·ψ_{j,l}(y)          [Eq. 2, 2-D decomposition]
   ```
2. **Tile the coarse subband** into overlapping `R×R` blocks, sliding by 1 pixel horizontally then vertically (`(M−R+1)×(N−R+1)` total blocks for an `M×N` image).
3. **Model the tampering-introduced blur** as a convolution with an unknown point-spread function (PSF) `h`:
   ```
   g(x,y) = (f*h)(x,y) + n(x,y)          [Eq. 3, duplicated region g = original f convolved with PSF h, plus noise]
   ```
   PSF assumed axially symmetric and energy-preserving: `h(x,y)=h(−x,−y)=h(y,x)`, `∫∫h(x,y)dxdy=1` (Eq. 4–5).
4. **Compute image moments** per block — raw moment `m_pq = ∫∫x^p·y^q·f(x,y)dxdy` (Eq. 6), central moment `μ_pq = ∫∫(x−x_t)^p(y−y_t)^q·f(x,y)dxdy` with centroid `(x_t,y_t)=(m_{10}/m_{00}, m_{01}/m_{00})` (Eq. 7–8). Under the convolution model, the central moments of the blurred/duplicated block relate to the original block's moments via the binomial convolution identity:
   ```
   μ_pq^{(g)} = Σ_{k=0}^p Σ_{j=0}^q C(p,k)·C(q,j)·μ_{kj}^{(f)}·μ_{p−k,q−j}^{(h)}          [Eq. 10]
   ```
5. **Blur-invariant construction** — features `B(p,q)` satisfying `B^{(f)} = B^{(f*h)} = B^{(g)}` (Eq. 11, invariant under the unknown blur), built recursively from central moments:
   ```
   B(p,q) = μ_pq − α·μ_pq·(1/μ_00)·Σ_{n=0}^{k} Σ_{i=m1}^{m2} C(t−2i,2i)·C(q,2i)·B(p−t+2i, q−2i)·μ_{t=2i,2i}          [Eq. 12]
   ```
   with `k = ⌊(p+q−4)/2⌋`, `t = 2(k−n+1)`, `m1 = max(0, ⌊(t−p+1)/2⌋)`, `m2 = min(t/2, ⌊q/2⌋)` (Eq. 13–14), and `α=1 ⟺ p∧q even`, `α=0 ⟺ p∨q odd` (Eq. 15) — a recursive construction rule, not a closed enumerated list; the paper uses **24 blur invariants up to 7th order** built from this recursion, `B={B_1,...,B_24}` (Eq. 16), giving a **72-dimensional** feature vector for RGB (24 per channel, concatenated: `B_rgb = {B_blue, B_red, B_green}`).
6. **Contrast-normalized invariants** (also invariant to contrast change, improving robustness): `B'_i = B_i / (R/2)^r·μ_00`, `R` = block size, `r` = order of `B_i` (Eq. 17).
7. **PCA dimensionality reduction**: project the `m`-dimensional invariant vector `X⃗` onto the top eigenvectors of its covariance/scatter, keeping only `m_0 ≪ m` components (Eq. 18–26, standard PCT/PCA formulation — eigen-decomposition of `R = E[X⃗X⃗ᵀ]`, keep eigenvectors with the largest eigenvalues `λ_i`).
8. **Block similarity measure**:
   ```
   S(B_i,B_j) = 1 / (1 + ρ(B_i,B_j))          [Eq. 27]
   ρ(B_i,B_j) = ( Σ_{k=1}^{dim} (B_i[k]−B_j[k])² )^{1/2}          [Eq. 28, Euclidean distance in feature space]
   ```
   Two blocks are candidate duplicates if `S(B_i,B_j) ≥ T` (a user/image-characteristic-dependent similarity threshold).
9. **Neighborhood consistency check** (rejects coincidental single-block matches — a *necessary but not sufficient* condition from step 8 alone): examine **16 neighboring blocks** within **maximum distance 4 pixels** — `x,y ∈ {−4,−3,...,4}`, i.e. `S(block(i+x_r,j+y_r), block(k+x_r,l+y_r)) ≥ T` for `r=1,...,16` (Eq. 29) — **and** spatial separation between the original candidate blocks must exceed a minimum distance `D`:
   ```
   sqrt((i−k)² + (j−l)²) ≤ D          [Eq. 30, note: paper states this as the CONDITION FOR REJECTING trivial near-identical neighbors in smooth regions — i.e., blocks closer than D are excluded from being flagged as "duplicated," to avoid flagging flat/smooth-region self-similarity as tampering]
   ```
10. **Duplicated-region map**: build a same-size binary matrix `Q`, set to `1` at positions where the neighborhood-consistency check (step 9) confirms genuine duplication; multiply elementwise with the original image to visualize the flagged **duplicated regions map**.

## Output

- **Pipeline A**: the extracted noise-residual image itself (not a scalar score on its own) — this is an **intermediate signal**, consumed by the `noise analysis` module's local-noise-level-comparison logic (see that module's `SKILL.md`) rather than an independent tampering score in its own right.
- **Pipeline B**: as *detection*, a compression-history classification (compressed/uncompressed) with an associated confidence from the fitted Laplacian goodness-of-fit; **the module's more important output for this engine is the awareness that this detector is defeatable** — see Key Findings — so any score this pipeline produces should be treated by the fusion layer as **low-trust by default**, not high-confidence evidence.
- **Pipeline C**: a **binary duplicated-region map** `Q` (direct spatial localization output, not requiring any post-hoc score-to-heatmap conversion) plus, implicitly, a whole-image scalar (fraction of flagged blocks) `(not explicitly defined in the source paper as a summary scalar — engineering recommendation: sum of Q, or largest connected-component size, for a fusion-layer scalar alongside the map itself)`.

## Key findings from papers

**Manipulation types detected best**: Pipeline C — un-scaled, un-rotated copy-move/cloning forgery, robust to the noise/blur/contrast changes commonly introduced during the copy-paste process itself (explicit claim, demonstrated qualitatively on 5 test images in Fig. 3–7). Pipeline A — general noise-inconsistency splicing (via the fed-forward `noise analysis` module). Pipeline B — detecting *that* an image has been through wavelet-based lossy compression, when the detector is *not* under adversarial attack.

**Documented failure cases / limitations**:
- **Pipeline B is completely, quantitatively defeated by its own paper's anti-forensic method**: starting from Lin et al.'s baseline classifier (99.6% accuracy at distinguishing SPIHT-compressed from never-compressed images, on a training set constructed from 244 UCID images), Stamm & Liu's dither attack causes that **same classifier to misclassify every single anti-forensically-modified image as "never-compressed" — a 100% success rate** — while introducing only modest visual distortion (**PSNR = 46.64dB** between the original-SPIHT-compressed and the anti-forensically-modified version, on the standard Lena test image). This is the single most important documented limitation in this entire module: **do not treat "no wavelet-compression-history artifact detected" as strong evidence of anything**, since a knowledgeable adversary can trivially manufacture that exact null result.
- Pipeline C: **86.67% blind detection accuracy** on a 15-image mixed authentic/forged test set (Kashyap & Joshi's own headline number) — not scaled/rotated-copy-move robust; the paper explicitly defers rotation/scale invariance to future work.
- Pipeline A (the review, mgaga2019): individual reviewed studies report method-specific results that are **not directly comparable** to each other (different datasets — NIST fingerprint DB vs. FVC2002 vs. self-acquired SD-OCT images; different noise types — Gaussian vs. speckle vs. salt-and-pepper); mgaga2019's own conclusion states SWT + weighted median is recommended **over** conventional DWT + plain median estimator, but this is a qualitative literature-synthesis recommendation, not a result from one single controlled head-to-head experiment in this corpus.

**Benchmark tables**:

| Paper | Dataset | Metric | Value | Conditions |
|---|---|---|---|---|
| Kashyap & Joshi 2013 | 15 authentic/forged JPEG images (blind mixed test) | Detection accuracy | **86.67%** | Proposed Haar-DWT + blur-invariant + PCA pipeline |
| Kashyap & Joshi 2013 | same | Runtime vs. Mahdian & Saic 2007 baseline | Substantially lower processing time at both R=8 and R=16 block sizes (Fig. 8: e.g. ~600s existing vs. ~50-200s proposed at various images) | Direct execution-time comparison, 5 test images |
| Stamm & Liu 2010 | 244 images, Uncompressed Colour Image Database (UCID), grayscale, SPIHT at 2 bits/pixel | Baseline classifier accuracy (Lin et al. 2009, being attacked) | 99.6% (never-compressed vs. SPIHT-compressed, on training set; 0% false positives on never-compressed) | Trained classifier, pre-attack |
| Stamm & Liu 2010 | same | Anti-forensic attack success rate | **100%** (every anti-forensically-modified image misclassified as never-compressed) | Full-scale test after dither applied |
| Stamm & Liu 2010 | Lena, SPIHT at 3 bits/pixel | Visual distortion cost | PSNR = 46.64dB (original-compressed vs. anti-forensically-modified) | "Very little visual distortion" |
| mgaga2019 (reviewing Yinping et al.) | NIST special database, 256×256 | PSNR improvement | +3.3% (σ=10), +4.7% (σ=15), +5.6% (σ=20) | Adaptive-threshold Shrink vs. conventional thresholding, db4 wavelet, level 3 |
| mgaga2019 (reviewing Zaki et al.) | SD-OCT fingertip samples | SNR | 20.59 (proposed NAWT) vs. 18.99 (Gaussian filter) vs. 17.99 (traditional WT) | sym4, level 4, speckle noise |
| mgaga2019 (reviewing Iqbal) | NIST DB | Qualitative | BayesShrink+level-2+Haar best for Gaussian/speckle; NeighShrink+level-1+Haar best for salt-and-pepper | Stationary WT, 4 wavelet families compared |
| mgaga2019 (reviewing Sasirekha et al.) | FVC2002 | MSE/RMSE/PSNR/SNR | e.g. soft+SYM4 noise=0.001: MSE 1.747893, PSNR 45.74625; hard+DB2: MSE 0.150877, PSNR 56.40315 | Modified (golden-ratio) universal threshold, full Tables I–VI in source |
| Zhu & Wang 2021 | Cameraman, 256×256, σ=10 noise | SNR / MSE | Hard: 28.3545/0.1214; Soft: 28.5821/0.1121; **Improved (piecewise): 29.6684/0.0972** | 3-level sym4, α=0.5, n=2 — improved function wins on both metrics |

## Implementation notes

- **SWT vs. DWT is a genuine, sourced recommendation, not a default assumption**: mgaga2019's review explicitly recommends the **stationary wavelet transform (SWT, translation-invariant, no downsampling)** over conventional decimated DWT specifically because "DWT has the inefficiency of not being translation invariant" — this matters directly for this engine's tamper-*localization* use case (Pipeline A feeding spatial noise-inconsistency maps), where shift-variant DWT residuals would introduce block-boundary artifacts inconsistent with true pixel-level tampering boundaries. Use SWT (via `pywt.swt2`) for any step requiring precise pixel-level localization; plain decimated DWT is acceptable for Pipeline C's coarse-subband block search, where exact shift-invariance is less critical than raw speed.
- **Wavelet family choice is genuinely disputed across sources, not settled**: Haar is cheapest/simplest and used by Kashyap & Joshi for speed in the copy-move search; **db2/sym4** are recommended by mgaga2019's review as "smoother... better for speckle noise" for the noise-estimation path; Stamm & Liu's underlying model is wavelet-family-agnostic (works for any tree-based or scalar-quantized wavelet compressor). Use Haar for Pipeline C (speed matters, coarse-subband search tolerates the extra blockiness), sym4/db2 for Pipeline A's noise-residual extraction (per mgaga2019's specific recommendation).
- **Decomposition level**: 2–3 levels recurs as the practically-validated range across sources (`mgaga2019`'s Table VII summary shows level 1–4 all appear across the four reviewed studies depending on noise type; `Zhu & Wang` use 3-level sym4). No single paper here validates decomposition level in a controlled sweep against ground truth for the *forensic* (as opposed to denoising-quality) use case — treat 2–3 as a reasonable starting default, not a rigorously-derived optimum for this specific task.
- **Odd image dimensions / boundary extension**: none of the four wavelet papers discuss boundary-handling modes explicitly in the extracted text — `(not specified in corpus — engineering recommendation: use symmetric/reflect padding, PyWavelets' default `'symmetric'` mode, to avoid introducing spurious high-frequency energy at image borders that could be mistaken for tampering evidence there).`
- **The `α=1.2, β=0.8`-style tunable constants throughout this module** (Zhu & Wang's `α,n`; Sasirekha et al.'s golden-ratio substitution) are each validated on a **single** test image/dataset in their respective source — treat as reasonable starting points requiring recalibration on this engine's own validation data, not universal constants.
- **No public reference code found in the extracted text** for any of the four papers in this module.
- **Recommended Python libraries**:
  - `pywt.swt2` (stationary/undecimated) and `pywt.dwt2`/`pywt.wavedec2` (decimated) for the core transform — `wavelet='haar'` for Pipeline C, `wavelet='sym4'` or `'db2'` for Pipeline A.
  - `numpy.median` + manual division by `0.6745` for the MAD noise estimator (Eq. 7).
  - `scipy.optimize.lstsq` or direct 2×2 linear solve (`numpy.linalg.solve`) for Stamm & Liu's weighted-least-squares Laplacian fit (Eq. 5–6).
  - `numpy.random` (exponential/Laplace sampling per bin, via inverse-CDF sampling matching Eq. 10–11) if ever reproducing the anti-forensic dither construction for red-team/robustness testing of this engine's own JPEG/wavelet modules.
  - `cv2.moments` gives raw/central image moments directly (though not the recursive blur-invariant construction of Eq. 12, which must be implemented manually) for Pipeline C.
  - `sklearn.decomposition.PCA` (or plain `numpy.linalg.eig` on the feature covariance, matching Eq. 18–26's derivation exactly) for the dimensionality reduction step.

## Key references

- **kashyap2013.pdf** — A. Kashyap, S.D. Joshi, "Detection of Copy-Move Forgery Using Wavelet Decomposition," IEEE, 2013. Source of: the full Haar-DWT + PSF/blur model (Eq. 1–5), the moment/central-moment/blur-invariant derivation (Eq. 6–17), the PCA formulation (Eq. 18–26), the block-similarity and 16-neighbor consistency check (Eq. 27–30), and the 86.67% accuracy / runtime benchmarks.
- **stamm2010.pdf** — M.C. Stamm, K.J.R. Liu, "Wavelet-Based Image Compression Anti-Forensics," ICIP 2010. Source of: the full compression-artifact model (Eq. 1–3), the anti-forensic Laplacian-parameter estimator and its iterative bias correction (Eq. 4–8), the complete dither-distribution derivation and its provable Laplacian-matching property (Eq. 9–13), and the 100%-attack-success / 46.64dB-PSNR benchmark.
- **mgaga2019.pdf** — S.S. Mgaga, N.P. Khanyile, J.-R. Tapamo, "A Review of Wavelet Transform based Techniques for Denoising Latent Fingerprint Images," IEEE, 2019. A **review paper** synthesizing four separate studies (Yinping et al., Zaki et al., Iqbal, Sasirekha et al.) — source of: the VisuShrink/BayesShrink/golden-ratio-threshold formulas (Eq. 3–4, 13), the signal/noise variance-separation derivation (Eq. 5–10), the weighted-median noise estimator (Eq. 14–16), and the SWT-over-DWT / db2-sym4-over-Haar recommendations (qualitative synthesis, not one controlled experiment).
- **Image-Denoising-by-Wavelet-Transform-Based-on-New-Threshold.pdf** — H. Zhu, X. Wang, Springer AISC 1303, 2021. Source of: the classical hard/soft threshold formulas (Eq. 1–2, shared definitionally with mgaga2019's Eq. 1–2) and the streamlined piecewise shrinkage function (Eq. 3) with its full SNR/MSE benchmark (Table 1).
- **Multi-stage-image-denoising-with-the-wavelet-transform.pdf** — C. Tian et al., Pattern Recognition 134, 2023. `[ML]`, marginal — CNN-based denoiser; not re-read in this pass, no forensic content.
- **cheng2021.pdf** — K. Cheng et al., J. Phys. Conf. Ser. 1757, 2021. Marginal — image fusion (not forgery detection); not re-read in this pass.
- **Image-Compression-Using-Hybrid-Radon-Transform-with-Discrete-Wavelet-Transform-Technique.pdf** — R. Nanmaran et al., Springer LNNS 954, 2024. Marginal — compression technique, not forensic; not re-read in this pass.
