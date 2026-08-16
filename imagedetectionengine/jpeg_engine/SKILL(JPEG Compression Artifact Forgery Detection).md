# JPEG Compression Artifact Forgery Detection

## Core mathematical principle

JPEG is a lossy, block-based (8×8 DCT) codec. Every JPEG decode→edit→re-encode cycle leaves a statistical fingerprint of quantization, rounding, and truncation error in the DCT coefficient distributions. Luo, Huang & Qiu (2010) formalize the three error sources precisely:

- **Quantization error**: for an original DCT coefficient `d1` with quantization step `q`, the dequantized coefficient `d1'` satisfies `d1' = kq` for integer `k` — i.e., dequantized JPEG coefficients cluster exactly on multiples of the quantization step.
- **Rounding error `ε(i,j) = DCT(ℜ)`**: modeled as approximately Gaussian, mean 0, variance **1/12** (from the Central Limit Theorem applied to the uniform `[-0.5,+0.5]` rounding distribution of 64 summed spatial-domain terms per DCT coefficient), so a **re-DCT'd, once-decompressed** image's coefficients `d2 = d1' + ε` spread in a narrow band around the true quantization-step multiples rather than landing exactly on them.
- **Truncation error**: pixel values exceeding `[0,255]` after inverse-DCT get clipped; empirically small (< 1% of pixels in most quality/dataset combinations tested, Luo et al. Table I) and therefore neglected in the rest of the analysis by restricting statistics to unsaturated 8×8 blocks.

**Property exploited (single vs. double compression).** A **never-compressed** bitmap's DCT coefficients follow a smooth, continuous distribution; a **once-JPEG-compressed-then-decompressed** bitmap's coefficients cluster tightly around multiples of a hidden quantization step `q` (spread only by the small rounding-error band). This distributional difference is directly measurable and is the basis of Luo et al.'s JPEG-history identification (Pipeline A).

**Property exploited (double compression / splicing).** When an image is compressed a *second* time with a different quantization step, the histogram of a DCT frequency develops **periodic artifacts** — missing bins or periodic peaks-and-valleys — because re-distributing values originally quantized at step `q_α` into new bins of step `q_β` is a non-injective, periodic remapping (Mahdian & Saic 2009, and independently confirmed in Wang & Zhang's histogram figures). When a forger splices content from a source with one compression history into a host with another and re-saves, the **spliced region and the background carry different double-quantization signatures** (or one is single-compressed while the other is double-compressed) — this is the exploitable inconsistency for splice localization.

## Input requirements

- **Format**: JPEG-derived image (has an 8×8 block-DCT structure to exploit); for closed-form methods (Luo, Mahdian) that read the JPEG quantization table directly from the file header, the image should ideally still be available as a **JPEG file**, not a re-saved bitmap, though Luo's *history-identification* feature `s` (Pipeline A.1) explicitly works on **bitmaps of unknown origin** to determine whether they were ever JPEG-compressed at all.
- **Preprocessing**: grayscale or luminance (Y) channel only — every closed-form method here (Luo, Mahdian) explicitly restricts to the **luminance channel**; Mahdian & Saic state explicitly that chrominance channels are down-sampled and quantized with larger steps in-camera, leaving "little information valuable" for this detection; 8×8 block alignment (assume the block grid is known/estimable — see Implementation Notes for the misaligned-grid case, which none of the closed-form methods in this corpus solve).
- **Reliable when**: sufficient block count for stable statistics (Luo et al. explicitly validate down to 8×8-pixel blocks, i.e. a *single* DCT block, with 88–96% accuracy depending on quality factor — see benchmark table); background quality factor is moderate-to-high (`QF ≥ ~85` for Luo's history-identification feature to cleanly separate compressed/uncompressed); for double-compression detection specifically, secondary quantization step **not** an integer multiple of the primary step (see below).
- **Unreliable / inapplicable when**:
  - **Very low quality factor (QF≈50)**: high-frequency coefficients quantize to zero, destroying the statistics both Luo's and Mahdian's methods rely on. Luo et al.'s own Table IV shows accuracy for quantization-step estimation dropping toward 41–63% at QF=50 for small (64×64) blocks, vs. 91%+ at QF=95.
  - **Integer quantization-step-ratio blind spot** (Mahdian & Saic, explicit and fundamental): if `Q_β(u,v)/Q_α(u,v)` is an integer, **no periodic double-quantization artifact is introduced at all** — Luo et al.'s own Table 1 confirms this empirically: at `Q_α=95, Q_β=70` (ratio not integer... but the paper notes at *other* specific pairs, e.g. where `q2` is an even multiple of `q1`, "detection accuracy is almost zero" because `Q_β(u,v)/Q_α(u,v)` is exactly an integer for the DCT frequencies used). This is a **fundamental, unavoidable blind spot** of every DQ-histogram-periodicity method, not an implementation weakness.
  - **`QF_second < QF_first`** (recompression at *lower* quality than the original): both Luo's Table 1 detection-accuracy matrix and Mahdian's method show substantially degraded accuracy in this regime (shorter histogram support, larger effective period) — this is the historically hardest case for every DQ-histogram method, closed-form or learned.
  - **Non-JPEG-derived content spliced into an otherwise-JPEG image and never itself compressed**: undetectable by this whole module family unless the composite is subsequently re-saved as JPEG at least once (in which case it becomes the "easy" single-vs-double case, see Key Findings).

## Step-by-step algorithm

Two fully closed-form, training-free algorithms (**A**, from Luo et al. 2010, and **B**, from Mahdian & Saic 2009) form the **PRIMARY, no-ML path** for this module — together they cover JPEG-history identification, quantization-step/table recovery, and double-compression detection. Two CNN-based algorithms (**C**, **D**) are documented for completeness and labelled `[ML — excluded from the no-ML engine]`.

### A. JPEG error analysis: history identification, quantization-step estimation, quantization-table detection (Luo, Huang & Qiu 2010) — PRIMARY

**A.1 — Identifying whether a bitmap was ever JPEG-compressed:**

1. For each DCT frequency position `(i,j)`, `0 ≤ i,j ≤ 7`, except DC `(0,0)`, let `p1(x)` be the pdf of coefficient `d1(i,j)` (never-recompressed) and `p2(x)` the pdf of `d2(i,j) = d1'(i,j) + ε(i,j)` (once-decompressed-and-re-DCT'd), where `d1'` is exactly quantized-step-multiple-valued and `ε` is the ≈Gaussian rounding-error term (mean 0, variance 1/12).
2. Define two disjoint regions of the real line:
   ```
   R1 = (-1, +1),    R2 = (-2,-1) ∪ (+1,+2)
   ```
3. Because `d1'` values concentrate at multiples of the quantization step `q ≥ 2` (the common regime), quantization pulls probability mass **into `R1`** (around zero/near-integer multiples) and **out of `R2`**, relative to a never-quantized coefficient population. Compute the 1-D feature:
   ```
   s = ∫_{R1} p_ac(y) dy / ∫_{R2} p_ac(y) dy          [Eq. 7]
   ```
   where `p_ac` is the empirical pdf of **all AC coefficients** of the test image (pooled across frequencies and blocks). A JPEG-compressed image has **larger `s`** than an uncompressed one (uncompressed images have their AC energy more evenly spread, so `∫_{R2}` is comparatively larger).
4. **Threshold** `s` using a minimum-risk classification rule (Theodoridis & Koutroumbas) fit on a training set of paired uncompressed/compressed images at each candidate image size. Reported thresholds (256×256 down to 8×8 blocks) range `t ≈ 0.29–0.38`, with corresponding false-positive rates 14.10% (256×256) down to 21.06% (8×8) — see benchmark table.

**A.2 — Estimating the quantization step per frequency (once identified as JPEG):**

1. Recover the denoised, rounding-error-corrected coefficient: since most `d2(i,j)` values fall within `±0.5` of the true quantization multiple with probability ≥91.50% (derived from the Gaussian rounding-error CDF, Eq. 11), simple **rounding** `[d2(i,j)]` recovers `d1'(i,j)` with high probability:
   ```
   P( [d2(i,j)] = d1'(i,j) ) = P( ε(i,j) = 0 ) = ∫_{-0.5}^{+0.5} p_ε(y) dy ≥ 91.50%          [Eq. 11]
   ```
2. Build the histogram `H` of `|[d2(i,j)]|` (absolute value, to merge symmetric ± bins) across all blocks, independently per frequency `(i,j)`.
3. **Empirical guard against the "ghost at 1" false estimate**: because ~8.5% of coefficients still round to `±1` even when the true quantization step is larger than 1 (residual rounding-error spillover), first test whether the step equals exactly `1`:
   ```
   if  H(1)/H(0) > t  AND  H(1) > H_max  →  q̂_{i,j} = 1          [Eq. 12]
   ```
   `t = 0.3` (empirically tuned in the paper across `t ∈ [0.10, 0.35]`, `t=0.3` gave the best results and is used for all reported experiments). Otherwise:
   ```
   q̂_{i,j} = argmin_k ( k | H(k) = H_max,  k ≥ 2 )          [Eq. 13]
   ```
   i.e., the smallest histogram-bin value achieving the maximum count, restricted to `k ≥ 2`.

**A.3 — Detecting the full quantization table (equivalently, the quality factor):**

1. Given a JPEG-decompressed bitmap `J1` suspected to have used quantization table `Q1`, recompress it with **every** candidate table from the standard quality-factor family `QF ∈ {1,...,100}`, producing `J2(i)` for `i=1,...,100`.
2. Compute pixel-domain similarity between `J1` and each `J2(i)`:
   ```
   R(J1, J2) = |E| / (M·N),   E = {(x,y) : J1(x,y) = J2(x,y), 1≤x≤M, 1≤y≤N}          [Eq. 24]
   ```
   i.e., the fraction of pixels that are *exactly* pixel-identical after recompression at that candidate table.
3. **Detected quality factor**:
   ```
   Q̂F = argmax_i ( R(J1, J2(i)) ),  i = 1,...,100          [Eq. 25]
   ```
   The correct table produces the highest exact-pixel-match rate because recompressing at the *true* table reproduces the *same* rounding/truncation behavior that generated `J1` in the first place.
4. **Standard quantization table formula** used to generate the 100 candidate tables from the base table `t` (IJG-standard luminance table given explicitly in the paper, transcribed in Implementation Notes below):
   ```
   Table_QF = { ⌊ t × 50/QF + 0.5 ⌋,          1 ≤ QF < 50
              { ⌊ t × (2 − QF/50) + 0.5 ⌋,    50 ≤ QF ≤ 100          [Eq. 26]
   ```
   (values less than 1 are floored up to 1).

### B. Double-quantization Fourier-periodicity detection (Mahdian & Saic 2009) — PRIMARY for splice/double-compression localization

1. **Double-quantization model**: for primary quantization matrix `Q_α` and secondary `Q_β`, the doubly-quantized coefficient is
   ```
   F^{Q_β}(u,v) = round( F^{Q_α}(u,v)·Q_α(u,v) / Q_β(u,v) )          [Eq. 4]
   ```
2. **Select 10 low-frequency DCT positions** (**luminance channel only**): `(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (0,2), (1,2), (0,3)`. Higher frequencies are excluded because they are frequently quantized entirely to zero, producing insufficient statistics.
3. For each of the 10 positions, build the **zero-mean histogram** of that frequency's DCT coefficients across all image blocks, and compute the magnitude of its **1-D FFT**: `|H_1|,...,|H_{10}|` (normalized to unit length). If the position is doubly-quantized, `|H_i|` exhibits **specific periodic peaks/artifacts**; if singly-quantized, it shows a smooth **decaying trend** instead.
4. **Trend removal (this paper's specific improvement over the Popescu & Farid generalized-Laplace curve-fitting baseline it compares against)**: rather than fitting a two-parameter generalized-Laplace model via nonlinear least squares (computationally heavier and paper-cited as producing more false positives on "non-perfect" real-image histograms), use a **local-minimum subtraction**:
   ```
   |H̃_i|(f) = |H_i|(f) − M_i(f)          [Eq. 5]
   ```
   `M_i(f) = min{ |H_i|(f), ..., |H_i|(f−n) }`, the minimum value over a trailing window of length `n` (a denoising **averaging filter** is applied to `|H_i|`, `i=2,...,10`, before this step; `i=1`/DC is treated as a special case because it alone shows a *clear peak* under double compression rather than a decaying trend under single compression). `n` (minimum-filter length) is **determined per quantization step in a training process** — the paper does not give a single fixed default value; `(engineering recommendation, not specified in corpus: calibrate n per deployment on a held-out set spanning the expected quantization-step range, following the paper's own stated procedure)`.
5. **Peak/feature construction**: build a feature vector from the resulting **local peaks** of `|H̃_i|` (post trend-removal) across the 10 frequencies.
6. `[ML — excluded]`: the paper's own classification stage uses a **Gaussian-kernel SVM** per quantization-step-of-interest (`k(x,y) = exp(-γ‖x-y‖²)`), trained with false-positive rate controlled to 1%, one classifier per quantization step `q ∈ {1,...,25}`. **Training-free substitute (engineering recommendation, not in corpus)**: threshold the **peak prominence** of `|H̃_i|` directly at the position(s) corresponding to a candidate secondary quantization step `q`, following the same "clear peak at DC ⟹ doubly-quantized" qualitative rule the paper itself describes for the un-trend-removed DC case (Section 5, "if the image is double compressed, typically the output... contains a specific clear peak"). This sacrifices the SVM's ability to integrate evidence across all 10 frequencies simultaneously but preserves a working, label-free detector.

**Combining A and B for this engine's primary (no-ML) pipeline**: run **A.1** first as a cheap global gate (is this image JPEG-derived at all?); if yes, run **A.3** to recover the estimated quantization table/QF (useful both as a standalone forensic feature — an image whose *locally* estimated table disagrees with its *globally* dominant table is suspicious — and as a condition input for calibrating other modules' thresholds elsewhere in this engine); run **B** for the actual double-compression/splice-localization signal, using **A.2**'s per-frequency quantization-step estimator to help interpret which candidate secondary step `q` to test peaks against.

### C. Histogram-feature CNN (Wang & Zhang 2016) `[ML — excluded from the no-ML engine]`

Documented for completeness. Feature: 9×11 hand-crafted vector per 8×8-aligned block, `X_B = {h_i(-5),...,h_i(5) | i=2,...,10}` — the histogram value at bins `{-5,...,5}` around the peak, for each of the 2nd–10th AC frequencies in zigzag order (Y channel only). Small 1-D CNN: 2 conv layers (kernel 3×1, 100 filters, stride 1) each followed by max-pooling (kernel 3×1, stride 2), then 3 fully-connected layers (1000-1000-2 neurons) with ReLU activations and dropout, ending in a 2-way softmax giving `[a,b] = P(\text{singly compressed}), P(\text{doubly compressed})`. Localization via a sliding `W×W` window (stride 8px) building a block posterior probability map. **The hand-crafted 9×11 histogram feature itself is training-free and reusable** as an alternative feature representation for Pipeline B's peak-detection step, even without the CNN classifier riding on top of it — `(engineering note: this repurposing is not validated in either source paper, but the feature extraction is mechanically identical to what Pipeline B already computes)`.

### D. Pixel/noise-residual CNN, aligned + non-aligned double-JPEG (Barni et al. 2017) `[ML — excluded from the no-ML engine]`

Documented for completeness — this is the capability gap the no-ML engine explicitly accepts. Three CNN variants on 64×64 patches: `C_hist` (CNN over concatenated DCT histograms, wins for **aligned** double-JPEG, A-DJPEG, but fails on non-aligned because DCT-domain periodicity is destroyed by grid misalignment); `C_pix` (CNN on raw pixels, fully self-learned); `C_noise` (CNN on high-pass/noise-residual-filtered pixels — most robust overall, outperforming prior periodicity-map methods on **non-aligned** double-JPEG, NA-DJPEG, in every tested scenario including `QF1 > QF2` and `QF1 = QF2`, and works even on bitmap/PNG-distributed images since it needs no JPEG bitstream access).

**Capability lost by excluding C and D from the no-ML engine**: `C_noise` is, per this corpus, the *only* documented technique that handles **non-aligned double-JPEG compression** (grid-shifted pastes) and the **`QF1 = QF2` case** at all — neither Luo's nor Mahdian's closed-form methods in Pipelines A/B are validated for (or designed to address) grid-misaligned splices or equal-quality double compression. A no-ML deployment of this module is therefore **blind to non-block-aligned splices** and to same-quality double compression; this should be explicitly surfaced to the fusion layer as a known coverage gap, not silently absorbed as "no evidence found."

## Output

- **Pipeline A.1**: scalar feature `s` (unbounded, larger = more evidence of JPEG compression history) compared against a size-dependent threshold `t`. `(Calibration to [0,1] not specified in corpus — engineering recommendation: logistic/Platt-style sigmoid fit per image-size bucket using the paper's own reported threshold/FPR pairs as calibration anchors.)`
- **Pipeline A.2**: per-frequency estimated quantization step `q̂_{i,j}` (integer) — a nuisance/conditioning parameter, not itself a tampering score, but a required input to Pipeline B and to other modules' QF-conditioned thresholds.
- **Pipeline A.3**: estimated quality factor `Q̂F ∈ {1,...,100}` (a conditioning parameter) plus, as a byproduct, `R(J1,J2(Q̂F))` (the max pixel-match fraction) as a **confidence** measure — a poor max match (`R` far below what's typical for genuine single-history images) suggests an inconsistent/mixed compression history, itself weak tampering evidence `(this specific use is an engineering extrapolation, not validated in the corpus)`.
- **Pipeline B**: per-frequency peak-prominence values, aggregable into a scalar double-compression score. **Localization**: none of Pipeline A/B natively produce a spatial heatmap the way the CNN methods (C, D) or the Ferrara/Bammey CFA methods do — Luo's and Mahdian's methods are validated at the **whole-image or single-block level**, not as a sliding-window localizer, though A.2/B can in principle be re-run per candidate block/region for a coarse heatmap `(block-wise re-application not itself validated end-to-end in the corpus — engineering extension)`.
- **Pipelines C/D** (`[ML]`, documented not implemented): softmax probability pair `[a,b]`, `b` = P(doubly compressed), directly usable as a per-block tampering score and, via the sliding-window BPPM, a heatmap — noted here only to make explicit what capability is forfeited under the no-ML constraint.

## Key findings from papers

**Manipulation types detected best**: JPEG-history identification (A.1) reliably separates compressed from uncompressed bitmaps even at small block sizes and high QF. Quantization-table/QF recovery (A.3) is robust and outperforms prior blocking-artifact methods (Fan's) especially for small images. Double-compression detection (B) is strongest when `QF_β > QF_α` (secondary/final quality higher than the spliced region's original quality) and when the quantization-step ratio is non-integer.

**Documented failure cases / limitations**: see Input Requirements — the integer-ratio blind spot and the `QF_second < QF_first` degradation are the two dominant, explicitly-acknowledged weaknesses shared by essentially all DQ-histogram methods (both Luo's and Mahdian's papers, and the CNN papers' own literature review, converge on this). High-frequency zeroing at low QF affects every method in this module.

**Benchmark tables**:

| Paper | Dataset | Metric | Value | Conditions |
|---|---|---|---|---|
| Luo 2010 | 5000 images (Corel/NJIT/NRCS/SYSU/UCID) | JPEG-identification accuracy | 92.36–96.92% (256×256) down to 81.95–88.36% (8×8 block) | Fixed QF ∈ {50,75,85,95,98}, Table II |
| Luo 2010 | same | JPEG-identification accuracy/FPR | 98.64% / 1.00% (256×256) to **95.08% / 5.80%** (8×8 block) | Random QF ∈ [50,98], Table III — outperforms MSDSt/Wang's/Liu's/Fan's baselines at every block size |
| Luo 2010 | 5000 images, 256×256 | Quantization-step est. accuracy | up to 99.76% (highest steps) | QF=95, full quantization-step confusion matrix given |
| Luo 2010 | same, QF=50 | Quantization-step est. accuracy | as low as 0.72–2.16% for high steps/high frequencies | QF=50 — severe degradation, high-frequency coefficients mostly quantized to 0 |
| Luo 2010 | 5000 images, 128×128 | Quantization-step est. accuracy | 81.97% (proposed) vs. 66.42% (Fridrich's [15]) | QF=85 — 16% improvement |
| Luo 2010 | 64×64/128×128, QF=95/90/85/75/50 | Quantization-step est. accuracy | Proposed generally 63–98%, beats method [15] in most cases | Table IV |
| Luo 2010 | 64×64 & 128×128, QF∈{10..90} | Quantization-table detection accuracy | up to 99.96% (128×128), 94.52–99.88% typical | Table V — vastly outperforms Fu et al.'s Benford's-law method [16], which collapses to near-0% at QF=50 |
| Mahdian 2009 | (own synthetic double-compression matrix, Luo et al.'s reproduction of comparable setup) | Detection accuracy | ranges 1–100% across `(Q_α, Q_β)` grid | **Near-0% at specific integer-ratio pairs**, e.g. row Qβ=70 vs Qα=95 → accuracy "1" (~1%) — confirms the fundamental blind spot |
| Wang & Zhang 2016 | UCID (train/val) + Dresden RAW (test) | Accuracy vs. block size/QF2 | 0.69–1.00, generally increasing with block size and QF2 | 1024×1024 near-perfect; 64×64 as low as 0.69–0.73 at low QF2 |
| Barni 2017 | (per prior summary — not re-extracted in this pass) | AUC | 0.6–0.97 depending on train/test QF gap and block size | `C_noise`, NA-DJPEG, cross-QF generalization |

## Implementation notes

- **Reading quantization tables without re-encoding**: standard JPEG files store their quantization tables directly in the file header (DQT marker) — `PIL.Image.quantization` (Pillow ≥ 9.1) or `jpeglib`/direct byte parsing exposes this without needing to decode-and-recompress. Use this for a fast QF estimate when the file is available as an actual `.jpg`; fall back to Luo's A.3 recompression-search method (Eq. 24–26) only when the image is a bitmap of unknown/stripped provenance.
- **The IJG standard base luminance table** `t` (needed for Eq. 26) as printed in Luo et al.:
  ```
  t = [[16,11,10,16,24,40,51,61],
       [12,12,14,19,26,58,60,55],
       [14,13,16,24,40,57,69,56],
       [14,17,22,29,51,87,80,62],
       [18,22,37,56,68,109,103,77],
       [24,35,55,64,81,104,113,92],
       [49,64,78,87,103,121,120,101],
       [72,92,95,98,112,100,103,99]]
  ```
- **Chroma subsampling**: only Y-channel statistics are used by both closed-form methods (A, B) — no chroma handling needed for this module, simplifying implementation relative to modules that need full-color analysis.
- **Zero-quantized high-frequency coefficients**: both A.2 and B degrade at low QF because high-frequency AC coefficients quantize to exactly zero, leaving no usable histogram — detect this condition (fraction of a frequency's coefficients equal to zero exceeding some threshold) and exclude that frequency from the feature/peak computation rather than letting a degenerate near-empty histogram silently corrupt the estimate `(exclusion rule not specified in corpus — engineering recommendation)`.
- **Histogram binning**: use integer-valued bins for DCT coefficient histograms (coefficients are already effectively integer/quantized-multiple-valued post-dequantization) — do not apply continuous/KDE binning, which would blur exactly the periodic bin structure both methods depend on.
- **No public reference code found in the extracted text** for Luo et al. or Mahdian & Saic. Both used **MATLAB's JPEG Toolbox** (Luo et al., explicitly cited as Sallee's `philsallee.com/jpegtbx`) for JPEG compression/decompression during their experiments.
- **Recommended Python libraries**:
  - `PIL.Image` / `Pillow` with `quality=` parameter for JPEG recompression sweeps (Pipeline A.3, B); `Image.quantization` attribute to read existing quantization tables directly.
  - `scipy.fft.dctn` / `cv2.dct` for block-DCT (should already be shared infrastructure with the Benford module).
  - `numpy.fft.fft` (1-D) for the per-frequency histogram FFT in Pipeline B step 3.
  - `numpy.histogram` for building `p_ac`, `H`, and the per-frequency coefficient histograms.
  - `scipy.ndimage.minimum_filter1d` for the local-minimum-subtraction trend removal (Eq. 5) — directly implements `M_i(f) = min` over a trailing window.

## Key references

- **luo2010.pdf** — W. Luo, J. Huang, G. Qiu, "JPEG Error Analysis and Its Applications to Digital Image Forensics," IEEE TIFS, vol. 5, no. 3, Sept. 2010. Source of: the full JPEG error model (quantization/rounding/truncation, Eq. 1–3); the JPEG-history-identification feature `s` (Eq. 7); the quantization-step estimator (Eq. 11–13); the quantization-table/QF detector (Eq. 24–26); the full benchmark tables II–V and the standard quantization-table matrix.
- **mahdian2009.pdf** — B. Mahdian, S. Saic, "Detecting Double Compressed JPEG Images," Institute of Information Theory and Automation of the ASCR, 2009. Source of: the double-quantization model (Eq. 4); the 10-frequency FFT-histogram-periodicity detector; the local-minimum trend-removal denoising step (Eq. 5); the integer-quantization-step-ratio blind spot; the SVM classification stage `[ML — excluded, training-free peak-prominence substitute recommended above]`.
- **s13635-016-0047-y.pdf** — Q. Wang, R. Zhang, "Double JPEG compression forensics based on a convolutional neural network," EURASIP J. Information Security, 2016:23. `[ML — excluded]`. Source of: the 9×11 hand-crafted DCT-histogram feature (reusable without the CNN) and the block-posterior-probability-map localization pattern.
- **barni2017.pdf** — M. Barni et al., "Aligned and Non-Aligned Double JPEG Detection Using Convolutional Neural Networks," J. Visual Communication and Image Representation, 2017. `[ML — excluded]`. Source of: the `C_hist`/`C_pix`/`C_noise` architecture comparison and the explicit identification of non-aligned double-JPEG and `QF1=QF2` as cases this corpus has no training-free solution for.

**Note — misfiled documents** (unchanged from prior review, not re-examined in this pass): `hussain2021.pdf` is Massimo Cacciari's "Nomes de Lugar: Confim" (a philosophy paper on borders/limits, Revista de Letras 2005) and `yasuda2018.pdf` is "Linear-Time Algorithm in Bayesian Image Denoising based on Gaussian Markov Random Field" (Yasuda et al., IEICE Trans. 2018) — neither is a JPEG forgery-detection paper. Both should be moved out of this folder or replaced with the correct source material; they are excluded from all analysis above.
