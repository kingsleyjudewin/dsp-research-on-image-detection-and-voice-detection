# Benford's Law Forgery Detection

## Core mathematical principle

**Underlying concept.** Benford's Law states that for many naturally-generated numerical populations, the leading (most significant) digit `d ∈ {1,...,b-1}` in base `b` is *not* uniformly distributed but follows a logarithmic law. The classical (base-10) form is:

```
p(d) = log10(1 + 1/d),   d = 1,...,9
```

Wang et al. (2009) give the formal derivation of *why*: if a positive random variable `x` is uniformly distributed on the log scale — i.e. `y = 10^x` with `x ~ Uniform(0, N)` — then the pdf of `y` is `p_Y(y) = 1/(N·ln10) · 1/y`, and integrating this pdf over the digit-`d` intervals of `y`'s decimal expansion analytically reproduces `p(d) = log10(1+1/d)` (Wang et al. Eq. 1–7). This is the mathematical precondition: **any population whose values are spread across orders of magnitude on a log-uniform-like scale will exhibit Benford conformance**, independent of the specific real-world process generating the numbers.

**Why block-DCT/DWT coefficients qualify.** Thai, Retraint & Cogranne (2012) derive, from first principles across the full camera acquisition chain (RAW Poisson-Gaussian sensor noise → bilinear demosaicking → gray-world white balance → gamma correction → block DCT), that pre-quantization DCT coefficients follow a **doubly-stochastic model**: conditioned on a block's variance `σ²`, coefficients are approximately zero-mean Gaussian (`f(I_ij|σ²) = 1/√(2πσ) · exp(-I_ij²/2σ²)`, their Eq. 10), and the block variance itself is modeled as a **Gamma distribution** `σ²_k ~ Gamma(α*, β*)` (fitted by moment-matching a sum of correlated Gamma variables, Eq. 11–15). Marginalizing over `σ²` gives the unconditional coefficient pdf as a **modified Bessel function of the second kind**, `f(I_ij) ∝ |I_ij|^{α*-1/2} · K_{α*-1/2}(·)` (Eq. 16–19) — a strict generalization in which the **Laplacian model is the special case `α*=1`** and the **Gaussian model is the limit `α*→∞`**. This is the theoretical justification (independently of any Benford argument) for why real DCT coefficients are sharply peaked, heavy-tailed, and validated by χ² goodness-of-fit against the Dresden Image Database (Nikon D200 subset) to fit better than the Generalized-Gamma model of Chang et al. Because Laplacian/generalized-Gaussian populations are known to produce Benford-conforming leading-digit statistics (this is *assumed*, not re-derived, by the Benford-specific papers below), quantized block-DCT (and, per Singh 2015, block-DWT/JPEG2000) coefficients inherit approximate Benford conformance.

**Property exploited: any operation that disturbs the coefficient-generation process breaks conformance.** Second/multiple lossy compression (re-quantization at a different step size), splicing (mixing coefficient populations from two different acquisition chains/compression histories), contrast enhancement/gamma remapping, histogram manipulation, and synthetic (GAN) generation (a feed-forward, largely FIR-filter-like process rather than the recursive/autoregressive-like natural image formation process — Bonettini et al.'s stated rationale) all measurably distort the empirical leading-digit histogram away from the fitted Benford curve. Because the "clean" curve is analytically known (or fittable in closed form), the **divergence between the empirical digit pmf and the fitted Benford curve is a compact, statistically-grounded, training-cheap tampering/synthesis feature.**

## Input requirements

- **Format**: still image, any codec on ingest; the detector operates on the **block-DCT domain** (primary, JPEG-native) or **block-DWT domain** (secondary, JPEG2000-native, per Singh 2015).
- **Preprocessing**:
  1. Convert to grayscale or operate on a single channel — every paper here (moin2017, singh2015, wang2009, bonettini2021) operates on **grayscale or a single (green/luminance) channel**; none of the five Benford papers analyze full RGB Benford statistics directly. `(not specified in the corpus for color — engineering recommendation: run per-channel or on luma Y only)`.
  2. Partition into non-overlapping `8×8` blocks (DCT) — Bonettini et al. explicitly state `K` non-overlapping `8×8`-pixel blocks.
  3. Apply 2D-DCT per block, then **quantize** each coefficient using a JPEG quantization step `Δ` drawn from a standard quantization matrix at a chosen quality factor (Bonettini's pipeline explicitly requires this quantization step — the digit statistic is computed on *quantized* coefficients, not raw DCT output).
  4. For DWT/JPEG2000 input (Singh 2015): decompose with **Daubechies 9/7** (lossy) or **Le Gall 5/3** (lossless) filters; **level 5** was used for their compression-ratio/double-compression analysis, **level 1** for glare detection (their Fig. 15 vs. Fig. 20–23 pipelines differ in decomposition depth — the paper does not explain why beyond noting "2nd, 3rd level can also be done but information loss is [higher]").
- **Bit depth / resolution**: not explicitly constrained by any paper; Bonettini et al.'s corpus is `256×256` RGB (converted to grayscale per-block); Moin/Singh use the UCID dataset (`512×384`-class uncompressed images).
- **Reliable when**: image has passed through **JPEG (or JPEG2000) block-transform quantization at least once** (this is what imposes/preserves Benford-fitting structure in the first place) and is not near-saturated/degenerate; quality factor ≥ ~80 for the strongest chi-square separation in Moin et al.'s tests (though their method still detects down to QF50, just with reduced deviation).
- **Unreliable / inapplicable when**:
  - Never-JPEG-compressed (pure RAW/PNG pipeline) images — no paper validates Benford features on uncompressed-domain coefficients directly (Bonettini's uncompressed-trained model in fact **fails** when tested against compressed images, see Key Findings).
  - Very low quality factor (QF≈50 or below) where high-frequency coefficients quantize to zero, removing their leading digit entirely (see Implementation Notes — "zero-coefficient problem", an ambiguity the corpus does not resolve).
  - JPEG2000 **double-compression** at the DWT level: Singh 2015's own Table II shows deviation ratios of `0`–`2.2204e-16` (i.e., numerically zero) for JPEG2000 double-compression at levels (3,3)/(3,5)/(5,3) — **the DWT-domain double-compression detector essentially does not work**, in contrast to the DCT-domain one, which shows clear deviation ratios of `0.08`–`0.39` for the equivalent JPEG case (Table I). This is an explicit negative result in the source paper, not an omission.
  - Copy-move/cut-paste tampering **within the same image** or from a similarly-compressed source — first-digit statistics are aggregate/global and largely insensitive to small localized splices (Singh 2015 explicitly demonstrates this: tampered vs. original image Benford curves are visually near-identical in their Fig. 9/10 copy-paste test).

## Step-by-step algorithm

There are **two distinct algorithmic families** in the corpus: (A) a **classifier-based multi-parameter divergence pipeline** (Bonettini et al. 2021 — PRIMARY, most complete and highest-performing), and (B) a **single-parameter chi-square / MSE deviation pipeline** (Moin et al. 2017 / Singh et al. 2015 — simpler, no feature-vector concatenation, historically earlier). Both are documented below; (A) is recommended as primary because it is the only one validated with a systematic ablation over features, bases, and quality factors, and achieves the best reported accuracy (99.83%, see Key Findings).

### A. Multi-parameter generalized-Benford divergence pipeline (Bonettini et al. 2021) — PRIMARY

1. **Block-DCT quantization.** Partition the (grayscale) image into `K` non-overlapping `8×8` blocks. Compute the 2D-DCT of each block. Quantize each coefficient at frequency `n` (zig-zag index) using a quantization step `Δ` taken from a standard JPEG quantization matrix at a chosen quality factor `QF ∈ J` (paper sweeps `J ⊆ {80, 85, 90, 95, 100}`).

2. **Leading-digit extraction in base `b`.** For the DCT coefficient `c_{n,Δ}(k)` at frequency `n`, quantization step `Δ`, block `k`, compute its first digit in base `b`:
   ```
   d_{b,n,Δ}(k) = floor( |c_{n,Δ}(k)| / b^floor(log_b |c_{n,Δ}(k)|) )
   ```
   This yields a value in `{1, ..., b-1}` for nonzero coefficients (see Implementation Notes for the zero-coefficient case, which the paper does not explicitly address).

3. **Empirical pmf.** Over all `K` blocks:
   ```
   p̂(d) = (1/K) · Σ_{k=1}^{K} 1_x(d(k)),    d ∈ {1,...,b-1}
   ```
   where `1_x(y) = 1` if `y=x` else `0`. Compute this separately for each combination of base `b ∈ B`, frequency `n ∈ N`, quantization step `Δ` (from `J`).

4. **Fit the generalized Benford curve** to the empirical pmf via least-squares:
   ```
   p_{b,n,Δ}^fit = argmin_p  Σ_{d=0}^{b-1} ( p̂_{b,n,Δ}(d) − p(d) )²
   ```
   where the model family is
   ```
   p(d) = β · log_b( 1 + 1/(γ + d^δ) )
   ```
   `β` = scale factor, `γ, δ` = shape parameters of the logarithmic curve. (Note: this is the same functional form as Wang et al.'s `B_g(d) = α1·log10(α3 + 1/d^α2)` — Bonettini generalizes the digit base from 10 to `b` and fits per-image rather than assuming fixed Laplacian/Gaussian parameter sets.)

5. **Compute divergence** between the empirical pmf `p̂` and the fitted curve `p^fit`, using **any one (or more) of three symmetrized divergences**:
   - **Jensen-Shannon**: `D^JS(p̂‖p) = D^KL(p̂‖p) + D^KL(p‖p̂)`, where `D^KL(p̂‖p) = Σ_{d=1}^{b-1} p̂(d)·log(p̂(d)/p(d))`.
   - **Rényi**: `D^R_α(p̂‖p) = 1/(1-α) · ( log S_α(p̂,p) + log S_α(p,p̂) )`.
   - **Tsallis**: `D^T_α(p̂‖p) = 1/(1-α) · ( 2 − S_α(p̂,p) − S_α(p,p̂) )`.
   - Both use the shared kernel `S_α(q,p) = Σ_{d=1}^{b-1} q(d)^α / p(d)^{α-1}`. The paper holds `α` **constant** in experiments (i.e., they do not sweep `α`; the exact fixed value is not stated in the extracted text — `(ambiguous in the corpus — the paper states α is "removed as a dependency... kept constant" without printing the numeric value used)`).
   - All three divergences were compared; the paper does not report one being categorically superior — JS is the simplest to implement and has no free parameter, making it the recommended default.

6. **Concatenate into a feature vector** across all swept bases `B`, frequencies `N`, and quantization steps `J`:
   ```
   φ_{B,N,J} = [ D^JS_{b,n,Δ}, D^R_{b,n,Δ}, D^T_{b,n,Δ} ]  for all b∈B, n∈N, Δ∈J
   ```
   - Bases tested: `B ⊆ {10, 20, 40, 60}` (1–4 elements, 15 combinations).
   - Frequencies tested: `N ⊆ {1,...,9}` in zig-zag order **after DC** (9 nested sets, each adding one more coefficient: `{1}, {1,2}, ..., {1,...,9}`).
   - Quantization steps: derived from JPEG quality factors `J ⊆ {80,85,90,95,100}` (5 arrays).
   - Total: **675 distinct feature-vector configurations** tested; dimensionality ranges from **3** (single base, single frequency, single divergence type region) up to **540**.

7. **Classification** `[ML — excluded from the no-ML engine]`: the paper feeds `φ` into a **Random Forest** (100 trees, Gini-index splitting, bootstrap sampling, scikit-learn defaults otherwise) trained with Leave-One-Group-Out cross-validation. **Training-free substitute (engineering recommendation, not in corpus)**: threshold a single scalar — e.g. the mean or max divergence across the swept `(b,n,Δ)` grid, or the divergence at the single best-performing configuration (`b=10`, all 9 frequencies, QF=100 per the paper's own ablation, see Key Findings) — against a calibrated cutoff. This sacrifices some accuracy (the paper shows 3-feature vectors already exceed 0.75 accuracy and 50-feature vectors exceed 0.97 *with* the Random Forest; a single-threshold rule without a classifier would be expected to sit below the 3-feature RF number, though this specific number is not measured in the paper) but removes the ML dependency.

### B. Single-statistic deviation pipeline (Moin et al. 2017 for contrast enhancement; Singh et al. 2015 for double-compression) — SECONDARY / lightweight fallback

**B1. Chi-square divergence for contrast-enhancement detection (Moin et al. 2017):**
1. Extract the **green channel** of a grayscale UCID-style image; JPEG-compress at quality factor `QF ∈ {50, 70, 90}`.
2. Compute block-DCT, extract first-significant-digit histogram `p̂(d)` per digit `d ∈ {1,...,9}` (implicitly base-10, single frequency band per the paper's simpler setup — it does not sweep multiple frequencies the way Bonettini does).
3. Compute the **chi-square divergence** against the classical Benford curve `p(d) = log10(1+1/d)`:
   ```
   χ² = Σ_d ( p̂(d) − p(d) )² / p(d)
   ```
   (standard Pearson chi-square form; the paper reports this per-image and averages over the dataset).
4. `[ML — excluded]` Feed `χ²` (and related per-digit/per-frequency features) into an **SVM**, 10-fold cross-validated. **Training-free substitute**: threshold `χ²` directly — the paper's own Table I shows this is separable by a fixed cutoff in the unaltered-vs-altered mean (see Key Findings table), though the paper itself only validates the SVM-classified version, not a raw threshold rule, so accuracy of a pure-threshold approach is `(not measured in the corpus — engineering extrapolation)`.
5. Reported to be robust even against an anti-forensic **local random dithering attack** that fully defeats a competing gap-bin histogram detector (Cao et al.) — see Key Findings.

**B2. MSE "deviation ratio" for double-compression / JPEG2000 / glare detection (Singh & Bansal 2015):**
1. Compute first-digit pmf of block-DCT (JPEG) or block-DWT (JPEG2000, level 1 or 5 per use case) coefficients for the image under test.
2. Compute the same for a **reference/original-compression** pmf (single-compression baseline).
3. **Deviation ratio** = mean squared error between the two first-digit probability curves:
   ```
   deviation_ratio = (1/9) · Σ_{d=1}^{9} ( p̂_test(d) − p̂_reference(d) )²
   ```
   (the paper describes this in prose as "mean square error between the probabilities of first digits of original and double-compressed images" without printing the formula as a numbered equation — the above is the direct algebraic reading of that description, so treat the exact averaging normalization `(1/9)` as `(not printed verbatim in the corpus — inferred from the MSE definition)`).
4. For **glare detection**: apply DWT (level 1) or DCT to the image, and inspect the first-digit histogram directly for an **anomalous spike at digit 5** — this is a qualitative visual signature (demonstrated on 2 example images, UCID00146/UCID00181), not a formal statistical test with a threshold. No general-purpose formula is given for "spike detection" beyond visual inspection of the histogram plot.
5. For **copy-move/splice detection at 1st-digit resolution**: the paper explicitly demonstrates this **fails** — tampered and original images produce visually near-identical Benford curves (their Fig. 9 vs Fig. 10). No higher-order-digit (2nd/3rd digit) extension is actually implemented or tested in this paper, despite the paper's introduction suggesting "taking more places into consideration" as a mitigation — treat that specific claim as **unimplemented/aspirational** in this source, not a validated technique.

## Output

- **Family A (Bonettini)**: per-image classifier output is a binary label (natural / GAN-generated) with an associated class probability from the Random Forest's vote fraction — `[0,1]`, where `1` = high confidence GAN/synthetic. Without the classifier (no-ML mode), the raw output is the **divergence value(s)** `D^JS`/`D^R`/`D^T` themselves — unbounded above, `0` = perfect conformance to the fitted Benford curve (most "natural"), larger values = greater deviation (more suspicious).
- **Family B1 (Moin)**: raw `χ²` statistic, `0` = perfect Benford conformance, larger = more suspicious. Paper's own empirical means: unaltered ≈ `0.0034` (cited from prior work, Acebo & Sbert) to `0.0112–0.0126` (their own UCID measurements across QF 50/70/90); contrast-enhanced ≈ `0.0051–0.0791` depending on gamma and QF (see Key Findings table) — i.e., roughly a **2×–20× increase** in χ² relative to the unaltered baseline, with the gap narrowing as gamma increases toward more extreme values in one direction and shrinking in the other (see below).
- **Family B2 (Singh)**: raw deviation ratio (MSE, unitless), `0` = no deviation from reference. Reported values `0.05–0.39` for genuine JPEG double-compression, statistically indistinguishable from `0` (`0`–`2.2×10⁻¹⁶`) for the JPEG2000/DWT case — see Input Requirements reliability note.

**Calibrating to a [0,1] fusion-layer probability**: none of the five papers specify a general-purpose calibration function mapping their raw statistic to a probability. `(Not specified in the corpus — engineering recommendation)`: fit a monotonic calibration (e.g. logistic/Platt-style sigmoid `σ(a·χ² + b)`, or empirical CDF percentile against a held-out calibration set of known-authentic images at the deployment's expected QF) per-statistic, per estimated-QF-bucket, since (a) the "unaltered" baseline itself shifts with QF (Moin's Table I: unaltered mean χ² is `0.0112` at QF90 vs `0.0126` at QF50 — not constant), and (b) no single fixed threshold is validated across QFs in any paper here.

## Key findings from papers

**Manipulation types detected best**: (1) GAN/synthetic image generation (Bonettini, 99.83% avg accuracy) — best-supported use case in this corpus; (2) contrast enhancement / gamma correction, including anti-forensically-integrated contrast enhancement designed to evade gap-bin detectors (Moin, Pd 93–99%); (3) double/multiple JPEG (DCT-domain) compression (Singh, deviation ratio 0.05–0.39 vs ~0 baseline).

**Documented failure cases / limitations**:
- Copy-move/splice tampering is **not** detected by first-digit statistics (Singh 2015, explicit negative result).
- JPEG2000/DWT-domain double-compression is **not** detected by the same deviation-ratio approach that works for JPEG/DCT (Singh 2015, deviation ratio ≈ 0 in Table II).
- Bonettini's classifier, trained on uncompressed-domain features, **collapses to near-random accuracy** when tested against images subsequently JPEG-recompressed at quality factors unseen during training (explicit result, "Resilience to JPEG compression" subsection) — training/calibration must match the deployment compression pipeline.
- Wang et al. (2009) demonstrate a **general anti-forensic vulnerability**: a manipulation (double compression, dithering) that breaks Benford conformance can be masked by a subsequent compensation operation (histogram equalization, rescaling) that *restores* conformance without undoing the tampering — modeled formally via their `H_M` (manipulation) / `H_C` (compensation) block-diagram system model (their Fig. 3), and demonstrated on cartoon-image and double-JPEG+equalization and dithering+rescaling cases.
- Moin et al.'s own gamma-sweep data (Table I below) shows the χ² divergence signal **shrinks as gamma moves away from 1 in the increasing direction** (γ=2.0 gives smaller χ² than γ=0.9 in two of three QF conditions) — i.e., detectability is **non-monotonic in gamma**, not simply "larger edits are easier to detect."

**Benchmark tables** (values transcribed directly from source papers):

| Paper | Dataset | Metric | Value | Conditions |
|---|---|---|---|---|
| Bonettini 2021 | 15 sub-datasets (CycleGAN + ProGAN), >200k images, 256×256 | Avg. accuracy | **99.83%** | Proposed method, best feature-vector config |
| Bonettini 2021 | same | Avg. accuracy | 89.64% | Baseline: fine-tuned Xception CNN `[ML]` |
| Bonettini 2021 | same | Avg. accuracy | 91.03% | Baseline: steganalysis rich-features + linear SVM `[ML]` |
| Bonettini 2021 | orange2apple (CycleGAN) | Accuracy | 98.13% (proposed) vs 97.64 (Xception) vs 88.80 (steg.) | — |
| Bonettini 2021 | lsun_bedroom (ProGAN) | Accuracy | 100.00% (proposed) vs 76.22 (Xception) vs 98.92 (steg.) | — |
| Bonettini 2021 | orange2apple | Accuracy | 94.50% → 82.01% → 65.93% | QF 100 → 95 → 90 (post-hoc recompression, retrained on compressed data) |
| Bonettini 2021 | Face test sets (StarGAN/GlowGAN/ProGAN/StyleGAN2) | Avg. accuracy | 87.96% | StarGAN 96–97%, GlowGAN 83–88%, ProGAN 79.75%, StyleGAN2 72.6–77.2% (hardest) |
| Bonettini 2021 | all 675 setups | Accuracy vs. feature length | >0.75 at 3 features; >0.97 at 50 features | Fig. 5 sweep |
| Moin 2017 | UCID (1338 images) | Mean χ² | 0.0112–0.0126 (unaltered) vs 0.0051–0.0791 range (altered, γ-dependent) | QF ∈ {50,70,90}, γ ∈ {0.5,0.9,1.5,2.0} |
| Moin 2017 | UCID | Pd / Pfa | 93.1–98.7% / 3.5–7.3% | Original contrast-enhancement mapping, all QF×γ combos |
| Moin 2017 | UCID | Pd / Pfa | 97.0–99.2% / 1.3–4.3% | **Anti-forensic-integrated** contrast enhancement — Benford+SVM still detects |
| Moin 2017 | UCID | Zero-height gap bins | 0 (both unaltered AND anti-forensic-attacked) vs 5.79–36.84 (naively altered) | Cao et al. baseline `[cited, not this engine's method]` — fully defeated by anti-forensic attack, unlike Benford+SVM |
| Singh 2015 | UCID, Cameraman/Lena test images | Deviation ratio (JPEG/DCT double-compression) | 0.0814–0.1440 (Cameraman) / 0.2510–0.3863 (Lena) | QF pairs (30,60)/(45,45)/(60,30) |
| Singh 2015 | same | Deviation ratio (JPEG2000/DWT double-compression) | ≈ 0 (0 to 2.22e-16) | Levels (3,3)/(3,5)/(5,3) — **method does not work in this domain** |
| Wang 2009 | Synthetic (Laplacian/Gaussian fits) | Fitted generalized-Benford params | Laplacian: α1=1.05, α2=1.352, α3=1.061; Gaussian: α1=1.08, α2=2.55, α3=1.15 | Table 1 |
| Thai 2012 | Dresden Image Database (Nikon D200) | χ² GOF statistic | Proposed Gamma-mixture model ≈ 10¹–10² vs Generalized-Gamma model ≈ 10²–10⁴ | Lower is better fit; proposed model fits ~1-2 orders of magnitude better |

## Implementation notes

- **Zero-coefficient problem (ambiguity)**: quantized high-frequency DCT coefficients are frequently exactly `0`, which has no defined leading digit under `d = floor(|c|/b^floor(log_b|c|))` (undefined/`-∞` for `log_b(0)`). **None of the five papers explicitly state how zero coefficients are excluded or handled** in the pmf computation — treat this as a genuine ambiguity in the corpus. Engineering recommendation: exclude zero coefficients from the per-block digit count (i.e., `K` in step 3 of Family A should be reinterpreted as "count of nonzero coefficients observed," not "count of blocks") and track what fraction of coefficients were excluded as a data-quality signal, since a very high zero-rate (heavy quantization) will make the pmf estimate noisy/unreliable regardless.
- **Frequency selection**: use zig-zag order **excluding DC** (frequency index 0) — all papers that specify this (Bonettini) start at AC frequency 1. Including more AC frequencies monotonically improves accuracy (Bonettini Fig. 6a/6b) up to the 9-frequency limit tested; there is no evidence in the corpus about frequencies beyond index 9.
- **Base selection**: `b=10` alone captures most of the achievable accuracy; adding more bases from `{20,40,60}` gives only marginal further improvement (Bonettini Fig. 6c/6d) — not worth the added computation for a first implementation.
- **Quality-factor conditioning**: because the "clean" Benford baseline itself shifts with JPEG quality factor (Moin's unaltered mean χ² varies 0.0109–0.0126 across QF 50/70/90), any deployed threshold must be conditioned on the image's **estimated** background quality factor (obtainable from the JPEG Compression module in this engine) rather than applied as one fixed global cutoff.
- **DCT convention**: use the standard JPEG-normalized 2D-DCT-II, `I_ij = (1/4)·T_i·T_j· ΣΣ Z(m,n)·cos((2m+1)iπ/16)·cos((2n+1)jπ/16)` with `T_k = 1/√2` for `k=0` else `1` (Thai et al. Eq. 8) — this is the exact form to reproduce libjpeg/Pillow-compatible coefficient magnitudes so the fitted Benford parameters transfer correctly.
- **Reference toolboxes/implementations cited**: none of the five papers link a public code repository in the extracted text. Bonettini et al. state they used **scikit-learn**'s Random Forest implementation (Gini index splitting, 100 trees, bootstrap sampling, otherwise default hyperparameters) and the **Marra et al. GAN-image corpus** as their dataset source.
- **Recommended Python libraries**:
  - `scipy.fft.dctn` (or `cv2.dct` per 8×8 block) for the block-DCT transform.
  - `numpy` for digit extraction (`np.floor(np.abs(c) / b**np.floor(np.log(np.abs(c))/np.log(b)))`) and pmf histogramming (`np.bincount` or `np.histogram`).
  - `scipy.optimize.curve_fit` or `scipy.optimize.least_squares` for the generalized-Benford curve fit (step 4 of Family A).
  - `scipy.stats.entropy` directly supports KL divergence (`scipy.stats.entropy(p̂, p)`), from which Jensen-Shannon is `0.5*(entropy(p̂, m) + entropy(p, m))` with `m=0.5*(p̂+p)` — note this is the *averaged* JS convention; Bonettini's Eq. 6 (`D^KL(p̂|p) + D^KL(p|p̂)`, unaveraged, no factor of 0.5) is a different normalization — implement their exact unaveraged form if reproducing their reported numbers.
  - `pywt.dwt2` / `pywt.wavedec2` for the JPEG2000/DWT variant (Singh et al.), with `wavelet='bior4.4'` as the closest standard PyWavelets approximation to the Daubechies 9/7 (CDF 9/7) biorthogonal filter used in JPEG2000, or `'db1'`/`'bior1.1'` (Le Gall 5/3-equivalent) for the lossless variant.
  - PIL/Pillow's `Image.quantization` or a manual JPEG quantization-matrix table (standard IJG tables scaled by quality factor) for step 1 of Family A.

## Key references

- **wang2009.pdf** — J. Wang, B.-H. Cha, S.-H. Cho, C.-C.J. Kuo, "Understanding Benford's Law and Its Vulnerability in Image Forensics," ICME 2009. Source of: the formal derivation of classical Benford's Law from log-uniform random variables (Eq. 1–7); the generalized Benford's Law functional form `B_g(d) = α1·log10(α3+1/d^α2)` and its fitted Laplacian/Gaussian parameter sets (Table 1); the manipulation/compensation vulnerability system model (Fig. 3) and its cartoon-image, double-JPEG+equalization, and dithering+rescaling demonstrations.
- **bonettini2021.pdf** — N. Bonettini, P. Bestagini, S. Milani, S. Tubaro, "On the Use of Benford's Law to Detect GAN-Generated Images," ICPR 2021. Source of: the primary multi-parameter divergence pipeline (digit extraction Eq. 2–4; generalized Benford fit Eq. 5, 11; JS/Rényi/Tsallis divergences Eq. 6–10; feature-vector concatenation Eq. 12); the Random Forest classifier setup; the full base/frequency/QF ablation and all accuracy benchmarks in the table above.
- **moin2017.pdf** — S.S. Moin, S. Islam, "Benford's Law for Detecting Contrast Enhancement," ICIIP 2017. Source of: the chi-square divergence statistic for contrast-enhancement detection; the SVM-based classification pipeline; the anti-forensic-integrated contrast enhancement robustness result (Tables III–IV) showing Benford+SVM survives an attack that fully defeats the Cao et al. gap-bin baseline.
- **singh2015.pdf** — N. Singh, R. Bansal, "Analysis of Benford's Law in Digital Image Forensics," IEEE 2015. Source of: the MSE "deviation ratio" statistic; the JPEG(DCT)-vs-JPEG2000(DWT) double-compression comparison (and the negative result for the DWT case); the copy-move-tampering negative result; the glare-detection digit-5-spike observation.
- **thai2012.pdf** — T.H. Thai, F. Retraint, R. Cogranne, "Statistical Model of Natural Images," ICIP 2012. Source of: the full acquisition-chain statistical model (RAW Poisson-Gaussian → demosaicking → white balance → gamma correction, Eq. 1–7); the Gamma-distributed block-variance model and its Bessel-function DCT coefficient distribution (Eq. 8–19), which is the theoretical justification for why DCT coefficients are Laplacian/generalized-Gaussian-like — the statistical prerequisite the Benford-specific papers build on.
- **shan2014.pdf** — M. Shan, "Research of Computer Forensics Based on Benford's Law," Advanced Materials Research Vols. 989-994 (2014). Background only: canonical exposition of the classical first-digit law and higher-order-digit tables (2nd/3rd/4th digit); not image-specific (applied to file-size statistics); no formal divergence/classification method. Not re-read in this pass — no new technical content beyond what was already extracted.
