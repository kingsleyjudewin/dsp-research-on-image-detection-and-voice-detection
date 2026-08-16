# CFA / Demosaicing Artifact Forgery Detection

## Core mathematical principle

**Underlying concept.** A Bayer-pattern camera sensor captures only one color value (R, G, or B) per pixel through its Color Filter Array; the other two channels at every pixel are reconstructed by a demosaicing (interpolation) algorithm applied to the sensor's neighboring samples. This interpolation is a periodic, near-linear predictive filtering process: **acquired** pixels (directly sensed) and **interpolated** pixels (computed from neighbors) sit on two complementary, spatially periodic lattices (a quincunx pattern for the green channel of a Bayer array — see Ferrara et al. Fig. 1). The demosaicing kernel imposes a statistically regular, locally-predictable correlation structure between neighboring pixels that is absent in raw sensor data and destroyed by any operation that overwrites, blends, or re-synthesizes pixel content after acquisition.

**Property exploited.** Ferrara et al. formalize this with a linear prediction-error model. For a 1-D signal along a row, the acquired samples are `s_A(x) = G(x)` for `x` even, undefined for `x` odd (Eq. 1); after demosaicing with interpolation kernel `h_u`, the full reconstructed signal is `s_R(x) = s_A(x)` for even `x` and `Σ_u h_u s_A(x+u)` for odd `x` (Eq. 2). Using a **prediction kernel `k_u`** (assumed equal to the true interpolation kernel `h_u` in the ideal case) to predict `s_R` from its neighbors, the **prediction error `e(x) = s_R(x) - s_P(x)`** is *identically zero at odd (interpolated) positions* but has nonzero variance at even (acquired) positions when demosaicing is genuinely present (Eq. 4–8). **Splicing, copy-moving, smoothing, or re-rendering content destroys this variance imbalance** (native acquired pixels get overwritten, or foreign content carries a differently-phased/absent CFA signature), so measuring the *local* imbalance between prediction-error variance at acquired vs. interpolated lattice positions — down to blocks as small as `2×2` — gives a spatially fine-grained forgery signal without needing to know a priori where tampering occurred.

**Complementary optical-artifact principle (Mayer & Stamm).** A physically distinct but related cue: **lateral chromatic aberration (LCA)** — the wavelength-dependent focal shift of a lens that causes the red and blue channels to be laterally displaced relative to green near the image periphery, modeled as `c = f(r,θ) = α(r-ζ) + ζ` (Eq. 1), an expansion/contraction about the optical center `ζ` with expansion coefficient `α`. A copy-pasted region carries the LCA displacement field of its *source* location/image, not the target's, so local LCA measurements that deviate from the image's global LCA model are forgery evidence — a signal from lens optics rather than sensor demosaicing, usable as an independent, fusable detector.

## Input requirements

- **Format**: color image with a genuine Bayer-CFA acquisition history (RGB, any container; **uncompressed TIFF preferred** — all quantitative results in Ferrara, Jeon, and Singh are reported on uncompressed/TIFF imagery, with JPEG treated as a degraded secondary condition).
- **Preprocessing**:
  - **CFA phase determination** (recommended precondition, not assumed): use Jeon et al.'s SVD-based estimator (Section: Step-by-step algorithm, Pipeline C) to determine which of the 4 Bayer configurations (`RGGB`/`GRBG`/`GBRG`/`BGGR`) is actually in use, rather than assuming one from EXIF/camera-model metadata, which may be stripped, wrong, or belong to an unfamiliar sensor.
  - **Channel extraction**: Ferrara's method operates on the **green channel only** (upsampled ×2 in a Bayer array, giving equal counts of acquired/interpolated samples for a square block). Bammey's method and Singh's method use **all three channels** jointly (inter-channel correlation).
  - **Block alignment**: partition into non-overlapping blocks whose size `B` is a multiple of the CFA period (minimum `2×2` for Bayer).
- **Reliable when**: image is uncompressed or JPEG quality **≥ ~95%** (Ferrara); demosaicing algorithm is close to linear/bilinear (all four methods perform best or are validated primarily on bilinear-demosaiced imagery); sufficient local texture is present (prediction error needs nonzero local signal variance to be diagnostic).
- **Unreliable / inapplicable when**:
  - **JPEG compression below ~90–95% quality** — the dominant, universally-reported failure mode. Ferrara: AUC on the ideal bilinear case collapses from 0.9975 (uncompressed) toward chance as quality drops to 85% ("with a quality factor of 85%, our algorithm is unable to discriminate between the presence and absence of CFA artifacts"). Jeon: estimation accuracy for the *proposed* method drops from 96.19% (unaltered) to as low as 19.74–34.78% under JPEG QF 70–100 depending on demosaicing algorithm (Table 8). Bammey: NFA-based detection percentage drops from 100% at QF100 to 67% at QF90 (their Fig. 6 table, see benchmark table below), and is explicitly stated to lose the ability to detect *small* forgeries with strong certainty as quality decreases, even though large-region detection with lower certainty persists to reasonably high compression.
  - **Flat/uniform and saturated regions** — near-zero prediction error regardless of CFA presence (Ferrara, explicit limitation): "the proposed method is less effective in the presence of either almost flat areas or sharp edges."
  - **Sharp edges** — can mimic the CFA-absent statistical signature (false positives), same Ferrara limitation.
  - **Non-Bayer sensors** (Foveon X3, some super-CCD arrays) — inapplicable to the whole method family (Ferrara, explicit).
  - **Copy-move forgeries that by chance preserve CFA grid phase** — Bammey et al. state explicitly there is a **1-in-4 probability** that a copy-paste operation preserves the same CFA grid phase, in which case grid-phase-mismatch-based detection (Bammey, Jeon) is blind to it; Ferrara's variance-imbalance feature is not phase-based and can still catch some such cases (demonstrated qualitatively on a copy-move example, Fig. 9, where a phase-preserving paste is missed and a phase-mismatched paste is caught).
  - **Aggressive resizing/rescaling/re-demosaicing** after tampering shifts or destroys the periodic CFA phase entirely (general limitation, not separately quantified in the corpus).
  - **LCA-specific**: image content near the optical center has near-zero LCA magnitude, so angular-error-based LCA metrics are undefined/unusable there (Mayer & Stamm, explicit "Shortcomings of Existing Metrics" section); forgeries whose local LCA differs from the global model in *magnitude but not angle* (e.g. radially-shifted copy-paste) defeat angle-only metrics (their stated motivation for proposing the Mahalanobis-distance metric instead).

## Step-by-step algorithm

Four distinct algorithms are documented. **Pipeline A (Ferrara et al.) is PRIMARY** — most directly validated for fine-grained (down to `2×2` block) forgery *localization*, fully specified end-to-end, and requires no labeled training data. Pipeline B (Bammey) is recommended as a **statistically-rigorous confirmatory layer** (gives false-alarm guarantees Ferrara's method does not). Pipeline C (Jeon) is a **CFA-phase-verification preprocessing step**. Pipeline D (Mayer & Stamm LCA) is an **independent, physically-distinct complementary detector**, fusable alongside A–C.

### A. Fine-grained CFA likelihood map (Ferrara, Bianchi, De Rosa, Piva 2012) — PRIMARY

1. **Extract the green channel** from the RGB image (or a suspected Bayer-pattern single channel more generally).

2. **Compute the prediction error** at every pixel using a bidimensional prediction filter `k_{u,v}`:
   ```
   e(x,y) = s(x,y) − Σ_{(u,v)≠(0,0)} k_{u,v} · s(x+u, y+v)      [Eq. 9]
   ```
   In the ideal case `k_{u,v} = h_{u,v}` (the true interpolation kernel), but in practice the in-camera algorithm is unknown; a **fixed bilinear predictor is the recommended default** (see Implementation Notes — Ferrara's own results show bilinear as the most robust choice when the true kernel is unknown, even though matching the true kernel gives a small further gain).

3. **Compute the locally-weighted variance** of the prediction error within a `(2K+1)×(2K+1)` window:
   ```
   σ²_e(x,y) = (1/c) · [ ( Σ_{i,j=-K}^{K} α_ij · e²(x+i,y+j) ) − (μ_e)² ]      [Eq. 10]
   ```
   where `α_ij = α'_ij / Σ α'_ij`, `α'_ij = W(i,j)` if `e(x+i,y+j)` belongs to the same acquired/interpolated class as `e(x,y)`, else `0`; `W(i,j)` is a `(2K+1)×(2K+1)` **Gaussian window with standard deviation `K/2`**; `c = 1 − Σ α²_ij` is a bias-correction scale factor making the estimator unbiased (`E[σ²_e(x,y)] = Var[e(x,y)]`); `μ_e` is the local weighted mean of `e`.

4. **Compute the feature `L`** on `B×B` non-overlapping blocks (`B` a multiple of the Bayer period; **smallest usable `B=2`**, i.e. `2×2` blocks):
   ```
   L(k,l) = log[ GM_A(k,l) / GM_I(k,l) ]      [Eq. 11]
   GM_A(k,l) = [ Π_{(i,j)∈B_{A_kl}} σ²_e(i,j) ]^(1/|B_{A_kl}|)      [Eq. 12, and analogously GM_I for interpolated positions]
   ```
   `GM_A`/`GM_I` = geometric mean of the local prediction-error variance at **acquired**/**interpolated** pixel positions within block `B_{k,l}`, respectively.

5. **Statistical model (Gaussian Mixture)**: under hypothesis `M1` (CFA present, i.e. authentic), `L(k,l) ~ N(μ1, σ1²)` with `μ1 > 0` (Eq. 13); under `M2` (CFA absent/destroyed, i.e. tampered), `L(k,l) ~ N(0, σ2²)` (Eq. 14, mean fixed at zero by assumption, not estimated). Validated against a **Generalized Gaussian Distribution** fit (`p(L) = (1/Z)·exp(-|L-μ|/η)^ν`, Eq. 19) on real camera data (Table I in the paper), confirming approximate Gaussianity is reasonable per-image (median GGD shape parameter `ν` close to 2 across cameras/predictors, e.g. 1.7–2.2).

6. **Estimate GMM parameters** (`μ1, σ1, σ2`; `μ2` fixed at `0`) globally per image via **Expectation-Maximization**: initialize `μ1, σ1²` to the sample mean/variance of the observed features, `σ2² = σ1²/10`, mixing weight `α=0.5`; iterate to convergence, defined as increase in log-likelihood `< 10⁻³` or after **500 iterations** (paper's exact stopping criteria).

7. **Compute the posterior probability of authenticity** per block via Bayes' rule with equal priors `Pr{M1}=Pr{M2}=1/2`:
   ```
   Pr{M1 | L(k,l)} = 1 / (1 + L(L(k,l)))      [Eq. 15–16]
   L(L(k,l)) = Pr{L(k,l)|M2} / Pr{L(k,l)|M1}      [Eq. 17, likelihood ratio]
   ```
   This yields a **forgery/likelihood map** at native `B×B` block resolution: low `Pr{M1}` = likely tampered, high = likely authentic.

8. **Denoise the map**: cumulate/filter the log-likelihood map with either a **mean filter or a 5×5 median filter** (median outperforms mean in the paper's experiments — see Key Findings). Filtering can alternatively be achieved by computing the feature on smaller blocks (`B=2` or `4`) and cumulating posterior probabilities onto larger `C×C` blocks (`C=8` recommended) via:
   ```
   L_cum(k',l') = Π_{k,l} Pr{L(k,l)|M2} / Π_{k,l} Pr{L(k,l)|M1}      [Eq. 18]
   ```
   (assuming conditional independence of blocks given `M1`/`M2`). The paper notes directly computing features at `8×8` gives *slightly better* results than computing at `2×2`/`4×4` and cumulating, but the difference is small — cumulation is the right choice when finer-grained resolution is separately needed elsewhere in the pipeline.

### B. Statistically-guaranteed grid-position detection (Bammey, Morel, von Gioi 2018) — confirmatory / false-alarm-controlled layer

1. **Per-grid-position filter estimation**: for each of the 4 candidate CFA grid positions `P ∈ {RG/GB, GR/BG, BG/GR, GB/RG}`, estimate **8 linear filters** by least squares — one per missing-channel/sampled-color pairing (`α_{R→g}`, `α_{R→b}`, `α_{GR→r}`, `α_{GR→b}`, `α_{B→r}`, `α_{B→g}`, `α_{GB→r}`, `α_{GB→b}`), using **all observed channels** (not just single-channel EM) to capture inter-channel correlation:
   ```
   A[u+Nv, s+Nt] = Σ_{x,y} G[x,y,0]·M[x+u,y+v]·M[x+s,y+t]
   b[u+Nv] = Σ_{x,y} G[x,y,0]·M[x+u,y+v]·I[x,y,1]
   ```
   solving `A·α = b` for each filter, where `M` is the mosaiced image and `G` the CFA-grid indicator.

2. **Per-block voting**: divide the image into blocks of size `b×b` (`b` even, large enough to contain full Bayer unit cells; **32×32 or 64×64** used in reported forgery-detection experiments; **2×2 or 8×8** used in false-alarm validation experiments). Each block votes for the grid position `P` whose estimated-filter reconstruction residual is smallest.

3. **Statistical significance test (a contrario / NFA)**: model the null hypothesis as white noise where each block votes for one of the 4 configurations independently with probability `1/4`. For `n` blocks in a window with `n_P` voting for position `P`, the **Number of False Alarms**:
   ```
   NFA(n_P, n) = 4z · Σ_{i=n_P}^{n} C(n,i) · (1/4)^i · (3/4)^(n-i)      [exact formula from paper]
   ```
   where `z` is the number of disjoint windows tested (`z=1` for a single whole-image test). A grid position is declared **meaningful only if `NFA(n_P, n) ≤ p_g`** for a chosen false-alarm budget `p_g` (paper's example: `p_g = 0.001`).

4. **Forgery detection**: subdivide the image into windows; find the globally-dominant grid position `P₀`. A window is flagged **forged** if it contains at least one significant position that is **not** `P₀` (i.e., a locally-significant grid position disagreeing with the image's dominant CFA phase).

### C. CFA phase/pattern verification via SVD (Jeon, Shin, Eom 2017) — preprocessing step

1. **Crop a square block `M×M`** at the image center (or, per this engine's needs, any suspected-authentic region).
2. **Decompose into 4 down-sampled sub-blocks** `A = [A_1^{mA(1)}, A_2^{mA(2)}, A_3^{mA(3)}, A_4^{mA(4)}]` corresponding to the 4 positions of the `2×2` Bayer cell (Eq. 1), each of size `M/2 × M/2`.
3. **Construct color-difference blocks** between each candidate R (or B) sub-block and its adjacent G sub-block:
   ```
   D_i^{mD(i)} = R_i^{mR(i)} − G_i^{mG(i)}      [Eq. 2]
   F_i^{mF(i)} = B_i^{mB(i)} − G_i^{mG(i)}      [Eq. 3]
   ```
   (color-difference is near-constant in flat regions, exposing acquired-vs-interpolated statistical asymmetry more cleanly than raw pixel values).
4. **SVD each difference block**: `J = UΣVᵀ` (Eq. 4); sum the **truncated (small) singular values** from index `t` to `M/2` (large singular values encode low-frequency background content, irrelevant; small ones encode high-frequency texture/edge content where the statistical asymmetry appears):
   ```
   S_{Di} = Σ_{n=t}^{M/2} λ_{Di}(n),    S_{Fi} = Σ_{n=t}^{M/2} λ_{Fi}(n)      [Eq. 5–6]
   ```
5. **Diagonal-pair similarity test**: for each `i`, let `d(i)` denote the diagonally-opposite position (`d(1)=4, d(2)=3`, etc.). Compute absolute differences:
   ```
   V_k^D = |S_{Dk} − S_{D_{d(k)}}|,    V_k^F = |S_{Fk} − S_{F_{d(k)}}|      [Eq. 7–8]
   ```
   The diagonal pair with the **larger** `V^D + V^F` is the R/B diagonal pair (larger difference ⇒ one member has the original-R statistical signature, the other has interpolated-R):
   ```
   b̃ = argmax_k [V_k^D + V_k^F]      [Eq. 9]
   ```
6. **Final position selection**: compare `S_{D_b̃} + S_{F_{d(b̃)}}` vs. `S_{D_{d(b̃)}} + S_{F_{d(b̃)}}` to pick between the two remaining candidate configurations sharing that diagonal pair:
   ```
   b = b̃ if S_{D_b̃} + S_{F_{d(b̃)}} > S_{D_{d(b̃)}} + S_{F_{d(b̃)}}, else d(b̃)      [Eq. 10]
   ```
7. **Parameters**: block size `M` tested at `32, 64, 128, 256, 512` (accuracy increases with `M`, see benchmark table); truncated singular-value cutoff `t = (M/2)/2` (i.e., the upper half of singular values by index, empirically fixed, not swept).

### D. Lateral chromatic aberration (LCA) inconsistency (Mayer & Stamm 2018) — complementary optical-artifact detector `(training-free — Neyman-Pearson hypothesis test, not a classifier)`

1. **Global LCA model**: `c = f(r,θ) = α(r-ζ) + ζ` (Eq. 1) relating a point `r` in a reference channel (green) to its corresponding point `c` in a comparison channel (red or blue); `α` = expansion coefficient, `ζ` = optical center.
2. **Efficient local displacement estimation via diamond search**: at each of `N` Shi-Tomasi corner keypoints `r_i`, find the displacement `d̂(r) = (m_max, n_max)` maximizing block-similarity (correlation coefficient) between a `W×W` reference-channel block and a candidate-shifted comparison-channel block, searching only the **Large Diamond Search Pattern (LDSP)** and **Small Diamond Search Pattern (SDSP)** neighborhoods (Algorithm 1) instead of the full `(2uΔ+1)²` exhaustive grid — reduces similarity calculations by **1–2 orders of magnitude** with no measurable added estimation error (validated against a synthetic checkerboard ground truth). Recommended parameters from the paper's own experiments: window `W=64×64`, max displacement `Δ=3`, upsample factor `u` up to `10`.
3. **Global model fit**: estimate `θ* = (α*, ζ*)` by nonlinear least squares (iterative Gauss-Newton) over all local displacement estimates: `θ* = argmin_θ Σ_i ‖d̂(r_i) − d(r_i,θ)‖²` (Eq. 5).
4. **LCA inconsistency vector**: `e(r) = α⁻¹·(d̂(r) − d(r,θ))` (Eq. 10) — the Cartesian difference between local and global LCA estimates, scaled by inverse expansion coefficient. Under authentic content, `e(r) = n` (pure observational noise, IID Gaussian, near-zero mean `μ0`); under forged (copy-pasted) content, `e(r) = n + δ`, where `δ` is an unknown **forgery offset** (Eq. 11–12).
5. **Concatenate green→red and green→blue inconsistency** into one 4D vector per keypoint: `e(r) = [e_x^{gr}, e_y^{gr}, e_x^{gb}, e_y^{gb}]ᵀ` (Eq. 13, with each component additionally scaled by its own channel-pair expansion coefficient `α^{gr}`/`α^{gb}`).
6. **Hypothesis test**: `H0: e(r)~N(μ0,Σ)` (authentic) vs. `H1: e(r)~N(μ0+δ,Σ)` (forged) (Eq. 14–15); derive the log-likelihood ratio from `N` IID 4D Gaussian observations (Eq. 16–17); estimate the unknown forgery offset by its maximum-likelihood value `δ̂ = ē − μ0` (Eq. 19, where `ē` is the sample mean of `e(r_i)` over the `N` keypoints in the tested region); substituting gives the final **Mahalanobis-distance detection statistic**:
   ```
   N·(ē − μ0)ᵀ·Σ⁻¹·(ē − μ0)   ≷_τ   (H1 if greater, H0 if less)      [Eq. 20]
   ```
   `μ0`, `Σ` (mean/covariance of observational noise) are estimated from LCA inconsistency vectors **outside** the tested region. This is a Neyman-Pearson-optimal detector under the stated Gaussian/IID assumptions — no classifier training required, only estimation of `μ0`/`Σ` from the image itself.

## Output

- **Pipeline A (Ferrara)**: `Pr{M1|L(k,l)} ∈ [0,1]` per `B×B` block — **probability the block is authentic (CFA present)**. Tampering score for fusion = `1 − Pr{M1|L(k,l)}`, i.e. `0` = confidently authentic, `1` = confidently tampered. Native output is a **heatmap**; reduce to a whole-image scalar via max, 95th-percentile, or fraction-of-blocks-below-threshold, per this engine's fusion-layer contract `(reduction rule not specified in the corpus — engineering recommendation)`.
- **Pipeline B (Bammey)**: binary **forged/not-forged flag per window**, with an associated `NFA` value (not a probability — an expected-false-alarm-count under the null). Smaller `NFA` = higher confidence. `(Not specified in corpus)`: to fold into a `[0,1]` fusion score, engineering recommendation is `score = 1 − clip(NFA / p_g, 0, 1)` at the chosen `p_g`, or a monotonic transform of `-log(NFA)`.
- **Pipeline C (Jeon)**: a **discrete CFA configuration label** (`RGGB`/`GRBG`/`GBRG`/`BGGR`), not itself a tampering score — feeds Pipeline A as a preprocessing/verification step. Can secondarily serve as a coarse global-consistency check: if different image regions vote for different CFA phases with high confidence, that disagreement is itself weak tampering evidence `(engineering extrapolation, not tested in the corpus for this purpose)`.
- **Pipeline D (Mayer & Stamm)**: the scaled Mahalanobis statistic `N·(ē−μ0)ᵀΣ⁻¹(ē−μ0)`, compared to threshold `τ` — unbounded above, `0` = no measured inconsistency. `(Not specified in corpus)`: normalize via a chi-square CDF (the statistic is asymptotically `χ²`-distributed with 4 degrees of freedom under `H0` for large `N`, by standard Mahalanobis-distance theory, though the paper does not itself state this normalization) to obtain a `[0,1]` p-value-like score for fusion.

## Key findings from papers

**Manipulation types detected best**: splicing/copy-paste with mismatched or destroyed CFA phase (all four papers); geometrically-transformed splices, since resizing/rotation is itself detectable as a demosaicing-artifact disruption (Ferrara Fig. 10 example); optically-inconsistent copy-paste via LCA (Mayer & Stamm), including cases (radially-shifted content) that defeat angle-only LCA metrics.

**Documented failure cases / limitations**: see Input Requirements reliability section above (JPEG compression is the dominant shared weakness across A/B/C; flat/saturated/sharp-edge regions defeat Ferrara specifically; 1-in-4 phase-preserving copy-move defeats phase-based methods B/C; LCA is undefined near the optical center and angle-only LCA metrics fail on magnitude-only inconsistencies).

**Benchmark tables**:

| Paper | Dataset | Metric | Value | Conditions |
|---|---|---|---|---|
| Ferrara 2012 | 400 images, 4 cameras (Canon EOS 450D, Nikon D50/D90/D7000) | AUC | 0.9845 (bilinear) – 0.9975 (bicubic/gradient-based) | Ideal case: known demosaicing kernel, uncompressed, 8×8 blocks |
| Ferrara 2012 | same | AUC | lower, "median predictor far worse than others" | Realistic case: unknown in-camera demosaicing, bilinear predictor recommended default |
| Ferrara 2012 | same | AUC vs. JPEG QF | drops sharply below QF95; unusable at QF85 | 8×8 blocks, 128×128 forged region |
| Ferrara 2012 | same | Comparison vs. DM/GC-B/GC-L baselines | Ferrara's method has highest AUC at every scenario (I)–(VI) tested | Fig. 7 |
| Bammey 2018 | Kodak dataset, 24 images, reinterpolated w/ contour stencils | Detection % / NFA | QF100: 100% / 10⁻³⁰⁰; QF99: 100% / 10⁻⁴⁰; QF97: 98% / 10⁻¹⁰; QF95: 81% / 10⁻⁶; QF90: 67% / 10⁻⁴ | Robustness-to-JPEG table |
| Bammey 2018 | 800,000 white-noise 32×32 images | False-alarm validation | Empirical false-alarm rate always below the theoretical NFA bound | Confirms the a contrario model is conservative/valid |
| Bammey 2018 | Tampered images from Christlein et al. dataset | NFA at detected forged windows | `< 10⁻¹⁰⁰` | Small forgeries, 32×32 windows of 4×4 blocks; NFA threshold set to 10⁻¹⁰ |
| Jeon 2017 | 1460 raw images, Dresden database | Estimation accuracy vs. block size | 91.20% (M=32) → 97.97% (M=512), avg. across 8 demosaicing algorithms | No post-processing |
| Jeon 2017 | same | Estimation accuracy vs. blur (σ=0.5→1.5) | 96.19% → 87.08% avg. | Gaussian blur post-processing, 256×256 blocks |
| Jeon 2017 | same | Estimation accuracy vs. sharpening | 91.25% → 94.81% avg. | Laplacian sharpening α=0.1→0.5 |
| Jeon 2017 | same | Estimation accuracy vs. JPEG QF | 27.53% (avg, QF100) down to 34.35%(QF70)-ish range, generally 19.74–34.78% per-method | JPEG compression QF 100/90/80/75/70 — **both proposed and conventional methods largely fail here** |
| Jeon 2017 | same | Computation time | 0.035s (proposed) vs. 0.176s (conventional) per 256×256 block | ~5× faster |
| Singh 2020 | UCID, 1338 uncompressed TIFF images | Decision error `P_e` | 0.0751 (proposed MTPM+SVM) vs. 0.10–0.35 (Ferrara/Dirik-Memon/Gallagher-Chen baselines) | Lower is better |
| Singh 2020 | same | Avg. computation time | 157s (proposed) vs. 160–205s (baselines) | Per test image; still far from real-time |
| Mayer 2018 | Dresden database, "Schoner Muehle"/"Reed" sets, 434 images | Similarity-calculation reduction | Diamond search: 14.4 calcs (upsample=5) vs. 961 (exhaustive) = 1.5%; 16.3 vs 3721 (upsample=10) = 0.4% | W=64×64, Δ=3 |
| Mayer 2018 | same | Runtime speedup | 59× (upsample=5) to 221× (upsample=10) | e.g. 0.099s vs 21.96s per displacement at upsample=10 |
| Mayer 2018 | synthetic checkerboard ground truth | LCA estimation error | No additional error vs. Gloe et al.'s exhaustive-search method | Diamond search matches exhaustive-search accuracy |

## Implementation notes

- **Predictor choice for Pipeline A**: use **bilinear** as the default fixed prediction kernel `k_{u,v}` when the true in-camera demosaicing algorithm is unknown — Ferrara's realistic-condition experiments (Fig. 5b, Fig. 4 histograms) show bilinear is the most robust choice across unknown demosaicing algorithms, even though matching the *true* kernel gives a small further AUC gain in the ideal/known case.
- **Block-size/resolution tradeoff**: computing the feature directly at `8×8` gives marginally better results than computing at `2×2`/`4×4` and cumulating onto `8×8` via Eq. 18 — but in realistic forgery scenarios, tampered regions are rarely larger than `8×8` pixels in the finest case, so cumulating from `2×2` is the practical choice when true fine-grained localization is required.
- **EM initialization/convergence** (Pipeline A): initialize `μ1, σ1²` from the sample mean/variance of `L`, `σ2² = σ1²/10`, mixing weight `0.5`; converge at `Δ(log-likelihood) < 10⁻³` or 500 iterations, whichever first.
- **NFA computation** (Pipeline B) is a **binomial tail sum** — implementable directly with `scipy.stats.binom.sf` rather than a manual summation loop for numerical stability at extreme values (`NFA` values as low as `10⁻³⁰⁰` require log-space computation; plain floating point will underflow).
- **SVD truncation** (Pipeline C): the paper fixes `t = (M/2)/2` (upper half of singular value indices) without an ablation over other cutoffs — treat as a fixed engineering default rather than a tuned optimum; `(the paper does not justify or sweep this choice — corpus ambiguity)`.
- **Diamond search corner selection** (Pipeline D): segment the image into non-overlapping blocks and select the single largest-corner-metric keypoint per block (Shi-Tomasi minimum eigenvalue) — this spatially distributes keypoints and avoids clumping in texture-rich regions while bounding compute cost, per the paper's explicit design note.
- **No public reference code found in the extracted text** for Ferrara, Bammey, Jeon, or Singh. Mayer & Stamm **do** cite a public repository: `misl.ece.drexel.edu/downloads` or `gitlab.com/mislgit/misl-lca-tifs`.
- **Recommended Python libraries**:
  - `numpy`/`scipy.ndimage` for the local weighted-variance convolution (Eq. 10) — implement `α_ij` as a precomputed Gaussian-windowed, lattice-masked kernel and apply via `scipy.signal.convolve2d` or `scipy.ndimage.correlate`.
  - `scipy.optimize.least_squares` or direct normal-equation solve (`numpy.linalg.lstsq`) for the EM M-step (Pipeline A) and the 8-filter least-squares estimation (Pipeline B).
  - `numpy.linalg.svd` for Pipeline C's SVD step.
  - `scipy.stats.binom` for the NFA binomial tail (Pipeline B).
  - `cv2.goodFeaturesToTrack` (Shi-Tomasi corners) and `scipy.optimize.least_squares` (Gauss-Newton global LCA fit) for Pipeline D; a manual diamond-search block-matching loop (no standard library implements this directly — MPEG motion-estimation libraries have analogous primitives but are not a drop-in fit for LCA's sub-pixel/upsampled search).
  - `numpy.linalg.inv`/`numpy.linalg.solve` for the Mahalanobis distance's `Σ⁻¹` term (Pipeline D, Eq. 20) — prefer `solve` over explicit `inv` for numerical stability.

## Key references

- **ferrara2012.pdf** — P. Ferrara, T. Bianchi, A. De Rosa, A. Piva, "Image Forgery Localization via Fine-Grained Analysis of CFA Artifacts," IEEE TIFS, vol. 7, no. 5, pp. 1566–1577, Oct. 2012. Source of: the complete prediction-error/GMM/EM/Bayesian-posterior pipeline (Pipeline A, Eq. 1–19); all ideal/realistic AUC benchmarks; the JPEG-quality degradation curve; the flat-region/sharp-edge/non-Bayer/copy-move limitations. (Duplicate file `ferrara2012 (1).pdf` — identical content.)
- **bammey2018.pdf** — Q. Bammey, J.-M. Morel, R. Grompone von Gioi, "Automatic Detection of Demosaicing Image Artifacts and its Use in Tampering Detection," IEEE MIPR 2018. Source of: the 8-filter least-squares grid-position estimator, the a contrario/NFA statistical framework (Pipeline B), and its JPEG-robustness/false-alarm-validation benchmarks.
- **jeon2017.pdf** — J.J. Jeon, H.J. Shin, I.K. Eom, "Estimation of Bayer CFA Pattern Configuration Based on Singular Value Decomposition," EURASIP J. Image and Video Processing, 2017:47. Source of: the SVD-based CFA phase identification pipeline (Pipeline C, Eq. 1–10) and its accuracy/robustness/timing benchmarks across 8 demosaicing algorithms.
- **singh2020.pdf** — G. Singh, K. Singh, "Digital Image Forensic Approach Based on the Second-Order Statistical Analysis of CFA Artifacts," Forensic Science International: Digital Investigation, vol. 32, 200899, 2020. Source of: the target-difference-image selection procedure, the intra-/inter-block MTPM construction (Eq. 1–10), the 648-dimensional feature vector, and the SVM-based classification results `[ML — excluded from the no-ML engine; the underlying MTPM feature extraction is training-free and reusable, but the paper's own classification stage requires a trained SVM. Training-free substitute (engineering recommendation, not in corpus): threshold a distance/divergence between the test image's MTPM feature vector and a reference "clean" MTPM statistic, analogous to the divergence-thresholding substitutes used elsewhere in this engine — not validated in this source]`.
- **mayer2018.pdf** — O. Mayer, M.C. Stamm, "Accurate and Efficient Image Forgery Detection Using Lateral Chromatic Aberration," IEEE TIFS, 2018 (early access). Source of: the LCA displacement model, the diamond-search efficient local-estimation algorithm (Pipeline D), the LCA-inconsistency Mahalanobis-distance optimal detector (Eq. 1–20), and its efficiency/accuracy benchmarks. A distinct physical mechanism (lens optics, not sensor demosaicing) — complementary to Pipelines A–C, fusable as an independent detector, weak/unusable near the image's optical center or on cameras with strong in-camera chromatic-aberration correction.
- **islam2020.pdf** — M.M. Islam, G. Karmakar, J. Kamruzzaman, M. Murshed, "A Robust Forgery Detection Method for Copy-Move and Splicing Attacks in Images," Electronics (MDPI), 9(9):1500, 2020. Marginal — general DCT+LBP+SVM forgery detector, not CFA-specific; `[ML]`; not re-read in this pass, no new content extracted.
- **An-Unpaired-Learning-Based-Method-for-Image-Despeckling.pdf** — Zafari & Jalali, 2025. Off-topic (SAR speckle-noise removal); not re-read in this pass.
- **Image-Interpolation-Using-Non-adaptive-Scaling-Algorithms-for-Multimedia-Applications-A-Survey.pdf** — Neetha, Moses, Selvathi, Springer LNEE 700, 2021. Background only (general resampling/upscaling survey, not demosaicing); not re-read in this pass.
- **A-new-robust-training-free-proactive-deepfake-detection-scheme-using-watermarking-and-identity-aware-hashing.pdf** — Lai et al., Expert Systems With Applications, 2026. Off-topic (proactive watermarking, not passive CFA forensics); not re-read in this pass.
