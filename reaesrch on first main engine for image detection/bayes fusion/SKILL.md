# Bayesian / Evidence Fusion for Multi-Modal Forgery Detection

## Core mathematical principle

**Underlying concept.** No single forensic detector (noise/PRNU, JPEG double-quantization, CFA/demosaicing, wavelet, Fourier/ghost, Benford, lighting, perspective) is reliable across all forgery types and image conditions — Phan-Ho & Retraint state this directly: PRNU/sensor-noise methods "work well for RAW or TIFF images but worse for detecting JPEG forgeries with low quality compression," DQ-coefficient methods "fail in detecting forgery images which is processed in RAW and resave in JPEG," and PRNU-based methods specifically "have high false detection rate on saturated and dark regions." Each detector is a **noisy, sometimes-contradictory partial vote**. A fusion layer combines these into one calibrated authenticity confidence that is more robust than any single cue and can expose disagreement explicitly rather than silently averaging it away.

**Why combining beats any single detector, and what each rule assumes.** Bayesian (Naive Bayes) fusion assumes conditional independence of detectors given the hypothesis and requires calibrated priors/likelihoods; it produces a single point-estimate probability. Dempster-Shafer (DST) fusion requires no priors, natively represents *ignorance* (mass can be assigned to "don't know," not forced onto one hypothesis), and produces a **belief-plausibility interval** rather than a point estimate — but its combination rule is known to behave poorly under high inter-source conflict (see below). MRF/MAP fusion adds a **spatial-coherence prior** on top of a Bayesian-style per-detector likelihood term, which matters specifically when detectors output spatial tampering *maps*, not just whole-image scores, because neighboring pixels' true labels are correlated in a way plain per-pixel Naive Bayes ignores.

## Input requirements

- **Contract each sub-detector must satisfy**: per this engine's other eight modules, each should return `{score ∈ [0,1] or log-likelihood-ratio, confidence/reliability ∈ [0,1], modality: str}`, where `confidence` reflects the detector's own self-assessed applicability to *this* image (e.g., the CFA module's confidence should be low on heavily-JPEG-compressed input; the perspective module should abstain/report near-zero confidence when vanishing-point estimation has few RANSAC inliers).
- **What must be true before fusion is valid**:
  - **Score-level fusion, not feature- or decision-level** — Phan-Ho & Retraint's own taxonomy (Section II.C) distinguishes feature-level (train one classifier on concatenated features — most powerful but `[ML]` and computationally demanding), score-level (aggregate independently-produced scalar scores/probabilities — **this is what this engine's modules already produce and what this fusion layer consumes**), and decision-level (combine already-thresholded binary outputs — cheap but loses the confidence information the fusion layer needs). Never hard-threshold a sub-detector's output before it reaches this layer.
  - **Calibration**: Naive Bayes fusion (Pipeline A) needs per-detector likelihoods `P(E_i|H_forged)`, `P(E_i|H_authentic)` estimated from a labeled validation set matching deployment conditions (compression level, resolution, camera mix) — Daniel's own paper states this plainly via its worked example (Table 1: likelihoods are estimated by counting how many of the known-fraud/known-genuine transactions triggered each rule).
- **Reliability conditions for each fusion rule**:
  - **Naive Bayes (Pipeline A)** reliable when detectors are **not strongly correlated** given the hypothesis; degrades (not catastrophically, per Daniel's citation of prior theoretical work on Naive Bayes's "surprising implausible efficacy" even under violated independence, but non-trivially) when several detectors key off the same underlying signal.
  - **DST (Pipeline C)** reliable when conflict `K` between sources is low-to-moderate; Phan-Ho & Retraint's own head-to-head experiment (see Key Findings) shows DST's advantage is scenario-specific, not universal, and its combination rule is explicitly flagged in both source papers as having known pathological behavior as `K→1`.
  - **MRF/MAP (Pipeline B)** reliable specifically when the detectors being fused output **spatial maps** with genuine neighbor-to-neighbor label correlation (adjacent pixels are very likely both-tampered or both-authentic) — it is the wrong tool for fusing whole-image scalar scores with no spatial structure.

## Step-by-step algorithm

### A. Weighted log-odds Naive Bayes — PRIMARY whole-image combiner (Daniel 2021)

1. Define hypotheses `H_f` (forged) and `H_g` (authentic), evidence set `E = {E_1,...,E_n}` (each `E_i` = one sub-detector's triggered/scored evidence).
2. **Exact Bayesian fusion** (intractable in general because the joint likelihood needs the full joint distribution):
   ```
   P(H_f|E) = P(E|H_f)P(H_f) / P(E)          [Eq. 1]
   ```
3. **Naive Bayes approximation** (assume conditional independence of detectors given the hypothesis):
   ```
   P(H_f|E) = P(H_f)·Π_{i=1}^n P(E_i|H_f) / P(E_1,...,E_n)          [Eq. 2]
   P(E) = P(H_f)·Π_i P(E_i|H_f) + P(H_g)·Π_i P(E_i|H_g)          [Eq. 3, 5]
   P(H_f|E) = (1/Z)·P(H_f)·Π_{i=1}^n P(E_i|H_f)          [Eq. 4a, Z = P(E)]
   ```
4. **Log-space computation** (avoid vanishing precision for large `n`):
   ```
   log-odds(forged|E) = log(P(H_f)/P(H_g)) + Σ_i log(P(E_i|H_f)/P(E_i|H_g))
   ```
   convert back to probability via sigmoid of the log-odds.
5. **Worked numeric example, transcribed exactly from the source** (illustrates the calibration procedure end-to-end on a 30-transaction/2-detector toy case — directly analogous to a small calibration set of 30 labeled images and 2 forensic modules):
   - Priors from base rate: `P(H_f)=7/30=0.23`, `P(H_g)=23/30=0.77`.
   - Per-detector likelihoods estimated by counting how often each detector triggers on known-forged vs. known-authentic samples: `P(E_1|H_f)=0.57, P(E_1|H_g)=0.26, P(E_2|H_f)=0.14, P(E_2|H_g)=0.09`.
   - `P(E|H_f) = 0.57×0.14 = 0.0798`; `P(E|H_g) = 0.26×0.09 = 0.0234`.
   - `P(E) = 0.23×0.0798 + 0.77×0.0234 = 0.0184+0.0180 = 0.0364`.
   - `P(H_f|E) = (0.0798×0.23)/0.0364 = 0.504`; `P(H_g|E)=0.495`.
6. **Combining more than two sources**: the combination is associative — `m_1⊕m_2⊕m_3 = (m_1⊕m_2)⊕m_3 = m_2⊕(m_1⊕m_3)`, so all `n` detectors can be folded in via repeated pairwise (or one single product/log-sum) combination without ambiguity about ordering.

### B. MRF/MAP fusion for spatial tampering maps (Phan-Ho & Retraint 2022)

1. Given decision/probability **maps** `m^{(k)}` from `K` forensic algorithms (not whole-image scalars — this pipeline is specifically for per-pixel/per-block outputs), find the tampering map `t ∈ {0,1}^N` (`N` = pixel count) maximizing the posterior:
   ```
   t̂ = argmax_{t∈[0,1]^N} P(t | m^{(k)}: k=1,...,K)          [Eq. 1]
   ```
2. Bayes' rule, dropping the constant `P(m)`, then assuming independence across pixels and across detectors:
   ```
   t̂ = argmax_t P(m^{(k)}:k|t)·P(t)
     = argmax_t Π_{i=1}^N P(m_i^{(k)}:k|t_i)·P(t)
     = argmax_t Π_{i=1}^N Π_{k=1}^K P(m_i^{(k)}|t_i)·P(t)          [Eq. 2–4]
   ```
3. **Spatial-coherence prior**, modeled as a Markov Random Field via Gibbs/Ising energy (Hammersley-Clifford equivalence between MRF and Gibbs form):
   ```
   P(t) = Z⁻¹·e^{−U(t)} = Z⁻¹·e^{−Σ_{c∈C}V_c(t)}          [Eq. 5]
   ```
   `V_c` = clique potential over clique `c` (a subset of mutually-neighboring pixels), `Z` = normalizing constant.
4. **Energy-minimization reformulation** (take negative log of Eq. 6, turning MAP into a minimization):
   ```
   minimize:  Σ_{c∈C} V_c(t) − Σ_{i=1}^N Σ_{k=1}^K log P(m_i^{(k)}|t_i)          [Eq. 7]
   ```
   solved with a **graph-cut solver** (the source paper uses the UGM MATLAB toolbox's max-flow/min-cut implementation).
5. **Concrete Ising-model instantiation used in the paper's own two experiments** (single-element and two-element cliques only):
   ```
   Σ_{c∈C} V_c(t) − Σ_i Σ_k log P(m_i^{(k)}|t_i)
       →  Σ_{i=1}^N Σ_{k=1}^K E_τ(m_i^{(k)}, t_i) + α·Σ_{i=1}^N t_i + β·Σ_{i=1}^N Σ_{j∈𝒩_i} |t_i − t_j|          [Eq. 8]
   ```
   `𝒩_i` = 4-neighborhood (top/bottom/left/right) of pixel `i`; `α` controls preference toward **sparser** tampering maps; `β` controls **interaction strength** between neighboring pixels (higher `β` = stronger smoothness pull). Per-pixel data term:
   ```
   E_τ(m_i, t_i) = −log max(Ψ_min, Ψ_τ(m_i,t_i))          [Eq. 23]
   Ψ_τ(m_i,t_i) = { 1 − m_i/(2τ),           t_i=0
                  { 1 + (m_i−1)/(2(1−τ)),   t_i=1          [Eq. 24]
   ```
   `Ψ_min ∈ [0,1]` is a floor to avoid `−log(0)`; `τ` is a threshold parameter shared with each individual detector's own binarization.
6. **Two fully-worked experimental instantiations from the same source** — directly reusable recipes:
   - Fusing `M^{PRNU}` (Eq. 16, a binary decision map from a Neyman-Pearson PRNU correlation test) and `M^{SF}` (Eq. 17, a statistical-feature/SCRM-based ensemble-classifier vote-score map, rescaled to `[0,1]`) via `m^{(1)}=M^{PRNU}, m^{(2)}=M^{SF}` into Eq. 8.
   - Fusing `M^{CP}` (copy-paste/CFA-demosaicing-artifact decision map) and `M^{CM}` (copy-move/SIFT-keypoint decision map) the same way.

### C. Dempster-Shafer evidence combination — conflict diagnostic and interval-valued alternative (Daniel 2021; Phan-Ho & Retraint 2022)

1. **Frame of discernment** `Ω = {H_f, H_g}` (or `{it, nt}` — is-tampered/not-tampered, per Phan-Ho & Retraint's pixel-level framing); **power set** `2^Ω = {∅, {H_f}, {H_g}, {H_f,H_g}}` (Eq. 6–8, Daniel).
2. **Basic mass assignment (BMA)** `m: 2^Ω → [0,1]`, with `m(∅)=0` and `Σ_{A∈2^Ω} m(A)=1` (Eq. 9–11). Mass on a **singleton** (`{H_f}`) means "this specific hypothesis is true"; mass on the **full set** `{H_f,H_g}` means genuine, explicit **uncertainty** — "one of these is true, but the source doesn't know which."
3. **Belief and plausibility**:
   ```
   bel(A) = Σ_{B⊆A} m(B)          [Eq. 12]
   pl(A) = Σ_{B∩A≠∅} m(B)          [Eq. 14]
   bel(A) ≤ P(A) ≤ pl(A)          [Eq. 18, the probability interval]
   pl(A) = 1 − bel(Ā)          [Eq. 17]
   ```
4. **Dempster's combination rule** for two independent sources' mass functions `m_1, m_2`:
   ```
   m_{1,2}(A) = (1/(1−K))·Σ_{B∩C=A≠∅} m_1(B)m_2(C)          [Eq. 21]
   K = Σ_{B∩C=∅} m_1(B)m_2(C)          [Eq. 22, conflict mass, K∈[0,1]]
   ```
   `K→0` = sources agree; `K→1` = sources in near-total disagreement, at which point `1/(1−K)` blows up and the combined result becomes numerically unstable and can be counter-intuitive (documented pathology in both source papers).
5. **Worked numeric example WITHOUT uncertainty** (both sources assign mass only to singletons — Daniel's Table 2, `m_1(H_f)=0.6, m_1(H_g)=0.4, m_2(H_f)=0.8, m_2(H_g)=0.2`):
   ```
   K = m_1(H_f)m_2(H_g) + m_1(H_g)m_2(H_f) = (0.6×0.2)+(0.4×0.8) = 0.44
   (m_1⊕m_2)(H_f) = 0.48/(1−0.44) = 0.8571
   (m_1⊕m_2)(H_g) = 0.08/(1−0.44) = 0.1428
   ```
   Since focal sets are singletons only, `P(H_f) = bel(H_f) = 0.8571` directly (no interval — Daniel's paper states explicitly: when focal sets are only singletons, belief/plausibility/probability all coincide).
6. **Worked numeric example WITH uncertainty** (each source now also assigns mass to the full set `{H_f,H_g}` — Daniel's Table 3, `m_1(H_f)=0.7, m_1(H_g)=0.1, m_1(H_f,H_g)=0.2, m_2(H_f)=0.3, m_2(H_g)=0.2, m_2(H_f,H_g)=0.5`):
   ```
   K = (m_1(H_f)m_2(H_g)) + (m_1(H_g)m_2(H_f)) = (0.6×0.2)+(0.4×0.3) = 0.17  [as computed in source, combining pairwise conflict terms]
   m_{1,2}(H_f) = 0.253,   pl(H_f) = m(H_f)+m(H_f,H_g) = 0.253+0.723 = 0.976
   ```
   Resulting **probability interval**: `0.25 ≤ P(H_f) ≤ 0.98` — genuinely wide, reflecting the substantial uncertainty mass both sources carried. **Reducing** each source's self-reported uncertainty (Daniel's Table 4 variant) narrows this to `0.40 ≤ P(H_f) ≤ 0.77` — demonstrating directly that **more confident sub-detectors produce a tighter, more actionable interval**, which is the operational value of carrying uncertainty mass explicitly rather than forcing every detector to output a bare point probability.
7. **`Bel(T) vs Bel(N)` pixel-level decision rule** (Phan-Ho & Retraint's DST instantiation for two forensic detectors, mass constructed directly from each detector's own decision-map confidence with **no unassigned uncertainty mass** in this specific instantiation — i.e. `m_i(it)=t_i, m_i(nt)=1−t_i`, `m_i(it,nt)=0`, Eq. 18–20):
   ```
   K = t_1n_2 + t_2n_1
   m_{1,2}(it) = t_1t_2/(1−K),   m_{1,2}(nt) = n_1n_2/(1−K)
   ```
   Decision: pixel flagged tampered if `Bel({it}) > Bel({nt}) + λ` for a chosen threshold `λ` (chosen heuristically in the source, not derived).

### D. PCA score fusion — zero-calibration cold-start fallback (Rao, Ghanekar, Chitnis, Dawkhar & Mishra 2025, from the `fourier transform ghost transform` module)

`(Documented in full in that module's SKILL.md; summarized here as the fusion-layer's fallback when no labeled calibration data yet exists for Pipeline A's likelihoods.)`

1. Stack each sub-detector's per-image scalar score into one feature vector.
2. Apply **PCA**, take the **first principal component** as the unified suspicion score, normalize to `[0,1]`.
3. Threshold (source paper uses `0.33`, empirically chosen, not derived — recalibrate for this engine).
4. **Needs no labeled data at all** — PCA is unsupervised, unlike Pipeline A's likelihood estimation — making this the correct fallback for early deployment before a calibration set exists, at the cost of no principled probabilistic interpretation and (per that module's own honesty caveat) **no quantitative validation in its source paper** beyond a qualitative "obvious clustering" claim on 10 images.

## Output

- **Final fused probability**: `P(H_f|E) ∈ [0,1]` from Pipeline A (point estimate) or the `[bel(H_f), pl(H_f)]` interval from Pipeline C (interval estimate) — **this engine should report both**: the Naive Bayes point estimate as the primary score, and the DST interval width `pl(H_f)−bel(H_f)` as an explicit **uncertainty measure**, since the two pipelines are answering related but different questions (point confidence vs. confidence bounded by explicit ignorance).
- **Log-odds**: the intermediate `Σ_i log(P(E_i|H_f)/P(E_i|H_g))` value from Pipeline A, exposed for explainability/audit (per-detector contribution is directly visible in a log-odds sum, unlike in a raw product).
- **Per-detector contributions**: `{modality: log-likelihood-ratio contribution}`.
- **Conflict score**: pairwise `K` (Eq. 22) computed between every pair of detectors, `max` or `mean` reported as a whole-fusion conflict summary.
- **Decision band**: `{authentic, forged, needs_review}` — `needs_review` should fire when the **conflict score** is high even if the aggregate point-probability looks confident, since high inter-detector conflict is itself diagnostic of an unusual image (double compression, anti-forensic manipulation, out-of-distribution content) per both source papers' explicit discussion of DST's conflict term as meaningful signal, not just noise to be normalized away.
- **Uncertainty/confidence interval**: use the DST belief-plausibility interval (Pipeline C) directly — `(the corpus does not specify any alternative confidence-interval method for the pure-Bayesian Pipeline A path; treat DST's interval as this engine's primary uncertainty-quantification mechanism, engineering recommendation)`.

## Confidence weighting

- **Exact weighted log-odds formula**: `log-odds(forged|E) = log(P(H_f)/P(H_g)) + Σ_i w_i·log(P(E_i|H_f)/P(E_i|H_g))`, where `w_i ∈ [0,1]` is detector `i`'s per-image reliability multiplier `(the plain Naive Bayes derivation in the corpus, Eq. 1–5 of Daniel's paper, has no explicit w_i term — this weighting is an engineering extension of that formula, motivated by the reliability-conditioning behavior documented per-module elsewhere in this engine, not itself present in the fusion papers)`.
- **How weights should adjust to image conditions, and which module's own paper motivates each condition**: down-weight PRNU/noise-analysis when the image shows large saturated/dark regions (Phan-Ho & Retraint's own stated motivation for why DST fusion helped in their PRNU+SF experiment — see Key Findings); down-weight CFA/demosaicing and Fourier-resampling detectors below ~JPEG quality 90–95% (both modules' own papers, cited in their respective SKILL.md files); down-weight the JPEG-compression module when the background quality factor is near-ceiling (QF≈95–100, per that module's own documented hard case); down-weight the perspective module when vanishing-point RANSAC inlier count is low (that module's own abstention recommendation); the lighting module should carry the **lowest baseline weight of all nine**, per that module's own explicit corpus-gap finding.
- **Condition-detection logic that must run before fusion**: an estimated-JPEG-quality-factor gate (feeds JPEG-compression, CFA, and Fourier modules' weight adjustment), a saturated/dark-region-fraction gate (feeds noise-analysis weighting), and a geometric-structure-sufficiency gate (feeds perspective-module weighting) are the three condition detectors explicitly motivated by this corpus's findings; `(their precise thresholds are set per-module in each module's own SKILL.md, not re-derived here)`.

## Prior probability

- **The corpus does not prescribe a real-world base rate for forgery prevalence.** Daniel's worked example uses `P(H_f)=0.23` purely because that specific toy dataset happened to contain 7 fraud cases out of 30 — it is a dataset artifact, not a claim about real-world fraud (or forgery) base rates.
- Setting the prior to `0.5` (uninformative) makes the fused output interpretable as a **pure likelihood-ratio comparison** — how much more consistent the evidence is with "forged" than "authentic" — without asserting any belief about how common forgeries are in the deployment population. `(This default, and the recommendation to expose it as a user-adjustable parameter per use-case, e.g. much lower for a generic photo-verification tool than for a forensic-investigation queue already pre-filtered for suspicion, is an engineering recommendation — not specified in the corpus.)`

## Key findings from papers

**Which fusion rule wins in which scenario, with the actual numbers** (Phan-Ho & Retraint's two head-to-head experiments — the corpus's single most direct evidence on this question):

| Experiment | Detectors fused | Dataset | Metric | PRNU/CP alone | SF/CM alone | DST fusion | MRF fusion |
|---|---|---|---|---|---|---|---|
| 1: PRNU + Statistical-Features (SF) | PRNU-based + SCRM-ensemble | UTT (3 cameras: Canon EOS-100D, Nikon D5200, Panasonic DMC-GM1), 10 forgery + 10 genuine images with heavy saturated/dark regions | **F1-score** | 0.0056 | 0.0057 | **0.0309 (best — ~10× either single detector)** | 0.0208 |
| 2: Copy-paste (CFA/demosaicing) + Copy-move (SIFT) | Le et al. (CFA) + Mahfoudi et al. (SIFT+dissimilarity) | Korus et al. "Realistic Tampering Dataset" | **F1-score** | 0.2323 (Le et al. alone) | 0.2948 (Mahfoudi et al. alone) | 0.0650 (**worst — fusion HURTS here**) | **0.3912 (best)** |

**Why DST won experiment 1**: the PRNU detector's dominant failure mode is high false-alarm rate specifically on saturated/dark regions; the SF detector doesn't share that failure mode. DST's conflict-arbitration mechanism lets it favor the region where the two detectors *agree* and suppress PRNU's known false positives — exactly the scenario DST's mass-on-uncertainty machinery is designed for.

**Why MRF won (and DST specifically failed) experiment 2**: the two detectors here are not so much *conflicting* as **complementary with disjoint blind spots** — Le et al.'s CFA method detects copy-paste but not copy-move; Mahfoudi et al.'s method detects copy-move but not copy-paste. Phan-Ho & Retraint's own diagnosis: "the DST is limited to combining the conflict evidence... there are usually the conflict parts in these maps... that is the reason why the decision maps generated from DST fusion... are usually all black" — DST's conflict-suppression behavior, which was exactly the *strength* in experiment 1, becomes a **liability** here because it discards exactly the complementary (non-overlapping, hence "conflicting" in DST's formal sense) evidence that MRF's spatial-smoothness prior is instead able to exploit and propagate. **Explicit paper conclusion**: no universal winner — the best technique is scenario-dependent, driven by (a) how much the detectors spatially/structurally correlate or have complementary vs. genuinely-conflicting coverage, (b) how much they conflict, and (c) whether reliable priors are available.

**Documented failure cases / limitations**: DST's `1/(1−K)` normalization is unstable and can produce counter-intuitive results as `K→1` (both source papers). Naive Bayes assumes conditional independence "that is not the case in most real problems" (Daniel, direct quote) — though cited prior theoretical work shows Naive Bayes classifiers work surprisingly well even when this assumption is violated. MRF/graph-cut fusion pays "a cost for building the edge structure... and the computing complexity for finding the optimization on the graph" (Phan-Ho & Retraint) — i.e., it is the most computationally expensive of the three rules.

## Implementation notes

- **Log-space computation is mandatory, not optional**, once more than a handful of detectors are combined via Naive Bayes — direct products of `n=8` per-module likelihoods will underflow long before reaching a meaningful precision; always sum log-likelihood-ratios and convert to probability via sigmoid only at the final step.
- **Clamp likelihood ratios** before taking their log — a detector reporting exactly `0` or `1` probability for one hypothesis (from a small calibration sample, or a saturating score function) produces `log(0)=−∞`, which will silently dominate and invalidate the entire fused sum; clip every per-detector probability to `[ε, 1−ε]` for a small `ε` before combining `(not specified in corpus — engineering recommendation, standard practice for numerically implementing any Naive Bayes combiner)`.
- **Detectors that abstain** (e.g. perspective module reporting low VP confidence, lighting module's structurally low trust) should contribute **zero** to the log-odds sum (equivalent to `log(1)=0`, i.e. a likelihood ratio of exactly 1, "no evidence either way") rather than being force-included with a degenerate score — this is the log-odds-sum analogue of DST's explicit uncertainty-mass mechanism, and achieves a similar effect without needing the full DST machinery for every module.
- **Correlated-detector double-counting**: several of this engine's eight forensic modules are *not* independent — the JPEG-compression and Benford modules both key off compression/quantization statistics; the wavelet and Fourier modules both key off frequency-domain irregularities. Naive Bayes fusion double-counts evidence from correlated detectors. `(Not resolved in the corpus itself — engineering recommendation, consistent with the standard Naive-Bayes-correlated-features caveat: either group correlated detectors into one combined "compression-domain evidence" block with a single combined likelihood before fusing, or explicitly model the covariance if enough labeled data supports it.)`
- **Do not use DST as the sole n-way combiner across all 8 detectors** — the two source-paper experiments both fuse exactly **two** detectors at a time; there is no corpus evidence about DST's behavior or stability fusing eight sources simultaneously, and the conflict-mass pathology as `K→1` is expected to compound with more sources. Use pairwise DST conflict scores as a **diagnostic layer** (Pipeline C, Output section) laid over the primary Naive Bayes fusion (Pipeline A), not as a replacement full-scale combiner.
- **Reference toolboxes cited in the corpus**: the UGM MATLAB toolbox (`cs.ubc.ca/~schmidtm/Software/UGM.html`) for the graph-cut/max-flow solver underlying Pipeline B. Python equivalents: `PyMaxflow` (Boykov-Kolmogorov max-flow, directly suited to the binary-labeling energy-minimization form of Eq. 7–8) or `pygco` (a Python wrapper around the same `gco` graph-cut library used broadly in the vision literature).
- **Recommended Python libraries**:
  - `numpy` for the core log-odds sum and sigmoid conversion.
  - `scipy.special.expit` for a numerically-stable sigmoid (log-odds → probability).
  - `PyMaxflow` / `pygco` for Pipeline B's graph-cut energy minimization.
  - Implement Dempster's combination rule (Pipeline C) directly — it is a handful of lines of arithmetic over a 4-element power set for the binary-hypothesis case used throughout this corpus; no specialized library is needed or was cited by either source paper.
  - `sklearn.decomposition.PCA` for Pipeline D (already noted in the Fourier/ghost module).

## Key references

- **A_Comparative_Study_of_Bayesian_and_Dempster-Shafer_Fusion_on_Image_Forgery_Detection.pdf** — A.-T. Phan-Ho, F. Retraint, "A Comparative Study of Bayesian and Dempster-Shafer Fusion on Image Forgery Detection," IEEE Access, vol. 10, pp. 99268–99281, 2022. **The most directly applicable paper in this module.** Source of: the fusion-level taxonomy (feature/score/decision), the full MRF/MAP energy-minimization formulation (Pipeline B, Eq. 1–8, 23–24), the DST pixel-level instantiation (Pipeline C step 7), and — critically — both fully-quantified head-to-head experiments (Table 3/4 in the source, reproduced above) that are this corpus's only direct evidence on when each fusion rule wins.
- **2104.07440v1.pdf** — F. Daniel, "Bayesian and Dempster-Shafer models for combining multiple sources of evidence in a fraud detection system," Lusis AI, March 2021 (arXiv:2104.07440v1). Source of: the full Naive Bayes log-odds derivation (Pipeline A, Eq. 1–5) with its complete worked numeric example (step 5), the DST fundamentals (mass/belief/plausibility, Eq. 6–20) with two complete worked numeric examples showing interval narrowing under reduced uncertainty (Pipeline C steps 5–6), and the associativity property enabling n-source combination (Eq. 26–27).
- **1307.5996v2.pdf** — Q. Wei, N. Dobigeon, J.-Y. Tourneret, "Bayesian Fusion of Multi-Band Images," University of Toulouse (arXiv:1307.5996v2). Not a forgery paper — fuses multi-sensor remote-sensing imagery — included for its clean, general worked example of the Bayesian estimation machinery: the observation model `z_p = F_p·x + e_p` (Eq. 1–2), and the **MAP vs. MMSE estimator distinction** (`x̂_MAP = argmax_x f(z|x)f(x)`, efficient but can converge to local optima, vs. the posterior-mean MMSE estimator, more accurate but requiring intractable high-dimensional integration, here approximated via Hamiltonian Monte Carlo). Generalizable takeaway for this engine: an explicit prior regularizes the fused estimate, and a hierarchical model can jointly infer per-source reliability/noise parameters alongside the fused result rather than fixing them by hand — the principled version of this module's per-detector confidence-weighting mechanism. Its MCMC-based MMSE approach is appropriate for offline/batch fusion, not real-time — treat as architectural inspiration only, not a directly-reusable formula for this engine's per-image fusion pass.
- **Image-Tampering-Detection-Using-Multi-Feature-Scoring-and-PCA-Based-Classification.pdf** (folder: `fourier transform ghost transform`) — A. Rao et al., CISCON 2025. Source of Pipeline D, the zero-calibration PCA score-fusion fallback — documented in full in that module's own `SKILL.md`, including its explicit lack of quantitative validation.
