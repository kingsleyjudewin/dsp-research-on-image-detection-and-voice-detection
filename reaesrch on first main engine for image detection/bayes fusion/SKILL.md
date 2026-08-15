# Bayesian / Evidence Fusion for Multi-Modal Forgery Detection

## Purpose
No single forensic detector (noise/PRNU, JPEG double-quantization, CFA/demosaicing, wavelet, Fourier/ghosts, Benford, lighting, perspective) is reliable across all forgery types and image conditions — each exploits one specific trace that is absent, weak, or masked in certain images (e.g. PRNU fails on saturated/dark regions; JPEG-artifact detectors fail on images that were never re-compressed; CFA cues vanish after resizing). Each detector therefore produces a noisy, sometimes contradictory, partial vote. A fusion layer combines these per-detector scores/beliefs into one calibrated authenticity confidence that is more robust and harder to evade than any single cue, and that can also produce an explicit uncertainty/conflict measure rather than a false-confident point estimate.

## Techniques found in the literature

### 1. "Bayesian Fusion of Multi-Band Images" — `1307.5996v2.pdf`
Wei, Dobigeon, Tourneret (Univ. Toulouse). Not a forgery paper — it fuses multi-sensor remote-sensing images (PAN/MS/HS) into one high-resolution scene — but it is the cleanest worked example of the **general Bayesian fusion machinery** the engine can reuse.
- Formulates each sensor's observation as a linear/noisy degradation of a shared latent truth: `z_p = F_p·x + e_p`, `e_p ~ N(0, Λ_p)`.
- Applies Bayes' rule to get the posterior over the true scene given all observations: `f(x|z) = f(z|x)f(x) / f(z)`.
- Two standard point estimators from the posterior:
  - MAP: `x̂_MAP = argmax_x f(z|x) f(x)` (penalized-likelihood form; efficient but can get stuck in local optima).
  - MMSE (posterior mean): more accurate but requires integrating the posterior — approximated here via MCMC (Gibbs sampling + a Hamiltonian Monte Carlo step) because the posterior is high-dimensional and non-conjugate.
- Key generalizable idea: an explicit **prior** regularizes the fused estimate and a **hierarchical model** lets you jointly infer nuisance parameters (e.g. per-source noise/reliability) alongside the fused result, rather than fixing them by hand.
- Tradeoff: statistically principled and handles heterogeneous/correlated sources well, but MCMC-based MMSE is computationally heavy — appropriate for offline/batch fusion, not necessarily per-frame real-time video.

### 2. "Bayesian and Dempster-Shafer models for combining multiple sources of evidence in a fraud detection system" — `2104.07440v1.pdf`
Fabrice Daniel (Lusis AI). Directly analogous problem to forgery fusion (binary hypothesis: fraudulent vs. genuine ↔ tampered vs. authentic), using rule/model outputs as evidence.

**Bayesian (Naive Bayes) approach:**
- Hypotheses `H_f` (fraud/forged) and `H_g` (genuine); evidence set `E = {E_1,...,E_n}` (each a triggered detector/rule).
- Exact fusion: `P(H_f|E) = P(E|H_f)P(H_f) / P(E)`.
- Because the joint likelihood `P(E_1,...,E_n|H_f)` is intractable, assume conditional independence of detectors given the hypothesis (Naive Bayes):
  `P(H_f|E) = P(H_f) · Π_i P(E_i|H_f) / P(E_1,...,E_n)`,
  with normalizer `Z = P(E) = P(H_f)Π_i P(E_i|H_f) + P(H_g)Π_i P(E_i|H_g)`.
- For large n, take logs to avoid vanishing precision (sum of log-likelihood-ratios instead of a product).
- Needs priors `P(H_f)`, `P(H_g)` and per-detector likelihoods `P(E_i|H_f)`, `P(E_i|H_g)` estimated from labeled history.
- Weakness: conditional independence rarely holds exactly for correlated detectors, though Naive Bayes is empirically robust to this violation.

**Dempster-Shafer (DST) approach** (used when priors/likelihoods aren't reliably estimable, only posterior-like scores are):
- Frame of discernment `Ω = {H_f, H_g}`; power set `2^Ω = {∅, {H_f}, {H_g}, {H_f,H_g}}`.
- Each detector `i` supplies a mass function `m_i` with `m_i(∅)=0`, `Σ_A m_i(A) = 1`; mass on `{H_f,H_g}` = unassigned uncertainty for that detector.
- Belief `bel(A) = Σ_{B⊆A} m(B)`; Plausibility `pl(A) = Σ_{B∩A≠∅} m(B)`; these bound the true probability: `bel(A) ≤ P(A) ≤ pl(A)` — giving an interval, not just a point estimate.
- **Dempster's combination rule** for two independent sources:
  `m_{1,2}(A) = (1/(1-K)) · Σ_{B∩C=A≠∅} m_1(B)m_2(C)`,
  where `K = Σ_{B∩C=∅} m_1(B)m_2(C)` is the **conflict mass** between the two sources (`K∈[0,1]`; `K→1` means near-total disagreement, and the `1/(1-K)` normalization becomes numerically unstable / can produce counterintuitive fused results in that regime — this is DST's well-known conflict pathology).
- Combines pairwise and is associative, so `n` sources are fused by repeated pairwise combination: `m_1⊕m_2⊕...⊕m_n`.
- Worked numeric example shows: with uncertainty mass explicitly modeled, reducing a detector's self-reported uncertainty narrows the resulting probability interval and can flip a borderline case across a decision threshold — i.e., DST's interval output is directly useful for a "needs human review" band, not just accept/reject.
- Paper's conclusion: DST wins when you lack reliable priors/likelihoods and want an explicit uncertainty interval; Bayes wins when you have enough labeled history to estimate priors/likelihoods and want a single calibrated probability.

### 3. "A Comparative Study of Bayesian and Dempster-Shafer Fusion on Image Forgery Detection" — `A_Comparative_Study_of_Bayesian_and_Dempster-Shafer_Fusion_on_Image_Forgery_Detection.pdf`
Phan-Ho & Retraint (IEEE Access, 2022). **This is the most directly applicable paper** — it fuses actual image-forensic detectors and empirically compares Bayesian vs. DST fusion on real tampering datasets.

Background taxonomy it establishes (useful vocabulary for the engine):
- **Fusion levels**: feature-level (concatenate features, one classifier — most powerful but expensive), score-level (combine each detector's independently-produced scalar score/probability — this is what our engine should use), decision-level (combine already-thresholded binary decisions — cheap but lossy).
- **Fusion method families**: rule-based (AND/OR/MIN/MAX/weighted sum), probability-based (Bayesian), evidence-reasoning (DST, and fuzzy set theory), classification-based (WMV, Behavior Knowledge Space, Naive Bayes combiner, SVM/KNN meta-classifiers, or fuzzy/Choquet integrals).

**Bayesian fusion, formulated as MAP over a Markov Random Field (pixel/region-level tampering map fusion):**
- Given decision/probability maps `m^(k)` from `K` forensic algorithms, find the tampering map `t` maximizing the posterior:
  `t̂ = argmax_t P(t | m^(1),...,m^(K))`.
- Assuming per-pixel and per-detector conditional independence: `t̂ = argmax_t Π_i Π_k P(m_i^(k)|t_i) · P(t)`.
- The prior `P(t)` is modeled as an MRF (Gibbs/Ising form: `P(t) = Z^-1 exp(-U(t))`), enforcing spatial smoothness (neighboring pixels likely share the same tampered/authentic label) — this turns MAP-Bayesian fusion into an **energy-minimization problem** solved with graph-cuts:
  `minimize  Σ_c V_c(t) − Σ_i Σ_k log P(m_i^(k)|t_i)`
  where `V_c` is a clique potential; solved with a graph-cut solver (UGM toolbox).
- Effectively: Bayesian fusion here = per-detector log-likelihood evidence terms + a spatial-coherence prior, optimized jointly.

**DST fusion of two concrete forensic detectors (PRNU-based + Statistical/Color-Rich-Model-based):**
- Each detector's per-pixel output is converted to a mass function on `Ω={it, nt}` (is-tampered / not-tampered): `m(it)=t_i`, `m(nt)=1-t_i`, i.e. the detector's own confidence score doubles directly as belief mass (no unassigned uncertainty mass allocated in this instantiation).
- Combined via Dempster's rule with conflict `K = t_1n_2 + t_2n_1`; final decision: pixel flagged tampered if `Bel({it}) > Bel({nt}) + λ` for a chosen threshold `λ`.

**Reported results / key empirical finding:**
- Fusing PRNU (strong overall, but high false-alarm rate on saturated/dark regions) with a Statistical-Feature detector (SRM-based, decent overall but poor spatial localization): **DST fusion outperformed Bayesian(MRF) fusion** here, because DST's conflict-handling let it favor the region where the two detectors *agreed* and suppress PRNU's known false positives in saturated/dark regions.
- Fusing a demosaicing-artifact detector with a SIFT copy-move detector for copy-paste + copy-move tampering: **MRF-based Bayesian fusion outperformed DST fusion**, because exploiting spatial neighbor-dependency (which MRF does natively and DST does not) mattered more than conflict-arbitration in that task.
- Explicit conclusion of the paper: **no universal winner — the best fusion technique is scenario-dependent**, driven by (a) how much the detectors spatially/structurally correlate, (b) how much they conflict, and (c) whether reliable priors are available. They flag known limitations of pure DST under high conflict and note transferable belief model (TBM) / Dezert-Smarandache theory (DSmT) as extensions for future work.

## Recommended approach for this engine

Given a fusion layer over 8 heterogeneous detectors (noise/PRNU, JPEG, CFA, wavelet, Fourier/ghost, Benford, lighting, perspective) that will disagree on different manipulation types and image conditions, synthesize the above into:

**1. Fusion level:** score-level fusion. Each sub-detector should be a self-contained module producing `(score ∈ [0,1] or log-likelihood-ratio, confidence/reliability)` — never raw features (too expensive to jointly retrain) and never a single thresholded bit (loses information the fusion layer needs to arbitrate). This matches the taxonomy in paper 3 and is what both Bayesian and DST formulations above actually consume.

**2. Primary fusion rule: weighted-log-odds Naive Bayes**, following paper 2's derivation, as the default combiner:
- Maintain, per detector `i`, empirically-estimated (or continually recalibrated) likelihoods `P(E_i|H_forged)` and `P(E_i|H_authentic)` from a labeled validation set (or per-detector ROC-derived calibration curves if scores are continuous, e.g. via Platt scaling/isotonic regression into a likelihood ratio).
- Fuse in log space to avoid underflow with 8 detectors:
  `log-odds(forged|E) = log(P(H_forged)/P(H_authentic)) + Σ_i log(P(E_i|H_forged)/P(E_i|H_authentic))`.
- Convert back to a probability via the sigmoid of the log-odds for the final confidence score.
- This is cheap, incremental (add/remove a detector without retraining others), interpretable (per-detector contribution is visible for explainability/audit), and matches how both forensic papers (2 and 3) actually deploy Bayesian fusion in practice.

**3. Add an MRF spatial-coherence prior only where the engine outputs a localization map** (not needed for whole-image authenticity scoring). If/when the engine produces per-region or per-pixel tampering maps (e.g. combining wavelet/CFA/PRNU spatial maps), reuse paper 3's energy-minimization formulation: combine per-detector log-likelihood terms with an Ising/Potts smoothness prior over neighboring blocks, solved via graph-cut. This is a natural extension of the same log-odds sum, not a separate framework.

**4. Use DST as a secondary/conflict-detection layer, not the primary fusion rule:**
- Run Dempster-Shafer combination alongside Naive Bayes specifically to compute the conflict mass `K` between detector pairs.
- Treat `K` as a diagnostic: high pairwise conflict between two normally-correlated detectors (e.g. JPEG-artifact detector says "authentic" while Benford's law says "forged" on a supposedly single-compression JPEG) is itself evidence of something unusual (double compression, anti-forensic manipulation, or an out-of-distribution image) and should be surfaced to a human reviewer or trigger a "low confidence / needs manual review" flag rather than being silently averaged away.
- Do not use DST as the sole final-score combiner across all 8 detectors: `K` compounds quickly with more sources and pushes toward the near-total-conflict regime where `1/(1-K)` normalization becomes unstable and Dempster's rule is known to produce counterintuitive results (documented failure mode in both papers 2 and 3).

**5. Reliability weighting.** Not all 8 detectors are equally trustworthy for a given image:
- Precompute per-detector operating conditions where it is known to be unreliable (PRNU on saturated/dark/flat regions; CFA cues after resizing/recompression; Benford's law on heavily denoised or low-texture images; lighting/perspective analysis on single-object or indoor-uniform-lighting images).
- Implement this as an input-conditioned reliability multiplier `r_i(image) ∈ [0,1]` applied to detector `i`'s log-likelihood-ratio contribution before summing (down-weight, don't hard-drop, so a detector can still break a tie). This operationalizes the "reliability index" concept paper 3 references from rule-based AND-fusion work, inside the Bayesian log-odds sum instead of a brittle boolean AND/OR rule.
- Optionally expose this per-image reliability estimate itself as a detector output (`confidence` field in the input/output contract below) so the fusion layer can compute it generically rather than hardcoding per-detector heuristics.

**6. Input/output contract:**
- Each sub-detector module returns: `{score: float [0,1], likelihood_ratio_or_logodds: float, confidence: float [0,1], modality: str}`. `confidence` reflects the detector's own assessment of applicability to this image (e.g. PRNU confidence low if the region is saturated).
- Fusion module returns: `{final_probability_forged: float, log_odds: float, per_detector_contributions: {modality: contribution}, conflict_score: float (max pairwise DST K across detector pairs), decision: {authentic|forged|needs_review}, flagged_conflicts: [(modality_a, modality_b, K)]}`. The `needs_review` decision fires when `conflict_score` exceeds a tuned threshold even if the aggregate probability looks confident — this surfaces exactly the failure mode DST is good at detecting.

**7. Pitfalls to avoid:**
- Don't assume detector independence blindly — several of the 8 detectors are correlated (e.g. JPEG artifacts and Benford's law both key off compression statistics; wavelet and Fourier both key off frequency-domain irregularities). Correlated detectors double-count evidence in Naive Bayes; consider grouping correlated detectors into a single "compression-domain evidence" block with one combined likelihood, or explicitly modeling covariance if labeled data supports it.
- Don't run DST as the sole n-way combiner across many sources — combine pairwise conflict diagnostics instead, or if using DST for the actual score, cap it to 2-3 highly-informative detectors as done in paper 3, not all 8 at once.
- Don't hard-threshold sub-detector outputs before fusion (decision-level fusion) — it throws away the confidence information the fusion layer needs; keep scores continuous into the fusion stage.
- Calibrate per-detector likelihoods/scores on a held-out set that matches deployment distribution (compression level, resolution, source camera mix) — likelihoods estimated from one dataset (as in paper 3's PRNU/SF experiment on three specific cameras) do not transfer automatically.
- Log per-detector contributions for every fused decision — needed both for explainability (forensic/legal use case) and for diagnosing which detector is driving false positives/negatives in production.

## References
- `1307.5996v2.pdf` — Qi Wei, Nicolas Dobigeon, Jean-Yves Tourneret, "Bayesian Fusion of Multi-Band Images," University of Toulouse (arXiv:1307.5996v2).
- `2104.07440v1.pdf` — Fabrice Daniel, "Bayesian and Dempster-Shafer models for combining multiple sources of evidence in a fraud detection system," Lusis AI, Paris (arXiv:2104.07440v1, March 2021).
- `A_Comparative_Study_of_Bayesian_and_Dempster-Shafer_Fusion_on_Image_Forgery_Detection.pdf` — Anh-Thu Phan-Ho, Florent Retraint, "A Comparative Study of Bayesian and Dempster-Shafer Fusion on Image Forgery Detection," IEEE Access, vol. 10, pp. 99268–99281, 2022.
