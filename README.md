# Predictive and Generative Modeling of Therapeutic Antibody Developability
> Using fine-tuned protein language models to predict and optimize antibody developability from sequence alone.
GitHub Repo for BMI702 Final Project, Sidhant Puntambekar and Jack Hwang

## Overview

Therapeutic antibody development is costly and time-intensive, with many candidates failing due to poor **developability**. Developability metrics refer to the set of biophysical properties an antibody must satisfy to be successfully manufactured and safely administered. This project investigates whether sequence-based protein language models can:

- **(Task A)** Predict developability metrics and downstream therapeutic approval from amino acid sequence alone
- **(Task B)** Enable lead optimization of developability properties through guided CDR3 sequence generation

By operating from sequence only (no experimental assays required), these tools are designed for early-stage lead evaluation — before expensive clinical trials.

---

## Methods

### Task A: Developability Prediction

**Approval Prediction Baseline**
- XGBoost classifier trained on 11 experimentally measured biophysical features (HIC, SMAC, HAC, PR_Ova, PR_CHO, SEC %Monomer, AC-SINS pH 6.0/7.4, Tm1, Tm2, Titer)
- Bootstrapped over 20 stratified 80:20 train-test splits
- Median AUPRC: **0.665** vs. null AUPRC: 0.593

**Developability Metric Prediction Baseline**
- Ridge regression on mean-pooled p-IgGen sequence embeddings
- Evaluated under 5-fold hierarchical cluster, IgG isotype-stratified cross-validation
- Best cross-validation Spearman ρ: PR_Ova (0.549), PR_CHO (0.496), AC-SINS pH7.4 (0.394), HIC (0.335)

**Planned: AbLang2 Fine-tuning**
- LoRA fine-tuning of AbLang2 encoder on GDPa1 for approval classification and per-metric regression
- Compared against XGBoost (experimental features upper bound) and p-IgGen Ridge (sequence-only baseline)

### Task B: Generative CDR3 Optimization

The generative pipeline operates in three stages:

1. **Oracle**: Ridge regression (α=0.1) trained on 480-dimensional AbLang2 full-sequence embeddings to predict HIC and AC-SINS (Spearman ρ = 0.32 and 0.445 on held-out test set)
2. **PLS Latent Space**: CDR3 embeddings compressed from 480d → 10d via Partial Least Squares, maximizing covariance with HIC and AC-SINS labels
3. **Flow Model**: FiLM-conditioned MLP velocity network trained with the conditional flow matching (CFM) objective, conditioned on masked framework embeddings and desired developability targets

**Inference**: For each test antibody, the flow model generates 50 CDR3 targets in PLS space; AbLang2 decoding is reweighted toward the top-5 targets to produce 20 candidate sequences; the oracle selects the best by composite HIC + AC-SINS z-score.

---

## Results

### Generative Pipeline (n=50 held-out antibodies)

| Metric | Parent | Baseline (AbLang2) | Guided (Flow + AbLang2) |
|---|---|---|---|
| Oracle HIC (↓) | 2.738 ± 0.048 | 2.466 ± 0.043 | 2.493 ± 0.046 |
| Oracle AC-SINS (↓) | 3.741 ± 1.215 | −0.545 ± 1.178 | −0.441 ± 1.044 |
| CamSol solubility (↑) | −0.004 ± 0.056 | 0.564 ± 0.071 | 0.649 ± 0.078 |
| SAP score (↓) | 0.449 ± 0.016 | 0.216 ± 0.019 | 0.242 ± 0.020 |

All improvements over parent are statistically significant (paired t-test, p < 0.001). Guided and unguided conditions perform comparably, suggesting AbLang2's language model prior is the primary driver of improvement at this data scale.

---

## Data

**PROPHET-Ab (GDPa1)** — [Arsiwala et al., mAbs 2025](https://doi.org/10.1080/19420862.2025.2593055)
- 246 therapeutic antibodies (106 FDA-approved, 135 clinical-stage, 5 withdrawn)
- Paired VH/VL sequences + 11 experimentally measured biophysical features
- 5-fold hierarchical cluster, IgG isotype-stratified splits provided

**OAS (Observed Antibody Space)** — [Olsen et al., Protein Science 2022](https://doi.org/10.1002/pro.4205) *(planned for flow model pretraining)*
- >2 billion antibody sequences from >80 studies

---

**Key dependencies:**
- `ablang2` — antibody language model backbone
- `scikit-learn` — Ridge regression, cross-validation
- `xgboost` — approval prediction baseline
- `torch` — flow model training
- `torchdiffeq` — ODE integration at inference

---

## Authors

**Jack Hwang** — Generative pipeline (PLS dimensionality reduction, conditional flow matching, steered AbLang2 generation)  
**Sidhant Puntambekar** — Predictive baselines (Ridge regression, XGBoost), AbLang2 fine-tuning  

Department of Biomedical Informatics, Harvard Medical School
