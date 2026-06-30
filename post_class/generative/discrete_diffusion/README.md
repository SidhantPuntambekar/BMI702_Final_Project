# OADM + ESM‑2 8M — Discrete‑Diffusion Baseline for Paired‑Antibody Generation

Notebook: `discrete_diffusion_esm2_antibody_baseline_final.ipynb`

This is a re‑implementation of the **unguided generative model** from Zhao, Moller,
Quintero‑Cadena & van Niekerk, *"Guided Generation for Developable Antibodies"* (ICML 2025
GenBio Workshop, arXiv:2507.02670), Section 1.1. It exists to serve as a **baseline** for a
controlled comparison against a Discrete Flow Matching (DFM) model built later. The two models
must run on identical data, tokenizer, length distribution, and evaluation code — only the
training objective and the sampler differ.

**Out of scope on purpose:** the paper's SVDD guidance and the HIC / AC‑SINS developability
predictors are *not* implemented here. Hook points where SVDD would attach are noted in the
generation cells.

---

## 1. What this is, in one paragraph

A masked discrete‑diffusion model over antibody sequences. The heavy and light chains of a pair
are concatenated as `<heavy>|<light>` (a literal pipe token marks the boundary) and fed to an
ESM‑2 8M architecture (`facebook/esm2_t6_8M_UR50D`, 6 layers, hidden dim 320) with a masked‑LM
head. Training uses the **order‑agnostic diffusion model (OADM / ARDM)** objective (Hoogeboom
2021; the objective EvoDiff uses): mask a random subset of amino‑acid positions, predict them,
reweight the loss per the ARDM ELBO. Generation runs the reverse process: start fully masked,
unmask one position at a time (most‑confident‑first), sample each token with a temperature, and
freeze it.

---

## 2. Quick start

### Smoke test (no data, runs anywhere with a GPU)
Open in Colab → **Runtime ▸ Run all**. With no `DATA_CSV` set, the notebook falls back to a tiny
**synthetic** corpus that exercises every code path end‑to‑end (train → generate → evaluate →
save). The synthetic sequences are **not real biology** — use this only to confirm the pipeline
runs. Do not draw conclusions from smoke‑test outputs.

### Real run (paper‑scale corpus)
1. Mount Drive (Cell 3).
2. Put `bulk_download.sh` at `DRIVE_ROOT/bulk_download.sh` (from the OAS website: select
   **Paired ▸ Download**).
3. Run **Stage 1** (download + extract paired sequences) — resumable; a unit whose parquet
   already exists is skipped, so a disconnect just resumes.
4. Run **Stage 2** (completeness filter + MMseqs2 clustering @ 90%) — cached behind a `.done`
   sentinel, so it runs once.
5. Run the training cell. Training is checkpointed and auto‑resumes (see §6).
6. Run the generation, metrics, and save cells.

`DATA_CSV` auto‑wires to the clustered representative CSV if it exists; otherwise it uses the
smoke corpus. You can also point `DATA_CSV` at any `heavy,light` CSV of your own.

---

## 3. What is faithful to the paper

- **OADM forward process.** For each sequence, over **amino‑acid positions only**
  (`<cls>/<eos>/<pad>/|` are never masked), draw `t ~ Uniform{1..D}`, mask `m = D − t + 1`
  randomly chosen positions, predict them.
- **ARDM ELBO reweighting.** Each masked‑token cross‑entropy is weighted by `D / (D − t + 1)`,
  matching EvoDiff's `OAMaskedCrossEntropyLoss`.
- **Architecture.** ESM‑2 8M via `EsmForMaskedLM`. The pipe token is added to the tokenizer and
  the input embeddings **and** the separate LM‑head bias are resized (the latter is an ESM quirk
  that silently breaks logits if missed).
- **Chain handling.** `<heavy>|<light>` concatenation with a dedicated pipe token; the pipe is
  never masked, so the H/L boundary is preserved.
- **Generation.** Reverse OADM; **minimum‑entropy** position selection (random‑order also
  available); token sampling restricted to the 20 canonical amino acids; sequence length sampled
  from the empirical training distribution.
- **Temperature.** Softmax temperature `p(x_i) = e^{x_i/T} / Σ e^{x_j/T}`. **Position selection
  uses the native (T=1) distribution; temperature is applied only to token sampling**, so the
  diversity/naturalness‑vs‑temperature sweep (paper Fig. 2) is well‑defined. (`gen_temperature=1.0`
  is the paper's favorable operating point.)

---

## 4. Deviations from the paper — read before publishing results

These are deliberate, documented choices. They change *which sequences* you train on or *which
init* you use, not the algorithm. Record in §9 which ones applied to your published run.

| Item | Paper | This notebook (as shipped) | Why / how to match the paper |
|---|---|---|---|
| **Initialization** | "re‑trained an ESM‑2 (8M) architecture" (ambiguous) | `init_from_pretrained = True` → **warm‑start from ESM‑2 weights** | Set `False` in the Config cell for from‑scratch training of the same architecture. Warm‑start is defensible (ESM‑2's MLM ≈ a single OADM step) and converges faster, but the from‑scratch reading is arguably closer to the text and to EvoDiff. |
| **Partial‑sequence removal** | AntiRef‑90 (Briney 2023) | **Length‑window proxy** (`H_LEN_RANGE = (90,160)`, `L_LEN_RANGE = (90,130)`) + MMseqs2 90% | A length window approximates "complete V‑domain." To match the paper's corpus exactly, intersect with the real AntiRef‑90 set offline. |
| **Clustering tool** | "MMseqs2 at a 90% threshold" | `mmseqs easy-linclust --min-seq-id 0.9 -c 0.8 --cov-mode 1` (default); `easy-cluster` is present, commented | `linclust` is the linear‑time approximation chosen for ~1.68M‑scale on Colab. Uncomment `easy-cluster` if you have the compute and want the cleaner clustering. Both are "MMseqs2 @ 0.9" but yield different representatives. |
| **Masking distribution** | OADM / ARDM | `random.randint(1, D)` → `m ∈ {1..D}` (the clean ARDM form) | EvoDiff's `np.random.randint(1, D)` happens to give `m ∈ {2..D}`. This is **intentionally** the more principled form, not a bug. Only change it if you need bit‑for‑bit EvoDiff parity. |
| **Corpus scale** | ~1.68M MMseqs2 clusters | knob (`DATA_CSV`); smoke‑test fallback if unset | Provide the real CSV and raise `EPOCHS`/`max_steps`. |

Sub‑paper‑critical detail: `max_length = 300`. The length windows cap a concatenated sequence at
`160 + 1 (pipe) + 130 + 2 (cls/eos) = 293 < 300`, so nothing is truncated today. If you widen the
length windows past ~295 total, raise `max_length` too — otherwise sequences are silently
truncated, which *would* change the data and require retraining.

---

## 5. Configuration knobs (Config cell)

| Field | Default | Meaning |
|---|---|---|
| `model_name` | `facebook/esm2_t6_8M_UR50D` | ESM‑2 8M architecture |
| `init_from_pretrained` | `True` | warm‑start vs from‑scratch (see §4) |
| `pipe_token` | `\|` | heavy/light separator |
| `max_length` | `300` | max concatenated token length (see §4) |
| `reweight` | `True` | ARDM ELBO reweighting `D/(D−t+1)` |
| `batch_size` | `32` | per‑step batch |
| `lr` | `1e-4` | AdamW learning rate |
| `weight_decay` | `0.01` | AdamW weight decay |
| `max_steps` | recomputed from `EPOCHS` | total optimizer steps (training cell sets this from `EPOCHS × ⌊pairs/batch⌋`) |
| `warmup_steps` | `200` (capped) | cosine‑schedule warmup |
| `grad_clip` | `1.0` | gradient‑norm clip |
| `amp` | `True` | mixed precision on GPU |
| `gen_decoding` | `min_entropy` | `min_entropy` (paper) or `random` |
| `gen_temperature` | `1.0` | sampling temperature |
| `gen_unmask_per_step` | `1` | 1 = faithful one‑token‑per‑step; >1 = faster, less faithful |
| `seed` | `0` | RNG seed (train/val split, masking, generation) |

`EPOCHS` is set just above the training call (default `3`) and drives `max_steps`. Raise it for a
more paper‑faithful model as compute allows.

---

## 6. Training, checkpointing, resume

- Optimizer: AdamW + cosine schedule with warmup; AMP on GPU; grad‑norm clip 1.0.
- **Full training state** (model/optimizer/scheduler/scaler/step) is saved to **local disk**
  (`/content/ckpt_local/ckpt_latest.pt`) every `ckpt_every` steps, and **mirrored sparsely** to
  Drive (`checkpoints/ckpt_latest.pt`) so a disconnect costs at most a few hundred steps.
- **Auto‑resume** on re‑run, but only if the checkpoint's data/model **fingerprint** matches the
  current corpus + tokenizer + model. A checkpoint from a different run (e.g. smoke → real) is
  rejected and training starts fresh — this is intentional and safe.
- `metrics.csv` (one row per eval: step, train_loss, val_loss, lr) and `loss_curve.png` are
  written to `checkpoints/`. These are your convergence evidence and the baseline numbers the DFM
  model is compared against.

**Tokenization cache gotcha.** Tokenized tensors are cached at
`checkpoints/tok_{train,val}_<count>_<vocab>_<maxlen>.pt`. The cache key is `(count, vocab,
max_length)` — **not** content. If you change the corpus content but the pair *count* lands the
same, delete the `tok_*.pt` files so you don't silently load stale tensors. (The training
checkpoint fingerprint *is* content‑aware, so only the tokenization cache needs manual clearing.)

---

## 7. Generation & evaluation

- `generate(model, n=..., decoding=..., temperature=..., unmask_per_step=...)` → list of
  `{"heavy", "light"}` dicts. Unconditional baseline sampling.
- `inpaint(model, heavy, light, mask_spans=[("H", start, end), ...])` → redesign a residue range
  on either chain while holding the rest fixed (for the guided experiments later; use ANARCI/IMGT
  numbering to get HCDR3 bounds on real templates).
- Built‑in sanity metrics: `% unique`, `% valid (AA‑only)`, amino‑acid composition KL vs.
  training, and mean chain lengths vs. training.
- **Naturalness (paper Fig. 2b / 5)** is stubbed: install `ablang2` / `p-IgGen` and uncomment the
  optional cell to score paired sequences with both models. These packages are heavy, hence
  optional.

To reproduce the temperature sweep, vary `gen_temperature` over ~0.5 → 2.0 and re‑run generation +
metrics (inference only — no retraining).

---

## 8. Outputs and where they live

All persistent paths sit under one Google Drive project root, set in the data cell:

```
DRIVE_ROOT = /content/drive/MyDrive/2026 Spring/BMI 702/post_class_project
```

| Path | What | Persistent? | Share with team? |
|---|---|---|---|
| `DRIVE_ROOT/baseline_release/` + `baseline_release.zip` | **The shareable bundle**: model weights, tokenizer, `length_pairs.json`, `config.json.bak` | Drive | **Yes — primary artifact** |
| `DRIVE_ROOT/oas_paired_repr.csv` (+ `.done`) | Final clustered training corpus (representatives) | Drive | Yes — pins the data |
| `DRIVE_ROOT/checkpoints/` | `ckpt_latest.pt`, `metrics.csv`, `loss_curve.png`, `tok_{train,val}_*.pt` | Drive | Yes — checkpoint, metrics, tokenized split |
| `DRIVE_ROOT/extracted/` | Per‑unit `heavy,light` parquet (intermediate) | Drive | Optional (regenerable) |
| `DRIVE_ROOT/bulk_download.sh` | OAS paired download script (input) | Drive | Yes, if teammates rebuild the corpus |
| `/content/oadm_esm2_8m_baseline/` | Local copy of the release before it's pushed to Drive | **Ephemeral** | No |
| `/content/ckpt_local/` | Local checkpoint mirror + metrics | **Ephemeral** | No |
| `/content/scratch/`, `/content/mmseqs_work/`, `/content/mmseqs*` | Raw downloads + MMseqs2 temp | **Ephemeral** | No |

Ephemeral `/content/...` paths are wiped when the Colab VM disconnects — never the source of truth.

---

## 9. Run provenance — FILL THIS IN for the run you publish

The README above describes the notebook; this section describes the *specific run* whose
checkpoint you hand to the team. Fill it in so a difference between OADM and DFM is never
misattributed to data prep.

- Date / author:
- Corpus: real OAS  ☐  /  smoke test  ☐   — total pairs (train / val):
- `init_from_pretrained`: True (warm‑start) ☐ / False (from‑scratch) ☐
- Clustering: `easy-linclust` ☐ / `easy-cluster` ☐ ; `--min-seq-id` ___ , `-c` ___
- AntiRef‑90 applied literally ☐ / length‑window proxy ☐ (ranges: H ___ , L ___)
- `EPOCHS` ___ → `max_steps` ___ ; `batch_size` ___ ; `lr` ___
- Final / min val loss:
- `gen_temperature`(s) used; generation regenerated with the fixed sampler? Yes ☐ / No ☐
- `transformers` version ___ ; `torch` version ___ ; MMseqs2 version ___
- Notebook file / commit:

---

## 10. Comparison contract for the DFM model (do not break this)

When you build the DFM baseline, **reuse**, do not re‑derive:

1. The **same tokenizer** (carries the added pipe token and resized vocab — token IDs must match).
2. The **same train/val split** — load the same `oas_paired_repr.csv` *and* the same
   `tok_{train,val}_*.pt` tensors rather than re‑splitting from a seed.
3. The **same `length_pairs.json`** for length sampling at generation time.
4. The **same evaluation code** — import the metric functions (and the AbLang2 / p‑IgGen
   scoring) from a shared module; do not copy‑paste them.

Only the **training objective** and the **sampler** change between baseline and DFM. Data,
tokenizer, length distribution, and evaluation stay fixed.

---

## 11. Retraining rules of thumb

- **Sampler / generation changes** (temperature, decoding order, `unmask_per_step`,
  naturalness scoring): inference only → **no retraining**.
- **Data, objective, or tokenization changes** (corpus, clustering, masking distribution,
  `max_length` that actually truncates, adding/removing tokens): **retrain**, and clear the
  `tok_*.pt` cache.

---

## 12. Environment

- `transformers >= 4.40`, `torch >= 2.1`, `datasets`, `pandas`, `numpy`, `tqdm`.
- MMseqs2 (downloaded in Stage 2) for the real corpus.
- GPU recommended (Colab T4 is sufficient for the baseline).
- Pin exact versions for a reproducible release — ESM tokenizer behavior and the LM‑head bias
  handling are version‑sensitive. Record the versions in §9.
