# How Validation Strategy Affects Generalization in Smartphone-Based HAR (ES 678)

This project measures how optimistic **random window-level CV** is compared with **participant-grouped CV** on the official
UCI HAR training partition (21 participants). An intermediate **segment-grouped CV** keeps exactly reconstructed runs of
50%-overlapping windows together. Every protocol is repeated over 10 split seeds. The models are a majority-class baseline,
logistic regression, an RBF SVM and a random forest, each with a fixed configuration. A gravity-feature ablation and a held-out
permutation check are supporting diagnostics. The final model is chosen with participant-grouped tuning only. A **separate, guarded
final stage** scores it on the official test set. That partition had already been scored in earlier iterations, so its result is
only a supplementary descriptive check (see `REVISION_AUDIT.md`).

## Dataset and attribution
Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013). *Human Activity Recognition Using
Smartphones*. UCI Machine Learning Repository. https://doi.org/10.24432/C54S4K. Licensed **CC BY 4.0**.
The original paper is Anguita et al., ESANN 2013, pp. 437–442.

`src/download_data.py` downloads the official UCI archive into `data/raw/UCI HAR Dataset/` and records the URL, timestamp and
SHA-256 in `data/raw/provenance.json`. The data is not committed.

## Setup and reproduction
```bash
python -m venv .venv && . .venv/bin/activate   # canonical runs: Python 3.13.18 on Kaggle
pip install -r requirements.txt                 # pinned to the Kaggle image (scikit-learn 1.6.1, numpy 2.1.3, pandas 2.3.3)
bash scripts/run_all.sh                         # 1) development analysis, training partition only
bash scripts/run_final_test.sh                  # 2) FINAL STAGE: run once, after reviewing step 1
bash scripts/build_paper.sh                     # 3) figures + verification + report/numbers.tex + report/paper.pdf
```
`run_all.sh` never scores the test set. `run_final_test.sh` writes `results/metrics/freeze_manifest.json`, which holds the
SHA-256 of every source file, the config and `selected_model.json`, before scoring. It refuses to run again once
`test_metrics.json` exists, unless you set `HAR_ALLOW_TEST_RERUN=1`. Because this repository already contains
`results/metrics/test_metrics.json`, reproducing the final stage from a fresh clone needs
`HAR_ALLOW_TEST_RERUN=1 bash scripts/run_final_test.sh`.

| Variable | Default | Purpose |
|---|---|---|
| `HAR_DATA_DIR` | `data/raw` | where the dataset is extracted |
| `HAR_RESULTS_DIR` | `results` | output folder |
| `HAR_JOBS` | `-1` | parallel workers (use 1–2 on small machines) |
| `HAR_PARTS` | all | subset of `validation,ablation,diagnostic,tune_fast,tune_rf,tune_rf2,select,figures` (for sharding) |
| `HAR_STAGE` | `dev` | names the timing/metadata files of a shard |
| `HAR_SMOKE` | `0` | `1` = 2 seeds, 20-tree forest, one-point grids; for pipeline checks only |

**Measured runtime** on a 4-vCPU Kaggle CPU kernel (from `results/logs/timings_*.json`):
- repeated validation, all features (4 models × 3 protocols × 10 seeds × 5 folds): 57 min
- repeated validation, no-gravity features (3 models): 64 min
- gravity permutation check: 5.5 min
- LR tuning: under 1 min; SVM tuning (25 settings): 6 min
- RF tuning, round 1 (12 settings): 2.2 h; bounded round 2 (14 settings, `tune_rf2`): 8 min
- summaries, selection and figures: seconds (run locally)
- final test stage: about 12 s on a 2-core laptop-class machine

Running everything in one kernel takes about 4.5 h; sharding brings the wall-clock time down to about 2.2 h.

**Note on Kaggle.** Kaggle is not needed for reproduction: the steps above run everything on a local machine.
For the record, the canonical development run was done on Kaggle CPU kernels only to get extra compute time. It was
split into five shards (`kaggle_entry.py <parts>`):
1. `validation,diagnostic,tune_fast`
2. `ablation`
3. `tune_rf`
4. `tune_fast` again, after the SVM grid was extended (see `REVISION_AUDIT.md`)
5. `tune_rf2`, the bounded RF grid extension (after the paper-v2 test run; training partition only)

Their outputs were merged into `results/`, and `summarize_repeated.py`, `select_model.py` and `make_figures.py`
were then run locally.

## Protocol
| Step | Script | Notes |
|---|---|---|
| Data audit | `src/data_audit.py` | counts, alignment, subject disjointness, NaN/inf, constant/duplicate checks, feature scale, majority share |
| Overlap audit | `src/sequence.py` | exact 64-sample overlap between consecutive raw windows → segments |
| Folds | `src/splits.py` | random-window `StratifiedKFold`; segment- and participant-grouped `StratifiedGroupKFold`; 5 folds |
| Repeated CV | `src/repeated_validation.py` | fixed configs, seeds 42–51, OOF predictions, full and no-gravity feature sets |
| Permutation check | `src/gravity_diagnostic.py` | gravity family permuted within held-out folds (seed 42) |
| Summaries | `src/summarize_repeated.py` | gaps, per-class drops, ablation effect, ranking stability |
| Tuning | `src/tune_final_model.py`, `src/select_model.py` | grouped CV (seed 42), macro-F1, grid-boundary flags; RF round 2 merged with round 1 |
| Final test | `src/final_evaluation.py`, `src/error_analysis.py` | guarded; refit on 7,352 windows, score 2,947 |
| Verification | `src/verify_audit.py` | independent recomputation from raw files and saved predictions → `metrics/verification.json`; fits no model, scores nothing |
| Figures/numbers | `src/make_figures.py`, `src/paper_numbers.py` | built only from saved outputs |

Leakage controls:
- Scalers are fitted inside each training fold through a `Pipeline`.
- The participant ID is used only as a group label, never as a feature.
- Tuning uses the training partition only.
- Each fold is checked for disjoint groups and for the presence of all six classes.

Two kinds of spread are reported, and neither is a confidence interval:
- the *across-seed SD*: the SD over the 10 seeds of the pooled out-of-fold score;
- the *within-seed fold SD*: the SD over a seed's 5 validation folds, averaged over seeds (`tables/within_seed_fold_sd.csv`).

## Outputs (`results/`)
- `repeated/`: fold and repeat metrics, OOF predictions (`.csv.gz`), fold assignments, fold diagnostics, gravity permutation
- `metrics/`: `data_audit.json`, `overlap_audit.json`, `tuning_*`, `selected_model.json`, `freeze_manifest.json`,
  `test_metrics.json`, `test_predictions.csv`, confusion matrices, `error_analysis.json`, `repeated_highlights.json`
- `tables/`: repeated summary, gaps, per-class drops, gravity ablation, ranking wins, fold diagnostics, tuning summary, test tables
- `figures/`: PDF and PNG figures
- `logs/`: `run_metadata_<stage>.json` (Python, package versions, CPU, seeds, config, timings) and Kaggle logs

`history/` keeps:
- the earlier runs that scored the test set (the local smoke test, and the first Kaggle run with its v1 paper), which are disclosed in the paper;
- `2026-11-14_paper_v2/` and `2026-11-14_paper_v3/`, full copies of papers v2 and v3 and their project state.

 `history/2026-11-12_svm_tuning_round1/` keeps the first SVM grid, whose optimum sat on a
corner of the grid.

## Paper
`report/paper.tex` (built to `report/paper.pdf`; the submission copy is `report/paper_vfinal.pdf`) uses the official ES 678 starter (`neurips_2026.sty`, unmodified; letter size, ≤ 4 pages).
`VIVA_NOTES.md` lists points to prepare.

## Tests
`pytest tests/` checks:
- shapes and row alignment
- the label set
- that train and test subjects are disjoint
- that no identifier is used as a feature
- that grouped folds are participant-disjoint and contain every class (3 seeds)
- the overlap reconstruction
- that segment folds keep overlapping windows together
- the size of the gravity feature family

## Limitations
- 30 participants, 9 of them in the test set.
- Laboratory setting, one waist-worn phone, six scripted activities.
- Engineered features only.
- Measures classification performance only; this is not a medical tool.
