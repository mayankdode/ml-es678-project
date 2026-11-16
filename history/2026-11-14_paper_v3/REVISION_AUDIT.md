# Revision audit: paper v2 → v3 (ES 678 HAR project, 2026-11-14)

Inputs:
- `HAR_Final_Revision_Brief_for_Claude-2.md`
- `LLM-Responses-on-paper-v2.txt`: three AI critiques of v2. They are AI feedback, not peer review.

Preserved copy of v2:
- `history/2026-11-14_paper_v2/` holds the paper (`paper-v2.pdf`, `.tex`, `numbers.tex`), code, config, results and the v2
  audit. That v2 audit covers the v1 → v2 changes and the reference checks made then.

Constraints on this revision:
- **The official test partition was not scored again.**
- No model, CNN or deep-learning baseline was added.
- No split, feature set or metric changed.
- The only new experiment is the bounded RF grid extension (§3), run on the training partition.

## 1. Corrections, most important first

| # | Issue (brief § / critique) | What was checked | What changed in v3 |
|---|---|---|---|
| 1 | Test-set reporting (§2; LLM-3) | Chronology rebuilt from `history/` and run metadata (§4 below). | Test score removed from the abstract. In the body it is labelled a *supplementary descriptive check on a previously accessed partition*. §6 now discloses that the selection changed from LR to SVM after the LR test scores had been seen. It also says that independence from those scores cannot be demonstrated, and that hashes document provenance but do not restore a blind holdout. Removed "the revision's changes follow reviewer requests". Added that the three test scores lie within one grouped fold SD of each other, so they do not rank the models. |
| 2 | Causal language in the overlap decomposition (§3, §12; LLM-1/2/3) | Shares recomputed from unrounded per-seed values: LR 22.8%, SVM 23.1%, RF 35.0% of the mean gap; per-seed range 18–43%. | Now called a "descriptive, protocol-based decomposition … not a causal attribution". The remainder is "associated with participant separation together with session, temporal-proximity and fold-composition effects", no longer "comes with shared participants". "Leakage through subject ID" wording avoided. |
| 3 | "SD" used for three quantities (§4; LLM-2/3) | `tables/within_seed_fold_sd.csv` and `verification.json` | *Across-seed SD* and *within-seed fold SD* are defined once in §3. Table 1 now has an **s** column and an **f** column for each protocol. Gaps are per-seed and paired, with ranges. The selected SVM's fold SD (0.025) is the tuned setting at seed 42. The 0.032 for the fixed SVM is the fold SD averaged over 10 seeds. The paper now labels each, which removes the apparent 0.025 vs 0.032 conflict. |
| 4 | Baseline explanation (§4) | Recounted from saved OOF predictions × fold assignments (fold majority below). | The paper explains F1 = 0 for five classes, accuracy equal to the laying share (19.1%), BA = 1/6, and why the segment-grouped baseline differs. |
| 5 | Gravity interpretation (§5; LLM-1/3) | Recomputed from `features.txt`: 40 tGravityAcc + 13 tGravityAccMag + 7 angle = 60; 501 remain; matches `results/metadata/gravity_features.csv`. Permutation code read: joint row shuffle of the block within each held-out fold, seed 42, 10 repeats. | Permutation results for all three models are reported; v2 omitted the SVM's (0.925 → 0.645). Removed "heavily used", "mostly transferable" and "participant-specific information". Ablation now means predictive contribution and permutation means sensitivity; the paper states that permutation creates out-of-distribution inputs. |
| 6 | RF optimum at the grid edge (§6; LLM-1) | One bounded extension (§3) | Reported. RF is now called provisional because its best setting is still at an edge of the explored grid. The grid was not expanded again. |
| 7 | Fixed comparison vs tuned selection (§6) | Configs checked in code and `config/experiment_config.json`: LR C=1; SVM C=1, γ=scale; RF 400 trees, `max_features="sqrt"`, leaf 1. | The paper says the two experiments differ and do not conflict. The SVM's 0.004 margin is called "a narrow numerical preference, not evidence of superiority". The RF uses √561 features per split, not all 561 as the brief guessed. |
| 8 | Missing related work (§7; LLM-3) | Dehghani et al. read from the arXiv PDF (§5) | Added, with its dataset described correctly. Bragança et al. re-checked against the full text: three merged datasets including UCI HAR, 59 subjects, 98% vs 85.37%, SHAP. Its description is now exact. |
| 9 | Structure and graduate-depth visibility (§8; LLM-2) | – | §4 is renamed *Results and Validation-Optimism Analysis*, with subsections *Validation-protocol comparison (RQ1–RQ2)*, *Activity-wise and feature diagnostics (RQ3)* and *Model selection and supplementary test-partition result*. |
| 10 | Dense prose (LLM-1/3) | – | Each paragraph now opens with its point, and the inline number lists are shorter. Terms are used consistently: random-window CV, segment-grouped CV, participant-grouped CV, macro-F1, across-seed SD, within-seed fold SD. |
| 11 | "gap gap" typo (§8; LLM-2) | Searched the v2 `.tex` and the PDF text: **not present** | No change needed. |
| 12 | Abstract length (§8; LLM-3) | v2 has 140 words of source text; v3 has 126 | Test score removed; overlap wording qualified. |
| 13 | AI disclosure wording (§10; LLM-2/3) | – | "Reviewers' revision instructions" is replaced by a statement that the drafts were critiqued by AI tools, which is AI feedback, not peer review. The tool names are left as a `[STUDENT: …]` field because the critique files label them only "LLM-1/2/3". The contribution statement is still left to the student, as the brief requires. LLM-2's suggested wording (that the student designed the hypotheses and audited the hashes) was **not** adopted: the repository does not show it. |
| 14 | Python version (§11; LLM-3) | `run_metadata_*.json`: 3.13.18 in every Kaggle development shard (`sys.version`); the test stage ran locally on 3.13.16; package versions identical | The paper's "Python 3.13.18" is correct for development. Nothing changed. |
| 15 | Missing verification script | `scripts/build_paper.sh` called `src/verify_audit.py`, which was not in the repository | Script written; it reproduces `results/metrics/verification.json` (§2 below). |

Figure 2 (test confusion matrix and per-class table) was removed to stay within four pages. It is supplementary under the
revised framing, and its main numbers stay in the text. The figure is still generated at
`results/figures/test_confusion_matrix.pdf`.

## 2. Verification, recomputed independently (`src/verify_audit.py` → `results/metrics/verification.json`)

The script reads only raw files and saved predictions. It fits no model and scores nothing.

**Overlap reconstruction.** Channels: body_acc, body_gyro and total_acc, each x/y/z (all nine released channels). Exact
equality, tolerance 0.
- 7,352 windows give 7,351 adjacent pairs.
- 6,752 pairs match: the last 64 samples of one window equal the first 64 of the next.
- 0 links cross a participant or activity boundary.
- Searching all pairs finds 6,752 exact matches, 0 of them non-adjacent.
- 600 segments: median 12 windows, min 1, max 29; 1 single-window segment, which forms its own group.
- Every segment lies within one participant and one activity. The segment IDs equal the saved `segments_train.csv`.
- Rows are ordered so each participant's block is contiguous: the subject ID changes at most once per subject.

**Folds**, across 10 seeds × 5 folds for each protocol:
- Every validation fold contains all six classes.
- Grouped protocols have disjoint groups.
- Validation windows with an overlapping neighbour in training: random-window 93.6%; segment-grouped and participant-grouped 0.
- Validation windows with a same-participant, same-activity window in training: segment-grouped 99.4%.

**Metrics**:
- Pooled OOF metrics recomputed from `oof_predictions_full.csv.gz` match `repeat_metrics_full.csv` to 1.1e-16.
- Gaps and decomposition shares are computed per seed before any rounding.
- Table 1 shows rounded values. The segment-gap text uses unrounded means: LR random − segment = 0.0125, which prints as 0.012, while 0.986 − 0.973 from the table gives 0.013.

**Majority baseline.** The class it predicted in each of the 50 folds per protocol:
- random-window: always LAYING;
- participant-grouped: always LAYING;
- segment-grouped: LAYING 31, STANDING 17, SITTING 1, WALKING 1.

The training shares of laying (19.1%) and standing (18.7%) are close, so the training-fold majority flips under segment grouping.

**Test (saved predictions only)**:
- Macro-F1 0.9608, accuracy 0.9613, BA 0.9599, equal to `test_metrics.json`.
- The confusion matrix sums to 2,947.
- Subjects match `subject_test.txt` and are disjoint from training.

**Leakage controls**, checked in code:
- `StandardScaler` sits inside the `Pipeline`.
- IDs are not used as features.
- Tuning uses the training partition only.
- `run_all.sh` does not call `final_evaluation.py`.
- 10 pytest tests pass.

## 3. Experiment rerun: bounded RF grid extension (training partition only)

| | |
|---|---|
| Why | The round-1 RF optimum (√d, leaf 4, 300 trees; 0.908) sat at the leaf-size edge and at the smaller feature-subset value. |
| Grid | `config.search_space_extensions.RandomForest_round2`: log2(d) × leaf {1,2,4,8,16} × trees {300,500}, plus √d × leaf {8,16} × trees {300,500}. That is 14 new settings, 26 in total. 0.5d was not extended: it scored about 0.02 below √d at every leaf size. |
| Folds | The same participant-grouped folds (`StratifiedGroupKFold`, seed 42). |
| Run | Kaggle CPU kernel, 2026-11-14, 8.2 min. Output backed up in `kaggle_runs/20261114-095735/`. Merged by `src/select_model.py`, which keeps `tuning_best_RandomForest_round1.json`. |
| Result | Best: log2(d), leaf 1, 500 trees, 0.926 (fold SD 0.030). This is still at the edge of the explored feature-subset and tree-count ranges; leaf 1 is the natural minimum. Reported as provisional and not expanded again. |
| Effect on selection | None. SVM 0.931 > LR 0.927 > RF 0.926. `selected_model.json` is byte-identical to the frozen file, and its SHA-256 matches `freeze_manifest.json`. |
| Timing | Run **after** the v2 test scoring, so it could not affect that score. The test was not rerun. |

## 4. Test-set access: reconstructed chronology

1. **Local smoke test, 2026-11-12, about 08:23.** Selected LR, C=1; test macro-F1 0.955.
   Record: `history/2026-11-12_local_smoke_run/`.
2. **Kaggle run 1 (v1 paper).** Selected LR, C=10; test macro-F1 0.952.
   Record: `history/2026-11-12_kaggle_run1/`.
3. **v2 revision.** The changes followed a written revision spec built from AI critiques of v1:
   - segment-grouped CV, 10 seeds, the gravity diagnostics and wider grids were added;
   - the first widened SVM grid peaked at a corner and was extended once;
   - the selection rule (best grouped macro-F1) was unchanged and picked the SVM;
   - the freeze manifest was written, then the test was scored once, 2026-11-12 18:45:47: SVM, 0.961.
4. **v3 (this revision).** Only the training-partition RF extension, verification and paper changes. **No test scoring.**

What the records cannot exclude:
- Steps 3–4 were carried out by an agent that had seen the earlier test scores.
- No change was chosen by a test score, but that cannot be proved after the fact. The paper now says so.

## 5. Reference verification

| Reference | Verified against | Status |
|---|---|---|
| Dehghani, Glatard & Shihab 2019, arXiv:1904.02666 | arXiv abstract page, and the full PDF text extracted locally | Verified: 17 subjects, 9 Xsens IMUs, 33 fitness activities (Baños et al. 2012 dataset), acceleration only; k-fold vs subject CV with overlapping and non-overlapping windows; k-fold inflates performance by about 10%, and by 16% with overlapping windows (abstract). arXiv lists v2 (2019-04-09), with no journal-ref and no comment. A published journal version under the same title was **not** found; a related article on overlapping windows by overlapping authors may exist but was not verified and is not cited. |
| Bragança et al. 2022, Sensors 22(6):2360 | Full-text XML (MDPI) | Re-verified: SHOAIB + WISDM + UCI-HAR merged (59 subjects), accelerometer only; k-CV 98% vs LOSO 85.37%; SHAP. |
| Rehman et al. 2024; Saeb et al. 2017; Anguita et al. 2013; Reyes-Ortiz et al. 2013; Pedregosa et al. 2011 | See §8 of the v2 audit (`history/2026-11-14_paper_v2/REVISION_AUDIT.md`) | Not re-fetched in this revision; wording unchanged. |

## 6. Changes after the test freeze

These files changed after the freeze; none of them affects any scored value:
- `src/common.py`, `src/tune_final_model.py`, `src/select_model.py` and `config/experiment_config.json` (RF round 2);
- `src/make_figures.py` (axis label "drop" → "gap"; figure height);
- `src/paper_numbers.py` (new macros only);
- `src/verify_audit.py` (new);
- `scripts/run_all.sh`, `scripts/build_paper.sh`, `kaggle_entry.py`, `README.md`.

These are unchanged since the freeze: `final_evaluation.py`, `error_analysis.py`, `selected_model.json`, `test_*`.

## 7. Still unresolved, or waiting on the student

- `[STUDENT NAME]`, `[ROLL NUMBER]` and `[REPOSITORY URL]` are still placeholders.
- `[STUDENT: name the tools and model versions]`: which AI tools wrote the v1 and v2 critiques.
- `[STUDENT: state truthfully which code, results and claims you inspected, reran, corrected or rewrote yourself.]`
- RF tuning is provisional (edge of the explored grid).
- Tuning uses one grouped split (seed 42); there is no nested CV and no LOSO.
- Segment grouping does not remove within-session or temporal dependence. The residual is not attributed to participant identity alone.
- The gravity diagnostics are not causal, and no participant-ID probe was run.
- The final test ran on local Python 3.13.16, while development ran on Kaggle 3.13.18. Library versions were identical.
- The bibliography is set in `\footnotesize` to fit four pages. The template `.sty` is unmodified.
