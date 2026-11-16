# Revision audit (ES 678 HAR project, revision of 2026-11-12)

This file records what changed in response to `HAR_Consolidated_Revision_Instructions_for_Claude.md`: what was rerun,
what could not be verified, and what is still open. All times are UTC. The paper is `report/paper.pdf` (4 pages
including references). The v1 paper, its numbers and its results are kept under `history/2026-11-12_kaggle_run1/`.

## 1. Review points addressed

| Spec § | Point | What was done | Where |
|---|---|---|---|
| 3.1 | Dataset and alignment checks | Row counts, label map, ID alignment, train/test subject disjointness, NaN/inf, constant features, duplicate rows and duplicate names (42 names, 126 columns; prefixed with the column index, order unchanged), no ID column in `X`. The `[-1,1]` bound was checked against the data (all 561 features, train) and the dataset README (line 57). Per-feature SD is 0.041–0.752. Class shares are computed from the files. | `src/data_audit.py`, `results/metrics/data_audit.json` |
| 3.2 | Leakage controls | Checked in code: `StandardScaler` sits inside a `Pipeline` for LR and SVM; RF gets features only; every protocol uses the 7,352 training rows only; grouped folds are asserted participant-disjoint with all six classes present; segment folds are asserted to have no overlap neighbour in training; tuning uses grouped folds on the training partition. The test set is touched only by `src/final_evaluation.py`. | `src/repeated_validation.py` (`build_folds`), `tests/` (10 tests pass) |
| 3.3 | Model details | LR: lbfgs, L2, C=1 fixed, max_iter 3000, no class weights. SVC: rbf, C=1, gamma='scale' fixed, `decision_function_shape='ovr'` (the default). libsvm trains 15 one-vs-one classifiers and predicts by voting; 'ovr' only reshapes the decision scores, and the paper now says so. probability=False. RF: 400 trees, sqrt features, leaf 1, random_state 42, no class weights. Dummy: `most_frequent`. | `config/experiment_config.json` |
| 3.3 | Standardisation wording | The paper now says features share a range but not a scale (SD 0.04–0.75), and that the scaler is fitted inside each training fold. | paper §2 |
| 3.4 | Environment | Python 3.13.18 is confirmed from `sys.version` in every Kaggle development shard. The final-test stage ran locally on Python 3.13.16 with identical numpy 2.1.3, pandas 2.3.3, scipy 1.16.3, scikit-learn 1.6.1, matplotlib 3.10.0 and joblib 1.6.0. | `results/logs/run_metadata_*.json` |
| 3.5 | Arithmetic audit | All paper numbers are LaTeX macros written by `src/paper_numbers.py` from saved outputs. Independent recomputation: the pooled OOF macro-F1 recomputed from `oof_predictions_full.csv.gz` matches the saved repeat metrics to 3e-16 for all 120 seed×protocol×model cells. Test macro-F1/accuracy/BA recomputed from `test_predictions.csv` give 0.9608 / 0.9613 / 0.9599, with 2,833 of 2,947 correct; the matrix sums to 2,947. v1's matrix also sums to 2,947, with 2,808 correct (0.953), so v1's number was internally correct. | `report/numbers.tex` |
| 4 | Test-set access | See §4 below. | paper §6 |
| 5.1 | Repeated validation | Split seeds 42–51, 5 folds, pooled OOF predictions per seed; mean and SD across seeds, labelled split-seed variability (not a CI); no p-values; mean within-seed fold SD reported too. | paper §3–4, `results/tables/repeated_summary.csv` |
| 5.2 | Fixed vs tuned kept apart | Table 1 and the protocol analysis use fixed configurations only; tuning and test results are in a separate paragraph and figure. | paper §4 |
| 5.3 | LOSO | Not run (optional). Listed as a limitation. | paper §5 |
| 5.4 | Cautious model claims | The ranking-win counts and the RF-gap ordering are reported with the caveat that the seeds reuse the same 21 people. The selection margin (0.004) is compared with the fold SD (0.025). | paper §4 |
| 6.1 | Robustness of the gap | The gap is positive in 10/10 seeds for all three models; seed SD and range are reported. | paper §4 |
| 6.2 | Test the RF/person-specific explanation | Gravity-family ablation (60 features) over all 10 seeds, plus permutation of the gravity family within held-out folds (seed 42, 10 permutations). The result is mixed and is reported as such (see §5). | paper §4, Fig. 1 |
| 6.3 | Overlap audit | Feasible; see §3 below. | paper §3 |
| 7.1 | Metrics | Macro-F1, accuracy and balanced accuracy for every model × protocol (Table 1) and for the test set. The paper explains why the test metrics are close but not identical (class shares 14–18%). | |
| 7.2 | Majority baseline | Accuracy 0.191 is the actual LAYING share of training; BA is 1/6; macro-F1 is 0.054. Under segment folds the majority class changes between folds, so its scores differ. | Table 1 |
| 7.3 | Figures | The clipped "Downstairs" label is fixed. The confusion matrix is redrawn at print size with readable fonts. Every page was inspected at 80 dpi. The new ablation figure is colour plus marker shape, with a legend. The per-seed protocol figure is in `results/figures/repeated_protocols.pdf` but not in the paper, for space; Table 1 carries the same means and SDs. | |
| 8 | Related work | Saeb 2017, Bragança 2022, Rehman 2024, Anguita 2013 and Reyes-Ortiz 2013 (UCI record) are cited and verified (§8 below). The paper frames itself as an audit, not a new phenomenon. | paper §1 |
| 9 | Graduate prose; section naming | Rewritten. Research questions RQ1–3 are explicit. The depth analysis sits inside Results under question-style headings; no section is called "Graduate Depth Analysis". | |
| 10 | AI use | The disclosure names the tool and what it did. The `[STUDENT: …]` statement is kept for the student to complete. | paper §6 |
| 11 | ≤ 4 pages, template | 4 pages including references; `neurips_2026.sty` is unmodified (see §7 on single vs two-column). | |
| 12 | Deliverables | paper.pdf/.tex, code, README with measured runtime, CSV/JSON outputs, figures, this file. | |

## 2. New experiments (all on the official training partition only)

| Experiment | Configuration | Seeds | Run |
|---|---|---|---|
| Repeated CV, all features | Dummy, LR, SVM, RF (fixed configs) × random-window / segment-grouped / participant-grouped, 5 folds | split seeds 42–51; model seed 42 | Kaggle `har-gen-validation`, 57 min |
| Repeated CV, no gravity | LR, SVM, RF on the 501 non-gravity features, same folds | 42–51 | Kaggle `har-gen-ablation`, 64 min |
| Gravity permutation | full-feature LR/SVM/RF; gravity columns permuted jointly within each validation fold; random-window and participant-grouped | folds seed 42; 10 permutations, rng seed 42 | Kaggle `har-gen-validation`, 5.5 min |
| Tuning (grouped CV, seed 42) | LR C {0.01, 0.1, 1, 10, 100}; SVM C {0.1, 1, 10, 100, 1000} × gamma {scale, 1e-4, 5e-4, 0.005, 0.01}; RF trees {300, 500} × max_features {sqrt, 0.5} × leaf {1, 2, 4} | 42 | Kaggle `har-gen-validation` (round 1), `har-gen-tune-svm2` (round 2), `har-gen-tune-rf` (2.2 h) |
| Final test | SVM C=100, gamma=5e-4, refitted on 7,352 windows | — | local, once, 2026-11-12 18:45:47 |

**SVM grid extension.** The first SVM grid (C up to 100; gamma 5e-4 to 0.01 plus scale) had its optimum at a corner
(C=100, gamma=5e-4). Following spec §5.2, the grid was extended one step in each direction, using training data only.
The optimum stayed at C=100, gamma=5e-4, now in the interior; its grid neighbours score 0.926–0.930. The round-1 files are in
`history/2026-11-12_svm_tuning_round1/`.

**RF boundary.** RF's optimum (300 trees, leaf 4) sits at the grid edge, but its neighbours are within 0.002, and RF
trails the SVM by 0.024, so the grid was not extended (that would cost about 2 h more). This is disclosed in the paper.

**Selection.** The rule was fixed before any run: the best grouped-CV macro-F1 wins. It picked the SVM (0.931 ± 0.025)
over LR (0.927) and RF (0.908). The margin, 0.004, is far below the fold SD.

## 3. Overlap audit: feasible

Using `train/Inertial Signals` (9 channels × 128 samples), samples 65–128 of a row exactly equal (tolerance 0) samples
1–64 of the next row for 6,752 of the 7,351 consecutive pairs.
- All 6,752 links stay within one participant and one activity.
- No row matches a non-adjacent row.
- Row order therefore preserves recording order.
- The links form 600 segments: median 12 windows, max 29; 4–8 segments per participant-activity.

Segment-grouped 5-fold CV keeps every segment in one fold. Measured over all folds:
- **Segment-grouped folds:** no validation window has an overlapping neighbour in training. 99.4% still have training windows from the same participant and activity.
- **Random-window folds:** 93.6% of validation windows have an overlapping neighbour in training.

**Result.** Separating overlap alone removes 0.012 (LR), 0.013 (SVM) and 0.025 (RF) of the macro-F1 gap. That is 23–35% of the
random − participant gap; the rest comes with shared participants. Outputs: `results/metadata/segments_train.csv`,
`results/metrics/overlap_audit.json`, `results/tables/fold_diagnostics_summary.csv`.

## 4. Test-set access: evidence and disclosure

Established from the session transcript and saved files:
1. **Local smoke test, 2026-11-12 about 08:23** (`HAR_SMOKE=1`; 20-tree RF, one-point grids). It selected LR, C=1, and scored
   test macro-F1 0.9550 (accuracy 0.9552, BA 0.9538). The agent saw this output and the confusion figure. The full run
   was pushed about 10 s later. The only additions in between were `requirements.txt` and `.gitignore`; code, grids and
   models were unchanged. Files: `history/2026-11-12_local_smoke_run/`.
2. **Kaggle run 1 (v1 paper).** It selected LR, C=10, and scored test macro-F1 0.9522 (accuracy 0.9528, 2,808/2,947 correct).
   Files: `history/2026-11-12_kaggle_run1/`.
3. **This revision.** Code and selection were frozen at 18:45:39 (`results/metrics/freeze_manifest.json`, SHA-256 of
   `src/*.py`, the config and `selected_model.json`). One scoring followed at 18:45:47: SVM, test macro-F1 0.9608. The
   guard in `final_evaluation.py` refuses re-runs.

What this cannot exclude: the agent that designed the revision had seen the earlier test scores. None of the revision's
changes was chosen by a test score. Every change follows a written item in the revision spec, and the selection rule
did not change. Even so, the holdout is not blind to the research process, and the paper says so.

**Exact wording in the paper (§6):**
> The official test partition has been scored three times. A local pipeline smoke test scored LR, C=1 (macro-F1
> 0.955); the first full run scored LR, C=10 (0.952); this revision scored the SVM above once. Code, grids and models
> did not change between the first two; the revision's changes follow reviewer requests, not test scores. As earlier
> scores were seen, the holdout is not fully unseen, so we report it as a supplementary check.

## 5. Results that changed from v1

| Item | v1 (single 5-fold split) | Revision (10 split seeds) |
|---|---|---|
| Gap, macro-F1, LR / SVM / RF | 0.059 / 0.055 / 0.077 | 0.055 / 0.056 / 0.072 (positive in 10/10 seeds for each) |
| Overlap vs participant share | not separated | overlap 23–35% of the gap |
| Selected model | LR, C=10 (grouped CV 0.927) | SVM, C=100, gamma=5e-4 (0.931); LR is 0.004 behind |
| Test macro-F1 / accuracy | 0.952 / 0.953 | 0.961 / 0.961 (BA 0.960) |
| Tuning grids | LR C {0.1, 1, 10}; SVM C {0.1, 1, 10} × gamma {scale, 0.01} | wider grids (§2); the SVM grid was extended once |
| RF explanation | "RF uses person-specific patterns more" (untested) | Tested with the gravity ablation and permutation. Without gravity features, grouped macro-F1 drops for every model, but the overall gap does not shrink (LR 0.055→0.056, RF 0.072→0.067, SVM 0.056→0.069). The sit/stand gaps shrink for LR and RF, mainly because random-window scores fall more. The laying gap grows for all models. So the person-specific hypothesis is only partly supported, and only for LR and RF. |

The test-score change comes from a different selected model, after a wider grid and an unchanged rule; it is not
a re-run of the same model.

## 6. Changes after the freeze (none affect scored values)

- `src/make_figures.py`: the confusion-matrix plot was resized and its fonts adjusted for print legibility, then redrawn
  from the saved `test_confusion_counts.csv`. The test set was not scored again.
- `src/paper_numbers.py`: added macros (RF boundary flag, test class-share range). It only reads saved files.
- `report/paper.tex`, `README.md` and this file were edited.

`freeze_manifest.json` therefore differs from the current `make_figures.py` and `paper_numbers.py` only.

## 7. Unresolved items

- **Layout.** The spec asks for a "conference-style two-column layout", but the official ES 678 starter
  (`neurips_2026.sty`, from es678_project_template.zip) is single-column. The course template was kept unmodified.
- **Tuning on one split.** Tuning used one grouped split (seed 42) of the same 21 people used for the protocol comparison; there is no nested selection.
- **LOSO.** Not run.
- **Participant-ID prediction probe.** Not run. The person-specific explanation rests on the ablation and permutation only.
- **Segment grouping limit.** It removes shared samples but not the correlation between nearby non-overlapping windows from the same recording.
- **Python patch version.** The final test ran on Python 3.13.16 locally; development ran on 3.13.18 on Kaggle. All libraries were identical.
- **Student fields.** `[STUDENT NAME]`, `[ROLL NUMBER]`, `[REPOSITORY URL]` and the `[STUDENT: …]` AI-use statement must be filled in by the student, who must also review and be able to defend the work.

## 8. Reference verification

| Reference | Verified against |
|---|---|
| Saeb et al. 2017, GigaScience 6(5):gix019, doi:10.1093/gigascience/gix019 | Crossref record (authors, title, volume, issue, year); no page range is listed, so the paper cites the article ID |
| Bragança et al. 2022, Sensors 22(6):2360, doi:10.3390/s22062360 | MDPI article page: k-CV vs LOSO on SHOAIB, WISDM, UCI-HAR; about 13-point overestimation; SHAP |
| Rehman et al. 2024, Algorithms 17(12):556, doi:10.3390/a17120556 | full text via Swansea Cronfa repository (MDPI page returned 403): PAMAP2, RF 89% k-fold vs 76% LOSO |
| Anguita et al. 2013, ESANN, pp. 437–442 | citation requested in the dataset README; page range checked on the ESANN proceedings PDF (es2013-84, printed pages 437–442) |
| Reyes-Ortiz et al. 2013, UCI repository, doi:10.24432/C54S4K | UCI dataset page |
| Pedregosa et al. 2011, JMLR 12:2825–2830 | standard scikit-learn citation (JMLR has no DOI and was not re-checked online) |
