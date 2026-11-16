# Viva preparation notes

Be ready to explain each point in your own words. The numbers come from `report/numbers.tex` and `results/`.

**Why random-window CV is wrong for claims about new users.**
- Every participant appears in both training and validation folds.
- 93.6% of validation windows also have an overlapping neighbour in training, sharing half their raw samples.
- The score therefore measures recognition of *seen* people, often on *seen* samples.

**StratifiedKFold vs StratifiedGroupKFold.**
- `StratifiedKFold` balances class shares across folds and ignores groups.
- `StratifiedGroupKFold` keeps every group (a participant or a segment) inside one fold while approximately balancing classes.
- `src/splits.py` checks every fold: groups are disjoint and all six classes are present.

**How segments are built (`src/sequence.py`).**
- Windows are 128 samples with a 64-sample step.
- If row i+1 continues row i, then the last 64 samples of i equal the first 64 of i+1, exactly, on all nine raw channels.
- 6,752 of 7,351 adjacent pairs match. No match crosses a participant or activity, and no non-adjacent rows match.
- Maximal chains of matches are segments: 600 in total, median 12 windows.

**Why segment grouping does not remove all dependence.**
- It only separates windows that share raw samples.
- 99.4% of validation windows still have training windows from the same person and activity, usually from the same session.
- So the segment → participant difference mixes person, session, temporal-proximity and fold-composition effects.
- The 23–35% figure is a descriptive protocol comparison, not a causal share.

**Why the scaler is fitted inside the pipeline.**
- Fitting `StandardScaler` on all data would leak validation-fold means and SDs into training.
- Inside the `Pipeline`, it is refitted on each training fold only.

**Metrics.**
- Macro-F1: the unweighted mean of the six per-class F1 scores. Each activity counts equally.
- Accuracy: the share of windows that are correct.
- Balanced accuracy: the mean per-class recall.
- Example: the majority baseline gets accuracy 19.1% (the laying share) but BA 1/6 and macro-F1 0.054.

**Two SDs.**
- *Across-seed SD*: how much the pooled score changes when the same 21 people are re-split (≤ 0.004).
- *Within-seed fold SD*: how much scores differ between held-out groups of 4–5 people (participant-grouped 0.032–0.038).
- Neither is a confidence interval: same 21 people, overlapping training sets.

**Gravity ablation vs permutation.**
- Ablation retrains without the 60 features. Grouped macro-F1 falls, so the features are predictive.
- Permutation shuffles the block in held-out folds. Scores collapse (to about 0.56–0.65), so the fitted models are sensitive to the block.
- Permutation breaks correlations with the other features and creates unrealistic inputs, so it is not a causal importance measure.
- Neither result shows that the features encode identity or body geometry.

**Why the test score is only supplementary.**
- The test partition was scored three times (0.955, 0.952, 0.961).
- The selected model changed (LR → SVM) after the earlier scores were seen.
- Hashes prove which code produced a score; they cannot make the partition unseen again.
- All conclusions rest on the training-partition CV.

**Why SVM vs LR is not decisive.**
- The tuned margin is 0.004, against a fold SD of about 0.025–0.037, on one split of the same 21 people.
- Among the fixed configurations, LR led in all 10 seeds.
- These are different experiments, not a contradiction.

**RF grid edge.**
- The first RF grid peaked at leaf 4, an edge of the grid. One bounded extension was run (log2 d features; leaves up to 16).
- The new best setting, 0.926, still lies on an edge, so RF tuning is reported as provisional.
- The grid was not expanded repeatedly, to avoid searching until a favourable score appears.
