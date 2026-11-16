#!/usr/bin/env bash
# Development analysis (official TRAINING partition only). Never scores the official test set.
# Env: HAR_DATA_DIR, HAR_RESULTS_DIR, HAR_SMOKE=1 (tiny grids, 2 seeds), HAR_JOBS, HAR_STAGE,
#      HAR_PARTS (subset of: validation,ablation,diagnostic,tune_fast,tune_rf,select,figures; default all).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
PARTS=",${HAR_PARTS:-validation,ablation,diagnostic,tune_fast,tune_rf,select,figures},"
has() { [[ "$PARTS" == *",$1,"* ]]; }
mkdir -p "${HAR_RESULTS_DIR:-results}/logs"
$PY src/download_data.py
$PY -m pytest -q tests
$PY src/data_audit.py
$PY src/sequence.py
FS=""; has validation && FS="full"; has ablation && FS="${FS:+$FS,}no_gravity"
[[ -n "$FS" ]] && HAR_FEATURESETS=$FS $PY src/repeated_validation.py
has diagnostic && $PY src/gravity_diagnostic.py
has tune_fast && HAR_TUNE_FAMILIES=LogReg,SVM_RBF $PY src/tune_final_model.py
has tune_rf && HAR_TUNE_FAMILIES=RandomForest $PY src/tune_final_model.py
has select && $PY src/summarize_repeated.py && $PY src/select_model.py
has figures && $PY src/make_figures.py
$PY src/run_metadata.py
echo DONE
