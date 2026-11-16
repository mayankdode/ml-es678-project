#!/usr/bin/env bash
# FINAL STAGE: run once, after the development analysis is complete and reviewed.
# Refits the selected model on all training windows and scores the official test set (guarded against re-runs).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export HAR_STAGE=final_test
$PY src/final_evaluation.py
$PY src/error_analysis.py
$PY src/make_figures.py
$PY src/run_metadata.py
echo DONE
