#!/usr/bin/env bash
# Regenerate paper numbers/figures from results/ and compile report/paper.pdf (needs pdflatex).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
$PY src/make_figures.py
$PY src/verify_audit.py >/dev/null   # independent re-computation -> results/metrics/verification.json
$PY src/paper_numbers.py
rm -f report/figures/*.pdf
cp results/figures/per_class_drop_ablation.pdf results/figures/test_confusion_matrix.pdf report/figures/
cd report
pdflatex -interaction=nonstopmode -halt-on-error paper.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error paper.tex >/dev/null
rm -f paper.aux paper.log paper.out
pdfinfo paper.pdf | grep Pages
