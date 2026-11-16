"""Kaggle CPU kernel entry: runs part of the development analysis (scripts/run_all.sh) as one shard.

Usage: kaggle_entry.py <stage> [--smoke], where <stage> is a comma list of run_all.sh parts.
Data go to /tmp, results to /kaggle/working/results. The official test set is never scored here.
"""
import os
import subprocess
import sys

stage = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "validation,ablation,diagnostic,tune_fast,tune_rf,tune_rf2,select,figures"
env = dict(os.environ, HAR_DATA_DIR="/tmp/har_data", HAR_RESULTS_DIR="/kaggle/working/results", PYTHON=sys.executable,
           HAR_PARTS=stage, HAR_STAGE=stage.replace(",", "+"))
if "--smoke" in sys.argv:
    env["HAR_SMOKE"] = "1"
sys.exit(subprocess.run(["bash", "scripts/run_all.sh"], env=env).returncode)
