"""Record software versions, seeds, hardware and timings for one stage (HAR_STAGE) of the pipeline."""
import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version

from common import CONFIG, SMOKE, STAGE, out_dir, save_json

PKGS = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "seaborn", "joblib"]


def main():
    timings = out_dir("logs") / f"timings_{STAGE}.json"
    save_json({"stage": STAGE, "finished_utc": datetime.now(timezone.utc).isoformat(), "smoke_mode": SMOKE,
               "python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
               "packages": {p: version(p) for p in PKGS},
               "seeds": {"model_and_tuning_split": CONFIG["seed"], "repeated_cv_split_seeds": CONFIG["repeat_seeds"]},
               "config": CONFIG, "timings_s": json.loads(timings.read_text()) if timings.exists() else {}},
              out_dir("logs") / f"run_metadata_{STAGE}.json")


if __name__ == "__main__":
    main()
