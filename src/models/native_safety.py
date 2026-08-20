from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class DependencyProbe:
    safe: bool
    detail: str


PROBES = {
    "xgboost": "from xgboost import XGBRegressor as M; M(n_estimators=2,n_jobs=1).fit([[0],[1],[2]],[0,1,2])",
    # A three-row smoke test passed on a Windows build that later raised a
    # native access violation on the platform's mixed-feature design matrix.
    # Exercise a representative preprocessing workload in the disposable
    # process so the unsafe binary is excluded before main-process training.
    "lightgbm": (
        "from src.benchmark.generators import generate_dataset; "
        "from src.data.cleaner import build_preprocessor; "
        "from src.utils.schema import suggest_features; "
        "from lightgbm import LGBMRegressor as M; "
        "d=generate_dataset('Apartments',rows=200,seed=7); "
        "f,_=suggest_features(d.frame,d.target); "
        "f=[c for c in f if c!='transaction_date']; "
        "x=d.frame[f]; z=build_preprocessor(x).fit_transform(x); "
        "M(n_estimators=20,verbosity=-1,n_jobs=1).fit(z,d.frame[d.target])"
    ),
    "catboost": "from catboost import CatBoostRegressor as M; M(iterations=2,verbose=False,allow_writing_files=False).fit([[0],[1],[2]],[0,1,2])",
}


@lru_cache(maxsize=None)
def probe_dependency(package: str, timeout_seconds: int = 20) -> DependencyProbe:
    """Exercise native import and fit in a disposable process so a crash cannot kill the app."""
    script = PROBES.get(package)
    if not script:
        return DependencyProbe(True, "No native preflight is required.")
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True,
            timeout=timeout_seconds, check=False,
        )
    except subprocess.TimeoutExpired:
        return DependencyProbe(False, f"Native dependency preflight exceeded {timeout_seconds}s.")
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or f"process exit code {completed.returncode}").strip().splitlines()[-1]
        return DependencyProbe(False, f"Native dependency failed isolated preflight: {detail[:300]}")
    return DependencyProbe(True, "Native import and minimal fit passed in an isolated process.")
