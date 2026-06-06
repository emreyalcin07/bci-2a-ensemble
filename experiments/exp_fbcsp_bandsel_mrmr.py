"""
exp_fbcsp_bandsel_mrmr.py
=========================

Faz 2 — Varyant 2:
    FBCSP (16 bant) -> TRAIN-ONLY band selection (top_k MI) ->
    StandardScaler -> mRMR(n_features) -> SVC(linear, OVR, probability)

mRMR (MID variant):
    relevance(f) = MI(f, y)
    redundancy(f, S) = avg_{s in S} MI(f, s)
    score(f) = relevance - redundancy

Iç CV grid:
    select_bands__top_k        : [4, 6, 8, 10]
    select_feats__n_features   : [20, 40, 60]
    svc__C                     : [0.01, 0.1, 1, 10]

CSP n_components: 4 (sabit, cache'lenir).

Leakage: tüm step'ler Pipeline içinde, train-only fit.
A0XE'ye DOKUNULMAZ.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import List

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _p in (str(PROJECT_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joblib import Memory  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from _fbcsp_common import (  # noqa: E402
    AllBandCSP,
    MRMRSelector,
    RANDOM_STATE,
    SubjectRunResult,
    TopKBandSelector,
    generate_bands,
    report_and_save,
    run_subject,
)

# --------------------------------------------------------------------------- #
# Pipeline                                                                    #
# --------------------------------------------------------------------------- #

_MEMORY = Memory(location=tempfile.mkdtemp(prefix="fbcsp_mrmr_cache_"), verbose=0)


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("csp_all", AllBandCSP(n_components=4, reg="ledoit_wolf")),
            ("select_bands", TopKBandSelector(top_k=8)),
            ("scaler", StandardScaler()),
            ("select_feats", MRMRSelector(n_features=20)),
            (
                "svc",
                SVC(
                    kernel="linear",
                    C=1.0,
                    probability=True,
                    decision_function_shape="ovr",
                    random_state=RANDOM_STATE,
                ),
            ),
        ],
        memory=_MEMORY,
    )


PARAM_GRID = {
    "select_bands__top_k": [4, 6, 8, 10],
    "select_feats__n_features": [20, 40, 60],
    "svc__C": [0.01, 0.1, 1.0, 10.0],
}


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description="FBCSP + band-sel + mRMR + linSVM.")
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--output", type=str, default="exp_fbcsp_bandsel_mrmr.csv")
    args = parser.parse_args()

    bands = generate_bands()
    print(f"Filter bank: {len(bands)} bands")
    for l, h in bands:
        print(f"  {l:>5.1f} - {h:>5.1f} Hz")
    print()

    rows: List[SubjectRunResult] = []
    t_total = time.time()
    try:
        for sid in args.subjects:
            res = run_subject(
                subject_id=sid,
                pipeline_factory=build_pipeline,
                param_grid=PARAM_GRID,
                bands=bands,
                outer_splits=args.outer_splits,
                inner_splits=args.inner_splits,
                n_jobs=args.n_jobs,
                verbose=True,
            )
            rows.append(res)
        total_elapsed = time.time() - t_total

        report_and_save(
            rows=rows,
            output_name=args.output,
            total_elapsed=total_elapsed,
            title="FBCSP + band-sel + mRMR + linSVM(OVR) - nested CV (5x5)",
        )
    finally:
        try:
            shutil.rmtree(_MEMORY.location, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
