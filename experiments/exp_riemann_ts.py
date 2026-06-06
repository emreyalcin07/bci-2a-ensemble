"""
exp_riemann_ts.py
=================

Faz 3 — Varyant 1: Tangent Space + Logistic Regression.

Pipeline:
    Covariances(estimator='oas')   # küçük örneklem icin shrinkage shrunk-cov
    -> TangentSpace(metric='riemann')
    -> StandardScaler
    -> LogisticRegression(max_iter=1000, multi_class='auto')   # softmax

Iç CV grid:
    lr__C : [0.01, 0.1, 1.0, 10.0]

Bant: tek geniş bant 8-30 Hz (data-independent, Pipeline dışında).

Leakage: Covariances, TangentSpace, StandardScaler, LR → hepsi train-only fit.
A0XE'ye DOKUNULMAZ.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _p in (str(PROJECT_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pyriemann.estimation import Covariances  # noqa: E402
from pyriemann.tangentspace import TangentSpace  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from _riemann_common import (  # noqa: E402
    RANDOM_STATE,
    SubjectRunResult,
    bandpass_single,
    report_and_save,
    run_subject,
)


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


PARAM_GRID = {
    "lr__C": [0.01, 0.1, 1.0, 10.0],
}


def main():
    parser = argparse.ArgumentParser(description="Riemann TS + LR nested CV.")
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--output", type=str, default="exp_riemann_ts.csv")
    args = parser.parse_args()

    print("Band: 8.0 - 30.0 Hz (single wide-band)")
    print()

    oof_dir = SCRIPT_DIR.parent / "results" / "oof" / "riemann_ts_1band"
    rows: List[SubjectRunResult] = []
    t_total = time.time()
    for sid in args.subjects:
        res = run_subject(
            subject_id=sid,
            pipeline_factory=build_pipeline,
            param_grid=PARAM_GRID,
            bandpass_fn=bandpass_single,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
            n_jobs=args.n_jobs,
            verbose=True,
            oof_dir=oof_dir,
        )
        rows.append(res)
    total_elapsed = time.time() - t_total

    report_and_save(
        rows=rows,
        output_name=args.output,
        total_elapsed=total_elapsed,
        title="Riemann Tangent Space + LR - nested CV (5x5)",
    )


if __name__ == "__main__":
    main()
