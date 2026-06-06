"""
exp_riemann_mdm.py
==================

Faz 3 — Varyant 2: Minimum Distance to Mean (MDM).

Pipeline:
    Covariances(estimator='oas')
    -> MDM(metric='riemann')

MDM hiperparametresizdir (sadece kovaryans estimator seçimi var, o da
sabit). Yine de raporlama tutarlılığı için nested CV iskeletini koruyup
trivial bir param_grid kullanıyoruz (overhead ihmal edilebilir, kapsam:
tek configürasyon).

predict_proba: pyriemann MDM softmax(-distances) ile sınıf olasılığı verir;
soft-voting ensemble için doğrudan kullanılabilir.

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

from pyriemann.classification import MDM  # noqa: E402
from pyriemann.estimation import Covariances  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from _riemann_common import (  # noqa: E402
    SubjectRunResult,
    bandpass_single,
    report_and_save,
    run_subject,
)


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("cov", Covariances(estimator="oas")),
            ("mdm", MDM(metric="riemann")),
        ]
    )


# Trivial grid — MDM hiperparametresiz. metric='riemann' tek seçenek olarak
# kalır; GridSearchCV en az 1 grid noktası bekler.
PARAM_GRID = {
    "mdm__metric": ["riemann"],
}


def main():
    parser = argparse.ArgumentParser(description="Riemann MDM nested CV.")
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--output", type=str, default="exp_riemann_mdm.csv")
    args = parser.parse_args()

    print("Band: 8.0 - 30.0 Hz (single wide-band)")
    print()

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
        )
        rows.append(res)
    total_elapsed = time.time() - t_total

    report_and_save(
        rows=rows,
        output_name=args.output,
        total_elapsed=total_elapsed,
        title="Riemann MDM - nested CV (5x5)",
    )


if __name__ == "__main__":
    main()
