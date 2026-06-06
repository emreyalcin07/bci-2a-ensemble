"""
main.py
=======

Deney çalıştırıcı. Komut satırından bir pipeline ve denek listesi seçilir;
sonuçlar results/tables/ altına CSV olarak kaydedilir.

Kullanım örnekleri (ileride):
    python main.py --pipeline csp_lda --subjects 1 2 3
    python main.py --pipeline csp_lda --all
    python main.py --pipeline riemann_mdm --all --nested-cv

Bu aşamada yalnızca iskelet; gerçek pipeline'lar experiments/ altında
ayrı modüller olarak tanımlanacak (örn. experiments/csp_lda.py).
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Dict, List

import pandas as pd

from data_loader import load_subject
from evaluation import CVResult, result_to_row, run_cv

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_pipeline(name: str):
    """experiments.<name> modülünü yükle ve build_pipeline() fabrikasını döndür.

    Her pipeline modülü şu arayüzü sağlamalı:
        - build_pipeline() -> sklearn-uyumlu estimator (predict_proba dahil)
        - PARAM_GRID: dict (opsiyonel; nested CV için)
    """
    mod = importlib.import_module(f"experiments.{name}")
    if not hasattr(mod, "build_pipeline"):
        raise AttributeError(f"experiments.{name} içinde build_pipeline() yok.")
    return mod


def run_pipeline_on_subject(pipeline_mod, subject_id: int) -> CVResult:
    """Tek bir deneğe pipeline'ı uygula ve CV sonucu döndür."""
    sd = load_subject(subject_id, session="T")
    if sd.y is None:
        raise RuntimeError(f"Subject A{subject_id:02d}T için etiket alınamadı.")
    factory = pipeline_mod.build_pipeline
    return run_cv(factory, sd.X, sd.y)


def main():
    parser = argparse.ArgumentParser(description="BCI Comp IV 2a deney çalıştırıcı.")
    parser.add_argument("--pipeline", required=True, help="experiments/ altındaki modül adı")
    parser.add_argument("--subjects", type=int, nargs="*", help="Denek listesi (1..9)")
    parser.add_argument("--all", action="store_true", help="Tüm denekleri çalıştır")
    parser.add_argument("--output", type=str, default=None, help="CSV çıktı dosyası adı")
    args = parser.parse_args()

    subjects: List[int] = list(range(1, 10)) if args.all else (args.subjects or [])
    if not subjects:
        parser.error("--subjects veya --all gerekli.")

    pipeline_mod = load_pipeline(args.pipeline)

    rows: List[Dict] = []
    for sid in subjects:
        print(f"[A{sid:02d}T] {args.pipeline} ... ", end="", flush=True)
        result = run_pipeline_on_subject(pipeline_mod, sid)
        rows.append(result_to_row(args.pipeline, sid, result))
        print(result.summary())

    df = pd.DataFrame(rows)
    out = args.output or f"{args.pipeline}.csv"
    df.to_csv(RESULTS_DIR / out, index=False)
    print(f"\nSonuçlar: {RESULTS_DIR / out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
