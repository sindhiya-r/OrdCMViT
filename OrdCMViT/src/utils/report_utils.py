"""
src/utils/report_utils.py
──────────────────────────
Helpers for summarizing training/validation/test results in the format
required by the slide deck described by the user.  This module is not
required for training itself but can be executed after a run to produce
textual tables and log entries that mirror the slides.

Usage example::

    from utils import report_utils
    report_utils.summarize_run("runs/ordcmvit/", "test", verbose=True)

The summary reads metrics json, result csv and prints human-readable tables.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _make_table1(metrics_path: Path) -> str:
    m = _load_json(metrics_path)
    lines = []
    lines.append("Table 1 — BI‑RADS Classification Performance on BCMID Test Set")
    lines.append("Model | Accuracy | QWK | AUC (Macro)")
    lines.append("-----|----------|-----|--------------")
    # the json is expected to contain keys for each model name
    for model_name, vals in m.items():
        acc = vals.get("acc", 0)
        qwk = vals.get("qwk", 0)
        auc = vals.get("auc_macro", 0)
        lines.append(f"{model_name} | {acc:.3f} | {qwk:.3f} | {auc:.3f}")
    lines.append("")
    return "\n".join(lines)


def _per_class_table(metrics_dict: Dict) -> str:
    lines = ["Table 2 — Per‑class Accuracy", "BI‑RADS | Accuracy", "------|---------"]
    for cls in range(1, 6):
        lines.append(f"{cls} | {metrics_dict.get(f'acc_birads{cls}', 0):.2f}")
    lines.append("")
    return "\n".join(lines)


def summarize_run(run_dir: str, split: str = "test", verbose: bool = False):
    r = Path(run_dir)
    metrics_path = r / f"{split}_metrics.json"
    results_csv = r / f"{split}_results.csv"

    if not metrics_path.exists() or not results_csv.exists():
        print(f"run directory {run_dir} missing metrics or results files")
        return

    m = _load_json(metrics_path)
    # if the json only contains a single entry (the final model), unwrap it
    if len(m) == 1:
        m = next(iter(m.values()))

    print(_make_table1(metrics_path))
    print(_per_class_table(m))

    print("Figure — Confusion Matrix:", (r / "confusion_matrix.png").as_posix())
    cams = list((r / "visualizations").glob("*_cam.png"))
    print("Figure — CAM Visualizations:")
    for c in cams[:6]:
        print("  ", c.as_posix())

    if verbose:
        print("\nNote: for full slide 7 and 8 outputs refer to the report document or "
              "see the notebook 'analysis.ipynb' in the repo.")
