#!/usr/bin/env python3
"""Plot TP fit means from a single CSV (fit_means_scope.csv).

This avoids refitting and uses a fixed time window via PLOT_START/PLOT_END.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


PLOTS_DIR = "plots_scope"
CSV_PATH = os.path.join(PLOTS_DIR, "fit_means_scope.csv")
PLOT_START = datetime(2026, 1, 24, 7, 0)
PLOT_END = None

DATA_LABELS = {
    "F1_high": "TP_Response_Inner_long_PM",
    "F2_high": "TP_Response_Outer_long_PM",
    "F1_low": "Inner_low_CB",
    "F2_low": "Outer_low_CB",
    "F3": "Test_pulse",
}

_TITLE_BY_KEY = {
    "F1_high": "Inner long PM TP response",
    "F2_high": "Outer long PM TP response",
    "F1_low": "Inner long PM",
    "F2_low": "Outer long PM",
    "F3": "Test pulse",
}

_COLOR_BY_KEY = {
    "F1_high": "tab:blue",
    "F2_high": "tab:orange",
    "F1_low": "tab:blue",
    "F2_low": "tab:orange",
    "F3": "tab:purple",
}


def _parse_datetime(dt_str: str) -> Optional[datetime]:
    if not dt_str:
        return None
    dt_str = dt_str.strip()
    try:
        return datetime.fromisoformat(dt_str)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    return None


def _resolve_plot_window() -> Tuple[Optional[datetime], Optional[datetime]]:
    start = PLOT_START
    end = PLOT_END
    env_start = os.getenv("TP_PLOT_START")
    env_end = os.getenv("TP_PLOT_END")
    if env_start:
        parsed = _parse_datetime(env_start)
        if parsed is None:
            print(f"Could not parse TP_PLOT_START='{env_start}' (expected YYYY-MM-DD HH:MM[:SS])")
        else:
            start = parsed
    if env_end:
        parsed = _parse_datetime(env_end)
        if parsed is None:
            print(f"Could not parse TP_PLOT_END='{env_end}' (expected YYYY-MM-DD HH:MM[:SS])")
        else:
            end = parsed
    return start, end


def _to_float(raw: str) -> Optional[float]:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _load_series(csv_path: str) -> Dict[str, List[Tuple[datetime, float, Optional[float]]]]:
    series: Dict[str, List[Tuple[datetime, float, Optional[float]]]] = {k: [] for k in DATA_LABELS.keys()}
    start, end = _resolve_plot_window()
    if not os.path.exists(csv_path):
        print(f"Missing CSV: {csv_path}")
        return series

    def _iter_rows():
        with open(csv_path, newline="") as handle:
            for line in handle:
                if line.lstrip().startswith("#"):
                    continue
                yield line

    reader = csv.DictReader(_iter_rows())
    for row in reader:
        ts = _parse_datetime(row.get("timestamp_iso", ""))
        if ts is None:
            continue
        if start and ts < start:
            continue
        if end and ts > end:
            continue
        for key, base in DATA_LABELS.items():
            val = _to_float(row.get(f"{base}_mV", ""))
            err = _to_float(row.get(f"{base}_err_mV", ""))
            if val is None:
                continue
            series[key].append((ts, val, err))
    return series


def _plot_series(ax, samples, color, title, ylim=None):
    if not samples:
        ax.set_visible(False)
        return
    ax.set_title(title)
    samples = sorted(samples, key=lambda x: x[0])
    times = [t for t, _, _ in samples]
    values = [v for _, v, _ in samples]
    errs = np.array([e if e is not None else np.nan for _, _, e in samples], dtype=float)
    ax.plot(times, values, marker="o", markersize=3.5, linestyle="none", color=color)
    ax.set_ylabel("Mean [mV]")
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)


def main() -> int:
    series = _load_series(CSV_PATH)
    if not any(series.values()):
        print("No data to plot.")
        return 0

    keys = ["F1_high", "F2_high", "F1_low", "F2_low", "F3"]
    fig, axes = plt.subplots(nrows=len(keys), ncols=1, figsize=(10, 13), sharex=True)

    for ax, key in zip(axes, keys):
        ylim = (1215, 1230) if key == "F3" else None
        _plot_series(
            ax,
            series.get(key, []),
            _COLOR_BY_KEY.get(key, "tab:gray"),
            _TITLE_BY_KEY.get(key, key),
            ylim=ylim,
        )

    axes[-1].set_xlabel("Measurement time")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, "fit_means_vs_time.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved mean-vs-time plot: {out_path}")
    if os.environ.get("TP_SHOW", "0") == "1":
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
