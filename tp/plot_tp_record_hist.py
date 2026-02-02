#!/usr/bin/env python3
"""Plot raw TP histograms from Record_*.csv files (no fitting)."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Make `src/` importable even when running from repo root.
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from tp.tp_cb_core import load_record_series, parse_timestamp  # noqa: E402


_REPO_DIR = Path(__file__).resolve().parents[2]
_NP02DATA_DIR = os.environ.get("NP02DATA_DIR", str(_REPO_DIR / "np02data"))

PLOTS_DIR = "plots_scope"
PLOT_START = datetime(2026, 1, 23, 15, 0)
PLOT_END = None


def _parse_datetime(dt_str: str) -> Optional[datetime]:
    if not dt_str:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(dt_str.strip(), fmt)
        except ValueError:
            continue
    return None


def _iter_record_files(root_dir: str) -> Iterable[str]:
    pattern = os.path.join(root_dir, "**", "Record_*.csv")
    return sorted(Path(p).as_posix() for p in Path(root_dir).glob("**/Record_*.csv"))


def _select_record(
    files: Sequence[str],
    start: Optional[datetime],
    end: Optional[datetime],
) -> Optional[str]:
    candidates: List[Tuple[datetime, str]] = []
    all_candidates: List[Tuple[datetime, str]] = []
    for path in files:
        ts = parse_timestamp(path)
        if ts is None:
            continue
        all_candidates.append((ts, path))
        if start and ts < start:
            continue
        if end and ts > end:
            continue
        candidates.append((ts, path))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    if all_candidates:
        print("No Record_*.csv files in the requested window; falling back to latest file.")
        return max(all_candidates, key=lambda item: item[0])[1]
    return None


def _resolve_plot_window(args_start: Optional[str], args_end: Optional[str]) -> Tuple[Optional[datetime], Optional[datetime]]:
    start = _parse_datetime(args_start) if args_start else None
    end = _parse_datetime(args_end) if args_end else None
    if start is None:
        start = PLOT_START
    if end is None:
        end = PLOT_END
    return start, end


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot TP Record_*.csv histograms (raw).")
    parser.add_argument("path", nargs="?", default=None, help="Record_*.csv file or root directory to search")
    parser.add_argument("--channels", nargs="*", default=["F1", "F2", "F3"], help="Channels to plot (F1/F2/F3/F4)")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD HH:MM[:SS]")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD HH:MM[:SS]")
    parser.add_argument("--logy", action="store_true", help="Use log scale on Y axis")
    parser.add_argument("--xmin", type=float, default=None, help="Minimum x to display")
    parser.add_argument("--xmax", type=float, default=None, help="Maximum x to display")
    parser.add_argument("--save", default=None, help="Save figure to this path (default under plots_scope)")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively")
    args = parser.parse_args()

    start, end = _resolve_plot_window(args.start, args.end)

    if args.path and os.path.isfile(args.path):
        record_path = args.path
    else:
        root_dir = args.path if args.path else _NP02DATA_DIR
        files = list(_iter_record_files(root_dir))
        record_path = _select_record(files, start, end)

    if record_path is None:
        print("No Record_*.csv files found.")
        return 1

    series = load_record_series(record_path, channels=tuple(args.channels))
    if not series:
        print(f"No histogram data found in {record_path}")
        return 1

    channels = [ch for ch in args.channels if ch in series]
    fig, axes = plt.subplots(
        nrows=len(channels),
        ncols=1,
        sharex=False,
        figsize=(10, max(3, 3 * len(channels))),
    )
    if isinstance(axes, np.ndarray):
        axes_list = axes.ravel().tolist()
    else:
        axes_list = [axes]

    ts = parse_timestamp(record_path)
    if ts:
        fig.suptitle(f"TIME {ts:%Y-%m-%d %H:%M:%S}")

    for ax, ch in zip(axes_list, channels):
        df = series.get(ch)
        if df is None or df.empty:
            ax.set_visible(False)
            continue
        x = df["BinCenter"].to_numpy(dtype=float)
        y = df["Population"].to_numpy(dtype=float)
        ax.step(x, y, where="mid", linewidth=1.0)
        ax.set_ylabel("Counts")
        ax.set_title(ch)
        ax.set_xlabel("Voltage [V]")
        ax.grid(True, which="both", alpha=0.25)
        if args.logy:
            ax.set_yscale("log")
        if args.xmin is not None or args.xmax is not None:
            xmin = args.xmin if args.xmin is not None else float(np.nanmin(x))
            xmax = args.xmax if args.xmax is not None else float(np.nanmax(x))
            ax.set_xlim(xmin, xmax)

    fig.tight_layout()

    save_path = args.save
    if not save_path:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        stem = Path(record_path).stem
        save_path = os.path.join(PLOTS_DIR, f"{stem}_hist.png")
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")

    show = args.show or (os.environ.get("TP_SHOW", "0") == "1")
    if show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
