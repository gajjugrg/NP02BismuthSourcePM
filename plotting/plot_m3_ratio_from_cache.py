import argparse
import os
from datetime import datetime
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from plot_m3_from_cache import collect_series, load_cache


def _parse_datetime(label: str, value: str) -> datetime:
    """Parse datetime inputs provided via CLI arguments."""
    value = value.strip()
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        try:
            return datetime.strptime(value, "%Y-%m-%d")
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid {label} datetime '{value}'. Use ISO format like 2025-10-01 or 2025-10-01T12:00."
            ) from exc


def compute_ratios(
    timestamps: List[datetime],
    m3_short: List[float],
    m3_long: List[float],
    min_ratio: Optional[float],
    max_ratio: Optional[float],
) -> Tuple[List[datetime], List[float]]:
    """Filter points and compute m3_short / m3_long ratios."""
    filtered_ts: List[object] = []
    ratios: List[float] = []
    for t, s, l in zip(timestamps, m3_short, m3_long):
        if not np.isfinite(s) or not np.isfinite(l) or l == 0:
            continue
        r = float(s) / float(l)
        if min_ratio is not None and r < min_ratio:
            continue
        if max_ratio is not None and r > max_ratio:
            continue
        filtered_ts.append(t)
        ratios.append(r)
    return filtered_ts, ratios


def plot_ratio_from_cache(
    cache_file: str,
    out_png: str,
    prefer_temp: bool,
    m3_short_min: float,
    m3_short_max: float,
    min_ratio: Optional[float],
    max_ratio: Optional[float],
    start: Optional[datetime],
    end: Optional[datetime],
) -> int:
    cache = load_cache(cache_file)
    ts, svals, lvals = collect_series(
        cache=cache,
        prefer_temp=prefer_temp,
        m3_short_min=m3_short_min,
        m3_short_max=m3_short_max,
    )
    if not ts:
        print("No valid m3 points found to plot (after m3_short cuts).")
        return 2

    if start or end:
        filtered_ts: List[datetime] = []
        filtered_short: List[float] = []
        filtered_long: List[float] = []
        for t, s, l in zip(ts, svals, lvals):
            if start and t < start:
                continue
            if end and t > end:
                continue
            filtered_ts.append(t)
            filtered_short.append(s)
            filtered_long.append(l)
        ts, svals, lvals = filtered_ts, filtered_short, filtered_long
        if not ts:
            print("No entries fall within the requested time window.")
            return 2

    ts_ratio, ratio_vals = compute_ratios(
        timestamps=ts,
        m3_short=svals,
        m3_long=lvals,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
    )
    if not ts_ratio:
        print("No valid ratios to plot after ratio cuts.")
        return 2

    plt.figure(figsize=(10, 5))
    plt.plot(ts_ratio, ratio_vals, marker="o", linestyle="none", color="tab:purple")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
    plt.gcf().autofmt_xdate()
    title = "m3_short / m3_long vs time"
    if prefer_temp:
        title += " (temp-corrected)"
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("m3_short / m3_long")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_dir = os.path.dirname(out_png) or "."
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved plot: {out_png}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot the time series of m3_short/m3_long from fit_cache.pkl"
    )
    parser.add_argument("--cache", default="fit_cache.pkl", help="Path to fit_cache.pkl")
    parser.add_argument(
        "--out",
        default="plots/m3_ratio_short_to_long.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--use-temp",
        action="store_true",
        help="Prefer temperature-corrected m3 if present",
    )
    parser.add_argument(
        "--min-m3-short",
        type=float,
        default=0.55,
        help="Lower cut on m3_short",
    )
    parser.add_argument(
        "--max-m3-short",
        type=float,
        default=1.38,
        help="Upper cut on m3_short",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=None,
        help="Optional lower bound on m3_short/m3_long",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=None,
        help="Optional upper bound on m3_short/m3_long",
    )
    parser.add_argument(
        "--start",
        type=lambda s: _parse_datetime("start", s),
        default=None,
        help="Inclusive window start (ISO date like 2025-10-01 or 2025-10-01T00:00).",
    )
    parser.add_argument(
        "--end",
        type=lambda s: _parse_datetime("end", s),
        default=None,
        help="Inclusive window end (ISO date like 2025-10-31 or 2025-10-31T23:59).",
    )
    args = parser.parse_args(argv)

    if args.start and args.end and args.start > args.end:
        parser.error("start must be earlier than or equal to end.")

    try:
        return plot_ratio_from_cache(
            cache_file=args.cache,
            out_png=args.out,
            prefer_temp=args.use_temp,
            m3_short_min=args.min_m3_short,
            m3_short_max=args.max_m3_short,
            min_ratio=args.min_ratio,
            max_ratio=args.max_ratio,
            start=args.start,
            end=args.end,
        )
    except FileNotFoundError:
        print(f"Cache file not found: {args.cache}")
        return 2
    except Exception as exc:
        print(f"Failed to generate ratio plot: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
