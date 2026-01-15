"""TP CrystalBall fitting -> JSON cache.

This script does *only* fitting (no plotting):
- Fits Crystal Ball to F1/F2 in low region (raw, 0.55-1.2 V)
- Fits Crystal Ball to F1/F2 in high region (>=1.2 V; F1 uses a narrow window)
- Fits Gaussian to F3 test pulse peak

Results are appended into a JSON file so reruns can skip already-fit timestamps.

Usage examples:
  /path/to/python TP_fit_CB_to_json.py
  /path/to/python TP_fit_CB_to_json.py --force
  /path/to/python TP_fit_CB_to_json.py --start "2025-12-01 00:00" --end "2025-12-03 23:59"

Notes:
- This intentionally mirrors the fit logic in TP_analysis_CrystallBallFit.py,
  but isolates fitting from plotting.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


ROOT_DIR_DEFAULT = os.environ.get("NP02DATA_DIR", "../np02data")
PLOTS_DIR_DEFAULT = "plots"
JSON_OUT_DEFAULT = os.path.join(PLOTS_DIR_DEFAULT, "tp_cb_fit_results.json")
PLOT_START = datetime(2025, 11, 21, 16, 00)

MV_SCALE = 1000.0
LOW_V_MIN = 0.55
LOW_V_MAX = 1.2
HIGH_FIT_WINDOW_F1 = 0.10

COLORS = {
    "F1": "tab:blue",
    "F2": "tab:orange",
    "F3": "tab:green",
}

DESCRIPTIONS = {
    "F1": "Inner long PM",
    "F2": "Outer long PM",
    "F3": "Test pulse",
}

FILES = {
    "F1": "F1.txt",
    "F2": "F2.txt",
    "F3": "F3.txt",
}

MONTH_MAP = {
    "Jan": "01",
    "Feb": "02",
    "Mar": "03",
    "Apr": "04",
    "May": "05",
    "Jun": "06",
    "Jul": "07",
    "Aug": "08",
    "Sep": "09",
    "Oct": "10",
    "Nov": "11",
    "Dec": "12",
}


def _parse_datetime(dt_str: str) -> Optional[datetime]:
    if not dt_str:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(dt_str.strip(), fmt)
        except ValueError:
            continue
    return None


def iter_measurement_dirs(root_dir: str) -> Iterator[str]:
    seen = set()
    pattern = f"{root_dir}/20??_[A-Za-z][a-z][a-z]/**/F1.txt"
    for f1_path in glob.iglob(pattern, recursive=True):
        directory = os.path.dirname(f1_path)
        if directory in seen:
            continue
        seen.add(directory)
        yield directory


def parse_timestamp(directory: str) -> Optional[datetime]:
    parts = directory.strip("/").split("/")
    idx = None
    for i, part in enumerate(parts):
        if re.fullmatch(r"\d{4}_[A-Za-z]{3}", part):
            idx = i
            break
    if idx is None or len(parts) <= idx + 3:
        return None

    year_month = parts[idx]
    day = parts[idx + 1]
    hour = parts[idx + 2]
    minute = parts[idx + 3]
    second = parts[idx + 4] if len(parts) > idx + 4 else "00"

    year_str, month_word = year_month.split("_", 1)
    month_str = MONTH_MAP.get(month_word.capitalize())
    if month_str is None:
        return None

    try:
        return datetime.strptime(
            f"{year_str}-{month_str}-{int(day):02d} {int(hour):02d}:{int(minute):02d}:{int(second):02d}",
            "%Y-%m-%d %H:%M:%S",
        )
    except ValueError:
        return None


def load_histogram(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, usecols=["BinCenter", "Population"])
    except Exception:
        return None
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["BinCenter", "Population"])
    if df.empty:
        return None
    return df.sort_values("BinCenter").reset_index(drop=True)


def crystal_ball(x: np.ndarray, amplitude: float, mean: float, sigma: float, alpha: float, n: float) -> np.ndarray:
    t = (x - mean) / sigma
    abs_alpha = np.abs(alpha)
    A = (n / abs_alpha) ** n * np.exp(-0.5 * abs_alpha * abs_alpha)
    B = n / abs_alpha - abs_alpha
    result = np.empty_like(t, dtype=float)
    core_mask = t > -abs_alpha
    tail_mask = ~core_mask
    result[core_mask] = np.exp(-0.5 * t[core_mask] * t[core_mask])
    if np.any(tail_mask):
        denom = np.maximum(B - t[tail_mask], 1e-12)
        result[tail_mask] = A * denom ** (-n)
    return amplitude * result


def gaussian(x: np.ndarray, amplitude: float, mean: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def _fit_cb_curve(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[Tuple[float, float, float, float, float], Optional[np.ndarray]]]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.allclose(y, 0):
        return None

    peak_idx = int(np.argmax(y))
    amplitude0 = max(float(y[peak_idx]), 1e-6)
    sigma0 = max((x.max() - x.min()) / 6.0, 1e-3)
    mean0 = float(x[peak_idx])
    alpha0 = 1.5
    n0 = 3.0

    lower = [0.0, float(x.min()), 1e-5, 0.1, 0.5]
    upper = [np.inf, float(x.max()), float((x.max() - x.min()) * 2.0), 10.0, 50.0]

    try:
        popt, pcov = curve_fit(
            crystal_ball,
            x,
            y,
            p0=[amplitude0, mean0, sigma0, alpha0, n0],
            bounds=(lower, upper),
            maxfev=40000,
        )
    except Exception:
        return None

    return (float(popt[0]), float(popt[1]), float(popt[2]), float(popt[3]), float(popt[4])), pcov


def _fit_f3_params(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[Tuple[float, float, float], Optional[np.ndarray]]]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.allclose(y, 0):
        return None

    peak_idx = int(np.argmax(y))
    amp0 = max(float(y[peak_idx]), 1e-6)
    mean0 = float(x[peak_idx])
    sigma0 = max(float((x.max() - x.min()) / 6.0), 1e-4)

    try:
        popt, pcov = curve_fit(
            gaussian,
            x,
            y,
            p0=[amp0, mean0, sigma0],
            bounds=([0.0, float(x.min()), 1e-6], [np.inf, float(x.max()), float((x.max() - x.min()) * 2.0)]),
            maxfev=20000,
        )
    except Exception:
        return None

    return (float(popt[0]), float(popt[1]), float(popt[2])), pcov


def _fit_f3_peak(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, Optional[float]]]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.allclose(y, 0):
        return None

    peak_idx = int(np.argmax(y))
    amp0 = max(float(y[peak_idx]), 1e-6)
    mean0 = float(x[peak_idx])
    sigma0 = max(float((x.max() - x.min()) / 6.0), 1e-4)

    try:
        popt, pcov = curve_fit(
            gaussian,
            x,
            y,
            p0=[amp0, mean0, sigma0],
            bounds=([0.0, float(x.min()), 1e-6], [np.inf, float(x.max()), float((x.max() - x.min()) * 2.0)]),
            maxfev=20000,
        )
    except Exception:
        return None

    mean = float(popt[1])
    mean_err = float(np.sqrt(pcov[1, 1])) if pcov is not None and pcov.shape[0] > 1 else None
    return mean, mean_err


def _mean_err_from_pcov(pcov: Optional[np.ndarray], mean_index: int) -> Optional[float]:
    if pcov is None:
        return None
    if pcov.ndim != 2 or pcov.shape[0] <= mean_index or pcov.shape[1] <= mean_index:
        return None
    val = pcov[mean_index, mean_index]
    if not np.isfinite(val) or val < 0:
        return None
    return float(np.sqrt(val))


def _to_millivolts_scalar(value_v: Optional[float]) -> Optional[float]:
    if value_v is None:
        return None
    if not np.isfinite(value_v):
        return None
    return float(value_v) * MV_SCALE


@dataclass(frozen=True)
class FitResult:
    mean_v: float
    err_v: Optional[float]
    params: Optional[Dict[str, float]] = None


def compute_means(series: Dict[str, pd.DataFrame]) -> Dict[str, FitResult]:
    out: Dict[str, FitResult] = {}

    # High region CB fits
    for label in ("F1", "F2"):
        df = series.get(label)
        if df is None or df.empty:
            continue
        centers = df["BinCenter"].to_numpy(dtype=float)
        counts = df["Population"].to_numpy(dtype=float)
        mask = centers >= 1.2
        if not mask.any():
            continue
        x = centers[mask]
        y = counts[mask]

        fit_x = x
        fit_y = y
        if label == "F1" and x.size > 3:
            peak_val = float(x[int(np.argmax(y))])
            narrow_mask = np.abs(x - peak_val) <= HIGH_FIT_WINDOW_F1
            if narrow_mask.any():
                fit_x = x[narrow_mask]
                fit_y = y[narrow_mask]

        fit = _fit_cb_curve(fit_x, fit_y)
        if fit is None:
            continue
        popt, pcov = fit
        mean = float(popt[1])
        mean_err = _mean_err_from_pcov(pcov, 1)
        out[f"{label}_high"] = FitResult(
            mean_v=mean,
            err_v=mean_err,
            params={
                "amplitude": float(popt[0]),
                "sigma": float(popt[2]),
                "alpha": float(popt[3]),
                "n": float(popt[4]),
            },
        )

    # Low region CB fits
    for label in ("F1", "F2"):
        df = series.get(label)
        if df is None or df.empty:
            continue
        v = df["BinCenter"].to_numpy(dtype=float)
        c = df["Population"].to_numpy(dtype=float)
        mask = (v >= LOW_V_MIN) & (v <= LOW_V_MAX)
        if not mask.any():
            continue
        x = v[mask]
        y = c[mask]

        fit = _fit_cb_curve(x, y)
        if fit is None:
            continue
        popt, pcov = fit
        mean = float(popt[1])
        mean_err = _mean_err_from_pcov(pcov, 1)

        # Keep the same rejection used in TP_analysis_CrystallBallFit.py
        if label == "F2":
            mean_mV = _to_millivolts_scalar(mean)
            if mean_mV is not None and mean_mV < 600.0:
                continue

        out[f"{label}_low"] = FitResult(mean_v=mean, err_v=mean_err)

        out[f"{label}_low"] = FitResult(
            mean_v=mean,
            err_v=mean_err,
            params={
                "amplitude": float(popt[0]),
                "sigma": float(popt[2]),
                "alpha": float(popt[3]),
                "n": float(popt[4]),
            },
        )

    # F3 peak
    df3 = series.get("F3")
    if df3 is not None and not df3.empty:
        x3 = df3["BinCenter"].to_numpy(dtype=float)
        y3 = df3["Population"].to_numpy(dtype=float)
        fit3 = _fit_f3_params(x3, y3)
        if fit3 is not None:
            popt3, pcov3 = fit3
            mean = float(popt3[1])
            mean_err = _mean_err_from_pcov(pcov3, 1)
            out["F3"] = FitResult(
                mean_v=mean,
                err_v=mean_err,
                params={
                    "amplitude": float(popt3[0]),
                    "sigma": float(popt3[2]),
                },
            )

    return out


def _plot_measurement_png(
    plots_dir: str,
    measurement_time: datetime,
    series: Dict[str, pd.DataFrame],
    means: Dict[str, FitResult],
) -> Optional[str]:
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=False)
    ax_high, ax_low, ax_f3 = axes

    plotted_any = False

    # High region
    for label in ("F1", "F2"):
        df = series.get(label)
        if df is None or df.empty:
            continue
        centers = df["BinCenter"].to_numpy(dtype=float)
        counts = df["Population"].to_numpy(dtype=float)
        mask = centers >= 1.2
        if not mask.any():
            continue
        x = centers[mask]
        y = counts[mask]
        ax_high.step(x * MV_SCALE, y, where="mid", color=COLORS[label], label=DESCRIPTIONS[label])

        key = f"{label}_high"
        fr = means.get(key)
        if fr is not None and fr.params is not None:
            x_fit = np.linspace(float(x.min()), float(x.max()), 400)
            y_fit = crystal_ball(
                x_fit,
                fr.params["amplitude"],
                fr.mean_v,
                fr.params["sigma"],
                fr.params["alpha"],
                fr.params["n"],
            )
            mean_mV = _to_millivolts_scalar(fr.mean_v)
            err_mV = _to_millivolts_scalar(fr.err_v) if fr.err_v is not None else None
            ax_high.plot(
                x_fit * MV_SCALE,
                y_fit,
                color=COLORS[label],
                linestyle="--",
                label=(
                    f"{DESCRIPTIONS[label]} fit (mean={mean_mV:.2f}"
                    + (f" ± {err_mV:.2f}" if err_mV is not None else "")
                    + f" mV, alpha={fr.params['alpha']:.2f}, n={fr.params['n']:.1f})"
                ),
            )

        plotted_any = True

    if plotted_any:
        ax_high.set_xlabel("Bin center [mV]")
        ax_high.set_ylabel("Counts")
        ax_high.set_title("F1/F2 raw high region")
        ax_high.grid(True, alpha=0.3)
        ax_high.legend(loc="best")
    else:
        ax_high.set_visible(False)

    # Low region
    plotted_low = False
    for label in ("F1", "F2"):
        df = series.get(label)
        if df is None or df.empty:
            continue
        v = df["BinCenter"].to_numpy(dtype=float)
        c = df["Population"].to_numpy(dtype=float)
        mask = (v >= LOW_V_MIN) & (v <= LOW_V_MAX)
        if not mask.any():
            continue
        x = v[mask]
        y = c[mask]
        ax_low.step(x * MV_SCALE, y, where="mid", color=COLORS[label], label=f"{DESCRIPTIONS[label]} raw")

        key = f"{label}_low"
        fr = means.get(key)
        if fr is not None and fr.params is not None:
            x_fit = np.linspace(float(x.min()), float(x.max()), 400)
            y_fit = crystal_ball(
                x_fit,
                fr.params["amplitude"],
                fr.mean_v,
                fr.params["sigma"],
                fr.params["alpha"],
                fr.params["n"],
            )
            mean_mV = _to_millivolts_scalar(fr.mean_v)
            err_mV = _to_millivolts_scalar(fr.err_v) if fr.err_v is not None else None
            ax_low.plot(
                x_fit * MV_SCALE,
                y_fit,
                color="k",
                linestyle="--",
                label=(
                    f"{DESCRIPTIONS[label]} CB fit (mean={mean_mV:.2f}"
                    + (f" ± {err_mV:.2f}" if err_mV is not None else "")
                    + f" mV, alpha={fr.params['alpha']:.2f}, n={fr.params['n']:.1f})"
                ),
            )
        plotted_low = True

    if plotted_low:
        ax_low.set_xlabel("Bin center [mV]")
        ax_low.set_ylabel("Counts")
        ax_low.set_title(f"Low region raw ({LOW_V_MIN:.2f}-{LOW_V_MAX:.2f} V)")
        ax_low.grid(True, alpha=0.3)
        ax_low.set_xlim(left=LOW_V_MIN * MV_SCALE, right=LOW_V_MAX * MV_SCALE)
        ax_low.legend(loc="best")
    else:
        ax_low.set_visible(False)

    # F3
    df3 = series.get("F3")
    if df3 is None or df3.empty:
        ax_f3.set_visible(False)
    else:
        x = df3["BinCenter"].to_numpy(dtype=float)
        y = df3["Population"].to_numpy(dtype=float)
        ax_f3.step(x * MV_SCALE, y, where="mid", color=COLORS["F3"], label=DESCRIPTIONS["F3"])
        fr = means.get("F3")
        if fr is not None and fr.params is not None:
            x_fit = np.linspace(float(x.min()), float(x.max()), 400)
            y_fit = gaussian(x_fit, fr.params["amplitude"], fr.mean_v, fr.params["sigma"])
            mean_mV = _to_millivolts_scalar(fr.mean_v)
            err_mV = _to_millivolts_scalar(fr.err_v) if fr.err_v is not None else None
            ax_f3.plot(
                x_fit * MV_SCALE,
                y_fit,
                color="tab:purple",
                linestyle="--",
                label=(
                    f"Peak fit (mean={mean_mV:.2f}"
                    + (f" ± {err_mV:.2f}" if err_mV is not None else "")
                    + " mV)"
                ),
            )
        ax_f3.set_xlim(170, 190)
        ax_f3.set_ylim(bottom=0)
        ax_f3.set_xlabel("Bin center [mV]")
        ax_f3.set_ylabel("Counts")
        ax_f3.set_title("Test pulse")
        ax_f3.grid(True, alpha=0.3)
        ax_f3.legend(loc="best")

    if not any(ax.get_visible() for ax in axes):
        plt.close(fig)
        return None

    fig.suptitle(f"{measurement_time:%Y-%m-%d %H:%M:%S}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, f"split_low_cb_{measurement_time.strftime('%Y%m%d_%H%M%S')}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit TP CrystalBall means and store in JSON for fast reruns.")
    parser.add_argument("--root-dir", default=ROOT_DIR_DEFAULT)
    parser.add_argument("--plots-dir", default=PLOTS_DIR_DEFAULT)
    parser.add_argument("--json-out", default=JSON_OUT_DEFAULT)
    parser.add_argument("--start", default=None, help="YYYY-MM-DD HH:MM[:SS]")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD HH:MM[:SS]")
    parser.add_argument("--force", action="store_true", help="Refit even if timestamp exists in JSON")
    parser.add_argument("--no-plots", action="store_true", help="Do not save per-measurement split PNGs")
    args = parser.parse_args()

    start = _parse_datetime(args.start) if args.start else PLOT_START
    end = _parse_datetime(args.end) if args.end else None

    # Ensure JSON default follows plots-dir if user changed plots-dir.
    json_out = args.json_out
    if json_out == JSON_OUT_DEFAULT:
        json_out = os.path.join(args.plots_dir, "tp_cb_fit_results.json")

    existing = _load_json(json_out)
    measurements = existing.get("measurements") if isinstance(existing.get("measurements"), dict) else {}
    if not isinstance(measurements, dict):
        measurements = {}

    directories = sorted(iter_measurement_dirs(args.root_dir))

    processed = 0
    skipped = 0
    try:
        for directory in directories:
            ts = parse_timestamp(directory)
            if ts is None:
                continue
            if start and ts < start:
                continue
            if end and ts > end:
                continue

            key = ts.isoformat()
            if not args.force and key in measurements:
                skipped += 1
                continue

            series: Dict[str, pd.DataFrame] = {}
            for label, fname in FILES.items():
                df = load_histogram(os.path.join(directory, fname))
                if df is not None:
                    series[label] = df
            if not series:
                continue

            means = compute_means(series)
            if not means:
                continue

            if not args.no_plots:
                out_png = _plot_measurement_png(args.plots_dir, ts, series, means)
                if out_png is not None:
                    print(f"Saved plot: {out_png}")

            measurements[key] = {
                k: {
                    "mean_v": v.mean_v,
                    "err_v": v.err_v,
                    **({"params": v.params} if v.params is not None else {}),
                }
                for k, v in means.items()
            }
            processed += 1

            # Persist on every measurement so Ctrl+C doesn't lose progress.
            out = {
                "schema_version": 1,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "config": {
                    "low_v_min": LOW_V_MIN,
                    "low_v_max": LOW_V_MAX,
                    "high_fit_window_f1": HIGH_FIT_WINDOW_F1,
                },
                "measurements": measurements,
            }
            _atomic_write_json(json_out, out)

            if processed % 25 == 0:
                print(f"Processed {processed} new fits (skipped {skipped})...")
    except KeyboardInterrupt:
        print("Interrupted; progress is saved to JSON.")

    print(f"Done. New fits: {processed}, skipped(existing): {skipped}")
    print(f"Wrote: {json_out}")


if __name__ == "__main__":
    main()
