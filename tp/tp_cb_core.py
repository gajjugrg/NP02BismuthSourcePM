"""Shared Crystal Ball / TP utilities.

These utilities are shared between:
- TP fitting cache writer: tp/TP_fit_CB_to_json.py
- One-shot plotting + fitting script: tp/TP_analysis_CrystallBallFit.py

Design notes:
- Scripts are typically run from the `src/` directory so `../np02data` resolves.
- Keep this module dependency-light (numpy/pandas/scipy only).
"""

from __future__ import annotations

import glob
import os
import re
from datetime import datetime
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


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
    """Yield measurement locations under `root_dir`.

    Supports two layouts:
    - New (2026+): Record_YYYY_Mmm_DD_HH_MM.csv files containing columns binCenter, F1..F4
    - Legacy: directories containing histogram files like F1.txt, F2.txt, ...
    """

    seen: set[str] = set()

    # New layout: Record_*.csv
    record_pattern = f"{root_dir}/**/Record_*.csv"
    for record_path in glob.iglob(record_pattern, recursive=True):
        if record_path in seen:
            continue
        seen.add(record_path)
        yield record_path

    # Legacy layout: .../F1.txt
    legacy_pattern = f"{root_dir}/20??_[A-Za-z][a-z][a-z]/**/F1.txt"
    for f1_path in glob.iglob(legacy_pattern, recursive=True):
        directory = os.path.dirname(f1_path)
        if directory in seen:
            continue
        seen.add(directory)
        yield directory


def parse_timestamp(measurement: str) -> Optional[datetime]:
    """Extract measurement timestamp.

    - New layout: Record_YYYY_Mmm_DD_HH_MM.csv (seconds assumed 00)
    - Legacy layout: directory structure YYYY_Mmm/DD/HH/MM/(SS optional)
    """

    base = os.path.basename(measurement)
    m = re.fullmatch(r"Record_(\d{4})_([A-Za-z]{3})_(\d{2})_(\d{2})_(\d{2})\.csv", base)
    if m:
        year_str, month_word, day_str, hour_str, minute_str = m.groups()
        month_str = MONTH_MAP.get(month_word.capitalize())
        if month_str is None:
            return None
        try:
            return datetime.strptime(
                f"{year_str}-{month_str}-{int(day_str):02d} {int(hour_str):02d}:{int(minute_str):02d}:00",
                "%Y-%m-%d %H:%M:%S",
            )
        except ValueError:
            return None

    parts = measurement.strip("/").split("/")
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


def load_record_series(
    record_csv_path: str,
    *,
    channels: Tuple[str, ...] = ("F1", "F2", "F3", "F4"),
) -> Dict[str, pd.DataFrame]:
    """Load a Record_*.csv file into per-channel histogram DataFrames.

    Expected columns:
    - binCenter (volts)
    - F1, F2, F3, F4 (counts; some channels may be empty)

    Returns a dict mapping channel name -> DataFrame with columns BinCenter, Population.
    """
    if not os.path.exists(record_csv_path):
        return {}
    try:
        df = pd.read_csv(record_csv_path)
    except Exception:
        return {}

    if df.empty:
        return {}

    # Normalize bin center column name.
    bin_col = None
    for candidate in ("binCenter", "BinCenter", "bin_center", "BIN_CENTER"):
        if candidate in df.columns:
            bin_col = candidate
            break
    if bin_col is None:
        return {}

    out: Dict[str, pd.DataFrame] = {}
    bin_vals = pd.to_numeric(df[bin_col], errors="coerce")

    for ch in channels:
        if ch not in df.columns:
            continue
        counts = pd.to_numeric(df[ch], errors="coerce")
        ch_df = pd.DataFrame({"BinCenter": bin_vals, "Population": counts}).dropna(subset=["BinCenter", "Population"])
        if ch_df.empty:
            continue
        out[ch] = ch_df.sort_values("BinCenter").reset_index(drop=True)

    return out


def load_histogram(path: str) -> Optional[pd.DataFrame]:
    """Load histogram with columns BinCenter, Population."""
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


def crystal_ball(
    x: np.ndarray,
    amplitude: float,
    mean: float,
    sigma: float,
    alpha: float,
    n: float,
) -> np.ndarray:
    """Crystal Ball lineshape with stable tail evaluation."""
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


def fit_crystal_ball(
    x: np.ndarray,
    y: np.ndarray,
    *,
    maxfev: int = 40000,
) -> Optional[Tuple[Tuple[float, float, float, float, float], Optional[np.ndarray]]]:
    """Return (params, covariance) where params = (A, mean, sigma, alpha, n)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.allclose(y, 0):
        return None

    peak_idx = int(np.argmax(y))
    amplitude0 = max(float(y[peak_idx]), 1e-6)
    sigma0 = max((float(x.max()) - float(x.min())) / 6.0, 1e-3)
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
            maxfev=maxfev,
        )
    except Exception:
        return None

    return (float(popt[0]), float(popt[1]), float(popt[2]), float(popt[3]), float(popt[4])), pcov


def fit_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    *,
    maxfev: int = 20000,
) -> Optional[Tuple[Tuple[float, float, float], Optional[np.ndarray]]]:
    """Return (params, covariance) where params = (A, mean, sigma)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.allclose(y, 0):
        return None

    peak_idx = int(np.argmax(y))
    amplitude0 = max(float(y[peak_idx]), 1e-6)
    mean0 = float(x[peak_idx])
    sigma0 = max(float((x.max() - x.min()) / 6.0), 1e-4)

    try:
        popt, pcov = curve_fit(
            gaussian,
            x,
            y,
            p0=[amplitude0, mean0, sigma0],
            bounds=([0.0, float(x.min()), 1e-6], [np.inf, float(x.max()), float((x.max() - x.min()) * 2.0)]),
            maxfev=maxfev,
        )
    except Exception:
        return None

    return (float(popt[0]), float(popt[1]), float(popt[2])), pcov
