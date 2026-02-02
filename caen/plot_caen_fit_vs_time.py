#!/usr/bin/env python3
"""CAEN peak-fit analysis (TP-style workflow, from CAEN RAW data).

This is a *script* (not a library module). It scans CAEN RAW spectra
(`CH<ch>@...<timestamp>.txt3`), fits the main peak in ADC space, and produces:

- Per-measurement spectrum+fit plots (optional)
- A stacked mean-vs-time plot (one subplot per channel)
- A CSV dump of fitted peak means vs time

Input layout
------------
- One or more CAEN run directories that contain a RAW/ folder.
    Each RAW/ contains files like:
        CH0@...20260123_154850.txt3
        CH1@...20260123_154850.txt3
        CH2@...20260123_154850.txt3

Fit model summary
-----------------
- CH0/CH1 (Inner/Outer Long): Crystal Ball by default (override with --fit-model-long)
- CH2 (Test Pulse): Gaussian (kept fixed to match the spectrum plotting conventions)

Outputs (written under --plots-dir)
----------------------------------
- caen_<timestamp>.png      : per-measurement spectrum plots (optional)
- caen_fit_means_vs_time.png            : stacked time-series of fitted means
- caen_fit_means_data.csv               : CSV dump of fitted mean values

Controlling the time window
---------------------------
- --start/--end OR env vars CAEN_PLOT_START / CAEN_PLOT_END (same format as TP):
    YYYY-MM-DD HH:MM[:SS]
- CAEN_SHOW=1 shows figures interactively (otherwise saves PNGs only).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import bisect

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd


# Make `src/` importable even when running from repo root.
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
_REPO_DIR = Path(__file__).resolve().parents[2]
_NP02DATA_DIR = os.environ.get("NP02DATA_DIR", str(_REPO_DIR / "np02data"))
_TEMP_CSV_PATH = os.environ.get("NP02_TEMP_CSV", os.path.join(_NP02DATA_DIR, "Temp_Jan26.csv"))
PLOT_START = datetime(2026, 1, 24, 7, 0)

# Reuse the parsing + fit logic from the canonical CAEN spectrum script.
from caen.plot_caen_espectrum import (  # noqa: E402
    _CH_FILE_RE,
    _CHANNEL_LABELS,
    _DEFAULT_FIT_HALF_WINDOW,
    _DEFAULT_PEAK_ADC,
    _TS_RE,
    _choose_default_run_dir,
    _crystal_ball_with_offset,
    _fit_crystal_ball_peak,
    _fit_gaussian_peak,
    _gaussian_with_offset,
    _parse_spectrum,
    _parse_timestamp,
)


# For TP-style high/low split fits on the long channels.
# User convention: treat "high" as >= 2700 ADC.
_HIGH_SPLIT_ADC = 2700.0
# Do not fit the long-channel high region above this cap.
_HIGH_MAX_ADC = 5000.0
# Avoid the near-zero pedestal dominating the "low" peak search on long channels.
_LOW_MIN_ADC = 1000.0
_LOW_FIT_HALF_WINDOW = 500.0

# Test pulse peak region (CH2): focus around ~300-320 ADC.
_TP_MIN_ADC = 300
_TP_MAX_ADC = 320
_TP_FIT_HALF_WINDOW = 10.0
_TP_MIN = 310.3
_TP_MAX = 310.5


@dataclass
class FitPoint:
    time: datetime
    mean: float
    mean_err: Optional[float]
    sigma: Optional[float]


def _format_gauss_fit(popt: np.ndarray, pcov: Optional[np.ndarray]) -> str:
    # popt = [A, mu, sigma, C]
    _A, mu, sigma, _C = popt
    return f"mean={mu:.2f} ± {abs(sigma):.2f} ADC"


def _format_cb_fit(
    popt: np.ndarray,
    pcov: Optional[np.ndarray],
    *,
    include_alpha_n: bool = True,
) -> str:
    # popt = [A, mu, sigma, alpha, n, C]
    _A, mu, sigma, alpha, n, _C = popt
    parts: List[str] = [f"mean={mu:.2f} ± {abs(sigma):.2f} ADC"]
    return ", ".join(parts)


def _fit_model_with_fallback(
    *,
    model: str,
    x: np.ndarray,
    y: np.ndarray,
    peak_guess: float,
    half_window: float,
) -> Optional[Tuple[str, np.ndarray, Optional[np.ndarray]]]:
    """Fit and return (model_used, popt, pcov). Tries requested model then the alternative."""

    def _try_gauss() -> Optional[Tuple[str, np.ndarray, Optional[np.ndarray]]]:
        fit = _fit_gaussian_peak(x, y, peak_guess=peak_guess, half_window=half_window)
        if fit is None:
            return None
        popt, pcov = fit
        return "gaussian", popt, pcov

    def _try_cb() -> Optional[Tuple[str, np.ndarray, Optional[np.ndarray]]]:
        fit = _fit_crystal_ball_peak(x, y, peak_guess=peak_guess, half_window=half_window)
        if fit is None:
            return None
        popt, pcov = fit
        return "crystalball", popt, pcov

    if model == "crystalball":
        out = _try_cb()
        return out if out is not None else _try_gauss()
    out = _try_gauss()
    return out if out is not None else _try_cb()


def _series_keys_for_channels(channels: Sequence[int]) -> List[str]:
    keys: List[str] = []
    for ch in channels:
        if int(ch) in (0, 1):
            keys.append(f"CH{int(ch)}_high")
            keys.append(f"CH{int(ch)}_low")
        else:
            keys.append(f"CH{int(ch)}")
    return keys


def _parse_datetime(dt_str: str) -> Optional[datetime]:
    if not dt_str:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(dt_str.strip(), fmt)
        except ValueError:
            continue
    return None


def _resolve_plot_window(args_start: Optional[str], args_end: Optional[str]) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Resolve plot start/end from CLI args or CAEN_PLOT_START/CAEN_PLOT_END."""
    start = _parse_datetime(args_start) if args_start else None
    end = _parse_datetime(args_end) if args_end else None

    env_start = os.getenv("CAEN_PLOT_START")
    env_end = os.getenv("CAEN_PLOT_END")

    if env_start and start is None:
        start = _parse_datetime(env_start)
        if start is None:
            print(f"Could not parse CAEN_PLOT_START='{env_start}' (expected YYYY-MM-DD HH:MM[:SS])")
    if env_end and end is None:
        end = _parse_datetime(env_end)
        if end is None:
            print(f"Could not parse CAEN_PLOT_END='{env_end}' (expected YYYY-MM-DD HH:MM[:SS])")

    if start is None:
        start = PLOT_START

    return start, end


def _load_temperature_series(
    csv_path: str,
    start_time: Optional[datetime],
    end_time: Optional[datetime],
) -> Tuple[List[datetime], List[float]]:
    if not csv_path or not os.path.exists(csv_path):
        return [], []
    try:
        df = pd.read_csv(csv_path, comment="#", header=None)
    except Exception:
        return [], []
    if df.shape[1] < 2:
        return [], []
    ts_raw = df.iloc[:, 0]
    temp_raw = df.iloc[:, 1]
    ts_parsed = pd.to_datetime(
        ts_raw.astype(str).str.strip(),
        format="%Y/%m/%d %H:%M:%S.%f",
        errors="coerce",
    )
    temp_vals = pd.to_numeric(temp_raw, errors="coerce")
    df_clean = pd.DataFrame({"timestamp": ts_parsed, "temperature": temp_vals}).dropna(
        subset=["timestamp", "temperature"]
    )
    times: List[datetime] = []
    temps: List[float] = []
    for _, row in df_clean.iterrows():
        ts = row["timestamp"].to_pydatetime()
        if start_time and ts < start_time:
            continue
        if end_time and ts > end_time:
            continue
        times.append(ts)
        temps.append(float(row["temperature"]))
    return times, temps


def _overlay_temperature(ax: Axes, temp_times: List[datetime], temp_vals: List[float], label_tracker: Dict[str, bool]) -> None:
    if not temp_times:
        return
    twin = ax.twinx()
    temp_times_arr = np.array(temp_times, dtype="datetime64[ns]")
    temp_vals_arr = -np.array(temp_vals, dtype=float)
    twin.plot(
        temp_times_arr,
        temp_vals_arr,
        color="tab:red",
        alpha=0.6,
        linewidth=1.0,
        label="Ambient temperature (inverted)",
    )
    if not label_tracker.get("temp_label"):
        twin.set_ylabel("Ambient temp [C] (inverted)", color="tab:red")
        label_tracker["temp_label"] = True
    twin.tick_params(axis="y", labelcolor="tab:red")
    twin.grid(False)


def _mean_err_from_pcov(pcov: Optional[np.ndarray], mean_index: int) -> Optional[float]:
    if pcov is None:
        return None
    if pcov.ndim != 2 or pcov.shape[0] <= mean_index or pcov.shape[1] <= mean_index:
        return None
    val = pcov[mean_index, mean_index]
    if not np.isfinite(val) or val < 0:
        return None
    return float(np.sqrt(val))


def _fit_peak_in_region(
    *,
    ch: int,
    x_adc: np.ndarray,
    y: np.ndarray,
    fit_model_long: str,
    region: str,
) -> Optional[Tuple[float, Optional[float], Optional[float]]]:
    """Fit a peak and return (mean, mean_err, sigma).

    region:
      - "high": x >= _HIGH_SPLIT_ADC (CH0/CH1 only)
      - "low" : _LOW_MIN_ADC <= x < _HIGH_SPLIT_ADC (CH0/CH1 only)
      - "full": whole spectrum (used for CH2)
    """

    region = region.lower().strip()
    if region not in {"high", "low", "full"}:
        raise ValueError(f"Unknown region: {region}")

    if region == "high":
        mask = (x_adc >= _HIGH_SPLIT_ADC) & (x_adc <= _HIGH_MAX_ADC)
    elif region == "low":
        mask = (x_adc < _HIGH_SPLIT_ADC) & (x_adc >= _LOW_MIN_ADC)
    else:
        # For CH2, constrain to the TP peak region around ~300-320 ADC.
        if int(ch) == 2:
            mask = (x_adc >= _TP_MIN_ADC) & (x_adc <= _TP_MAX_ADC)
        else:
            mask = np.isfinite(x_adc)

    if not np.any(mask):
        return None

    x_r = x_adc[mask]
    y_r = y[mask]

    if int(ch) == 2:
        model = "gaussian"
    else:
        model = fit_model_long

    # Peak guess & fit window.
    if region == "low":
        # Discover the low-peak position inside the low region.
        idx = int(np.argmax(y_r))
        peak_guess = float(x_r[idx])
        half_window = float(_LOW_FIT_HALF_WINDOW)
    else:
        # For high/full, allow peak position to move measurement-to-measurement.
        # (This is particularly important for CH2 where the TP peak can shift.)
        idx = int(np.argmax(y_r))
        peak_guess = float(x_r[idx])
        if int(ch) == 2:
            half_window = float(_TP_FIT_HALF_WINDOW)
        else:
            half_window = float(_DEFAULT_FIT_HALF_WINDOW.get(int(ch), 200.0))

    def _try_crystalball() -> Optional[Tuple[float, Optional[float], Optional[float]]]:
        fit = _fit_crystal_ball_peak(x_r, y_r, peak_guess=peak_guess, half_window=half_window)
        if fit is None:
            return None
        popt, pcov = fit
        _A, mean, sigma, _alpha, _n, _C = popt
        mean_err = _mean_err_from_pcov(pcov, 1)
        return float(mean), mean_err, float(abs(sigma))

    def _try_gaussian() -> Optional[Tuple[float, Optional[float], Optional[float]]]:
        fit = _fit_gaussian_peak(x_r, y_r, peak_guess=peak_guess, half_window=half_window)
        if fit is None:
            return None
        popt, pcov = fit
        _A, mean, sigma, _C = popt
        mean_err = _mean_err_from_pcov(pcov, 1)
        return float(mean), mean_err, float(abs(sigma))

    # Try requested model first, then fall back to the alternative.
    if model == "crystalball":
        out = _try_crystalball()
        if out is not None:
            return out
        return _try_gaussian()

    out = _try_gaussian()
    if out is not None:
        return out
    return _try_crystalball()


def _iter_complete_measurements_in_run_dir(
    run_dir: str,
    channels: Sequence[int],
    timestamp_filter: Optional[str],
) -> List[Tuple[datetime, Dict[int, str]]]:
    """Return all (timestamp, {ch: path}) measurements that have all channels.

    CAEN files can have small timestamp skews between channels (often ~1s).
    We therefore match channels by nearest timestamps within a tolerance.
    """

    raw_dir = os.path.join(run_dir, "RAW")
    if not os.path.isdir(raw_dir):
        return []

    required = [int(c) for c in channels]
    if not required:
        return []

    # Collect all candidate files per channel with parsed datetimes.
    per_ch: Dict[int, List[Tuple[datetime, str]]] = {ch: [] for ch in required}
    for name in os.listdir(raw_dir):
        m = _CH_FILE_RE.match(name)
        if not m:
            continue
        ch = int(m.group(1))
        if ch not in per_ch:
            continue
        suffix = m.group(2)
        ts_matches = _TS_RE.findall(suffix)
        if not ts_matches:
            continue
        ts = ts_matches[-1]
        dt = _parse_timestamp(ts)
        if dt is None:
            continue
        per_ch[ch].append((dt, os.path.join(raw_dir, name)))

    for ch in list(per_ch.keys()):
        per_ch[ch].sort(key=lambda t: t[0])

    # If any required channel has no files, nothing to match.
    if any(len(per_ch[ch]) == 0 for ch in required):
        return []

    tolerance_seconds = 2.0

    def _closest_path(ch: int, target: datetime) -> Optional[str]:
        items = per_ch[ch]
        times = [t for t, _p in items]
        idx = bisect.bisect_left(times, target)
        candidates: List[Tuple[float, str]] = []
        if 0 <= idx < len(items):
            dt, p = items[idx]
            candidates.append((abs((dt - target).total_seconds()), p))
        if idx - 1 >= 0:
            dt, p = items[idx - 1]
            candidates.append((abs((dt - target).total_seconds()), p))
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0])
        if candidates[0][0] <= tolerance_seconds:
            return candidates[0][1]
        return None

    ref_ch = required[0]
    out: List[Tuple[datetime, Dict[int, str]]] = []
    seen: set[Tuple[int, ...]] = set()

    for ref_dt, ref_path in per_ch[ref_ch]:
        ch_map: Dict[int, str] = {ref_ch: ref_path}
        ok = True
        for ch in required:
            if ch == ref_ch:
                continue
            p = _closest_path(ch, ref_dt)
            if p is None:
                ok = False
                break
            ch_map[ch] = p
        if not ok:
            continue

        # Deduplicate by the tuple of selected paths.
        key = tuple(sorted(hash(v) for v in ch_map.values()))
        if key in seen:
            continue
        seen.add(key)

        # Apply timestamp filter at the measurement level (allowing per-channel skews).
        if timestamp_filter is not None:
            if not any(timestamp_filter in os.path.basename(p) for p in ch_map.values()):
                continue

        out.append((ref_dt, ch_map))

    out.sort(key=lambda t: t[0])
    return out


def _iter_channel_files_in_run_dir(
    run_dir: str,
    channels: Sequence[int],
    timestamp_filter: Optional[str],
) -> List[Tuple[datetime, int, str]]:
    """Return a flat list of (timestamp, ch, path) for all matching RAW files.

    This is used for "allow-partial" mode, where we fit whichever channels exist
    (instead of requiring a complete CH0/CH1/CH2 set per timestamp).
    """

    raw_dir = os.path.join(run_dir, "RAW")
    if not os.path.isdir(raw_dir):
        return []

    allowed = {int(c) for c in channels}
    out: List[Tuple[datetime, int, str]] = []

    for name in os.listdir(raw_dir):
        m = _CH_FILE_RE.match(name)
        if not m:
            continue
        ch = int(m.group(1))
        if ch not in allowed:
            continue

        suffix = m.group(2)
        if timestamp_filter is not None and timestamp_filter not in suffix:
            continue

        ts_matches = _TS_RE.findall(suffix)
        if not ts_matches:
            continue
        ts = ts_matches[-1]
        dt = _parse_timestamp(ts)
        if dt is None:
            continue
        out.append((dt, ch, os.path.join(raw_dir, name)))

    out.sort(key=lambda t: t[0])
    return out


def _fit_one_channel(
    ch: int,
    path: str,
    fit_model_long: str,
) -> Optional[FitPoint]:
    spectrum = _parse_spectrum(path)
    x_adc = spectrum.adc.astype(float)
    mask = spectrum.adc != 0
    x_adc = x_adc[mask]
    y = spectrum.counts[mask]

    if ch not in _DEFAULT_PEAK_ADC:
        return None

    peak_guess = _DEFAULT_PEAK_ADC[ch]
    half_window = _DEFAULT_FIT_HALF_WINDOW.get(ch, 200.0)

    # Keep current CAEN convention:
    # - CH2 (Test Pulse) is Gaussian
    # - CH0/CH1 use CrystalBall unless overridden by caller
    if ch == 2:
        model = "gaussian"
    else:
        model = fit_model_long

    if model == "crystalball":
        fit = _fit_crystal_ball_peak(x_adc, y, peak_guess=peak_guess, half_window=half_window)
        if fit is None:
            return None
        popt, _ = fit
        _amplitude, mean, sigma, _alpha, _n, _offset = popt
        return FitPoint(time=datetime.min, mean=float(mean), mean_err=None, sigma=float(abs(sigma)))

    fit = _fit_gaussian_peak(x_adc, y, peak_guess=peak_guess, half_window=half_window)
    if fit is None:
        return None
    popt, _ = fit
    _amplitude, mean, sigma, _offset = popt
    return FitPoint(time=datetime.min, mean=float(mean), mean_err=None, sigma=float(abs(sigma)))


def _plot_measurement_png(
    plots_dir: str,
    measurement_time: datetime,
    ch_map: Dict[int, str],
    channels: Sequence[int],
    fit_model_long: str,
) -> Optional[str]:
    """Save a per-measurement split plot (TP-style): high, low, and test pulse."""

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(11, 10), sharex=False)
    ax_high, ax_low, ax_tp = axes

    colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}

    # Long channels: overlay CH0/CH1 in the high and low regions.
    for ch in (0, 1):
        if int(ch) not in channels:
            continue
        path = ch_map.get(int(ch))
        if not path:
            continue

        spectrum = _parse_spectrum(path)
        x_adc = spectrum.adc.astype(float)
        mask_nonzero = spectrum.adc != 0
        x_adc = x_adc[mask_nonzero]
        y = spectrum.counts[mask_nonzero]

        ch_label = _CHANNEL_LABELS.get(int(ch), f"CH{ch}")

        # High region
        mask_high = (x_adc >= _HIGH_SPLIT_ADC) & (x_adc <= _HIGH_MAX_ADC)
        if np.any(mask_high):
            ax_high.step(
                x_adc[mask_high],
                y[mask_high],
                where="mid",
                linewidth=1.0,
                color=colors.get(int(ch), None),
                label=ch_label,
            )
            x_r = x_adc[mask_high]
            y_r = y[mask_high]
            peak_guess = float(x_r[int(np.argmax(y_r))])
            half_window = float(_DEFAULT_FIT_HALF_WINDOW.get(int(ch), 200.0))

            out = _fit_model_with_fallback(model=fit_model_long, x=x_r, y=y_r, peak_guess=peak_guess, half_window=half_window)
            if out is not None:
                model_used, popt, pcov = out
                x_fit = np.linspace(float(peak_guess - half_window), float(peak_guess + half_window), 2000)
                if model_used == "crystalball":
                    y_fit = _crystal_ball_with_offset(x_fit, *popt)
                    fit_label = f"{ch_label} fit ({_format_cb_fit(popt, pcov, include_alpha_n=True)})"
                else:
                    y_fit = _gaussian_with_offset(x_fit, *popt)
                    fit_label = f"{ch_label} fit ({_format_gauss_fit(popt, pcov)})"
                ax_high.plot(
                    x_fit,
                    y_fit,
                    color="k",
                    linestyle="--",
                    label=fit_label,
                )

        # Low region
        mask_low = (x_adc < _HIGH_SPLIT_ADC) & (x_adc >= _LOW_MIN_ADC)
        if np.any(mask_low):
            ax_low.step(
                x_adc[mask_low],
                y[mask_low],
                where="mid",
                linewidth=1.0,
                color=colors.get(int(ch), None),
                label=f"{ch_label} raw",
            )
            x_r = x_adc[mask_low]
            y_r = y[mask_low]
            peak_guess = float(x_r[int(np.argmax(y_r))])
            half_window = float(_LOW_FIT_HALF_WINDOW)

            out = _fit_model_with_fallback(model=fit_model_long, x=x_r, y=y_r, peak_guess=peak_guess, half_window=half_window)
            if out is not None:
                model_used, popt, pcov = out
                x_fit = np.linspace(float(peak_guess - half_window), float(peak_guess + half_window), 2000)
                if model_used == "crystalball":
                    y_fit = _crystal_ball_with_offset(x_fit, *popt)
                    fit_label = f"{ch_label} fit ({_format_cb_fit(popt, pcov, include_alpha_n=False)})"
                else:
                    y_fit = _gaussian_with_offset(x_fit, *popt)
                    fit_label = f"{ch_label} fit ({_format_gauss_fit(popt, pcov)})"
                ax_low.plot(
                    x_fit,
                    y_fit,
                    color="k",
                    linestyle="--",
                    label=fit_label,
                )

    ax_high.set_xlabel("Bin center [ADC]")
    ax_high.set_ylabel("Counts")
    ax_high.set_title("Test Pulse response")
    ax_high.grid(True, alpha=0.3)
    ax_high.set_xlim(left=_HIGH_SPLIT_ADC, right=_HIGH_MAX_ADC)
    ax_high.legend(loc="best")

    ax_low.set_xlabel("Bin center [ADC]")
    ax_low.set_ylabel("Counts")
    ax_low.set_title(f"Purity monitor")
    ax_low.grid(True, alpha=0.3)
    ax_low.set_xlim(left=_LOW_MIN_ADC, right=_HIGH_SPLIT_ADC)
    ax_low.legend(loc="best")

    # Test pulse channel (CH2): fit full spectrum (Gaussian by convention).
    if 2 in [int(c) for c in channels]:
        path = ch_map.get(2)
        if path:
            spectrum = _parse_spectrum(path)
            x_adc = spectrum.adc.astype(float)
            mask_nonzero = spectrum.adc != 0
            x_adc = x_adc[mask_nonzero]
            y = spectrum.counts[mask_nonzero]
            ax_tp.step(
                x_adc,
                y,
                where="mid",
                linewidth=1.0,
                color=colors.get(2, None),
                label=_CHANNEL_LABELS.get(2, "Test pulse"),
            )
            # Constrain the TP fit around ~300-320 ADC.
            tp_mask = (x_adc >= _TP_MIN_ADC) & (x_adc <= _TP_MAX_ADC)
            peak_guess: Optional[float] = None
            half_window: Optional[float] = None
            if np.any(tp_mask):
                x_tp = x_adc[tp_mask]
                y_tp = y[tp_mask]
                idx = int(np.argmax(y_tp))
                peak_guess = float(x_tp[idx])
                half_window = float(_TP_FIT_HALF_WINDOW)
                fit = _fit_gaussian_peak(x_tp, y_tp, peak_guess=peak_guess, half_window=half_window)
            else:
                fit = None
            if fit is not None:
                popt, pcov = fit
                assert peak_guess is not None
                assert half_window is not None
                x_fit = np.linspace(float(peak_guess - half_window), float(peak_guess + half_window), 2000)
                y_fit = _gaussian_with_offset(x_fit, *popt)
                ax_tp.plot(
                    x_fit,
                    y_fit,
                    color="k",
                    linestyle="--",
                    label=f"Peak fit ({_format_gauss_fit(popt, pcov)})",
                )

    ax_tp.set_xlim(_TP_MIN_ADC, _TP_MAX_ADC)
    ax_tp.set_ylim(bottom=0)
    ax_tp.set_xlabel("Bin center [ADC]")
    ax_tp.set_ylabel("Counts")
    ax_tp.set_title("Test pulse")
    ax_tp.grid(True, alpha=0.3)
    ax_tp.legend(loc="best")

    fig.suptitle(measurement_time.strftime("%Y-%m-%d %H:%M:%S"))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, f"caen_fit_{measurement_time.strftime('%Y%m%d_%H%M%S')}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _write_csv(
    out_path: str,
    series: Dict[str, List[FitPoint]],
    series_keys: Sequence[str],
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # Build a time-indexed table.
    all_times = sorted({p.time for k in series_keys for p in series.get(str(k), [])})
    by_key: Dict[Tuple[datetime, str], FitPoint] = {}
    for k in series_keys:
        for p in series.get(str(k), []):
            by_key[(p.time, str(k))] = p

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["timestamp_iso"]
        for k in series_keys:
            # Human-friendly prefixes.
            if k.startswith("CH0"):
                prefix = k.replace("CH0", "CH0_InnerLong")
            elif k.startswith("CH1"):
                prefix = k.replace("CH1", "CH1_OuterLong")
            elif k == "CH2":
                label = _CHANNEL_LABELS.get(2)
                prefix = f"CH2_{label}" if label else "CH2"
            else:
                prefix = k
            header.append(f"{prefix}_mean_adc")
            # User convention: use fitted sigma as the 'error' column.
            header.append(f"{prefix}_sigma_adc")
            # Keep the statistical fit uncertainty on the mean as an extra column.
            header.append(f"{prefix}_mean_err_adc")
        writer.writerow(header)
        for t in all_times:
            row = [t.isoformat()]
            for k in series_keys:
                p = by_key.get((t, str(k)))
                row.append(f"{p.mean:.6g}" if p is not None else "")
                row.append(f"{p.sigma:.6g}" if p is not None and p.sigma is not None else "")
                row.append(f"{p.mean_err:.6g}" if p is not None and p.mean_err is not None else "")
            writer.writerow(row)


def main() -> int:
    caen_root_env = os.environ.get("CAEN_DATA_DIR")
    if caen_root_env is None:
        repo_root = Path(__file__).resolve().parents[2]
        caen_root_env = str((repo_root / "data" / "caen").resolve())

    parser = argparse.ArgumentParser(description="Fit CAEN peaks vs time (TP-style plot)")
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Run dir containing RAW/ or a CAEN root dir containing multiple runs",
    )
    parser.add_argument("--channels", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Optional timestamp filter like 20260123_154850 (restricts to matching RAW files)",
    )
    parser.add_argument(
        "--fit-model-long",
        choices=["gaussian", "crystalball"],
        default="crystalball",
        help="Model for CH0/CH1 peak fitting (CH2 is always Gaussian).",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Force scanning all run subdirectories under the CAEN root (instead of picking one default run).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Fit whichever channel files exist (do not require complete CH0+CH1+CH2 sets per timestamp).",
    )
    parser.add_argument("--start", default=None, help="YYYY-MM-DD HH:MM[:SS]")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD HH:MM[:SS]")
    parser.add_argument("--plots-dir", default="plots_caen", help="Output directory for PNG/CSV")
    parser.add_argument("--save", default=None, help="Save mean-vs-time PNG to this path (default under --plots-dir)")
    parser.add_argument("--no-measurement-plots", action="store_true", help="Skip per-measurement spectrum PNGs")
    parser.add_argument("--csv-out", default=None, help="CSV output path (default under --plots-dir)")
    parser.add_argument("--show", action="store_true", help="Show figures interactively")

    args = parser.parse_args()

    debug = os.environ.get("CAEN_DEBUG", "0") == "1"

    start, end = _resolve_plot_window(args.start, args.end)

    # Decide which run dirs to scan.
    # Default behavior: scan the whole CAEN root so we truly iterate over all files.
    path = args.path or caen_root_env

    def _discover_run_dirs(root: str) -> List[str]:
        if os.path.isdir(os.path.join(root, "RAW")):
            return [root]
        if not os.path.isdir(root):
            raise ValueError(f"Path does not exist: {root}")

        found: List[str] = []
        # Walk recursively so we don't miss nested run structures.
        for dirpath, dirnames, _filenames in os.walk(root):
            # Don't descend into RAW/ itself.
            if "RAW" in dirnames:
                found.append(dirpath)
                # Still continue searching siblings.
            dirnames[:] = [d for d in dirnames if d != "RAW"]
        return sorted(set(found))

    run_dirs = _discover_run_dirs(path)

    if not run_dirs:
        raise ValueError(f"No run directories found under: {path}")

    series_keys = _series_keys_for_channels(args.channels)
    series: Dict[str, List[FitPoint]] = {k: [] for k in series_keys}

    if debug:
        print(f"[debug] scanning {len(run_dirs)} run dir(s)")

    for run_dir in run_dirs:
        if args.allow_partial:
            files = _iter_channel_files_in_run_dir(run_dir, args.channels, args.timestamp)
            if debug:
                print(f"[debug] {run_dir}: {len(files)} channel file(s) (allow-partial)")

            for dt, ch, p in files:
                if start and dt < start:
                    continue
                if end and dt > end:
                    continue

                spectrum = _parse_spectrum(p)
                x_adc = spectrum.adc.astype(float)
                mask_nonzero = spectrum.adc != 0
                x_adc = x_adc[mask_nonzero]
                y = spectrum.counts[mask_nonzero]

                if int(ch) in (0, 1):
                    fit_h = _fit_peak_in_region(ch=int(ch), x_adc=x_adc, y=y, fit_model_long=args.fit_model_long, region="high")
                    if debug:
                        print(f"[debug] {dt.isoformat()} CH{int(ch)} high fit: {fit_h}")
                    if fit_h is not None:
                        mean, mean_err, sigma = fit_h
                        series[f"CH{int(ch)}_high"].append(FitPoint(time=dt, mean=mean, mean_err=mean_err, sigma=sigma))

                    fit_l = _fit_peak_in_region(ch=int(ch), x_adc=x_adc, y=y, fit_model_long=args.fit_model_long, region="low")
                    if debug:
                        print(f"[debug] {dt.isoformat()} CH{int(ch)} low fit: {fit_l}")
                    if fit_l is not None:
                        mean, mean_err, sigma = fit_l
                        series[f"CH{int(ch)}_low"].append(FitPoint(time=dt, mean=mean, mean_err=mean_err, sigma=sigma))

                elif int(ch) == 2:
                    fit_tp = _fit_peak_in_region(ch=2, x_adc=x_adc, y=y, fit_model_long=args.fit_model_long, region="full")
                    if debug:
                        print(f"[debug] {dt.isoformat()} CH2 full fit: {fit_tp}")
                    if fit_tp is not None:
                        mean, mean_err, sigma = fit_tp
                        series["CH2"].append(FitPoint(time=dt, mean=mean, mean_err=mean_err, sigma=sigma))

                else:
                    fit_full = _fit_peak_in_region(ch=int(ch), x_adc=x_adc, y=y, fit_model_long=args.fit_model_long, region="full")
                    if fit_full is not None:
                        mean, mean_err, sigma = fit_full
                        series[f"CH{int(ch)}"].append(FitPoint(time=dt, mean=mean, mean_err=mean_err, sigma=sigma))

        else:
            measurements = _iter_complete_measurements_in_run_dir(run_dir, args.channels, args.timestamp)
            if debug:
                print(f"[debug] {run_dir}: {len(measurements)} complete measurement(s)")
            for dt, ch_map in measurements:
                if start and dt < start:
                    continue
                if end and dt > end:
                    continue

                if debug:
                    print(f"[debug] measurement {dt.isoformat()} files: {[os.path.basename(p) for p in ch_map.values()]}")

                # Optional per-measurement diagnostic plot, similar to TP's split plot.
                if not args.no_measurement_plots:
                    out_png = _plot_measurement_png(
                        args.plots_dir,
                        dt,
                        ch_map,
                        args.channels,
                        fit_model_long=args.fit_model_long,
                    )
                    if out_png is not None:
                        print(f"Saved plot: {out_png}")

                for ch in args.channels:
                    p = ch_map.get(int(ch))
                    if not p:
                        continue
                    spectrum = _parse_spectrum(p)
                    x_adc = spectrum.adc.astype(float)
                    mask_nonzero = spectrum.adc != 0
                    x_adc = x_adc[mask_nonzero]
                    y = spectrum.counts[mask_nonzero]

                    # CH0/CH1: fit both high and low regions.
                    if int(ch) in (0, 1):
                        fit_h = _fit_peak_in_region(ch=int(ch), x_adc=x_adc, y=y, fit_model_long=args.fit_model_long, region="high")
                        if debug:
                            print(f"[debug] CH{int(ch)} high fit: {fit_h}")
                        if fit_h is not None:
                            mean, mean_err, sigma = fit_h
                            series[f"CH{int(ch)}_high"].append(FitPoint(time=dt, mean=mean, mean_err=mean_err, sigma=sigma))

                        fit_l = _fit_peak_in_region(ch=int(ch), x_adc=x_adc, y=y, fit_model_long=args.fit_model_long, region="low")
                        if debug:
                            print(f"[debug] CH{int(ch)} low fit: {fit_l}")
                        if fit_l is not None:
                            mean, mean_err, sigma = fit_l
                            series[f"CH{int(ch)}_low"].append(FitPoint(time=dt, mean=mean, mean_err=mean_err, sigma=sigma))

                    # CH2: full-spectrum gaussian (TP test pulse channel analogue).
                    elif int(ch) == 2:
                        fit_tp = _fit_peak_in_region(ch=2, x_adc=x_adc, y=y, fit_model_long=args.fit_model_long, region="full")
                        if debug:
                            print(f"[debug] CH2 full fit: {fit_tp}")
                        if fit_tp is not None:
                            mean, mean_err, sigma = fit_tp
                            series["CH2"].append(FitPoint(time=dt, mean=mean, mean_err=mean_err, sigma=sigma))

                    else:
                        # Fallback: treat as a single full-spectrum fit for other channels.
                        fit_full = _fit_peak_in_region(ch=int(ch), x_adc=x_adc, y=y, fit_model_long=args.fit_model_long, region="full")
                        if fit_full is not None:
                            mean, mean_err, sigma = fit_full
                            series[f"CH{int(ch)}"].append(FitPoint(time=dt, mean=mean, mean_err=mean_err, sigma=sigma))

    any_points = any(series[k] for k in series)
    if not any_points:
        print("No fit points found.")
        return 0

    # CSV dump (TP-style).
    csv_out = args.csv_out
    if not csv_out:
        csv_out = os.path.join(args.plots_dir, "caen_fit_means_data.csv")
    _write_csv(csv_out, series, series_keys)
    print(f"Wrote plot data: {csv_out}")

    # Mean-vs-time plot (match TP_analysis_CrystallBallFit.py formatting).
    desired_order = ["CH0_high", "CH1_high", "CH0_low", "CH1_low", "CH2"]
    plot_keys: List[str] = []
    for k in desired_order:
        if k in series_keys:
            plot_keys.append(k)
    for k in series_keys:
        if k not in plot_keys:
            plot_keys.append(str(k))

    # Hard-coded titles/colors (TP-style) for known CAEN series.
    _TITLE_BY_KEY: Dict[str, str] = {
        "CH0_high": "Inner long PM TP response",
        "CH1_high": "Outer long PM TP response",
        "CH0_low": "Inner long PM",
        "CH1_low": "Outer long PM",
        "CH2": "Test pulse",
    }
    _COLOR_BY_KEY: Dict[str, str] = {
        "CH0_high": "tab:blue",
        "CH1_high": "tab:orange",
        "CH0_low": "tab:blue",
        "CH1_low": "tab:orange",
        "CH2": "tab:purple",
    }

    n = len(plot_keys)
    fig_h = 13 if n == 5 else max(3, 2.8 * n)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, fig_h), sharex=True)
    if isinstance(axes, np.ndarray):
        axes_list = axes.ravel().tolist()
    else:
        axes_list = [axes]
    start_time_candidates: List[datetime] = []
    for pts in series.values():
        if pts:
            start_time_candidates.append(min(pts, key=lambda fp: fp.time).time)
    if start:
        start_time_candidates.append(start)
    start_time = min(start_time_candidates) if start_time_candidates else None
    temp_times, temp_vals = _load_temperature_series(_TEMP_CSV_PATH, start_time, end)
    temp_label_tracker: Dict[str, bool] = {}

    def _plot_series(
        ax: Axes,
        pts: List[FitPoint],
        color: str,
        title: str,
        ylim: Optional[Tuple[float, float]] = None,
        show_band: bool = False,
    ) -> None:
        if not pts:
            ax.set_visible(False)
            return
        ax.set_title(title)
        pts = sorted(pts, key=lambda fp: fp.time)
        times = [p.time for p in pts]
        values = [p.mean for p in pts]
        errs = np.array(
            [
                (p.sigma if p.sigma is not None else p.mean_err)
                if (p.sigma is not None or p.mean_err is not None)
                else np.nan
                for p in pts
            ],
            dtype=float,
        )
        ax.plot(times, values, marker="o", markersize=3.5, linestyle="none", color=color)  # type: ignore[arg-type]
        finite_mask = np.isfinite(errs)
        if finite_mask.any() and show_band:
            vals_arr = np.array(values, dtype=float)
            band = errs
            lower = vals_arr - band
            upper = vals_arr + band
            ax.fill_between(times, lower, upper, color=color, alpha=0.12, linewidth=0)  # type: ignore[arg-type]
        ax.set_ylabel("Mean [ADC]")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        _overlay_temperature(ax, temp_times, temp_vals, temp_label_tracker)

    for ax, k in zip(axes_list, plot_keys):
        ylim = (_TP_MIN, _TP_MAX) if k == "CH2" else None
        _plot_series(
            ax,
            series.get(str(k), []),
            _COLOR_BY_KEY.get(str(k), "tab:gray"),
            _TITLE_BY_KEY.get(str(k), str(k)),
            ylim=ylim,
            show_band=False,  # Set to False to disable ±σ band
        )

    axes_list[-1].set_xlabel("Measurement time")
    axes_list[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()

    save_path = args.save
    if not save_path:
        save_path = os.path.join(args.plots_dir, "caen_fit_means_vs_time.png")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Saved mean-vs-time plot: {save_path}")

    show = args.show or (os.environ.get("CAEN_SHOW", "0") == "1")
    if show:
        plt.show()
    plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
