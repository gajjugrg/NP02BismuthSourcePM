"""Test Pulse (TP) analysis with Crystal Ball fits.

This is a *script* (not a library module). It scans TP histogram measurements,
fits peak means, and produces both per-measurement plots and a mean-vs-time plot.

Supported input layouts
-----------------------
1) New (2026+): single CSV per measurement
     - File name encodes time: Record_<YEAR>_<MONTH>_<DAY>_<HOUR>_<MINUTE>.csv
         Example: Record_2026_Jan_24_07_32.csv
     - Columns: binCenter, F1, F2, F3, F4 (F4 may be empty)

2) Legacy: one directory per measurement containing histogram files
     - F1.txt, F2.txt, F3.txt (etc.) with columns BinCenter, Population

Fit model summary
-----------------
- F1/F2 ("TP response", high region): Crystal Ball fit above 1.2 V.
    For F1 a narrow window around the peak is used for stability.
- F1/F2 ("purity monitor", low region): Crystal Ball fit in LOW_V_MIN..LOW_V_MAX.
    F2 low fits with mean < 600 mV are rejected (noise/failed fits).
- F3 ("test pulse"): Gaussian fit.

Outputs (written under PLOTS_DIR)
--------------------------------
- split_low_cb_<timestamp>.png : per-measurement 3-panel plot (high/low/F3)
- fit_means_vs_time.png        : stacked time-series of fitted means
- fit_means_data_low_cb.csv    : CSV dump of the fitted mean values

Controlling the time window
---------------------------
- TP_PLOT_START / TP_PLOT_END (env vars) override PLOT_START/PLOT_END.
    Format: YYYY-MM-DD HH:MM[:SS]
- TP_SHOW=1 shows figures interactively (otherwise saves PNGs only).
"""
import sys
from pathlib import Path

# Allow running this script directly (e.g. `python tp/TP_analysis_CrystallBallFit.py`) by ensuring
# the `src/` directory is on sys.path so `import tp.*` works.
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
import os
import glob
import re
from datetime import datetime, timezone
from typing import Dict, Optional, List, Tuple, Union

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tp.tp_cb_core import (
    crystal_ball,
    fit_crystal_ball,
    fit_gaussian,
    gaussian,
    iter_measurement_dirs,
    load_histogram,
    load_record_series,
    parse_timestamp,
)

_REPO_DIR = Path(__file__).resolve().parents[2]

# Root directory containing measurements.
# - If NP02DATA_DIR is set, it is used directly.
# - Otherwise default to <repo>/np02data so the script works when run from repo root.
NP02DATA_DIR = os.environ.get('NP02DATA_DIR', str(_REPO_DIR / 'np02data'))
ROOT_DIR = NP02DATA_DIR
PLOTS_DIR = 'plots_scope'
PLOT_START = datetime(2026, 1, 24, 7, 0)
PLOT_END = None
MV_SCALE = 1000.0  # volts to millivolts
TEMP_CSV_PATH = os.environ.get('NP02_TEMP_CSV', os.path.join(NP02DATA_DIR, 'Temp_Jan26.csv'))
LOW_V_MIN = 0.55
LOW_V_MAX = 1.2
HIGH_FIT_WINDOW_F1 = 0.10  # V window around F1 high peak for CB fit

FILES = {
    'F1': 'F1.txt',  # inner long PM
    'F2': 'F2.txt',  # outer long PM
    'F3': 'F3.txt',  # test pulse
}

COLORS = {
    'F1': 'tab:blue',
    'F2': 'tab:orange',
    'F3': 'tab:green',
}

DESCRIPTIONS = {
    'F1': 'Inner long PM',
    'F2': 'Outer long PM',
    'F3': 'Test pulse',
}

DATA_LABELS = {
    # These names are used for the CSV header in write_plot_data().
    'F1_high': 'TP_Response_Inner_long_PM',
    'F2_high': 'TP_Response_Outer_long_PM',
    'F1_low': 'Inner_low_CB',
    'F2_low': 'Outer_low_CB',
    'F3': 'Test_pulse',
}

def _parse_datetime(dt_str: str) -> Optional[datetime]:
    """Parse datetime from 'YYYY-MM-DD HH:MM[:SS]' strings."""
    if not dt_str:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(dt_str.strip(), fmt)
        except ValueError:
            continue
    return None

def _resolve_plot_window() -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Resolve plot start/end from constants or env vars TP_PLOT_START / TP_PLOT_END.
    """
    start = PLOT_START
    end = PLOT_END
    env_start = os.getenv("TP_PLOT_START")
    env_end = os.getenv("TP_PLOT_END")
    if env_start:
        parsed = _parse_datetime(env_start)
        if parsed:
            start = parsed
        else:
            print(f"Could not parse TP_PLOT_START='{env_start}' (expected YYYY-MM-DD HH:MM[:SS])")
    if env_end:
        parsed = _parse_datetime(env_end)
        if parsed:
            end = parsed
        else:
            print(f"Could not parse TP_PLOT_END='{env_end}' (expected YYYY-MM-DD HH:MM[:SS])")
    return start, end
    
def _to_millivolts_scalar(value_v: Optional[float]) -> Optional[float]:
    if value_v is None or not np.isfinite(value_v):
        return None
    return value_v * MV_SCALE


def _fit_cb_curve(
    x: np.ndarray,
    y: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float, float], Optional[np.ndarray]]]:
    """Back-compat wrapper for older call sites within this file."""
    res = fit_crystal_ball(x, y)
    if res is None:
        return None
    popt, pcov = res
    x_fit = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 400)
    y_fit = crystal_ball(x_fit, *popt)
    return x_fit, y_fit, popt, pcov


MeanWithErr = Tuple[float, Optional[float]]
RowValue = Union[datetime, Optional[float]]


def _plot_high_region(ax, series: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[float, Optional[float]]]:
    plotted = False
    mean_map: Dict[str, MeanWithErr] = {}
    for label in ('F1', 'F2'):
        df = series.get(label)
        if df is None or df.empty:
            continue
        centers = df['BinCenter'].to_numpy(dtype=float)
        counts = df['Population'].to_numpy(dtype=float)
        mask = centers >= 1.2  # reuse original high cutoff
        if not mask.any():
            continue
        x = centers[mask]
        y = counts[mask]
        # Fit F1 high with a narrow window around its peak to stabilize the CB fit.
        fit_x = x
        fit_y = y
        if label == 'F1' and x.size > 3:
            peak_val = x[np.argmax(y)]
            narrow_mask = np.abs(x - peak_val) <= HIGH_FIT_WINDOW_F1
            if narrow_mask.any():
                fit_x = x[narrow_mask]
                fit_y = y[narrow_mask]
        label_text = DESCRIPTIONS[label]
        ax.step(x * MV_SCALE, y, where='mid', color=COLORS[label], label=label_text)
        fit = _fit_cb_curve(fit_x, fit_y)
        if fit is not None:
            x_fit, y_fit, (_, mean, sigma, alpha, n), pcov = fit
            mean_map[f'{label}_high'] = (mean, float(abs(sigma)))
            mean_mV = _to_millivolts_scalar(mean)
            sigma_mV = _to_millivolts_scalar(float(abs(sigma)))
            ax.plot(
                x_fit * MV_SCALE,
                y_fit,
                color='k',
                linestyle='--',
                label=(
                    f'{label_text} fit (mean={mean_mV:.2f}'
                    + (f' ± {sigma_mV:.2f}' if sigma_mV is not None else '')
                    + f' mV)'
                ),
            )
        plotted = True

    if not plotted:
        ax.set_visible(False)
        return {}

    ax.set_xlabel('Bin center [mV]')
    ax.set_ylabel('Counts')
    ax.set_title('Test Pulse response')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    return mean_map


def _plot_low_region(ax, series: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[float, Optional[float]]]:
    # Low region: raw F1 and F2, no scaling/subtraction, fit CB in 0.4-1.2 V.
    plotted = False
    mean_map: Dict[str, MeanWithErr] = {}
    for label in ('F1', 'F2'):
        df = series.get(label)
        if df is None or df.empty:
            continue
        v = df['BinCenter'].to_numpy(dtype=float)
        c = df['Population'].to_numpy(dtype=float)
        mask = (v >= LOW_V_MIN) & (v <= LOW_V_MAX)
        if not mask.any():
            continue
        x = v[mask]
        y = c[mask]
        ax.step(x * MV_SCALE, y, where='mid', color=COLORS[label], label=f"{DESCRIPTIONS[label]} raw")
        fit = _fit_cb_curve(x, y)
        if fit is not None:
            x_fit, y_fit, (_, mean, sigma, alpha, n), pcov = fit
            mean_mV = _to_millivolts_scalar(mean)
            sigma_mV = _to_millivolts_scalar(float(abs(sigma)))
            if label == 'F2' and mean_mV is not None and mean_mV < 600.0:
                ax.plot([], [], ' ', label="F2 fit rejected (<600 mV)")
            else:
                mean_map[f'{label}_low'] = (mean, float(abs(sigma)))
                ax.plot(
                    x_fit * MV_SCALE,
                    y_fit,
                    color='k',
                    linestyle='--',
                    label=(
                        f"{DESCRIPTIONS[label]} fit (mean={mean_mV:.2f}"
                        + (f' ± {sigma_mV:.2f}' if sigma_mV is not None else '')
                        + " mV)"
                    ),
                )
        plotted = True

    if not plotted:
        ax.set_visible(False)
        return {}
    ax.set_xlabel('Bin center [mV]')
    ax.set_ylabel('Counts')
    ax.set_title(f'Purity monitor ({LOW_V_MIN:.1f}-{LOW_V_MAX:.1f} V)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=LOW_V_MIN * MV_SCALE, right=LOW_V_MAX * MV_SCALE)
    ax.legend(loc='best')
    return mean_map


def _fit_f3_peak(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
    res = fit_gaussian(x, y)
    if res is None:
        return None
    (amp, mean, sigma), pcov = res
    x_fit = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 400)
    y_fit = gaussian(x_fit, amp, mean, sigma)
    return x_fit, y_fit, mean, float(abs(sigma))


def _plot_f3(ax, series: Dict[str, pd.DataFrame]) -> Optional[Tuple[float, Optional[float]]]:
    df = series.get('F3')
    if df is None or df.empty:
        ax.set_visible(False)
        return None
    x = df['BinCenter'].to_numpy(dtype=float)
    y = df['Population'].to_numpy(dtype=float)
    x_mv = x * MV_SCALE
    ax.step(x_mv, y, where='mid', color=COLORS['F3'], label=DESCRIPTIONS['F3'])
    mean_val = None
    mean_err_val: Optional[float] = None
    fit = _fit_f3_peak(x, y)
    if fit is not None:
        x_fit, y_fit, mean, sigma = fit
        mean_val = mean
        mean_err_val = sigma
        mean_mV = _to_millivolts_scalar(mean)
        sigma_mV = _to_millivolts_scalar(sigma)
        ax.plot(
            x_fit * MV_SCALE,
            y_fit,
            color='k',
            linestyle='--',
            label=(
                f'Peak fit (mean={mean_mV:.2f}'
                + (f' ± {sigma_mV:.2f}' if sigma_mV is not None else '')
                + ' mV)'
            ),
        )
    ax.set_xlim(1100, 1400)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Bin center [mV]')
    ax.set_ylabel('Counts')
    ax.set_title('Test pulse')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    return (mean_val, mean_err_val) if mean_val is not None else None


def plot_measurement(measurement_time: datetime, series: Dict[str, pd.DataFrame]) -> Dict[str, MeanWithErr]:
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=False)
    ax_high, ax_low, ax_f3 = axes

    high_means = _plot_high_region(ax_high, series)
    low_means = _plot_low_region(ax_low, series)
    f3_mean = _plot_f3(ax_f3, series)

    if not any(ax.get_visible() for ax in axes):
        plt.close(fig)
        return {}

    fig.suptitle(f"{measurement_time:%Y-%m-%d %H:%M:%S}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(
        PLOTS_DIR,
        f"scope_{measurement_time.strftime('%Y%m%d_%H%M%S')}.png"
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")

    means: Dict[str, MeanWithErr] = {}
    means.update(low_means)
    for key, value in high_means.items():
        means[key] = value
    if f3_mean is not None:
        means['F3'] = f3_mean
    return means


def _load_temperature_series(csv_path: str, start_time: Optional[datetime]) -> Tuple[List[datetime], List[float]]:
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
    ts_parsed = pd.to_datetime(ts_raw.astype(str).str.strip(), format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
    temp_vals = pd.to_numeric(temp_raw, errors="coerce")
    df_clean = pd.DataFrame({"timestamp": ts_parsed, "temperature": temp_vals}).dropna(subset=["timestamp", "temperature"])
    times = []
    temps = []
    for _, row in df_clean.iterrows():
        ts = row["timestamp"].to_pydatetime()
        if start_time and ts < start_time:
            continue
        times.append(ts)
        temps.append(float(row["temperature"]))
    return times, temps


def _overlay_temperature(ax, temp_times, temp_vals, label_tracker: Dict[str, bool]):
    if not temp_times:
        return
    twin = ax.twinx()
    twin.plot(temp_times, [-v for v in temp_vals], color='tab:red', alpha=0.6, linewidth=1.0, label='Ambient temperature (inverted)')
    if not label_tracker.get('temp_label'):
        twin.set_ylabel('Ambient temp [°C] (inverted)', color='tab:red')
        label_tracker['temp_label'] = True
    twin.tick_params(axis='y', labelcolor='tab:red')
    twin.grid(False)


def plot_fit_means(mean_records: Dict[str, List[Tuple[datetime, float, Optional[float]]]], show_band: bool = False):
    keys_present = any(mean_records.get(k) for k in DATA_LABELS.keys())
    if not keys_present:
        return

    plot_start, plot_end = _resolve_plot_window()
    filtered_records: Dict[str, List[Tuple[datetime, float, Optional[float]]]] = {}
    for key, samples in mean_records.items():
        if samples:
            filtered_records[key] = [
                (t, v, e)
                for t, v, e in samples
                if ((plot_start is None or t >= plot_start) and (plot_end is None or t <= plot_end))
            ]
        else:
            filtered_records[key] = []

    fig, axes = plt.subplots(5, 1, figsize=(10, 13), sharex=True)
    ax_f1_high, ax_f2_high, ax_f1_low, ax_f2_low, ax_f3 = axes
    start_time_candidates = []
    for samples in filtered_records.values():
        if samples:
            start_time_candidates.append(min(samples, key=lambda x: x[0])[0])
    if plot_start:
        start_time_candidates.append(plot_start)
    start_time = min(start_time_candidates) if start_time_candidates else None

    # Optional temperature overlay (right axis). This is useful for diagnosing drifts.
    temp_times, temp_vals = _load_temperature_series(TEMP_CSV_PATH, start_time)
    temp_label_tracker = {}

    def _plot_series(ax, samples, color, title, ylim=None):
        if not samples:
            ax.set_visible(False)
            return
        ax.set_title(title)
        samples = sorted(samples, key=lambda x: x[0])
        times = [t for t, _, _ in samples]
        values = [v * MV_SCALE for _, v, _ in samples]
        errs = np.array([e * MV_SCALE if e is not None else np.nan for _, _, e in samples], dtype=float)
        ax.plot(times, values, marker='o', markersize=3.5, linestyle='none', color=color)
        finite_mask = np.isfinite(errs)
        if finite_mask.any() and show_band:
            vals_arr = np.array(values, dtype=float)
            # Visualize uncertainty as a ±σ band (when covariance is available).
            band = errs
            lower = vals_arr - band
            upper = vals_arr + band
            ax.fill_between(times, lower, upper, color=color, alpha=0.12, linewidth=0)
        ax.set_ylabel('Mean [mV]')
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        _overlay_temperature(ax, temp_times, temp_vals, temp_label_tracker)

    _plot_series(ax_f1_high, filtered_records.get('F1_high', []), 'tab:blue', 'Inner long PM TP response')
    _plot_series(ax_f2_high, filtered_records.get('F2_high', []), 'tab:orange', 'Outer long PM TP response')
    _plot_series(ax_f1_low, filtered_records.get('F1_low', []), 'tab:blue', 'Inner long PM')
    _plot_series(ax_f2_low, filtered_records.get('F2_low', []), 'tab:orange', 'Outer long PM')
    _plot_series(ax_f3, filtered_records.get('F3', []), 'tab:purple', 'Test pulse', ylim=(1215, 1230))

    axes[-1].set_xlabel('Measurement time')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, 'fit_means_vs_time.png')
    fig.savefig(out_path, dpi=150)
    if os.environ.get("TP_SHOW", "0") == "1":
        plt.show()
    plt.close(fig)
    print(f"Saved mean-vs-time plot: {out_path}")


def write_plot_data(rows: List[Dict[str, RowValue]]):
    if not rows:
        return

    def _row_timestamp(row: Dict[str, RowValue]) -> datetime:
        ts = row.get('timestamp')
        return ts if isinstance(ts, datetime) else datetime.min

    rows = sorted(rows, key=_row_timestamp)
    data_path = os.path.join(PLOTS_DIR, 'fit_means_scope.csv')
    with open(data_path, 'w', newline='') as f:
        f.write("# Derived with low-region CB fits (no scaling)\n")
        f.write(f"# Generated at {datetime.now(timezone.utc).isoformat()}\n")
        writer = csv.writer(f)
        header = ['timestamp_iso']
        for k in DATA_LABELS.keys():
            header.append(DATA_LABELS[k] + '_mV')
            header.append(DATA_LABELS[k] + '_err_mV')
        writer.writerow(header)
        for row in rows:
            ts = row.get('timestamp')
            if not isinstance(ts, datetime):
                continue
            timestamp = ts.isoformat()
            values = []
            for key in DATA_LABELS.keys():
                raw_val = row.get(key)
                raw_err = row.get(f"{key}_err")
                mv_val = _to_millivolts_scalar(float(raw_val) if isinstance(raw_val, (int, float, np.floating)) else None)
                mv_err = _to_millivolts_scalar(float(raw_err) if isinstance(raw_err, (int, float, np.floating)) else None)
                values.append(f"{mv_val:.3f}" if mv_val is not None else '')
                values.append(f"{mv_err:.3f}" if mv_err is not None else '')
            writer.writerow([timestamp] + values)
    print(f"Wrote plot data: {data_path}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_start, plot_end = _resolve_plot_window()
    directories: List[str] = sorted(iter_measurement_dirs(ROOT_DIR))
    if not directories:
        print(
            f"No measurement directories found under ROOT_DIR='{ROOT_DIR}'. "
            "Check NP02DATA_DIR or run from src/."
        )
    mean_records: Dict[str, List[Tuple[datetime, float, Optional[float]]]] = {}
    data_rows: List[Dict[str, RowValue]] = []
    # Each item is either a Record_*.csv file (new format) or a legacy measurement directory.
    for measurement in directories:
        measurement_time = parse_timestamp(measurement)
        if measurement_time is None:
            continue
        if plot_start and measurement_time < plot_start:
            continue
        if plot_end and measurement_time > plot_end:
            continue
        series: Dict[str, pd.DataFrame] = {}
        if os.path.isfile(measurement) and os.path.basename(measurement).startswith("Record_") and measurement.lower().endswith(".csv"):
            series = load_record_series(measurement)
        else:
            for label, fname in FILES.items():
                df = load_histogram(os.path.join(measurement, fname))
                if df is not None:
                    series[label] = df
        if not series:
            continue
        means = plot_measurement(measurement_time, series)
        for key, (mean_val, err_val) in means.items():
            mean_records.setdefault(key, []).append((measurement_time, mean_val, err_val))
        row: Dict[str, RowValue] = {'timestamp': measurement_time}
        for key in DATA_LABELS.keys():
            if key in means:
                row[key] = means[key][0]
                row[f"{key}_err"] = means[key][1]
            else:
                row[key] = None
                row[f"{key}_err"] = None
        data_rows.append(row)
    plot_fit_means(mean_records)
    write_plot_data(data_rows)


if __name__ == '__main__':
    main()
