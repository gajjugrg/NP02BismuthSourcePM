"""
Variant of TP_analysis that treats the low-voltage region differently:
- No F1/F2 scaling/subtraction in the low region
- Fit Crystal Ball functions to F1 and F2 in 0.4- 1.2 V
- Split plots show F1/F2 high, F1/F2 low (raw), and F3 as before
- Time plot shows F1_high, F2_high, combined low (<1.2 V) for F1 and F2, and F3
"""
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
from scipy.optimize import curve_fit

NP02DATA_DIR = os.environ.get('NP02DATA_DIR', '../np02data')
ROOT_DIR = NP02DATA_DIR
PLOTS_DIR = 'plots'
PLOT_START = datetime(2025, 11, 21, 16, 00)
PLOT_END = None
MV_SCALE = 1000.0  # volts to millivolts
TEMP_CSV_PATH = os.environ.get('NP02_TEMP_CSV', os.path.join(NP02DATA_DIR, 'Temp.csv'))
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
    'F1_high': 'TP_Response_Inner_long_PM',
    'F2_high': 'TP_response_Outer_long_PM',
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


def crystal_ball(x, amplitude, mean, sigma, alpha, n):
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


def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def _fit_cb_curve(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float, float], Optional[np.ndarray]]]:
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
    lower = [0.0, x.min(), 1e-5, 0.1, 0.5]
    upper = [np.inf, x.max(), (x.max() - x.min()) * 2.0, 10.0, 50.0]
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
    x_fit = np.linspace(x.min(), x.max(), 400)
    y_fit = crystal_ball(x_fit, *popt)
    return x_fit, y_fit, tuple(popt), pcov


MONTH_MAP = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
}


def iter_measurement_dirs(root_dir: str):
    seen = set()
    pattern = f"{root_dir}/20??_[A-Za-z][a-z][a-z]/**/F1.txt"
    for f1_path in glob.iglob(pattern, recursive=True):
        directory = os.path.dirname(f1_path)
        if directory in seen:
            continue
        seen.add(directory)
        yield directory


def parse_timestamp(directory: str) -> Optional[datetime]:
    parts = directory.strip('/').split('/')
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
    second = parts[idx + 4] if len(parts) > idx + 4 else '00'
    year_str, month_word = year_month.split('_', 1)
    month_str = MONTH_MAP.get(month_word.capitalize())
    if month_str is None:
        return None
    try:
        timestamp = datetime.strptime(
            f"{year_str}-{month_str}-{int(day):02d} {int(hour):02d}:{int(minute):02d}:{int(second):02d}",
            '%Y-%m-%d %H:%M:%S'
        )
    except ValueError:
        return None
    return timestamp


def load_histogram(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, usecols=['BinCenter', 'Population'])
    except Exception:
        return None
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['BinCenter', 'Population'])
    if df.empty:
        return None
    return df.sort_values('BinCenter').reset_index(drop=True)


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
            mean_err = float(np.sqrt(pcov[1, 1])) if pcov is not None and pcov.shape[0] > 1 else None
            mean_map[f'{label}_high'] = (mean, mean_err)
            mean_mV = _to_millivolts_scalar(mean)
            mean_err_mV = _to_millivolts_scalar(mean_err) if mean_err is not None else None
            ax.plot(
                x_fit * MV_SCALE,
                y_fit,
                color=COLORS[label],
                linestyle='--',
                label=(
                    f'{label_text} fit (mean={mean_mV:.2f}'
                    + (f' ± {mean_err_mV:.2f}' if mean_err_mV is not None else '')
                    + f' mV, alpha={alpha:.2f}, n={n:.1f})'
                ),
            )
        plotted = True

    if not plotted:
        ax.set_visible(False)
        return {}

    ax.set_xlabel('Bin center [mV]')
    ax.set_ylabel('Counts')
    ax.set_title('F1/F2 raw high region')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    return mean_map


def _plot_low_region(ax, series: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[float, Optional[float]]]:
    """
    Low region: raw F1 and F2, no scaling/subtraction, fit CB in 0.4-1.2 V.
    """
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
            mean_err = float(np.sqrt(pcov[1, 1])) if pcov is not None and pcov.shape[0] > 1 else None
            mean_mV = _to_millivolts_scalar(mean)
            mean_err_mV = _to_millivolts_scalar(mean_err) if mean_err is not None else None
            if label == 'F2' and mean_mV is not None and mean_mV < 600.0:
                ax.plot([], [], ' ', label="F2 CB fit rejected (<600 mV)")
            else:
                mean_map[f'{label}_low'] = (mean, mean_err)
                ax.plot(
                    x_fit * MV_SCALE,
                    y_fit,
                    color='k',
                    linestyle='--',
                    label=(
                        f"{DESCRIPTIONS[label]} CB fit (mean={mean_mV:.2f}"
                        + (f' ± {mean_err_mV:.2f}' if mean_err_mV is not None else '')
                        + f" mV, alpha={alpha:.2f}, n={n:.1f})"
                    ),
                )
        plotted = True

    if not plotted:
        ax.set_visible(False)
        return {}
    ax.set_xlabel('Bin center [mV]')
    ax.set_ylabel('Counts')
    ax.set_title(f'Low region raw ({LOW_V_MIN:.1f}-{LOW_V_MAX:.1f} V)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=LOW_V_MIN * MV_SCALE, right=LOW_V_MAX * MV_SCALE)
    ax.legend(loc='best')
    return mean_map


def _fit_f3_peak(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, float, Optional[float]]]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.allclose(y, 0):
        return None
    peak_idx = int(np.argmax(y))
    amp0 = max(float(y[peak_idx]), 1e-6)
    mean0 = float(x[peak_idx])
    sigma0 = max((x.max() - x.min()) / 6.0, 1e-4)
    try:
        popt, pcov = curve_fit(
            gaussian,
            x,
            y,
            p0=[amp0, mean0, sigma0],
            bounds=([0.0, x.min(), 1e-6], [np.inf, x.max(), (x.max() - x.min()) * 2.0]),
            maxfev=20000,
        )
    except Exception:
        return None
    x_fit = np.linspace(x.min(), x.max(), 400)
    y_fit = gaussian(x_fit, *popt)
    _, mean, _ = popt
    mean_err = float(np.sqrt(pcov[1, 1])) if pcov is not None and pcov.shape[0] > 1 else None
    return x_fit, y_fit, mean, mean_err


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
        x_fit, y_fit, mean, mean_err = fit
        mean_val = mean
        mean_err_val = mean_err
        mean_mV = _to_millivolts_scalar(mean)
        mean_err_mV = _to_millivolts_scalar(mean_err) if mean_err is not None else None
        ax.plot(
            x_fit * MV_SCALE,
            y_fit,
            color='tab:purple',
            linestyle='--',
            label=(
                f'Peak fit (mean={mean_mV:.2f}'
                + (f' ± {mean_err_mV:.2f}' if mean_err_mV is not None else '')
                + ' mV)'
            ),
        )
    ax.set_xlim(170, 190)
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
        f"split_low_cb_{measurement_time.strftime('%Y%m%d_%H%M%S')}.png"
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


def plot_fit_means(mean_records: Dict[str, List[Tuple[datetime, float, Optional[float]]]]):
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

    temp_times, temp_vals = _load_temperature_series(TEMP_CSV_PATH, start_time)
    temp_label_tracker = {}

    def _plot_series(ax, samples, color, title, ylim=None):
        if not samples:
            ax.set_visible(False)
            return
        samples = sorted(samples, key=lambda x: x[0])
        times = [t for t, _, _ in samples]
        values = [v * MV_SCALE for _, v, _ in samples]
        errs = np.array([e * MV_SCALE if e is not None else np.nan for _, _, e in samples], dtype=float)
        ax.plot(times, values, marker='o', markersize=3.5, linestyle='none', color=color)
        finite_mask = np.isfinite(errs)
        if finite_mask.any():
            vals_arr = np.array(values, dtype=float)
            band = 2.0 * errs
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
    _plot_series(ax_f1_low, filtered_records.get('F1_low', []), 'tab:blue', 'Inner long PM low')
    _plot_series(ax_f2_low, filtered_records.get('F2_low', []), 'tab:orange', 'Outer long PM low')
    _plot_series(ax_f3, filtered_records.get('F3', []), 'tab:purple', 'Test pulse', ylim=(181.2, 183))

    axes[-1].set_xlabel('Measurement time')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, 'fit_means_vs_time.png')
    plt.show()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved mean-vs-time plot: {out_path}")


def write_plot_data(rows: List[Dict[str, RowValue]]):
    if not rows:
        return

    def _row_timestamp(row: Dict[str, RowValue]) -> datetime:
        ts = row.get('timestamp')
        return ts if isinstance(ts, datetime) else datetime.min

    rows = sorted(rows, key=_row_timestamp)
    data_path = os.path.join(PLOTS_DIR, 'fit_means_data_low_cb.csv')
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
    directories: List[str] = sorted(iter_measurement_dirs(ROOT_DIR))
    mean_records: Dict[str, List[Tuple[datetime, float, Optional[float]]]] = {}
    data_rows: List[Dict[str, RowValue]] = []
    for directory in directories:
        measurement_time = parse_timestamp(directory)
        if measurement_time is None or (PLOT_START and measurement_time < PLOT_START):
            continue
        series = {}
        for label, fname in FILES.items():
            df = load_histogram(os.path.join(directory, fname))
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
