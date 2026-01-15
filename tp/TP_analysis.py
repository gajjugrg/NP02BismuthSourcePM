import os
import glob
import re
from datetime import datetime, timezone
from typing import Dict, Optional, List, Tuple

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit


NP02DATA_DIR = os.environ.get('NP02DATA_DIR', '../np02data')
ROOT_DIR = NP02DATA_DIR
PLOTS_DIR = 'plots'
PLOT_START = datetime(2025, 11, 7, 15, 30)
SPLIT_THRESHOLD = 1.2
LOW_REGION_MIN = 0.6
HIGH_REGION_MAX = 1.5
LONG_X_SCALE = 1
LONG_Y_SCALE = 0.57
F3_FIT_WINDOW = 0.004
TEMP_CSV_PATH = os.environ.get('NP02_TEMP_CSV', os.path.join(NP02DATA_DIR, 'Temp.csv'))
MV_SCALE=1000.0  # volts to millivolts
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
    'F1_low': 'PM_signal_data',
    'F3': 'Test_pulse',
}


def _to_millivolts(values: np.ndarray) -> np.ndarray:
    return values * MV_SCALE


def _to_millivolts_scalar(value_v: Optional[float]) -> Optional[float]:
    if value_v is None or not np.isfinite(value_v):
        return None
    return value_v * MV_SCALE

def crystal_ball(x, amplitude, mean, sigma, alpha, n):
    """
    Crystal Ball lineshape: Gaussian core with power-law tail on the low side.
    Parameters:
        amplitude: overall scale
        mean: peak position
        sigma: width (>0)
        alpha: where tail starts in sigma units (>0)
        n: tail power (>0)
    """
    t = (x - mean) / sigma
    abs_alpha = np.abs(alpha)
    # Constants for the tail
    A = (n / abs_alpha) ** n * np.exp(-0.5 * abs_alpha * abs_alpha)
    B = n / abs_alpha - abs_alpha
    result = np.empty_like(t, dtype=float)
    core_mask = t > -abs_alpha
    tail_mask = ~core_mask
    # Gaussian core
    result[core_mask] = np.exp(-0.5 * t[core_mask] * t[core_mask])
    # Power-law tail with safe floor to avoid invalid powers
    if np.any(tail_mask):
        denom = np.maximum(B - t[tail_mask], 1e-12)
        result[tail_mask] = A * denom ** (-n)
    return amplitude * result

MONTH_MAP = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
}


def iter_measurement_dirs(root_dir: str):
    """
    Yield directories that contain at least an F1.txt histogram file.
    """
    seen = set()
    pattern = f"{root_dir}/20??_[A-Za-z][a-z][a-z]/**/F1.txt"
    for f1_path in glob.iglob(pattern, recursive=True):
        directory = os.path.dirname(f1_path)
        if directory in seen:
            continue
        seen.add(directory)
        yield directory


def parse_timestamp(directory: str) -> Optional[datetime]:
    """
    Extract a timestamp from the directory structure: YYYY_Mmm/DD/HH/MM/(SS optional).
    """
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
    """
    Load histogram data while preserving the natural voltage span.
    """
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


def collect_series(directory: str) -> Dict[str, pd.DataFrame]:
    """
    Load requested channel histograms.
    """
    series = {}
    for label, filename in FILES.items():
        df = load_histogram(os.path.join(directory, filename))
        if df is not None:
            series[label] = df
    return series


def _fit_crystal_ball_curve(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float, float]]]:
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
        popt, _ = curve_fit(
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
    return x_fit, y_fit, tuple(popt)


def _scale_background(signal_df: Optional[pd.DataFrame], bg_df: Optional[pd.DataFrame]) -> Optional[Dict[str, np.ndarray]]:
    if signal_df is None or bg_df is None:
        return None
    signal_df = signal_df.dropna(subset=['BinCenter', 'Population'])
    bg_df = bg_df.dropna(subset=['BinCenter', 'Population'])
    if signal_df.empty or bg_df.empty:
        return None
    voltage = signal_df['BinCenter'].to_numpy(dtype=float)
    count = signal_df['Population'].to_numpy(dtype=float)
    mask = np.isfinite(voltage) & np.isfinite(count)
    voltage = voltage[mask]
    count = count[mask]
    bg_voltage = (bg_df['BinCenter'].to_numpy(dtype=float) * LONG_X_SCALE)
    bg_count = (bg_df['Population'].to_numpy(dtype=float) * LONG_Y_SCALE)
    mask_bg = np.isfinite(bg_voltage) & np.isfinite(bg_count)
    bg_voltage = bg_voltage[mask_bg]
    bg_count = bg_count[mask_bg]
    if voltage.size == 0 or bg_voltage.size == 0:
        return None
    order = np.argsort(bg_voltage)
    scaled_bg = np.interp(voltage, bg_voltage[order], bg_count[order], left=0.0, right=0.0)
    signal = count - scaled_bg
    return {
        'voltage': voltage,
        'count': count,
        'scaled_bg': scaled_bg,
        'signal': signal,
    }


def _plot_low_region(ax, series: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    scaled = _scale_background(series.get('F1'), series.get('F2'))
    if scaled is None:
        ax.set_visible(False)
        return {}
    mask = (scaled['voltage'] < SPLIT_THRESHOLD) & (scaled['voltage'] >= LOW_REGION_MIN)
    if not mask.any():
        ax.set_visible(False)
        return {}
    v = scaled['voltage'][mask]
    raw = scaled['count'][mask]
    bg = scaled['scaled_bg'][mask]
    signal = scaled['signal'][mask]
    v_mV = v * MV_SCALE
    ax.step(v_mV, raw, where='mid', color=COLORS['F1'], label='F1 raw (signal)')
    ax.step(v_mV, bg, where='mid', color=COLORS['F2'], label='F2 scaled (background)')
    ax.step(v_mV, signal, where='mid', color='tab:red', label='Signal - background')
    mean_map: Dict[str, float] = {}
    fit_signal = _fit_crystal_ball_curve(v, np.clip(signal, 0, None))
    if fit_signal is not None:
        x_fit, y_fit, (_, mean, sigma, alpha, n) = fit_signal
        mean_map['F1_low'] = mean
        mean_mV = _to_millivolts_scalar(mean)
        ax.plot(
            x_fit * MV_SCALE,
            y_fit,
            color='black',
            linestyle='--',
            label=f'Signal CB fit (mean={mean_mV:.2f} mV, α={alpha:.2f}, n={n:.1f})',
        )
    ax.set_xlabel('Bin center [mV]')
    ax.set_ylabel('Counts')
    ax.set_title(f'F1 with scaled background ({LOW_REGION_MIN:.1f}-{SPLIT_THRESHOLD:.1f} V)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=LOW_REGION_MIN * MV_SCALE, right=SPLIT_THRESHOLD * MV_SCALE)
    ax.legend(loc='best')
    return mean_map


def _plot_high_region(ax, series: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    plotted = False
    mean_map: Dict[str, float] = {}
    for label in ('F1', 'F2'):
        df = series.get(label)
        if df is None or df.empty:
            continue
        centers = df['BinCenter'].to_numpy(dtype=float)
        counts = df['Population'].to_numpy(dtype=float)
        mask = (centers >= SPLIT_THRESHOLD) & (centers <= HIGH_REGION_MAX)
        if not mask.any():
            continue
        x = centers[mask]
        y = counts[mask]
        x_mV = x * MV_SCALE
        label_text = DESCRIPTIONS[label]
        ax.step(x_mV, y, where='mid', color=COLORS[label], label=label_text)
        fit = _fit_crystal_ball_curve(x, y)
        if fit is not None:
            x_fit, y_fit, (_, mean, sigma, alpha, n) = fit
            mean_map[f'{label}_high'] = mean
            mean_mV = _to_millivolts_scalar(mean)
            ax.plot(
                x_fit * MV_SCALE,
                y_fit,
                color=COLORS[label],
                linestyle='--',
                label=f'{label_text} CB fit (mean={mean_mV:.2f} mV, α={alpha:.2f}, n={n:.1f})',
            )
        plotted = True

    if not plotted:
        ax.set_visible(False)
        return {}

    ax.set_xlabel('Bin center [mV]')
    ax.set_ylabel('Counts')
    ax.set_title(f'F1/F2 raw ({SPLIT_THRESHOLD:.1f}-{HIGH_REGION_MAX:.1f} V)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=SPLIT_THRESHOLD * MV_SCALE, right=HIGH_REGION_MAX * MV_SCALE)
    ax.legend(loc='best')
    return mean_map


def _fit_f3_peak(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    if x.size < 3:
        return None
    peak_idx = int(np.argmax(y))
    peak_center = x[peak_idx]
    for factor in (1, 2, 3, 4):
        window = F3_FIT_WINDOW * factor
        mask = np.abs(x - peak_center) <= window
        if mask.sum() >= 3:
            fit = _fit_crystal_ball_curve(x[mask], y[mask])
            if fit is not None:
                x_fit, y_fit, (_, mean, sigma, alpha, n) = fit
                return x_fit, y_fit, mean
    return None


def _plot_f3(ax, series: Dict[str, pd.DataFrame]) -> Optional[float]:
    df = series.get('F3')
    if df is None or df.empty:
        ax.set_visible(False)
        return None
    x = df['BinCenter'].to_numpy(dtype=float)
    y = df['Population'].to_numpy(dtype=float)
    x_mv = x * MV_SCALE
    ax.step(x_mv, y, where='mid', color=COLORS['F3'], label=DESCRIPTIONS['F3'])
    mean_val = None
    fit = _fit_f3_peak(x, y)
    if fit is not None:
        x_fit, y_fit, mean = fit
        mean_val = mean
        mean_mV = _to_millivolts_scalar(mean)
        ax.plot(x_fit * MV_SCALE, y_fit, color='tab:purple', linestyle='--', label=f'Peak fit (mean={mean_mV:.2f} mV)')
    ax.set_xlim(170, 190)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Bin center [mV]')
    ax.set_ylabel('Counts')
    ax.set_title('Test pulse')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    return mean_val


def plot_measurement(measurement_time: datetime, series: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    Build the stacked plot (F1/F2 low, F1/F2 high, F3).
    """
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=False)
    ax_high, ax_low, ax_f3 = axes

    high_means = _plot_high_region(ax_high, series)
    low_means = _plot_low_region(ax_low, series)
    f3_mean = _plot_f3(ax_f3, series)

    if not any(ax.get_visible() for ax in axes):
        plt.close(fig)
        return {}

    fig.suptitle(f"{measurement_time:%Y-%m-%d %H:%M:%S}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(
        PLOTS_DIR,
        f"split_{measurement_time.strftime('%Y%m%d_%H%M%S')}.png"
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")
    means: Dict[str, float] = {}
    means.update(low_means)
    for key, value in high_means.items():
        means[key] = value
    if f3_mean is not None:
        means['F3'] = f3_mean
    return means


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    directories: List[str] = sorted(iter_measurement_dirs(ROOT_DIR))
    mean_records: Dict[str, List[Tuple[datetime, float]]] = {}
    data_rows: List[Dict[str, Optional[float]]] = []
    for directory in directories:
        measurement_time = parse_timestamp(directory)
        if measurement_time is None or measurement_time < PLOT_START:
            continue
        series = collect_series(directory)
        if not series:
            continue
        means = plot_measurement(measurement_time, series)
        for key, value in means.items():
            mean_records.setdefault(key, []).append((measurement_time, value))
        row = {'timestamp': measurement_time}
        for key in DATA_LABELS.keys():
            row[key] = means.get(key)
        data_rows.append(row)
    plot_fit_means(mean_records)
    plot_fit_variations(mean_records)
    write_plot_data(data_rows)


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


def plot_fit_means(mean_records: Dict[str, List[Tuple[datetime, float]]]):
    keys_present = any(mean_records.get(k) for k in DATA_LABELS.keys())
    if not keys_present:
        return

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    ax_f1_high, ax_f2_high, ax_f1_low, ax_f3 = axes
    start_time_candidates = []
    for samples in mean_records.values():
        if samples:
            start_time_candidates.append(min(samples, key=lambda x: x[0])[0])
    start_time = min(start_time_candidates) if start_time_candidates else None

    temp_times, temp_vals = _load_temperature_series(TEMP_CSV_PATH, start_time)
    temp_label_tracker = {}

    samples_f1_high = mean_records.get('F1_high')
    if samples_f1_high:
        samples_f1_high = sorted(samples_f1_high, key=lambda x: x[0])
        times = [t for t, _ in samples_f1_high]
        values = [v * MV_SCALE for _, v in samples_f1_high]
        ax_f1_high.plot(times, values, marker='o', linestyle=None, color='tab:blue')
        ax_f1_high.set_ylabel('Mean [mV]')
        ax_f1_high.set_title('Inner long PM')
        ax_f1_high.grid(True, alpha=0.3)
        _overlay_temperature(ax_f1_high, temp_times, temp_vals, temp_label_tracker)
    else:
        ax_f1_high.set_visible(False)

    samples_f2_high = mean_records.get('F2_high')
    if samples_f2_high:
        samples_f2_high = sorted(samples_f2_high, key=lambda x: x[0])
        times = [t for t, _ in samples_f2_high]
        values = [v * MV_SCALE for _, v in samples_f2_high]
        ax_f2_high.plot(times, values, marker='o', linestyle=None, color='tab:orange')
        ax_f2_high.set_ylabel('Mean [mV]')
        ax_f2_high.set_title('Outer long PM')
        ax_f2_high.grid(True, alpha=0.3)
        _overlay_temperature(ax_f2_high, temp_times, temp_vals, temp_label_tracker)
    else:
        ax_f2_high.set_visible(False)

    samples_low = mean_records.get('F1_low')
    if samples_low:
        samples_low = sorted(samples_low, key=lambda x: x[0])
        times = [t for t, _ in samples_low]
        values = [v * MV_SCALE for _, v in samples_low]
        ax_f1_low.plot(times, values, marker='o', linestyle=None, color='tab:green')
        ax_f1_low.set_ylabel('Mean [mV]')
        ax_f1_low.set_title('PM signal data')
        ax_f1_low.grid(True, alpha=0.3)
        _overlay_temperature(ax_f1_low, temp_times, temp_vals, temp_label_tracker)
    else:
        ax_f1_low.set_visible(False)

    samples_f3 = mean_records.get('F3')
    if samples_f3:
        samples_f3 = sorted(samples_f3, key=lambda x: x[0])
        times = [t for t, _ in samples_f3]
        values = [v * MV_SCALE for _, v in samples_f3]
        ax_f3.plot(times, values, marker='o', linestyle=None, color='tab:purple')
        ax_f3.set_ylabel('Mean [mV]')
        ax_f3.set_title('Test pulse')
        ax_f3.set_ylim(180.4, 180.6)
        ax_f3.grid(True, alpha=0.3)
        _overlay_temperature(ax_f3, temp_times, temp_vals, temp_label_tracker)
    else:
        ax_f3.set_visible(False)

    axes[-1].set_xlabel('Measurement time')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()
    out_path = os.path.join(PLOTS_DIR, 'fit_means_vs_time_.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved fit-means plot: {out_path}")


def plot_fit_variations(mean_records: Dict[str, List[Tuple[datetime, float]]]):
    if not any(mean_records.get(k) for k in DATA_LABELS.keys()):
        return

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    ax_f1_high, ax_f2_high, ax_f1_low, ax_f3 = axes

    all_times = [t for samples in mean_records.values() for (t, _) in samples]
    start_time = min(all_times) if all_times else None
    temp_times, temp_vals = _load_temperature_series(TEMP_CSV_PATH, start_time)
    temp_label_tracker = {}
    ref_label = start_time.strftime('%Y-%m-%d %H:%M') if start_time else 'first available point'

    def _plot_percent(ax, key, color, label):
        samples = mean_records.get(key)
        if not samples:
            ax.set_visible(False)
            return False
        samples = sorted(samples, key=lambda x: x[0])
        ref = samples[0][1]
        if ref is None or ref == 0 or not np.isfinite(ref):
            ax.set_visible(False)
            return False
        times = [t for t, _ in samples]
        values = [((val / ref) - 1.0) * 100.0 for _, val in samples]
        ax.plot(times, values, marker='o', linestyle='-', color=color, label=label)
        ax.set_ylabel('diff [%]')
        ax.grid(True, alpha=0.3)
        _overlay_temperature(ax, temp_times, temp_vals, temp_label_tracker)
        return True

    if _plot_percent(ax_f1_high, 'F1_high', 'tab:blue', 'Inner long PM'):
        ax_f1_high.set_title('Percent variation: inner long PM')

    if _plot_percent(ax_f2_high, 'F2_high', 'tab:orange', 'Outer long PM'):
        ax_f2_high.set_title('Percent variation: outer long PM')

    if _plot_percent(ax_f1_low, 'F1_low', 'tab:green', 'PM signal data'):
        ax_f1_low.set_title('Percent variation: PM signal data')

    if _plot_percent(ax_f3, 'F3', 'tab:purple', 'F3 test pulse'):
        ax_f3.set_title('Percent variation: Test pulse')

    axes[-1].set_xlabel('Measurement time')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    fig.text(0.5, 0.04, f"Percent differences referenced to {ref_label}", ha='center', fontsize=10)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()
    out_path = os.path.join(PLOTS_DIR, 'fit_percent_variation.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved percent-variation plot: {out_path}")


def write_plot_data(rows: List[Dict[str, Optional[float]]]):
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r['timestamp'])
    data_path = os.path.join(PLOTS_DIR, 'fit_means_data.csv')
    with open(data_path, 'w', newline='') as f:
        f.write("# Derived from np02_analysis_fixed_scaling_split_hist.py\n")
        f.write(f"# Generated at {datetime.now(timezone.utc).isoformat()}\n")
        writer = csv.writer(f)
        writer.writerow(['timestamp_iso'] + [DATA_LABELS[k] + '_mV' for k in DATA_LABELS.keys()])
        for row in rows:
            timestamp = row['timestamp'].isoformat()
            values = []
            for key in DATA_LABELS.keys():
                mv_val = _to_millivolts_scalar(row.get(key))
                values.append(f"{mv_val:.3f}" if mv_val is not None else '')
            writer.writerow([timestamp] + values)
    print(f"Wrote plot data: {data_path}")


if __name__ == '__main__':
    main()
