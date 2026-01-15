import os
import csv
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

PLOTS_DIR = 'plots'

NP02DATA_DIR = os.environ.get('NP02DATA_DIR', '../np02data')

# Simple usage: set the input CSV here.
# Examples:
#   '../np02data/TempMarch2025.csv'
#   '../np02data/TempAugust2025.csv'
INPUT_CSV = os.environ.get('NP02_TEMP_INPUT_CSV', os.path.join(NP02DATA_DIR, 'TempMarch2025.csv'))

# Smoothing parameters
SMOOTH_MINUTES = 200
POLYORDER = 2


def load_time_series_csv(path: str) -> Tuple[List[datetime], List[float]]:
    """Load a time-series CSV with rows like
    'YYYY-MM-DD HH:MM:SS.sss, value' or 'YYYY/MM/DD HH:MM:SS, value'.
    Returns (times, values). Ignores badly formatted lines.
    """
    times: List[datetime] = []
    vals: List[float] = []
    if not os.path.exists(path):
        return times, vals
    fmts = [
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
        '%Y/%m/%d %H:%M:%S',
        '%Y-%m-%d',
        '%Y/%m/%d',
    ]
    with open(path, 'r') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            sep = ',' if ',' in ln else (';' if ';' in ln else None)
            if not sep:
                continue
            tstr, vstr = ln.split(sep, 1)
            tstr = tstr.strip()
            vstr = vstr.strip()
            t = None
            for fmt in fmts:
                try:
                    t = datetime.strptime(tstr, fmt)
                    break
                except Exception:
                    continue
            if t is None:
                continue
            try:
                v = float(vstr)
            except Exception:
                continue
            times.append(t)
            vals.append(v)
    return times, vals


def savgol_smooth(times: List[datetime], values: List[float], smooth_minutes: float = 60.0,
                  polyorder: int = 2) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Apply Savitzky–Golay smoothing on a uniform grid.
    - times, values: raw series
    - smooth_minutes: target smoothing window (minutes) to cover with the filter
    - polyorder: polynomial order for savgol
    Returns (t_grid, v_smooth, info_dict)
    """
    if len(times) < max(7, polyorder + 3):
        # Not enough points: return original series as numpy arrays
        t = np.array([t.timestamp() for t in times], dtype=float)
        v = np.array(values, dtype=float)
        return t, v, {'method': 'identity', 'reason': 'insufficient_points'}

    # Sort by time and drop non-finite values
    order = np.argsort(np.array([t.timestamp() for t in times]))
    t_arr = np.array([times[i].timestamp() for i in order], dtype=float)
    v_arr = np.array([values[i] for i in order], dtype=float)
    mask = np.isfinite(t_arr) & np.isfinite(v_arr)
    t_arr = t_arr[mask]
    v_arr = v_arr[mask]
    if t_arr.size < max(7, polyorder + 3):
        return t_arr, v_arr, {'method': 'identity', 'reason': 'insufficient_points_after_filter'}

    # Interpolate NaNs (if any) and set up uniform grid
    # Compute median dt (seconds)
    dts = np.diff(t_arr)
    dts = dts[np.isfinite(dts) & (dts > 0)]
    if dts.size == 0:
        return t_arr, v_arr, {'method': 'identity', 'reason': 'non_increasing_time'}
    dt = float(np.median(dts))

    t_min, t_max = float(t_arr[0]), float(t_arr[-1])
    n_steps = max(2, int(np.floor((t_max - t_min) / dt)))
    t_grid = np.linspace(t_min, t_max, n_steps)
    v_interp = np.interp(t_grid, t_arr, v_arr)

    # Choose window length in samples to approximately cover smooth_minutes
    window_samples = int(round((smooth_minutes * 60.0) / dt))
    # Must be odd and >= polyorder + 2
    if window_samples % 2 == 0:
        window_samples += 1
    window_samples = max(window_samples, polyorder + 3)
    # Put an upper bound to avoid over-large convolution kernels
    window_samples = min(window_samples, max(301, polyorder + 3))

    try:
        from scipy.signal import savgol_filter
        v_smooth = savgol_filter(v_interp, window_length=window_samples, polyorder=polyorder, mode='interp')
        info = {
            'method': 'savgol',
            'dt_seconds': dt,
            'window_samples': window_samples,
            'polyorder': polyorder,
            'smooth_minutes': smooth_minutes,
            'n_points': int(t_grid.size),
        }
        return t_grid, v_smooth, info
    except Exception as e:
        # Fallback: simple moving average
        k = window_samples
        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1
        kernel = np.ones(k, dtype=float) / float(k)
        v_pad = np.pad(v_interp, (k//2, k//2), mode='edge')
        v_smooth = np.convolve(v_pad, kernel, mode='valid')[:v_interp.size]
        info = {
            'method': 'moving_average',
            'dt_seconds': dt,
            'window_samples': k,
            'smooth_minutes': smooth_minutes,
            'n_points': int(t_grid.size),
            'fallback_error': str(e),
        }
        return t_grid, v_smooth, info


def plot_raw_and_smoothed(times: List[datetime], values: List[float],
                          t_grid: np.ndarray, v_smooth: np.ndarray,
                          out_path: str, title: Optional[str] = None,
                          info: Optional[dict] = None) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Prepare raw series ordered by time
    order = np.argsort(np.array([t.timestamp() for t in times]))
    times_sorted = [times[i] for i in order]
    vals_sorted = [values[i] for i in order]

    plt.figure(figsize=(10, 4))
    plt.plot(times_sorted, vals_sorted, '.', color='tab:blue', alpha=0.6, label='Raw')
    # Smoothed: plot using grid converted to datetime
    t_grid_dt = [datetime.fromtimestamp(x) for x in t_grid]
    plt.plot(t_grid_dt, v_smooth, '-', color='tab:orange', linewidth=2.0, alpha=0.9,
             label='Smoothed')
    plt.gcf().autofmt_xdate()
    plt.xlabel('Time')
    plt.ylabel('Temperature (C)')
    ttl = title or 'Temperature: Raw vs Smoothed'
    if info and 'method' in info:
        meth = info.get('method')
        if meth == 'savgol':
            ttl += f" — SavGol (win={info.get('window_samples')}, poly={info.get('polyorder')})"
        elif meth == 'moving_average':
            ttl += f" — Moving average (win={info.get('window_samples')})"
    plt.title(ttl)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()


def save_smoothed_csv(t_grid: np.ndarray, v_smooth: np.ndarray, out_csv_path: str) -> None:
    """Save smoothed series to CSV with columns: timestamp, smoothed_temperature_C."""
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['timestamp', 'smoothed_temperature_C'])
        for t_sec, v in zip(t_grid, v_smooth):
            try:
                ts = datetime.fromtimestamp(float(t_sec)).strftime('%Y-%m-%d %H:%M:%S')
                vv = float(v)
            except Exception:
                continue
            if np.isfinite(vv):
                w.writerow([ts, f"{vv:.6f}"])


def main():
    src = INPUT_CSV
    times, vals = load_time_series_csv(src)
    if not times:
        print(f"[INFO] No points in {src}; nothing to do.")
        return
    t_grid, v_smooth, info = savgol_smooth(times, vals, smooth_minutes=SMOOTH_MINUTES, polyorder=POLYORDER)
    base = os.path.splitext(os.path.basename(src))[0]
    out_path = os.path.join(PLOTS_DIR, f'{base}_raw_vs_smoothed.png')
    plot_raw_and_smoothed(times, vals, t_grid, v_smooth, out_path,
                          title=f'{base}: Raw vs Smoothed', info=info)
    print(f'Saved overlay plot: {out_path}')
    out_csv = os.path.join(PLOTS_DIR, f'{base}_smoothed.csv')
    save_smoothed_csv(t_grid, v_smooth, out_csv)
    print(f'Saved smoothed CSV: {out_csv}')


if __name__ == '__main__':
    main()
