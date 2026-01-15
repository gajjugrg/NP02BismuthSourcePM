import os
import glob
import csv
import calendar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from datetime import datetime
from typing import Optional
import pickle
from typing import Optional

NP02DATA_DIR = os.environ.get('NP02DATA_DIR', '../np02data')

SAVE_FILES = True  # set to False to skip writing plots to disk
# Control whether to reapply ratio-based tau correction on each run
# Keep this False for idempotent behavior (only recompute when inputs change)
FORCE_REAPPLY_RATIO_CORR = False
# Never write back to fit_cache.pkl (read-only processing)
WRITE_CACHE_BACK = False

# Directory to save plots
PLOTS_DIR = 'plots'
if SAVE_FILES:
    os.makedirs(PLOTS_DIR, exist_ok=True)

# Cuts for valid points
M3_SHORT_MIN = 0.550
M3_SHORT_MAX = 1.38

# Exclusion windows (e.g., known 50 µs readout period to omit from analysis)
# March 21, 2025: 14:19–15:17 local
EXCLUDE_WINDOWS = [
    (datetime(2025, 3, 21, 14, 19, 0), datetime(2025, 3, 21, 15, 17, 0)),
]

def _in_excluded(t: datetime) -> bool:
    for s, e in EXCLUDE_WINDOWS:
        if s <= t <= e:
            return True
    return False

# File to cache fit parameters so we don't re-fit on reruns
FIT_CACHE_FILE = 'fit_cache.pkl'
fit_cache = {}
# In-memory model store to avoid refitting multiple times per run
_local_ratio_models = {}
if os.path.exists(FIT_CACHE_FILE):
    try:
        with open(FIT_CACHE_FILE, 'rb') as f:
            fit_cache = pickle.load(f)
        if not isinstance(fit_cache, dict):
            print(f"[WARN] {FIT_CACHE_FILE} is not a dict. Resetting cache.")
            fit_cache = {}
    except Exception as e:
        print(f"[WARN] Failed to load {FIT_CACHE_FILE}: {e}. Starting with empty cache.")
        fit_cache = {}

def get_delta_t_and_scaling_factor(measurement_time: datetime):
    """
    Use a fixed timing/scaling for all times.
    Returns (delta_t, scaling_factor).
    Note:
    - 16 is the drift distance (cm)
    - 0.1635 is the drift velocity (cm/us) at 520 V/cm
    """
    # Always use 520 V/cm parameters
    delta_t = 0.09785932722  #  16.0 / 0.1635; 16 cm drift length / 0.1635 cm/us drift velocity at 520 V/cm
    scaling_factor = 0.92
    return delta_t, scaling_factor

# ---- Small utility: linear fit with uncertainties ----
def _linear_fit_with_uncertainty(x: np.ndarray, y: np.ndarray):
    """Return slope, intercept, their 1-sigma uncertainties, and a fit curve grid.
    Uses numpy.polyfit with cov=True to estimate parameter covariance.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return None
    try:
        import warnings as _warn
        with _warn.catch_warnings():
            _warn.simplefilter('ignore')
            (m, b), cov = np.polyfit(x, y, 1, cov=True)
        sigma_m = float(np.sqrt(cov[0, 0])) if cov.size >= 4 else np.nan
        sigma_b = float(np.sqrt(cov[1, 1])) if cov.size >= 4 else np.nan
        xfit = np.linspace(float(x.min()), float(x.max()), 200)
        yfit = m * xfit + b
        return m, b, sigma_m, sigma_b, xfit, yfit
    except Exception:
        return None
def _load_time_series_temperature(path: str):
    """Load a time-series temperature CSV with rows like
    'YYYY-MM-DD HH:MM:SS.sss, value' or 'YYYY/MM/DD HH:MM:SS, value'.
    Returns (times: list[datetime], temps: list[float]). Ignores bad lines.
    """
    times = []
    temps = []
    if not path or not os.path.exists(path):
        return times, temps
    fmts = [
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
        '%Y/%m/%d %H:%M:%S',
        '%Y-%m-%d',
        '%Y/%m/%d',
    ]
    try:
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
                temps.append(v)
    except Exception as e:
        print(f"[WARN] Failed to read temperature time-series {path}: {e}")
    return times, temps

def plot_temperature_file(path: str, save: bool = True, out_name: str = 'temperature_timeseries.png'):
    """Plot the temperature time-series CSV and save to plots/ if save is True.
    Falls back to showing the plot if not saving.
    """
    times, vals = _load_time_series_temperature(path)
    if not times:
        print(f"[INFO] No temperature points parsed from {path}.")
        return
    # Sort by time
    order = np.argsort(np.array([t.timestamp() for t in times]))
    times = [times[i] for i in order]
    vals = [vals[i] for i in order]
    plt.figure(figsize=(10, 4))
    plt.plot(times, vals, marker='.', linestyle='-', alpha=0.8)
    plt.gcf().autofmt_xdate()
    plt.xlabel('Time')
    plt.ylabel('Temperature (C)')
    plt.title(os.path.basename(path))
    plt.tight_layout()
    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        out = os.path.join(PLOTS_DIR, out_name)
        plt.savefig(out, dpi=150)
        print(f"Saved temperature plot: {out}")
    else:
        plt.show()
    plt.close()

_temp_file_candidates = [
    (
        os.path.join(NP02DATA_DIR, 'Oct15_Nov5.csv'),
        'temp_timeseries_oct15_nov5.png'
    ),
    (
        os.path.join(NP02DATA_DIR, 'TempMarch2025_smoothed.csv'),
        'temp_timeseries_march2025.png'
    ),
]
for _path, _out in _temp_file_candidates:
    if os.path.exists(_path):
        plot_temperature_file(_path, save=True, out_name=_out)
        break

# ---------------- Temperature vs (m3_long / m3_short) ---------------- #

def _load_fit_cache(path: str = FIT_CACHE_FILE):
    try:
        with open(path, 'rb') as f:
            d = pickle.load(f)
        return d if isinstance(d, dict) else {}
    except Exception as e:
        print(f"[WARN] Failed to load cache {path}: {e}")
        return {}

def _nearest_temp_for_time(times, temps, t: datetime):
    """Linearly interpolate temperature at time t from a time series.
    - Returns None if inputs are empty/mismatched or if t is outside the series range (no extrapolation).
    - Accepts lists of datetimes (times) and floats (temps); sorts internally.
    """
    if not times or not temps or len(times) != len(temps):
        return None
    # Sort by time
    order = np.argsort(np.array([ti.timestamp() for ti in times]))
    ts_sorted = np.array([times[i].timestamp() for i in order], dtype=float)
    vs_sorted = np.array([temps[i] for i in order], dtype=float)
    if ts_sorted.size == 0:
        return None
    if ts_sorted.size == 1:
        v0 = float(vs_sorted[0])
        return v0 if np.isfinite(v0) else None
    tt = float(t.timestamp())
    # Find right bracket index; do not extrapolate outside range
    idx = int(np.searchsorted(ts_sorted, tt))
    if idx == 0 or idx >= ts_sorted.size:
        return None
    t0, t1 = ts_sorted[idx - 1], ts_sorted[idx]
    v0, v1 = vs_sorted[idx - 1], vs_sorted[idx]
    if not (np.isfinite(t0) and np.isfinite(t1) and np.isfinite(v0) and np.isfinite(v1)):
        return None
    if t1 <= t0:
        return None
    w = (tt - t0) / (t1 - t0)
    v = float(v0 + w * (v1 - v0))
    return v if np.isfinite(v) else None

def _format_time_window_label(tmin: datetime, tmax: datetime) -> str:
    """Return a human-readable label for a time window."""
    if not isinstance(tmin, datetime) or not isinstance(tmax, datetime):
        return 'selected window'
    if tmin.date() == tmax.date():
        return tmin.strftime('%b %d, %Y')
    if tmin.year == tmax.year:
        return f"{tmin.strftime('%b %d')}–{tmax.strftime('%b %d, %Y')}"
    return f"{tmin.strftime('%b %d, %Y')}–{tmax.strftime('%b %d, %Y')}"

def plot_temp_vs_m3_ratio_from_cache(temp_series_csv: str,
                                     cache_file: str = FIT_CACHE_FILE,
                                     use_temp_corrected: bool = False,
                                     out_name: str = 'temp_vs_m3_ratio.png'):
    """
    Plot Temperature (C) vs q_l/q_s using entries in fit_cache.pkl.
    - temp_series_csv: path to time-series CSV with rows 'YYYY/MM/DD HH:MM:SS, value'
    - use_temp_corrected: prefer 'm3_*_temp_corr' when available
    - out_name: output PNG filename in plots/
    """
    cache = _load_fit_cache(cache_file)
    times, temps = _load_time_series_temperature(temp_series_csv)
    if not times:
        print(f"[INFO] No temperature series found at {temp_series_csv}; skipping T vs ratio plot.")
        return
    pts_T = []
    pts_R = []
    seen_dirs = set()
    for key, entry in cache.items():
        if not isinstance(entry, dict):
            continue
        meta = entry.get('meta', {})
        ts = meta.get('timestamp') or entry.get('timestamp')
        if not ts:
            continue
        try:
            t = datetime.fromisoformat(str(ts))
        except Exception:
            continue
        # Skip excluded time windows
        if _in_excluded(t):
            continue
        directory = meta.get('directory') or (key if isinstance(key, str) else None)
        if directory in seen_dirs:
            continue
        # Choose m3 values
        if use_temp_corrected:
            m3s = meta.get('m3_short_temp_corr', meta.get('m3_short'))
            m3l = meta.get('m3_long_temp_corr', meta.get('m3_long'))
        else:
            m3s = meta.get('m3_short')
            m3l = meta.get('m3_long')
        if m3s is None or m3l is None:
            if 'short' in entry and m3s is None:
                m3s = entry['short'].get('m3')
            if 'long' in entry and m3l is None:
                m3l = entry['long'].get('m3')
        if m3s is None or m3l is None:
            continue
        try:
            m3s = float(m3s)
            m3l = float(m3l)
        except Exception:
            continue
        # Apply short m3 cut
        if not (M3_SHORT_MIN < m3s < M3_SHORT_MAX):
            continue
        # Find nearest temperature sample
        Tval = _nearest_temp_for_time(times, temps, t)
        if Tval is None:
            continue
        if m3s == 0 or not np.isfinite(m3s) or not np.isfinite(m3l):
            continue
        ratio = m3l / m3s
        if not np.isfinite(ratio):
            continue
        pts_T.append(Tval)
        pts_R.append(ratio)
        if directory:
            seen_dirs.add(directory)

    if not pts_T:
        print("[INFO] No valid points for Temperature vs q_l/q_s.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(pts_T, pts_R, 'o', alpha=0.8)
    plt.xlabel('Temperature (C)')
    plt.ylabel('Ratio = Q_l/ Q_s')
    title = 'Temperature vs q_l/q_s'
    if use_temp_corrected:
        title += ' (temp-corrected m3)'
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, out_name)
    plt.savefig(out_path, dpi=150)
    plt.close()

# Example call: adjust the path below to your temp CSV if needed
_temp_series_candidates = [
    os.path.join(NP02DATA_DIR, 'TempMarch2025.csv'),
]
## Ratio plot disabled per request
## for _pt in _temp_series_candidates:
##     if os.path.exists(_pt):
##         plot_temp_vs_m3_ratio_from_cache(_pt, use_temp_corrected=False, out_name='temp_vs_m3_ratio.png')
##         break

# ---------------- Overlay: Temperature, m3_short, m3_long vs time ---------------- #

def plot_temp_and_m3_over_time(temp_series_csv: str,
                               cache_file: str = FIT_CACHE_FILE,
                               use_temp_corrected: bool = False,
                               out_name: str = 'temp_m3_overlay.png'):
    """
    Overlay plot versus time: temperature (C) as a line on the right y-axis, and
    m3_short and m3_long on the left y-axis, using entries from fit_cache.pkl.
    - temp_series_csv: time-series CSV with rows 'YYYY/MM/DD HH:MM:SS, value'
    - use_temp_corrected: if True, prefer m3_*_temp_corr
    Saves to plots/out_name.
    """
    cache = _load_fit_cache(cache_file)
    # Load temperature time series
    times_T, vals_T = _load_time_series_temperature(temp_series_csv)
    if not times_T:
        print(f"[INFO] No temperature series found at {temp_series_csv}; skipping overlay plot.")
        return
    # Sort temperature series
    order_T = np.argsort(np.array([t.timestamp() for t in times_T]))
    times_T = [times_T[i] for i in order_T]
    vals_T = [vals_T[i] for i in order_T]

    # Bound plotting window to the temperature series coverage
    tmin = min(times_T)
    tmax = max(times_T)
    window_label = _format_time_window_label(tmin, tmax)

    # Collect m3 series within the temperature time window only
    rows = []  # (time, m3_short, m3_long)
    seen_dirs = set()
    for key, entry in cache.items():
        if not isinstance(entry, dict):
            continue
        meta = entry.get('meta', {})
        ts = meta.get('timestamp') or entry.get('timestamp')
        if not ts:
            continue
        try:
            t = datetime.fromisoformat(str(ts))
        except Exception:
            continue
        # Skip excluded windows
        if _in_excluded(t):
            continue
        directory = meta.get('directory') or (key if isinstance(key, str) else None)
        if directory in seen_dirs:
            continue
        # keep only points that fall within the temperature series time window (e.g., March)
        if (t < tmin) or (t > tmax):
            continue
        # m3 values
        if use_temp_corrected:
            m3s = meta.get('m3_short_temp_corr', meta.get('m3_short'))
            m3l = meta.get('m3_long_temp_corr', meta.get('m3_long'))
        else:
            m3s = meta.get('m3_short')
            m3l = meta.get('m3_long')
        if m3s is None or m3l is None:
            short = entry.get('short') if isinstance(entry, dict) else None
            long  = entry.get('long') if isinstance(entry, dict) else None
            if m3s is None and isinstance(short, dict):
                m3s = short.get('m3')
            if m3l is None and isinstance(long, dict):
                m3l = long.get('m3')
        if m3s is None or m3l is None:
            continue
        try:
            m3s = float(m3s)
            m3l = float(m3l)
        except Exception:
            continue
        if not (M3_SHORT_MIN < m3s < M3_SHORT_MAX):
            continue
        rows.append((t, m3s, m3l))
        if directory:
            seen_dirs.add(directory)

    if not rows:
        print("[INFO] No valid m3 points found in cache for overlay plot.")
        return
    rows.sort(key=lambda r: r[0])
    times_m = [r[0] for r in rows]
    m3s_vals = [r[1] for r in rows]
    m3l_vals = [r[2] for r in rows]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    # Left axis: m3_short and m3_long
    ax1.plot(times_m, m3s_vals, 'o', label='m3_short', color='tab:blue')
    ax1.plot(times_m, m3l_vals, 'o', label='m3_long', color='tab:green')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Pulse height [V]')
    # Right axis: Temperature line
    ax2 = ax1.twinx()
    ax2.plot(times_T, vals_T, '-', label='Temperature (C)', color='tab:red', alpha=0.7)
    ax2.set_ylabel('Temperature (C)')

    fig.autofmt_xdate()
    title = f'Temperature and m3 overlay vs time ({window_label})'
    if use_temp_corrected:
        title += ' (m3 temp-corrected)'
    fig.suptitle(title)
    fig.tight_layout()

    # Merge legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='best')

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, out_name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# Generate the overlay using the provided file
_overlay_temp_csv = os.path.join(NP02DATA_DIR, 'TempMarch2025.csv')
## Combined overlay disabled per request
## if os.path.exists(_overlay_temp_csv):
##     plot_temp_and_m3_over_time(_overlay_temp_csv, use_temp_corrected=False, out_name='temp_m3_overlay.png')

def plot_temp_and_single_m3_over_time(temp_series_csv: str,
                                      series: str = 'short',
                                      cache_file: str = FIT_CACHE_FILE,
                                      use_temp_corrected: bool = False,
                                      out_name: str = 'temp_m3_short_overlay.png'):
    """
    Overlay Temperature (right y-axis) with a single m3 series (left y-axis), either
    'short' or 'long', limited to the time span covered by the temperature CSV.
    """
    cache = _load_fit_cache(cache_file)
    times_T, vals_T = _load_time_series_temperature(temp_series_csv)
    if not times_T:
        print(f"[INFO] No temperature series found at {temp_series_csv}; skipping single overlay plot.")
        return
    order_T = np.argsort(np.array([t.timestamp() for t in times_T]))
    times_T = [times_T[i] for i in order_T]
    vals_T = [vals_T[i] for i in order_T]
    tmin, tmax = min(times_T), max(times_T)
    window_label = _format_time_window_label(tmin, tmax)

    # Collect selected m3 series within window
    rows = []  # (time, m3_value)
    seen_dirs = set()
    sel = series.lower()
    for key, entry in cache.items():
        if not isinstance(entry, dict):
            continue
        meta = entry.get('meta', {})
        ts = meta.get('timestamp') or entry.get('timestamp')
        if not ts:
            continue
        try:
            t = datetime.fromisoformat(str(ts))
        except Exception:
            continue
        # Skip excluded windows
        if _in_excluded(t):
            continue
        if t < tmin or t > tmax:
            continue
        directory = meta.get('directory') or (key if isinstance(key, str) else None)
        if directory in seen_dirs:
            continue
        if sel == 'short':
            val = meta.get('m3_short_temp_corr' if use_temp_corrected else 'm3_short')
            if val is None and 'short' in entry and isinstance(entry['short'], dict):
                val = entry['short'].get('m3')
        else:
            val = meta.get('m3_long_temp_corr' if use_temp_corrected else 'm3_long')
            if val is None and 'long' in entry and isinstance(entry['long'], dict):
                val = entry['long'].get('m3')
        if val is None:
            continue
        try:
            v = float(val)
        except Exception:
            continue
        # Apply cut only for short series; for long, require finite and positive
        if sel == 'short':
            if not (M3_SHORT_MIN < v < M3_SHORT_MAX):
                continue
        else:
            if not np.isfinite(v) or v <= 0:
                continue
        rows.append((t, v))
        if directory:
            seen_dirs.add(directory)

    if not rows:
        print(f"[INFO] No valid m3_{sel} points for overlay plot.")
        return
    rows.sort(key=lambda r: r[0])
    times_m = [r[0] for r in rows]
    mvals = [r[1] for r in rows]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(times_m, mvals, 'o', label=f'm3_{sel}', color='tab:blue' if sel == 'short' else 'tab:green')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Pulse height [V]')
    ax2 = ax1.twinx()
    ax2.plot(times_T, vals_T, '-', label='Temperature (C)', color='tab:red', alpha=0.7)
    ax2.set_ylabel('Temperature (C)')
    fig.autofmt_xdate()
    title = f'Temperature and m3_{sel} overlay vs time ({window_label})'
    if use_temp_corrected:
        title += ' (m3 temp-corrected)'
    fig.suptitle(title)
    fig.tight_layout()
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='best')
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, out_name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# Separate overlays disabled per request
# if os.path.exists(_overlay_temp_csv):
#     plot_temp_and_single_m3_over_time(_overlay_temp_csv, series='short', use_temp_corrected=False, out_name='temp_m3_short_overlay.png')
#     plot_temp_and_single_m3_over_time(_overlay_temp_csv, series='long',  use_temp_corrected=False, out_name='temp_m3_long_overlay.png')

# Oct 15 – Nov 5 temperature overlay with long PM m3
_oct15_nov5_temp_csv = os.path.join(NP02DATA_DIR, 'Oct15_Nov5.csv')
if os.path.exists(_oct15_nov5_temp_csv):
    plot_temp_and_single_m3_over_time(
        _oct15_nov5_temp_csv,
        series='long',
        cache_file='fit_cache_october.pkl',
        use_temp_corrected=False,
        out_name='temp_m3_long_oct15_nov5_overlay.png'
    )

# ---------------- Scatter: Temperature vs m3_short (March window) ---------------- #

def plot_temp_vs_m3_short_from_cache(temp_series_csv: str,
                                     cache_file: str = FIT_CACHE_FILE,
                                     use_temp_corrected: bool = False,
                                     out_name: str = 'temp_vs_m3_short_march.png',
                                     start_date: Optional[datetime] = None,
                                     end_date: Optional[datetime] = None,
                                     do_fit: bool = True):
    """
    Scatter plot of Temperature (C) vs m3_short using cache entries, limited to
    the time span covered by the provided temperature CSV (e.g., March).
    - use_temp_corrected: prefer m3_short_temp_corr if present.
    """
    cache = _load_fit_cache(cache_file)
    times_T, vals_T = _load_time_series_temperature(temp_series_csv)
    if not times_T:
        print(f"[INFO] No temperature series found at {temp_series_csv}; skipping temp vs m3_short plot.")
        return
    # Window bounds from the temperature file (month coverage), optionally tightened
    tmin, tmax = min(times_T), max(times_T)
    if isinstance(start_date, datetime) and start_date > tmin:
        tmin = start_date
    if isinstance(end_date, datetime) and end_date < tmax:
        tmax = end_date
    window_label = _format_time_window_label(tmin, tmax)

    xs_T = []
    ys_m3s = []
    seen_dirs = set()
    for key, entry in cache.items():
        if not isinstance(entry, dict):
            continue
        meta = entry.get('meta', {})
        ts = meta.get('timestamp') or entry.get('timestamp')
        if not ts:
            continue
        try:
            t = datetime.fromisoformat(str(ts))
        except Exception:
            continue
        if t < tmin or t > tmax:
            continue
        # Skip excluded windows
        if _in_excluded(t):
            continue
        directory = meta.get('directory') or (key if isinstance(key, str) else None)
        if directory in seen_dirs:
            continue
        # Select m3_short (optionally temp-corrected)
        val = meta.get('m3_short_temp_corr' if use_temp_corrected else 'm3_short')
        if val is None and 'short' in entry and isinstance(entry['short'], dict):
            val = entry['short'].get('m3')
        if val is None:
            continue
        try:
            m3s = float(val)
        except Exception:
            continue
        if not (M3_SHORT_MIN < m3s < M3_SHORT_MAX):
            continue
        # Temperature at nearest timestamp
        Tval = _nearest_temp_for_time(times_T, vals_T, t)
        if Tval is None:
            continue
        xs_T.append(Tval)
        ys_m3s.append(m3s)
        if directory:
            seen_dirs.add(directory)

    if not xs_T:
        print("[INFO] No valid points for Temperature vs m3_short in the selected month window.")
        return

    # Filter out any negative values just in case
    xs_T = [x for x, y in zip(xs_T, ys_m3s) if (np.isfinite(x) and np.isfinite(y) and y > 0 and x >= 0)]
    ys_m3s = [y for x, y in zip(xs_T, ys_m3s) if (np.isfinite(x) and np.isfinite(y) and y > 0 and x >= 0)]
    if not xs_T:
        print("[INFO] No positive values left after filtering for Temperature vs m3_short.")
        return
    plt.figure(figsize=(8, 5))
    plt.plot(xs_T, ys_m3s, 'o', alpha=0.8, label='data')
    # Optional straight-line fit
    if do_fit and len(xs_T) >= 2:
        X = np.asarray(xs_T, dtype=float)
        Y = np.asarray(ys_m3s, dtype=float)
        fit = _linear_fit_with_uncertainty(X, Y)
        if fit is not None:
            m, b, sm, sb, xfit, yfit = fit
            plt.plot(xfit, yfit, 'r--', label=f'fit: y = ({m:.4g}±{sm:.2g})·T + ({b:.4g}±{sb:.2g})')
    plt.xlabel('Temperature (C)')
    plt.ylabel('m3_short [V]')
    title = f'Temperature vs m3_short ({window_label})'
    if use_temp_corrected:
        title += ' (temp-corrected)'
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, out_name)
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")
    plt.close()

# Auto-generate Temperature vs m3_short scatter for March if the CSV exists
if os.path.exists(_overlay_temp_csv):
    plot_temp_vs_m3_short_from_cache(
        _overlay_temp_csv,
        use_temp_corrected=False,
        out_name='temp_vs_m3_short_march_17_end.png',
        start_date=datetime(2025, 3, 17, 0, 0, 0),
        end_date=datetime(2025, 3, 31, 23, 59, 59),
        do_fit=True,
    )

def plot_temp_vs_m3_long_from_cache(temp_series_csv: str,
                                    cache_file: str = FIT_CACHE_FILE,
                                    use_temp_corrected: bool = False,
                                    out_name: str = 'temp_vs_m3_long_march.png',
                                    start_date: Optional[datetime] = None,
                                    end_date: Optional[datetime] = None,
                                    do_fit: bool = True):
    """
    Scatter plot of Temperature (C) vs m3_long using cache entries, limited to
    the time span covered by the provided temperature CSV (e.g., March).
    - use_temp_corrected: prefer m3_long_temp_corr if present.
    """
    cache = _load_fit_cache(cache_file)
    times_T, vals_T = _load_time_series_temperature(temp_series_csv)
    if not times_T:
        return
    tmin, tmax = min(times_T), max(times_T)
    if isinstance(start_date, datetime) and start_date > tmin:
        tmin = start_date
    if isinstance(end_date, datetime) and end_date < tmax:
        tmax = end_date

    xs_T = []
    ys_m3l = []
    seen_dirs = set()
    for key, entry in cache.items():
        if not isinstance(entry, dict):
            continue
        meta = entry.get('meta', {})
        ts = meta.get('timestamp') or entry.get('timestamp')
        if not ts:
            continue
        try:
            t = datetime.fromisoformat(str(ts))
        except Exception:
            continue
        if t < tmin or t > tmax:
            continue
        # Skip excluded windows
        if _in_excluded(t):
            continue
        directory = meta.get('directory') or (key if isinstance(key, str) else None)
        if directory in seen_dirs:
            continue
        # Select m3_long (optionally temp-corrected)
        val = meta.get('m3_long_temp_corr' if use_temp_corrected else 'm3_long')
        if val is None and 'long' in entry and isinstance(entry['long'], dict):
            val = entry['long'].get('m3')
        if val is None:
            continue
        try:
            m3l = float(val)
        except Exception:
            continue
        if not (np.isfinite(m3l) and m3l >= 0.4):
            continue
        Tval = _nearest_temp_for_time(times_T, vals_T, t)
        if Tval is None:
            continue
        xs_T.append(Tval)
        ys_m3l.append(m3l)
        if directory:
            seen_dirs.add(directory)

    if not xs_T:
        return

    # Filter out any negative values
    xs_T = [x for x, y in zip(xs_T, ys_m3l) if (np.isfinite(x) and np.isfinite(y) and y > 0 and x >= 0)]
    ys_m3l = [y for x, y in zip(xs_T, ys_m3l) if (np.isfinite(x) and np.isfinite(y) and y > 0 and x >= 0)]
    if not xs_T:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(xs_T, ys_m3l, 'o', alpha=0.8, label='data')
    if do_fit and len(xs_T) >= 2:
        X = np.asarray(xs_T, dtype=float)
        Y = np.asarray(ys_m3l, dtype=float)
        fit = _linear_fit_with_uncertainty(X, Y)
        if fit is not None:
            m, b, sm, sb, xfit, yfit = fit
            plt.plot(xfit, yfit, 'r--', label=f'fit: y = ({m:.4g}±{sm:.2g})·T + ({b:.4g}±{sb:.2g})')
    plt.xlabel('Temperature (C)')
    plt.ylabel('m3_long [V]')
    title = 'Temperature vs m3_long (Mar 17–31 window)'
    if use_temp_corrected:
        title += ' (temp-corrected)'
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, out_name)
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")
    plt.close()

if os.path.exists(_overlay_temp_csv):
    plot_temp_vs_m3_long_from_cache(
        _overlay_temp_csv,
        use_temp_corrected=False,
        out_name='temp_vs_m3_long_march_17_end.png',
        start_date=datetime(2025, 3, 17, 0, 0, 0),
        end_date=datetime(2025, 3, 31, 23, 59, 59),
        do_fit=True,
    )

# ---------------- PRM Top lifetime CSV overlaid with cache tau (March) ---------------- #

def _load_prm_top_lifetime_csv(path: str):
    """Best-effort loader for a CSV with time and lifetime columns.
    Returns (times: list[datetime], taus: list[float]) or ([], []) on failure.
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return [], []
    if df.empty:
        return [], []
    # Pick a datetime-like column
    time_col = None
    for c in df.columns:
        s = pd.to_datetime(df[c], errors='coerce')
        if s.notna().sum() >= max(3, len(df) // 4):
            time_col = c
            times = list(s.dropna().astype('datetime64[ns]'))
            break
    if time_col is None:
        # fallback: try first column
        try:
            s = pd.to_datetime(df.iloc[:, 0], errors='coerce')
            times = list(s.dropna().astype('datetime64[ns]'))
        except Exception:
            return [], []
    # Align tau column length with parsed times by filtering same index
    if time_col is not None:
        idx = pd.to_datetime(df[time_col], errors='coerce').notna()
        df_valid = df[idx]
    else:
        idx = pd.to_datetime(df.iloc[:, 0], errors='coerce').notna()
        df_valid = df[idx]

    # Pick a numeric tau-like column (not the datetime col)
    tau_col = None
    for c in df_valid.columns:
        if c == time_col:
            continue
        try:
            vals = pd.to_numeric(df_valid[c], errors='coerce')
        except Exception:
            continue
        if vals.notna().sum() >= max(3, len(vals) // 4):
            tau_col = c
            taus = list(vals.dropna().astype(float))
            break
    # Fallback to second column
    if tau_col is None and df_valid.shape[1] >= 2:
        try:
            vals = pd.to_numeric(df_valid.iloc[:, 1], errors='coerce')
            taus = list(vals.dropna().astype(float))
        except Exception:
            return [], []
    # Ensure lengths match by trimming to min length
    n = min(len(times), len(taus))
    return list(times[:n]), list(taus[:n])

def plot_prm_top_vs_cache_tau_march(csv_path: str,
                                    cache_file: str = FIT_CACHE_FILE,
                                    year: int = 2025,
                                    out_name: str = 'prm_top_tau_overlay_march.png'):
    """
    Overlay PRM top lifetime (from CSV) with tau from fit_cache over March of a given year.
    - csv_path: path to prm_Top_lifetime_data_GainOnly.csv
    - year: calendar year to filter (default 2025)
    Saves to plots/out_name.
    """
    # Load external series
    t_ext, tau_ext = _load_prm_top_lifetime_csv(csv_path)
    if not t_ext:
        return
    # Filter to March of the specified year
    mask_march = [ (t.year == year and t.month == 3) for t in t_ext ]
    ext_times = [t for t, m in zip(t_ext, mask_march) if m]
    ext_taus  = [v for v, m in zip(tau_ext, mask_march) if m]
    if not ext_times:
        return

    # Load cache taus and m3 values; filter to March of the same year, applying short m3 cuts
    cache = _load_fit_cache(cache_file)
    cache_pts = []  # (time, tau)
    ratio_pts = []  # (time, q_l/ q_s)
    seen_dirs = set()
    for key, entry in cache.items():
        if not isinstance(entry, dict):
            continue
        meta = entry.get('meta', {})
        if 'timestamp' not in meta or 'tau' not in meta:
            continue
        try:
            t = datetime.fromisoformat(str(meta['timestamp']))
        except Exception:
            continue
        # Skip excluded windows
        if _in_excluded(t):
            continue
        if t.year != year or t.month != 3:
            continue
        directory = meta.get('directory') or (key if isinstance(key, str) else None)
        if directory in seen_dirs:
            continue
        # Apply short m3 cut if available
        m3s = meta.get('m3_short')
        if m3s is None:
            ent2 = cache.get(key)
            if ent2 and 'short' in ent2:
                m3s = ent2['short'].get('m3')
        try:
            m3s = float(m3s) if m3s is not None else None
        except Exception:
            m3s = None
        if (m3s is None) or not (M3_SHORT_MIN < m3s < M3_SHORT_MAX):
            continue
        tau_val = meta['tau']
        try:
            tau_val = float(tau_val)
        except Exception:
            continue
        cache_pts.append((t, tau_val))
        # Also collect ratio q_l/ q_s for overlay
        m3l = meta.get('m3_long')
        if m3l is None:
            ent2 = cache.get(key)
            if ent2 and 'long' in ent2:
                m3l = ent2['long'].get('m3')
        try:
            m3l = float(m3l) if m3l is not None else None
        except Exception:
            m3l = None
        if (m3l is not None) and np.isfinite(m3l) and (m3l >= 0.4):
            r = m3l / m3s if (m3s and np.isfinite(m3s) and m3s != 0) else None
            if r is not None and np.isfinite(r):
                ratio_pts.append((t, r))
        if directory:
            seen_dirs.add(directory)

    # Sort
    ext_idx = np.argsort([t.timestamp() for t in ext_times])
    ext_times = [ext_times[i] for i in ext_idx]
    ext_taus  = [ext_taus[i] for i in ext_idx]
    cache_pts.sort(key=lambda x: x[0])
    cache_times = [p[0] for p in cache_pts]
    cache_taus  = [p[1] for p in cache_pts]

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(ext_times, ext_taus, '-o', label='PRM Top ', color='tab:blue', alpha=0.8)
    if cache_times:
        ax1.plot(cache_times, cache_taus, 'o', label='tau', color='tab:orange')
    ax1.set_yscale('log')
    ax1.set_xlabel('Time')
    ax1.set_ylabel(r'Electron Lifetime $\tau_e$ (ms)')
    ax1.set_title(fr'Electron Lifetime $\tau_e$ — March {year}')
    fig.autofmt_xdate()

    axes_for_legend = [ax1]

    # Overlay ratio (q_l/ q_s) on a secondary y-axis
    ax2 = None
    if ratio_pts:
        r_times = [t for t, _ in ratio_pts]
        r_vals  = [v for _, v in ratio_pts]
        ax2 = ax1.twinx()
        ax2.plot(r_times, r_vals, 'o', color='tab:green', label='Q_l/ Q_s')
        ax2.set_ylabel('Q_l/ Q_s')
        axes_for_legend.append(ax2)

    # Overlay temperature (C) on a third y-axis if available
    ax3 = None
    try:
        if '_overlay_temp_csv' in globals() and os.path.exists(_overlay_temp_csv):
            tT, vT = _load_time_series_temperature(_overlay_temp_csv)
            if tT:
                filt = [(tt.year == year and tt.month == 3) for tt in tT]
                tT = [tt for tt, keep in zip(tT, filt) if keep]
                vT = [vv for vv, keep in zip(vT, filt) if keep]
                if tT:
                    order = np.argsort([tt.timestamp() for tt in tT])
                    tT = [tT[i] for i in order]
                    vT = [vT[i] for i in order]
                    ax3 = ax1.twinx()
                    # Offset the right spine to avoid overlap with ax2 (if present)
                    ax3.spines['right'].set_position(('outward', 60))
                    ax3.plot(tT, vT, '-', color='tab:red', alpha=0.7, label='Temperature (C)')
                    ax3.set_ylabel('Temperature (C)')
                    axes_for_legend.append(ax3)
    except Exception:
        pass

    # Merge legends from all axes
    handles = []
    labels = []
    for ax in axes_for_legend:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    if handles:
        ax1.legend(handles, labels, loc='best')

    fig.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, out_name)
    fig.savefig(out_path, dpi=1500)
    print(f"Saved plot: {out_path}")
    if not SAVE_FILES:
        plt.show()
    plt.close(fig)

_prm_top_csv = os.path.join(NP02DATA_DIR, 'prm_Top_lifetime_data_GainOnly.csv')
if os.path.exists(_prm_top_csv):
    plot_prm_top_vs_cache_tau_march(_prm_top_csv, year=2025, out_name='prm_top_tau_overlay_march.png')

# Compare PRM Top lifetime with temperature-corrected tau (March)
def plot_prm_top_vs_corrected_tau_march(csv_path: str,
                                        cache_file: str = FIT_CACHE_FILE,
                                        year: int = 2025,
                                        out_name: str = 'prm_top_vs_tau_corrected_march.png'):
    """Overlay PRM top lifetime with temperature-corrected tau (tau_temp_corr_ratio)
    from cache, restricted to March of the given year.
    """
    # Load external series
    t_ext, tau_ext = _load_prm_top_lifetime_csv(csv_path)
    if not t_ext:
        return
    # Filter to March of the specified year
    mask_march = [ (t.year == year and t.month == 3) for t in t_ext ]
    ext_times = [t for t, m in zip(t_ext, mask_march) if m]
    ext_taus  = [v for v, m in zip(tau_ext, mask_march) if m]
    if not ext_times:
        return

    # Load cache corrected tau and filter to March/year with cuts
    cache = _load_fit_cache(cache_file)
    pts = []  # (time, tau_corr)
    seen = set()
    for key, entry in cache.items():
        if not isinstance(entry, dict):
            continue
        meta = entry.get('meta', {})
        if 'timestamp' not in meta or 'tau_temp_corr_ratio' not in meta:
            continue
        try:
            t = datetime.fromisoformat(str(meta['timestamp']))
        except Exception:
            continue
        # Skip excluded windows
        if _in_excluded(t):
            continue
        if t.year != year or t.month != 3:
            continue
        directory = meta.get('directory') or (key if isinstance(key, str) else None)
        if directory in seen:
            continue
        # Apply the same m3 cuts for consistency
        m3s = meta.get('m3_short')
        if m3s is None and 'short' in entry:
            m3s = entry['short'].get('m3')
        m3l = meta.get('m3_long')
        if m3l is None and 'long' in entry:
            m3l = entry['long'].get('m3')
        try:
            m3s = float(m3s) if m3s is not None else None
            m3l = float(m3l) if m3l is not None else None
        except Exception:
            continue
        if (m3s is None) or not (M3_SHORT_MIN < m3s < M3_SHORT_MAX):
            continue
        if (m3l is None) or not (np.isfinite(m3l) and m3l >= 0.4):
            continue
        tau_c = meta['tau_temp_corr_ratio']
        try:
            tau_c = float(tau_c)
        except Exception:
            continue
        # Require positive, finite tau for log-scale plotting
        if not (np.isfinite(tau_c) and tau_c > 0):
            continue
        pts.append((t, tau_c))
        if directory:
            seen.add(directory)

    if not pts:
        return

    # Sort
    ext_idx = np.argsort([t.timestamp() for t in ext_times])
    ext_times = [ext_times[i] for i in ext_idx]
    ext_taus  = [ext_taus[i] for i in ext_idx]
    pts.sort(key=lambda x: x[0])
    corr_times = [p[0] for p in pts]
    corr_taus  = [p[1] for p in pts]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(ext_times, ext_taus, '-o', label='PRM Top (gain only)', color='tab:blue', alpha=0.8)
    plt.plot(corr_times, corr_taus, 'o', label='Tau (temp-corrected via ratio)', color='tab:orange')
    plt.yscale('log')
    plt.xlabel('Time')
    plt.ylabel(r'Electron Lifetime $\tau_e$ (ms)')
    plt.title(fr'PRM Top vs corrected $\tau_e$ — March {year}')
    plt.gcf().autofmt_xdate()
    plt.legend(loc='best')
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, out_name)
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")
    # Also echo the tau raw vs corrected plot path if present, per request
    try:
        tau_cmp_path = os.path.join(PLOTS_DIR, 'tau_raw_vs_corrected.png')
        if os.path.exists(tau_cmp_path):
            print(f"Saved plot: {tau_cmp_path}")
    except Exception:
        pass
    if not SAVE_FILES:
        plt.show()
    plt.close()

def plot_prm_top_vs_corrected_tau_overlay(prm_csv: str,
                                          results,
                                          year: int = 2025,
                                          month: int = 3,
                                          out_name: str = 'prm_top_vs_tau_corrected_march.png'):
    """Overlay PRM top lifetime with in-memory corrected tau results for March of given year.
    - prm_csv: path to prm_Top_lifetime_data_GainOnly.csv
    - results: list of (time, tau_raw, tau_corr, directory) from apply_ratio_temp_correction_to_tau
    """
    if not results:
        print("[INFO] No corrected tau results to overlay with PRM Top.")
        return
    # Load external series
    t_ext, tau_ext = _load_prm_top_lifetime_csv(prm_csv)
    if not t_ext:
        print(f"[INFO] No PRM Top series found at {prm_csv}")
        return
    mask_march = [ (t.year == year and t.month == month) for t in t_ext ]
    ext_times = [t for t, m in zip(t_ext, mask_march) if m]
    ext_taus  = [v for v, m in zip(tau_ext, mask_march) if m]
    if not ext_times:
        print(f"[INFO] No PRM Top data for {year}-{month:02d}.")
        return
    # Corrected tau points (already filtered to March inside computation)
    rows = [(t, tau_corr) for (t, _tau_raw, tau_corr, _dir) in results
            if (t.year == year and t.month == month and np.isfinite(tau_corr) and (tau_corr > 0))]
    if not rows:
        print("[INFO] No corrected tau points in March to overlay.")
        return
    rows.sort(key=lambda r: r[0])
    corr_times = [r[0] for r in rows]
    corr_taus  = [float(r[1]) for r in rows]
    # Sort external series
    order = np.argsort([t.timestamp() for t in ext_times])
    ext_times = [ext_times[i] for i in order]
    ext_taus  = [ext_taus[i] for i in order]
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ext_times, ext_taus, 'o', label=r'$\tau$ Classical UV Lamp', color='tab:blue', alpha=0.8)
    # Direct points without moving average smoothing
    ax.plot(corr_times, corr_taus, 'o', label=r'$\tau$ Bi Source PM (corrected)', color='tab:orange', alpha=0.9)

    ax.set_yscale('log')
    ax.set_xlabel('Time')
    ax.set_ylabel(r'Electron Lifetime $\tau_e$ (ms)')
    try:
        mname = calendar.month_name[month]
    except Exception:
        mname = f'Month {month}'
    ax.set_title(rf'$\tau$ Classical UV Lamp vs Bi Source PM (corrected)  — {mname} {year}')
    fig.autofmt_xdate()
    ax.legend(loc='best')
    fig.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, out_name)
    fig.savefig(out_path, dpi=500)
    print(f"Saved plot: {out_path}")
    if not SAVE_FILES:
        plt.show()
    plt.close(fig)

# ---------------- Scatter: Temperature vs (m3_long / m3_short) within temperature month ---------------- #

def plot_temp_vs_ratio_within_month(temp_series_csv: str,
                                    cache_file: str = FIT_CACHE_FILE,
                                    out_name: str = 'temp_vs_m3_ratio_march.png',
                                    start_date: Optional[datetime] = None,
                                    end_date: Optional[datetime] = None):
    """
    Temperature (C) vs (Q_l/Q_s) using cache entries whose timestamps fall within the
    temperature CSV coverage (e.g., a month). Plots original and corrected ratios
    (corrected using the ln(R) vs T model), and overlays model-derived fits.
    Uses interpolated temperature lookup.
    """
    cache = _load_fit_cache(cache_file)
    times_T, vals_T = _load_time_series_temperature(temp_series_csv)
    if not times_T:
        print(f"[INFO] No temperature series found at {temp_series_csv}; skipping ratio plot.")
        return
    # Window bounds from the temperature file, optionally restricted
    tmin, tmax = min(times_T), max(times_T)
    if isinstance(start_date, datetime) and start_date > tmin:
        tmin = start_date
    if isinstance(end_date, datetime) and end_date < tmax:
        tmax = end_date

    # Fit ln(R) vs T model once for this window
    model = get_ratio_temp_model(temp_series_csv, cache_file, start_date=tmin, end_date=tmax)
    T_ref = float(model['T_ref']) if isinstance(model, dict) and ('T_ref' in model) else None
    a = float(model['a']) if isinstance(model, dict) and ('a' in model) else None
    b = float(model['b']) if isinstance(model, dict) and ('b' in model) else None

    xs_T = []       # Temperature
    ys_R = []       # Original ratio
    ys_Rcorr = []   # Corrected ratio
    seen_dirs = set()
    for key, entry in cache.items():
        if not isinstance(entry, dict):
            continue
        meta = entry.get('meta', {})
        ts = meta.get('timestamp') or entry.get('timestamp')
        if not ts:
            continue
        try:
            t = datetime.fromisoformat(str(ts))
        except Exception:
            continue
        if t < tmin or t > tmax:
            continue
        # Skip excluded windows
        if _in_excluded(t):
            continue
        directory = meta.get('directory') or (key if isinstance(key, str) else None)
        if directory in seen_dirs:
            continue
        # Pull m3 values (original)
        m3s = meta.get('m3_short')
        m3l = meta.get('m3_long')
        if m3s is None or m3l is None:
            if 'short' in entry and m3s is None:
                m3s = entry['short'].get('m3')
            if 'long' in entry and m3l is None:
                m3l = entry['long'].get('m3')
        if m3s is None or m3l is None:
            continue
        try:
            m3s = float(m3s)
            m3l = float(m3l)
        except Exception:
            continue
        # Validity cuts
        if not (M3_SHORT_MIN < m3s < M3_SHORT_MAX):
            continue
        if not (np.isfinite(m3l) and m3l >= 0.4):
            continue
        # Interpolated temperature at this time
        Tval = _nearest_temp_for_time(times_T, vals_T, t)
        if Tval is None:
            continue
        ratio = m3l / m3s
        if not np.isfinite(ratio):
            continue
        xs_T.append(float(Tval))
        ys_R.append(float(ratio))
        if (T_ref is not None) and (b is not None):
            lnR = float(np.log(ratio))
            lnR_corr = lnR - b * (float(Tval) - T_ref)
            ys_Rcorr.append(float(np.exp(lnR_corr)))
        if directory:
            seen_dirs.add(directory)

    if not xs_T:
        print("[INFO] No valid points for Temperature vs (Q_l/ Q_s) in the selected month window.")
        return

    plt.figure(figsize=(8, 5))
    X = np.asarray(xs_T, dtype=float)
    Y = np.asarray(ys_R, dtype=float)
    plt.plot(X, Y, 'o', alpha=0.65, label='Original ratio')
    if ys_Rcorr:
        Yc = np.asarray(ys_Rcorr, dtype=float)
        plt.plot(X, Yc, 'o', alpha=0.65, label=(f'Corrected ratio @ T_ref={T_ref:.2f} C' if T_ref is not None else 'Corrected ratio'))
        # Overlay model-derived fits if model is present
        if (a is not None) and (b is not None) and (T_ref is not None):
            xfit = np.linspace(float(np.min(X)), float(np.max(X)), 200)
            yfit_orig = np.exp(a + b * (xfit - T_ref))
            yfit_corr = np.exp(a) * np.ones_like(xfit)
            plt.plot(xfit, yfit_orig, 'b--', label='Original fit (from ln-model)')
            plt.plot(xfit, yfit_corr, 'g--', label='Corrected fit (constant @ T_ref)')
    plt.xlabel('Temperature (C)')
    plt.ylabel('Q_l/Q_s')
    plt.title('Temperature vs (Q_l/Q_s) — original and corrected')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, out_name)
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")
    plt.close()

# (moved) Auto-generate Temperature vs ratio scatter is placed after model helpers

# ---------------- Temperature correction of tau via ratio trend ---------------- #

def fit_ratio_temp_dependence_from_cache(temp_series_csv: str,
                                         cache_file: str = FIT_CACHE_FILE,
                                         min_points: int = 20,
                                         month: Optional[int] = None,
                                         year: Optional[int] = None,
                                         start_date: Optional[datetime] = None,
                                         end_date: Optional[datetime] = None):
    """Fit ln(R) vs T where R = Q_l/ Q_s and store the model in cache.
    Returns the model dict or None on failure.
    """
    cache = _load_fit_cache(cache_file)
    times_T, vals_T = _load_time_series_temperature(temp_series_csv)
    if not times_T:
        print(f"[INFO] No temperature series at {temp_series_csv}")
        return None

    Ts = []
    lnRs = []
    seen_dirs = set()
    for key, entry in cache.items():
        if not isinstance(entry, dict):
            continue
        meta = entry.get('meta', {})
        ts = meta.get('timestamp') or entry.get('timestamp')
        if not ts:
            continue
        try:
            t = datetime.fromisoformat(str(ts))
        except Exception:
            continue

        directory = meta.get('directory') or (key if isinstance(key, str) else None)
        if directory in seen_dirs:
            continue
        # Skip excluded windows
        if _in_excluded(t):
            continue
        # Optional month/year filter for model fit
        if (month is not None and t.month != month) or (year is not None and t.year != year):
            continue
        # Optional absolute time window (inclusive)
        if (start_date is not None and t < start_date):
            continue
        if (end_date is not None and t > end_date):
            continue

        m3s = meta.get('m3_short')
        m3l = meta.get('m3_long')
        if m3s is None and 'short' in entry:
            m3s = entry['short'].get('m3')
        if m3l is None and 'long' in entry:
            m3l = entry['long'].get('m3')
        try:
            m3s = float(m3s) if m3s is not None else None
            m3l = float(m3l) if m3l is not None else None
        except Exception:
            continue

        # Validity cuts
        if (m3s is None) or not (M3_SHORT_MIN < m3s < M3_SHORT_MAX):
            continue
        if (m3l is None) or not (np.isfinite(m3l) and m3l >= 0.4):
            continue

        Tval = _nearest_temp_for_time(times_T, vals_T, t)
        if Tval is None:
            continue

        R = m3l / m3s
        if not np.isfinite(R) or R <= 0:
            continue

        Ts.append(float(Tval))
        lnRs.append(float(np.log(R)))
        if directory:
            seen_dirs.add(directory)

    if len(Ts) < min_points:
        print(f"[INFO] Not enough points for ratio-vs-temp fit (n={len(Ts)})")
        return None

    Ts = np.asarray(Ts, float)
    lnRs = np.asarray(lnRs, float)
    T_ref = 17.8
    dT = Ts - T_ref
    try:
        import warnings as _warn
        with _warn.catch_warnings():
            _warn.simplefilter('ignore')
            (b, a), cov = np.polyfit(dT, lnRs, 1, cov=True)
        err_b = float(np.sqrt(cov[0, 0])) if cov.size >= 4 else np.nan
        err_a = float(np.sqrt(cov[1, 1])) if cov.size >= 4 else np.nan
    except Exception:
        print("[INFO] Ratio fit failed")
        return None

    # Create a compact model signature to detect whether corrected values are up-to-date
    model_sig = f"Tref={T_ref:.3f}|b={b:.6g}|n={int(len(Ts))}"
    model = {'T_ref': T_ref, 'a': float(a), 'b': float(b),
             'err_a': err_a, 'err_b': err_b, 'n_points': int(len(Ts)),
             'model_sig': model_sig}
    # Do not write model back to cache file by default
    if WRITE_CACHE_BACK:
        cache['__temp_ratio_correction__'] = model
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
    if start_date or end_date:
        sd = start_date.isoformat(sep=' ') if isinstance(start_date, datetime) else 'None'
        ed = end_date.isoformat(sep=' ') if isinstance(end_date, datetime) else 'None'
        print(f"[INFO] Fitted ln(R) vs T (window {sd} to {ed}): b={b:.4g}±{err_b:.1g} (n={len(Ts)}) at T_ref={T_ref:.2f} C")
    else:
        print(f"[INFO] Fitted ln(R) vs T: b={b:.4g}±{err_b:.1g} (n={len(Ts)}) at T_ref={T_ref:.2f} C")
    return model

def get_ratio_temp_model(temp_series_csv: str,
                         cache_file: str = FIT_CACHE_FILE,
                         month: Optional[int] = None,
                         year: Optional[int] = None,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None):
    """Return a cached in-memory ratio-vs-T model for this temp CSV+month/year, fitting once if needed."""
    global _local_ratio_models
    sd_key = start_date.isoformat() if isinstance(start_date, datetime) else None
    ed_key = end_date.isoformat() if isinstance(end_date, datetime) else None
    key = f"{os.path.abspath(temp_series_csv)}|m={month}|y={year}|sd={sd_key}|ed={ed_key}"
    if key in _local_ratio_models and isinstance(_local_ratio_models[key], dict):
        return _local_ratio_models[key]
    model = fit_ratio_temp_dependence_from_cache(temp_series_csv, cache_file,
                                                 month=month, year=year,
                                                 start_date=start_date, end_date=end_date)
    if isinstance(model, dict):
        _local_ratio_models[key] = model
    return model

# Auto-generate Temperature vs ratio scatter for the temperature month if CSV exists
if os.path.exists(_overlay_temp_csv):
    plot_temp_vs_ratio_within_month(
        _overlay_temp_csv,
        out_name='temp_vs_m3_ratio_march_17_end.png',
        start_date=datetime(2025, 3, 17, 0, 0, 0),
        end_date=datetime(2025, 3, 31, 23, 59, 59),
    )


def apply_ratio_temp_correction_to_tau(temp_series_csv: str,
                                       cache_file: str = FIT_CACHE_FILE,
                                       model: Optional[dict] = None,
                                       month: Optional[int] = 3,
                                       year: Optional[int] = None):
    """Compute ratio-based temperature correction to tau (in-memory only).
    Returns a list of (time, tau_raw, tau_corr, directory) filtered by optional month/year.
    Does not write back to the cache file.
    """
    cache = _load_fit_cache(cache_file)
    if not isinstance(model, dict):
        model = get_ratio_temp_model(temp_series_csv, cache_file)
        if not isinstance(model, dict):
            return
    T_ref = float(model['T_ref'])
    b = float(model['b'])
    model_sig = str(model.get('model_sig', f"Tref={T_ref:.3f}|b={b:.6g}"))
    times_T, vals_T = _load_time_series_temperature(temp_series_csv)
    if not times_T:
        print(f"[INFO] No temperature series at {temp_series_csv}")
        return

    updated = 0
    results = []
    for key, entry in list(cache.items()):
        if not isinstance(entry, dict):
            continue
        meta = entry.get('meta')
        if not isinstance(meta, dict):
            continue
        ts = meta.get('timestamp')
        if not ts:
            continue
        try:
            t = datetime.fromisoformat(str(ts))
        except Exception:
            continue
        # Skip excluded windows
        if _in_excluded(t):
            continue
        # Optional month/year filter
        if (month is not None and t.month != month) or (year is not None and t.year != year):
            continue

        Tval = _nearest_temp_for_time(times_T, vals_T, t)
        if Tval is None:
            continue

        m3s = meta.get('m3_short')
        m3l = meta.get('m3_long')
        if m3s is None and 'short' in entry:
            m3s = entry['short'].get('m3')
        if m3l is None and 'long' in entry:
            m3l = entry['long'].get('m3')
        try:
            m3s = float(m3s) if m3s is not None else None
            m3l = float(m3l) if m3l is not None else None
        except Exception:
            continue
        if (m3s is None) or not (M3_SHORT_MIN < m3s < M3_SHORT_MAX):
            continue
        if (m3l is None) or not (np.isfinite(m3l) and m3l >= 0.4):
            continue
        # Current inputs
        R = m3l / m3s
        if not (np.isfinite(R) and R > 0):
            continue

        lnR_corr = np.log(R) - b * (float(Tval) - T_ref)
        R_corr = float(np.exp(lnR_corr))

        delta_t, scaling = get_delta_t_and_scaling_factor(t)
        try:
            tau_corr = -delta_t / np.log(R_corr / scaling)  # ms
        except Exception:
            continue
        if np.isfinite(tau_corr) and (tau_corr > 0):
            # Collect for plotting; do not write back to cache
            try:
                tau_raw = float(meta.get('tau')) if meta.get('tau') is not None else np.nan
            except Exception:
                tau_raw = np.nan
            directory = meta.get('directory') or (key if isinstance(key, str) else None)
            results.append((t, tau_raw, float(tau_corr), directory))
            updated += 1

    if updated:
        print(f"[INFO] Computed tau_temp_corr_ratio (in-memory) for {updated} March entries.")
    return results

def compute_corrected_tau_with_error(temp_series_csv: str,
                                     cache_file: str = FIT_CACHE_FILE,
                                     model: Optional[dict] = None,
                                     month: Optional[int] = 3,
                                     year: Optional[int] = None,
                                     ratio_err_frac: float = 0.01,
                                     use_poisson_for_upper: bool = True,
                                     cl: float = 0.90):
    """Compute corrected tau and +-1% ratio-derived bounds.
    Returns list of (time, tau_nom_ms, tau_low_ms, tau_high_ms, directory).
    Skips non-finite or non-positive taus (for log-scale plotting).
    """
    cache = _load_fit_cache(cache_file)
    if not isinstance(model, dict):
        model = get_ratio_temp_model(temp_series_csv, cache_file)
        if not isinstance(model, dict):
            return []
    T_ref = float(model['T_ref'])
    b = float(model['b'])
    # Temperature series for nearest lookup
    times_T, vals_T = _load_time_series_temperature(temp_series_csv)
    if not times_T:
        return []
    out = []
    seen_dirs = set()
    for key, entry in cache.items():
        if not isinstance(entry, dict):
            continue
        meta = entry.get('meta', {})
        ts = meta.get('timestamp') or entry.get('timestamp')
        if not ts:
            continue
        try:
            t = datetime.fromisoformat(str(ts))
        except Exception:
            continue
        # Skip excluded windows
        if _in_excluded(t):
            continue
        if (month is not None and t.month != month) or (year is not None and t.year != year):
            continue
        directory = meta.get('directory') or (key if isinstance(key, str) else None)
        if directory in seen_dirs:
            continue

        m3s = meta.get('m3_short')
        m3l = meta.get('m3_long')
        if m3s is None and 'short' in entry:
            m3s = entry['short'].get('m3')
        if m3l is None and 'long' in entry:
            m3l = entry['long'].get('m3')
        try:
            m3s = float(m3s) if m3s is not None else None
            m3l = float(m3l) if m3l is not None else None
        except Exception:
            continue
        if (m3s is None) or not (M3_SHORT_MIN < m3s < M3_SHORT_MAX):
            continue
        if (m3l is None) or not (np.isfinite(m3l) and m3l >= 0.4):
            continue
        if not (np.isfinite(m3s) and np.isfinite(m3l) and m3s != 0):
            continue

        Tval = _nearest_temp_for_time(times_T, vals_T, t)
        if Tval is None:
            continue

        R = m3l / m3s
        if not (np.isfinite(R) and R > 0):
            continue

        # Corrected ratio and +-1%
        lnR_corr = np.log(R) - b * (float(Tval) - T_ref)
        R_corr = float(np.exp(lnR_corr))
        R_corr_hi = R_corr * (1.0 + float(ratio_err_frac))
        R_corr_lo = R_corr * max(1e-9, (1.0 - float(ratio_err_frac)))

        delta_t, scaling = get_delta_t_and_scaling_factor(t)
        # Nominal tau must be positive/finite; otherwise treat as no finite estimate
        x_nom = R_corr / scaling
        if x_nom <= 0:
            continue
        ln_nom = np.log(x_nom)
        if not np.isfinite(ln_nom) or ln_nom >= 0:
            # Nominal implies tau -> negative/undefined; skip point for symmetric error plot
            # (Use Poisson lower-limit plot for these cases.)
            continue
        tau_nom = -delta_t / ln_nom 
        # Lower bound from (1 - err)
        x_lo = R_corr_lo / scaling
        ln_lo = np.log(x_lo) if x_lo > 0 else np.nan
        if not np.isfinite(ln_lo) or ln_lo >= 0:
            # No finite lower bound — skip this point in symmetric error plot
            continue
        tau_lo = -delta_t / ln_lo 
        # Upper bound from (1 + err); may be invalid (one-sided)
        x_hi = R_corr_hi / scaling
        tau_hi = np.nan
        if x_hi > 0:
            ln_hi = np.log(x_hi)
            if np.isfinite(ln_hi) and ln_hi < 0:
                tau_hi = -delta_t / ln_hi 
        # If upper still invalid and allowed, try Poisson-based lnR uncertainty
        if (not np.isfinite(tau_hi)) and use_poisson_for_upper:
            # z for one-sided/two-sided? Use two-sided symmetric around ln x for an 'upper error bar'
            z_by_cl = {0.80: 1.281551566, 0.90: 1.644853627, 0.95: 1.959963985}
            z = z_by_cl.get(round(cl, 2), 1.644853627)
            nS, nL = _extract_counts_from_entry(entry, meta)
            sigma_lnR = None
            if nS is not None and nL is not None and nS > 0 and nL > 0:
                sigma_lnR = np.sqrt(1.0/float(nL) + 1.0/float(nS))
            else:
                # fallback to fractional uncertainty if counts absent
                eps = float(ratio_err_frac)
                sigma_lnR = max(eps*np.sqrt(2.0), eps*2.0)
            ln_x_nom = ln_nom  # ln(Rcorr/alpha)
            ln_x_hi = ln_x_nom + z * sigma_lnR
            if ln_x_hi < 0:
                tau_hi = -delta_t / ln_x_hi 
        # Ensure ordering and build output; if tau_hi invalid, omit upper error (set to NaN)
        tau_low = min(tau_lo, tau_nom, tau_hi) if np.isfinite(tau_hi) else min(tau_lo, tau_nom)
        tau_high = max(tau_lo, tau_nom, tau_hi) if np.isfinite(tau_hi) else np.nan
        out.append((t, float(tau_nom), float(tau_low), (float(tau_high) if np.isfinite(tau_high) else np.nan), directory))
        if directory:
            seen_dirs.add(directory)
    out.sort(key=lambda r: r[0])
    return out

def compute_corrected_tau_with_model_sigma(temp_series_csv: str,
                                           cache_file: str = FIT_CACHE_FILE,
                                           model: Optional[dict] = None,
                                           month: Optional[int] = 3,
                                           year: Optional[int] = None):
    """Compute corrected tau and bounds from ln(R) vs T slope uncertainty.
    For each point, compute tau_nom using b, and tau bounds using (b ± err_b).
    Returns list of (time, tau_nom, tau_low, tau_high, directory).
    """
    cache = _load_fit_cache(cache_file)
    if not isinstance(model, dict):
        model = get_ratio_temp_model(temp_series_csv, cache_file)
        if not isinstance(model, dict):
            return []
    T_ref = float(model['T_ref'])
    b = float(model['b'])
    err_b = model.get('err_b')
    try:
        sb = float(err_b) if err_b is not None else None
    except Exception:
        sb = None
    times_T, vals_T = _load_time_series_temperature(temp_series_csv)
    if not times_T:
        return []
    out = []
    seen_dirs = set()
    for key, entry in cache.items():
        if not isinstance(entry, dict):
            continue
        meta = entry.get('meta', {})
        ts = meta.get('timestamp') or entry.get('timestamp')
        if not ts:
            continue
        try:
            t = datetime.fromisoformat(str(ts))
        except Exception:
            continue
        # Skip excluded windows and month/year filter
        if _in_excluded(t):
            continue
        if (month is not None and t.month != month) or (year is not None and t.year != year):
            continue
        directory = meta.get('directory') or (key if isinstance(key, str) else None)
        if directory in seen_dirs:
            continue
        # m3 values and cuts
        m3s = meta.get('m3_short')
        m3l = meta.get('m3_long')
        if m3s is None and 'short' in entry:
            m3s = entry['short'].get('m3')
        if m3l is None and 'long' in entry:
            m3l = entry['long'].get('m3')
        try:
            m3s = float(m3s) if m3s is not None else None
            m3l = float(m3l) if m3l is not None else None
        except Exception:
            continue
        if (m3s is None) or not (M3_SHORT_MIN < m3s < M3_SHORT_MAX):
            continue
        if (m3l is None) or not (np.isfinite(m3l) and m3l >= 0.4):
            continue
        if not (np.isfinite(m3s) and np.isfinite(m3l) and m3s != 0):
            continue
        # Temperature
        Tval = _nearest_temp_for_time(times_T, vals_T, t)
        if Tval is None:
            continue
        R = m3l / m3s
        if not (np.isfinite(R) and R > 0):
            continue
        # Nominal corrected ratio
        lnR = np.log(R)
        dt = float(Tval) - T_ref
        lnR_corr = lnR - b * dt
        R_corr = float(np.exp(lnR_corr))
        delta_t, scaling = get_delta_t_and_scaling_factor(t)
        x_nom = R_corr / scaling
        if x_nom <= 0:
            continue
        ln_x_nom = np.log(x_nom)
        if not np.isfinite(ln_x_nom) or ln_x_nom >= 0:
            continue
        tau_nom = -delta_t / ln_x_nom
        # Bounds from slope uncertainty if available
        tau_candidates = [tau_nom]
        if (sb is not None) and np.isfinite(sb) and (sb >= 0):
            for b_eff in (b - sb, b + sb):
                lnR_c = lnR - b_eff * dt
                R_c = float(np.exp(lnR_c))
                x_c = R_c / scaling
                if x_c > 0:
                    ln_x_c = np.log(x_c)
                    if np.isfinite(ln_x_c) and ln_x_c < 0:
                        tau_c = -delta_t / ln_x_c
                        if np.isfinite(tau_c) and tau_c > 0:
                            tau_candidates.append(float(tau_c))
        # Build asymmetric range if we have more than one candidate
        if len(tau_candidates) >= 1:
            tau_low = min(tau_candidates)
            tau_high = max(tau_candidates) if len(tau_candidates) >= 2 else np.nan
            out.append((t, float(tau_nom), float(tau_low), (float(tau_high) if np.isfinite(tau_high) else np.nan), directory))
            if directory:
                seen_dirs.add(directory)
    out.sort(key=lambda r: r[0])
    return out

def plot_corrected_tau_with_model_sigma(temp_series_csv: str,
                                        cache_file: str = FIT_CACHE_FILE,
                                        model: Optional[dict] = None,
                                        month: Optional[int] = 3,
                                        year: Optional[int] = 2025,
                                        out_name: str = 'tau_corrected_with_fit_sigma.png'):
    """Plot corrected tau with error bars derived from ln(R) vs T slope uncertainty (log y-axis)."""
    rows = compute_corrected_tau_with_model_sigma(temp_series_csv, cache_file, model=model,
                                                  month=month, year=year)
    if not rows:
        print("[INFO] No corrected tau points with slope-uncertainty error bars to plot.")
        return
    times = [r[0] for r in rows]
    tau_nom = np.array([r[1] for r in rows], dtype=float)
    tau_lo = np.array([r[2] for r in rows], dtype=float)
    tau_hi = np.array([r[3] for r in rows], dtype=float)
    yerr_low = np.maximum(0.0, tau_nom - tau_lo)
    yerr_high = np.where(np.isfinite(tau_hi), np.maximum(0.0, tau_hi - tau_nom), 0.0)
    yerr = np.vstack([yerr_low, yerr_high])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(times, tau_nom, yerr=yerr, fmt='o', color='tab:orange', ecolor='tab:orange',
                elinewidth=1, capsize=2, label='Tau (temp-corrected) ± slope σ')
    ax.set_yscale('log')
    ax.set_xlabel('Time')
    ax.set_ylabel(r'Electron Lifetime $\tau_e$ (ms)')
    ax.set_title(f'Temperature-corrected tau with ± slope σ — {"March " + str(year) if month == 3 else "Selected window"}')
    fig.autofmt_xdate()
    ax.legend(loc='best')
    fig.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, out_name)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")
    plt.close(fig)

def _extract_counts_from_entry(entry: dict, meta: dict):
    """Best-effort extraction of event counts (nS, nL) from cache entry.
    Returns tuple (n_short, n_long) or (None, None) if unavailable.
    """
    def _pick(d: dict):
        if not isinstance(d, dict):
            return None
        for k in ['n', 'N', 'count', 'counts', 'entries', 'num_events', 'n_events', 'events', 'total']:
            if k in d:
                try:
                    v = float(d[k])
                    if np.isfinite(v) and v > 0:
                        return int(v)
                except Exception:
                    pass
        return None
    # Try nested short/long blocks
    nS = _pick(entry.get('short'))
    nL = _pick(entry.get('long'))
    # Try meta-level hints
    for kS in ['n_short', 'short_n', 'nS', 'short_count']:
        if nS is None and kS in meta:
            try:
                v = float(meta[kS]);
                if np.isfinite(v) and v > 0:
                    nS = int(v)
            except Exception:
                pass
    for kL in ['n_long', 'long_n', 'nL', 'long_count']:
        if nL is None and kL in meta:
            try:
                v = float(meta[kL]);
                if np.isfinite(v) and v > 0:
                    nL = int(v)
            except Exception:
                pass
    return nS, nL

def compute_tau_one_sided_lower_CL_poisson(temp_series_csv: str,
                                           cache_file: str = FIT_CACHE_FILE,
                                           model: Optional[dict] = None,
                                           month: Optional[int] = 3,
                                           year: Optional[int] = None,
                                           cl: float = 0.90,
                                           fallback_ratio_err_frac: float = 0.01):
    """Compute one-sided lower confidence limits on tau using Poisson error on ln R.
    - Uses sigma_lnR ≈ sqrt(1/nL + 1/nS) when counts (nS, nL) are available in cache.
    - Falls back to a fixed fractional ratio error (fallback_ratio_err_frac) when counts are absent.
    Returns list of (time, tau_lower_ms, n_short, n_long, directory). Skips points with no finite bound.
    """
    # z for one-sided CL (Normal approximation)
    # Use exact constant for common CLs; otherwise approximate via inverse erfc if desired.
    z_by_cl = {0.80: 0.841621233, 0.90: 1.281551566, 0.95: 1.644853627, 0.975: 1.959963985}
    # One-sided 90% -> 1.2816, 95% -> 1.6449. Default 0.90
    z = z_by_cl.get(round(cl, 3), 1.281551566)

    cache = _load_fit_cache(cache_file)
    if not isinstance(model, dict):
        model = get_ratio_temp_model(temp_series_csv, cache_file)
        if not isinstance(model, dict):
            return []
    T_ref = float(model['T_ref'])
    b = float(model['b'])
    times_T, vals_T = _load_time_series_temperature(temp_series_csv)
    if not times_T:
        return []
    out = []
    seen_dirs = set()
    for key, entry in cache.items():
        if not isinstance(entry, dict):
            continue
        meta = entry.get('meta', {})
        ts = meta.get('timestamp') or entry.get('timestamp')
        if not ts:
            continue
        try:
            t = datetime.fromisoformat(str(ts))
        except Exception:
            continue
        # Skip excluded windows
        if _in_excluded(t):
            continue
        if (month is not None and t.month != month) or (year is not None and t.year != year):
            continue
        directory = meta.get('directory') or (key if isinstance(key, str) else None)
        if directory in seen_dirs:
            continue

        m3s = meta.get('m3_short')
        m3l = meta.get('m3_long')
        if m3s is None and 'short' in entry:
            m3s = entry['short'].get('m3')
        if m3l is None and 'long' in entry:
            m3l = entry['long'].get('m3')
        try:
            m3s = float(m3s) if m3s is not None else None
            m3l = float(m3l) if m3l is not None else None
        except Exception:
            continue
        if (m3s is None) or not (M3_SHORT_MIN < m3s < M3_SHORT_MAX):
            continue
        if (m3l is None) or not (np.isfinite(m3l) and m3l >= 0.4):
            continue
        if not (np.isfinite(m3s) and np.isfinite(m3l) and m3s != 0):
            continue

        Tval = _nearest_temp_for_time(times_T, vals_T, t)
        if Tval is None:
            continue

        R = m3l / m3s
        if not (np.isfinite(R) and R > 0):
            continue

        lnR_corr = np.log(R) - b * (float(Tval) - T_ref)
        # Determine sigma_lnR
        nS, nL = _extract_counts_from_entry(entry, meta)
        if nS is not None and nL is not None and nS > 0 and nL > 0:
            sigma_lnR = np.sqrt(1.0 / float(nL) + 1.0 / float(nS))
        else:
            # Fallback to fractional error on ratio -> sigma_lnR ~ sqrt( (σ_R/R)^2 )
            eps = float(fallback_ratio_err_frac)
            # If both legs contribute similarly in ln-space, use sqrt(2)*eps; be conservative with 2*eps
            sigma_lnR = max(eps * np.sqrt(2.0), eps * 2.0)

        # One-sided lower bound on ln(R/alpha)
        delta_t, scaling = get_delta_t_and_scaling_factor(t)
        ln_x_nom = lnR_corr - np.log(scaling)  # ln(Rcorr) - ln(alpha) = ln(Rcorr/alpha)
        ln_x_low = ln_x_nom - z * sigma_lnR
        if ln_x_low >= 0:
            # No finite lower limit on tau at this CL
            continue
        tau_lower = -delta_t / ln_x_low 
        if np.isfinite(tau_lower) and tau_lower > 0:
            out.append((t, float(tau_lower), int(nS) if nS else None, int(nL) if nL else None, directory))
            if directory:
                seen_dirs.add(directory)
    out.sort(key=lambda r: r[0])
    return out

def plot_tau_lower_limits_poisson(temp_series_csv: str,
                                  cache_file: str = FIT_CACHE_FILE,
                                  model: Optional[dict] = None,
                                  month: Optional[int] = 3,
                                  year: Optional[int] = 2025,
                                  cl: float = 0.98,
                                  out_name: str = 'tau_lower_limit_poisson.png'):
    """Plot one-sided lower limits on tau using Poisson lnR errors (log y-axis)."""
    rows = compute_tau_one_sided_lower_CL_poisson(temp_series_csv, cache_file, model=model,
                                                  month=month, year=year, cl=cl)
    if not rows:
        print("[INFO] No Poisson lower-limit tau points to plot.")
        return
    times = [r[0] for r in rows]
    tau_lo = np.array([r[1] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, tau_lo, '^', color='tab:purple', label=f'tau lower limit (CL={int(cl*100)}%)')
    ax.set_yscale('log')
    ax.set_xlabel('Time')
    ax.set_ylabel('Electron Lifetime lower limit (ms)')
    ax.set_title(f'One-sided Poisson lower limits on tau — {"March " + str(year) if month == 3 else "Selected window"}')
    fig.autofmt_xdate()
    ax.legend(loc='best')
    fig.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, out_name)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")
    plt.close(fig)

def plot_corrected_tau_with_error(temp_series_csv: str,
                                  cache_file: str = FIT_CACHE_FILE,
                                  model: Optional[dict] = None,
                                  month: Optional[int] = 3,
                                  year: Optional[int] = 2025,
                                  ratio_err_frac: float = 0.01,
                                  use_poisson_for_upper: bool = True,
                                  cl: float = 0.90,
                                  out_name: str = 'tau_corrected_with_error.png'):
    """Plot corrected tau with +-1% ratio-derived error bars (log y-axis)."""
    rows = compute_corrected_tau_with_error(temp_series_csv, cache_file, model=model,
                                            month=month, year=year, ratio_err_frac=ratio_err_frac,
                                            use_poisson_for_upper=use_poisson_for_upper, cl=cl)
    if not rows:
        print("[INFO] No corrected tau points with error bars to plot.")
        return
    times = [r[0] for r in rows]
    tau_nom = np.array([r[1] for r in rows], dtype=float)
    tau_lo = np.array([r[2] for r in rows], dtype=float)
    tau_hi = np.array([r[3] for r in rows], dtype=float)
    # Build asymmetric error bars; draw custom upward arrows for infinite uppers
    yerr_low = tau_nom - tau_lo
    yerr_high = np.where(np.isfinite(tau_hi), np.maximum(0.0, tau_hi - tau_nom), 0.0)
    yerr = np.vstack([yerr_low, yerr_high])

    fig, ax = plt.subplots(figsize=(10, 5))
    is_upper_inf = ~np.isfinite(tau_hi)
    n_onesided = int(np.sum(is_upper_inf))
    lbl = 'Tau (temp-corrected) ±1% ratio'
    if n_onesided > 0:
        if use_poisson_for_upper:
            lbl += f' (Poisson upper CL where finite; {n_onesided} upper limits)'
        else:
            lbl += f' ({n_onesided} upper limits)'
    ax.errorbar(times, tau_nom, yerr=yerr, fmt='o', color='tab:orange', ecolor='tab:orange',
                elinewidth=1, capsize=2, label=lbl)
    # Draw explicit upward arrows for points with infinite upper bound
    arrow_scale = 1.5  # 50% taller arrow on log scale
    for t, y, inf in zip(times, tau_nom, is_upper_inf):
        if inf and np.isfinite(y) and y > 0:
            y_top = y * arrow_scale
            try:
                ax.annotate('', xy=(t, y_top), xytext=(t, y),
                            arrowprops=dict(arrowstyle='-|>', color='tab:orange', lw=1))
            except Exception:
                pass
    ax.set_yscale('log')
    ax.set_xlabel('Time')
    ax.set_ylabel('Electron Lifetime (ms)')
    ax.set_title(f'Temperature-corrected tau with ±1% ratio band — {"March " + str(year) if month == 3 else "Selected window"}')
    fig.autofmt_xdate()
    ax.legend(loc='best')
    fig.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, out_name)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")
    plt.close(fig)
def save_corrected_ratio_to_csv(rows, out_path: str = None):
    """Save corrected ratio time series to CSV.
    Columns: timestamp_iso, ratio_corrected
    Default: plots/MarchCorrected.csv
    """
    if not rows:
        print("[INFO] No corrected ratio results to save.")
        return None
    if out_path is None:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        out_path = os.path.join(PLOTS_DIR, 'MarchCorrected.csv')
    try:
        with open(out_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['timestamp_iso', 'ratio_corrected'])
            for (t, r_corr, _directory) in rows:
                ts = t.isoformat() if hasattr(t, 'isoformat') else str(t)
                try:
                    rc = float(r_corr) if np.isfinite(r_corr) else ''
                except Exception:
                    rc = ''
                w.writerow([ts, rc])
        print(f"Saved corrected ratio CSV: {out_path}")
        return out_path
    except Exception as e:
        print(f"[WARN] Failed to save corrected ratio CSV: {e}")
        return None


def compute_corrected_ratio_series(temp_series_csv: str,
                                   cache_file: str = FIT_CACHE_FILE,
                                   month: Optional[int] = 3,
                                   year: Optional[int] = None,
                                   model: Optional[dict] = None):
    """Compute temperature-corrected ratio R_corr = exp(ln(R) - b*(T - T_ref)) for cache entries.
    Returns a list of (time, R_corr, directory). Does not write to cache.
    """
    cache = _load_fit_cache(cache_file)
    # Get or fit the ratio-vs-temp model (once per run)
    if not isinstance(model, dict):
        model = get_ratio_temp_model(temp_series_csv, cache_file)
        if not isinstance(model, dict):
            return []
    T_ref = float(model['T_ref'])
    b = float(model['b'])
    # Load temperature series
    times_T, vals_T = _load_time_series_temperature(temp_series_csv)
    if not times_T:
        return []
    rows = []
    seen_dirs = set()
    for key, entry in cache.items():
        if not isinstance(entry, dict):
            continue
        meta = entry.get('meta', {})
        ts = meta.get('timestamp') or entry.get('timestamp')
        if not ts:
            continue
        try:
            t = datetime.fromisoformat(str(ts))
        except Exception:
            continue
        # Skip excluded windows
        if _in_excluded(t):
            continue
        if (month is not None and t.month != month) or (year is not None and t.year != year):
            continue
        directory = meta.get('directory') or (key if isinstance(key, str) else None)
        if directory in seen_dirs:
            continue
        # m3 values with cuts
        m3s = meta.get('m3_short')
        m3l = meta.get('m3_long')
        if m3s is None and 'short' in entry:
            m3s = entry['short'].get('m3')
        if m3l is None and 'long' in entry:
            m3l = entry['long'].get('m3')
        try:
            m3s = float(m3s) if m3s is not None else None
            m3l = float(m3l) if m3l is not None else None
        except Exception:
            continue
        if (m3s is None) or not (M3_SHORT_MIN < m3s < M3_SHORT_MAX):
            continue
        if (m3l is None) or not (np.isfinite(m3l) and m3l >= 0.4):
            continue
        if not (np.isfinite(m3s) and np.isfinite(m3l) and m3s != 0):
            continue
        # Temperature lookup
        Tval = _nearest_temp_for_time(times_T, vals_T, t)
        if Tval is None:
            continue
        # Corrected ratio
        R = m3l / m3s
        if not (np.isfinite(R) and R > 0):
            continue
        lnR_corr = np.log(R) - b * (float(Tval) - T_ref)
        R_corr = float(np.exp(lnR_corr))
        rows.append((t, R_corr, directory))
        if directory:
            seen_dirs.add(directory)
    rows.sort(key=lambda r: r[0])
    return rows

def plot_corrected_ratio_vs_prm_tau(prm_csv: str,
                                    temp_series_csv: str,
                                    cache_file: str = FIT_CACHE_FILE,
                                    month: Optional[int] = 3,
                                    year: Optional[int] = 2025,
                                    out_name: str = 'corrected_ratio_vs_prm_tau_march.png',
                                    model: Optional[dict] = None):
    """Overlay corrected ratio (q_l/q_s at T_ref) vs time with PRM Top tau vs time.
    Left axis: corrected ratio (linear). Right axis: PRM Top lifetime (ms, log).
    Saves plots/out_name.
    """
    # Compute corrected ratio series
    rows = compute_corrected_ratio_series(temp_series_csv, cache_file, month=month, year=year, model=model)
    if not rows:
        print("[INFO] No corrected ratio points to plot.")
        return
    times_R = [r[0] for r in rows]
    R_corr  = [r[1] for r in rows]
    # Load PRM series and filter to month/year
    t_ext, tau_ext = _load_prm_top_lifetime_csv(prm_csv)
    if not t_ext:
        print(f"[INFO] No PRM Top series at {prm_csv}")
        return
    mask = [(t.year == year if year is not None else True) and (t.month == month if month is not None else True)
            for t in t_ext]
    ext_times = [t for t, keep in zip(t_ext, mask) if keep]
    ext_taus  = [v for v, keep in zip(tau_ext, mask) if keep]
    if not ext_times:
        print("[INFO] No PRM Top points in selected month/year.")
        return
    # Sort
    order_R = np.argsort([t.timestamp() for t in times_R])
    times_R = [times_R[i] for i in order_R]
    R_corr  = [R_corr[i] for i in order_R]
    order_E = np.argsort([t.timestamp() for t in ext_times])
    ext_times = [ext_times[i] for i in order_E]
    ext_taus  = [ext_taus[i] for i in order_E]


    # Keep only positive, finite taus for log plotting
    _msk = [np.isfinite(v) and (v > 0) for v in ext_taus]
    ext_times = [t for t, keep in zip(ext_times, _msk) if keep]
    ext_taus  = [v for v, keep in zip(ext_taus, _msk) if keep]



    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(times_R, R_corr, 'o', color='tab:green', alpha=0.8, label='Corrected ratio (Q_L/Q_S @ T_ref)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Corrected ratio (unitless)', color='tab:green')
    ax1.tick_params(axis='y', labelcolor='tab:green')
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(ext_times, ext_taus, '-o', color='tab:blue', alpha=0.8, label='PRM Top lifetime (ms)')
    ax2.set_ylabel('PRM Top lifetime', color='tab:blue')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Merge legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='best')
    fig.autofmt_xdate()
    fig.suptitle(f'Corrected ratio vs PRM Top lifetime — {"March " + str(year) if month == 3 else "Selected window"}')
    fig.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, out_name)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")
    plt.close(fig)

def plot_tau_raw_vs_corrected(cache_file: str = FIT_CACHE_FILE,
                              results=None,
                              out_name: str = 'tau_raw_vs_corrected.png'):
    """Overlay tau (raw) and corrected tau vs time.
    - If 'results' is provided (list of (time, tau_raw, tau_corr, directory)), plot that.
    - Otherwise, look for 'tau_temp_corr_ratio' saved in cache and plot those.
    """
    times = []
    tau_raw = []
    tau_corr = []
    if results:
        rows = [(t, tr, tc) for (t, tr, tc, _dir) in results
                if (t is not None and np.isfinite(tr) and np.isfinite(tc) and (tr > 0) and (tc > 0))]
        if not rows:
            print("[INFO] No valid corrected tau results to plot.")
            return
        rows.sort(key=lambda r: r[0])
        times = [r[0] for r in rows]
        tau_raw = [float(r[1]) for r in rows]
        tau_corr = [float(r[2]) for r in rows]
    else:
        cache = _load_fit_cache(cache_file)
        pts = []  # (time, tau_raw, tau_corr)
        seen = set()
        for key, entry in cache.items():
            if not isinstance(entry, dict):
                continue
            meta = entry.get('meta', {})
            if 'timestamp' not in meta:
                continue
            if 'tau' not in meta or 'tau_temp_corr_ratio' not in meta:
                continue
            try:
                t = datetime.fromisoformat(str(meta['timestamp']))
            except Exception:
                continue
            # Only plot March entries (those are the ones corrected)
            if t.month != 3:
                continue
            directory = meta.get('directory') or (key if isinstance(key, str) else None)
            if directory in seen:
                continue
            try:
                tr = float(meta['tau'])
                tc = float(meta['tau_temp_corr_ratio'])
            except Exception:
                continue
            if not (np.isfinite(tr) and np.isfinite(tc) and (tr > 0) and (tc > 0)):
                continue
            pts.append((t, tr, tc))
            if directory:
                seen.add(directory)
        if not pts:
            print("[INFO] No entries with saved tau_temp_corr_ratio in cache.")
            return
        pts.sort(key=lambda x: x[0])
        times = [p[0] for p in pts]
        tau_raw = [p[1] for p in pts]
        tau_corr = [p[2] for p in pts]

    plt.figure(figsize=(10, 5))
    plt.plot(times, tau_raw, 'o', label=r'$\tau$ (original)', color='tab:blue')
    plt.plot(times, tau_corr, 'o', label=r'$\tau$ (temp-corrected)', color='tab:orange')
    plt.yscale('log')
    plt.xlabel('Time')
    plt.ylabel(r'Electron Lifetime $\tau_e$ (ms)')
    plt.title(r'$\tau$: original vs temperature-corrected')
    plt.gcf().autofmt_xdate()
    plt.legend(loc='best')
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, out_name)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=500)
    print(f"Saved plot: {out_path}")
    plt.close()


# Fit, apply (idempotent), and plot comparison if the temperature CSV is present
if os.path.exists(_overlay_temp_csv):
    # Fit once with restricted window (March 17–31), reuse across computations
    _march_start = datetime(2025, 3, 17, 0, 0, 0)
    _march_end   = datetime(2025, 3, 31, 23, 59, 59)
    print('[INFO] Model fit window: March 17–31, 2025 (applied to full March and August).')
    _model_once = get_ratio_temp_model(_overlay_temp_csv, month=3, year=2025,
                                       start_date=_march_start, end_date=_march_end)
    # Compute correction for the full month of March 2025 using this model
    _computed = apply_ratio_temp_correction_to_tau(_overlay_temp_csv, model=_model_once, month=3, year=2025)
    # Save corrected ratio series (MarchCorrected.csv)
    _ratio_rows = compute_corrected_ratio_series(_overlay_temp_csv, FIT_CACHE_FILE, month=3, year=2025, model=_model_once)
    save_corrected_ratio_to_csv(_ratio_rows, out_path=os.path.join(PLOTS_DIR, 'MarchCorrected.csv'))
    plot_tau_raw_vs_corrected(FIT_CACHE_FILE, results=_computed, out_name='tau_raw_vs_corrected.png')
    # Overlay corrected tau vs PRM Top lifetime if available
    if os.path.exists(_prm_top_csv):
        plot_prm_top_vs_corrected_tau_overlay(_prm_top_csv, results=_computed, year=2025,
                                              out_name='prm_top_vs_tau_corrected_march.png')
        # Plot corrected tau with ± slope sigma error bars
        plot_corrected_tau_with_model_sigma(_overlay_temp_csv, model=_model_once, month=3, year=2025,
                                            out_name='tau_corrected_with_fit_sigma.png')

# Also handle August 2025 if the temperature CSV is present
_overlay_temp_csv_aug = os.path.join(NP02DATA_DIR, 'TempAugust2025_smoothed.csv')
if os.path.exists(_overlay_temp_csv_aug):
    # Raw temp series for August
    try:
        plot_temperature_file(_overlay_temp_csv_aug, save=True, out_name='temp_timeseries_august.png')
    except Exception:
        pass
    # Use the March model to correct August values when available; otherwise try to build it
    if '_model_once' in globals() and isinstance(_model_once, dict):
        _model_once_aug = _model_once
    else:
        if os.path.exists(_overlay_temp_csv):
            _march_start = datetime(2025, 3, 17, 0, 0, 0)
            _march_end   = datetime(2025, 3, 31, 23, 59, 59)
            _model_once_aug = get_ratio_temp_model(_overlay_temp_csv, month=3, year=2025,
                                                   start_date=_march_start, end_date=_march_end)
        else:
            print('[INFO] March model not available; falling back to August-specific fit for correction.')
            _model_once_aug = get_ratio_temp_model(_overlay_temp_csv_aug, month=8, year=2025)
    _computed_aug = apply_ratio_temp_correction_to_tau(_overlay_temp_csv_aug, model=_model_once_aug, month=8, year=2025)
    # Save corrected ratio series for August
    _ratio_rows_aug = compute_corrected_ratio_series(_overlay_temp_csv_aug, FIT_CACHE_FILE, month=8, year=2025, model=_model_once_aug)
    save_corrected_ratio_to_csv(_ratio_rows_aug, out_path=os.path.join(PLOTS_DIR, 'AugustCorrected.csv'))
    # Plots for August
    try:
        plot_temp_vs_ratio_within_month(_overlay_temp_csv_aug, use_temp_corrected=False, out_name='temp_vs_m3_ratio_august.png')
    except Exception:
        pass
    plot_tau_raw_vs_corrected(FIT_CACHE_FILE, results=_computed_aug, out_name='tau_raw_vs_corrected_august.png')
    if os.path.exists(_prm_top_csv):
        plot_prm_top_vs_corrected_tau_overlay(_prm_top_csv, results=_computed_aug, year=2025, month=8,
                                              out_name='prm_top_vs_tau_corrected_august.png')
