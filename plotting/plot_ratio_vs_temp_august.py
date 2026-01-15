import os
import pickle
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# Plot ratio (m3_long / m3_short) vs temperature for the whole August 2025.

FIT_CACHE_FILE = 'fit_cache.pkl'
NP02DATA_DIR = os.environ.get('NP02DATA_DIR', '../np02data')
TEMP_CSV = os.environ.get('NP02_TEMP_CSV_AUGUST', os.path.join(NP02DATA_DIR, 'TempAugust2025.csv'))

# Cuts for valid points
M3_SHORT_MIN = 0.550
M3_SHORT_MAX = 1.38

# Normalization factor (alpha) used in some analyses; here it only rescales 
scaling_factor = 0.92

# Full August 2025 window
START = datetime(2025, 8, 1, 0, 0, 0)
END   = datetime(2025, 8, 31, 23, 59, 59)

# No known exclusions for August; keep list for symmetry
EXCLUDE_WINDOWS = []  # e.g., [(datetime(2025, 8, 10, 0, 0, 0), datetime(2025, 8, 10, 1, 0, 0))]

def in_excluded(t: datetime) -> bool:
    for s, e in EXCLUDE_WINDOWS:
        if s <= t <= e:
            return True
    return False


def load_fit_cache(path: str):
    try:
        with open(path, 'rb') as f:
            d = pickle.load(f)
        return d if isinstance(d, dict) else {}
    except Exception as e:
        print(f"[WARN] Failed to load cache {path}: {e}")
        return {}


def load_temp_series(path: str):
    times = []
    vals = []
    if not os.path.exists(path):
        return times, vals
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
                vals.append(v)
    except Exception as e:
        print(f"[WARN] Failed to read temperature series {path}: {e}")
    return times, vals


def nearest_temp(times, temps, t: datetime):
    if not times:
        return None
    arr = np.array([ti.timestamp() for ti in times], dtype=float)
    idx = int(np.argmin(np.abs(arr - t.timestamp())))
    try:
        return float(temps[idx])
    except Exception:
        return None


# Load inputs
cache = load_fit_cache(FIT_CACHE_FILE)
times_T, vals_T = load_temp_series(TEMP_CSV)

if not times_T:
    print(f"[INFO] No temperature series at {TEMP_CSV}; nothing to plot.")
else:
    # Ensure temperature series is sorted
    orderT = np.argsort(np.array([t.timestamp() for t in times_T]))
    times_T = [times_T[i] for i in orderT]
    vals_T = [vals_T[i] for i in orderT]

    X_T = []      # temperature
    Y_R = []      # ratio = m3_long / m3_short
    T_ratio = []  # timestamps corresponding to the ratio points (for ratio vs time)
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
        if not (START <= t <= END):
            continue
        if in_excluded(t):
            continue
        directory = meta.get('directory') or (key if isinstance(key, str) else None)
        if directory in seen_dirs:
            continue

        # Get m3 values
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

        Tval = nearest_temp(times_T, vals_T, t)
        if Tval is None or not np.isfinite(Tval):
            continue
        R = (m3l / m3s) / scaling_factor
        if not (np.isfinite(R) and R > 0):
            continue

        X_T.append(float(Tval))
        Y_R.append(float(R))
        T_ratio.append(t)
        if directory:
            seen_dirs.add(directory)

    if not X_T:
        print("[INFO] No valid points for ratio vs temperature in August 2025.")
    else:
        X = np.asarray(X_T, dtype=float)
        Y = np.asarray(Y_R, dtype=float)
        # Linear fit: R = m*T + b
        try:
            (m, b), cov = np.polyfit(X, Y, 1, cov=True)
            sm = float(np.sqrt(cov[0, 0])) if cov.size >= 4 else np.nan
            sb = float(np.sqrt(cov[1, 1])) if cov.size >= 4 else np.nan
        except Exception:
            m = b = sm = sb = np.nan

        xfit = np.linspace(float(X.min()), float(X.max()), 200)
        yfit = m * xfit + b if np.isfinite(m) and np.isfinite(b) else None

        plt.figure(figsize=(8, 5))
        plt.plot(X, Y, 'o', alpha=0.8, label='data')
        if yfit is not None:
            plt.plot(xfit, yfit, 'r--', label=f'fit: y = ({m:.4g}±{sm:.2g})·T + ({b:.4g}±{sb:.2g})')
        plt.xlabel('Temperature (C)')
        plt.ylabel('m3_long / m3_short')
        plt.title('Ratio vs Temperature — August 2025 (linear fit)')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()

        # Also: Ratio vs Time scatter over the same window
        if T_ratio:
            plt.figure(figsize=(10, 4))
            ord_idx = np.argsort([ti.timestamp() for ti in T_ratio])
            times_sorted = [T_ratio[i] for i in ord_idx]
            ratios_sorted = [Y_R[i] for i in ord_idx]
            plt.plot(times_sorted, ratios_sorted, 'o', alpha=0.8, label='data')
            plt.gcf().autofmt_xdate()
            plt.xlabel('Time')
            plt.ylabel('m3_long / m3_short')
            plt.title('Ratio vs Time — August 2025')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            plt.tight_layout()

        plt.show()

