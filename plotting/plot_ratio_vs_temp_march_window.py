import os
import pickle
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# temperature from a CSV, filters March 17–31, and shows a linear fit.

FIT_CACHE_FILE = 'fit_cache.pkl'
NP02DATA_DIR = os.environ.get('NP02DATA_DIR', '../np02data')
TEMP_CSV = os.environ.get('NP02_TEMP_CSV_MARCH', os.path.join(NP02DATA_DIR, 'TempMarch2025_smoothed.csv'))
PLOTS_DIR = 'plots'

# Cuts for valid points. This is to remove the obvious outliers when the short PM is not working.
M3_SHORT_MIN = 0.550 
M3_SHORT_MAX = 1.38

scaling_factor = 0.92 # This value comes from calibration analyses.

# Time window for plotting
START = datetime(2025, 3, 17, 0, 0, 0)
END   = datetime(2025, 3, 31, 23, 59, 59)

# Exclusion windows (remove periods where readout window was 50 us)
# Per note: March 21 2:19 PM changed to 50 us; reset back at 3:17 PM.
EXCLUDE_WINDOWS = [
    (datetime(2025, 3, 21, 14, 19, 0), datetime(2025, 3, 21, 15, 17, 0)),
]

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
    """Return linearly interpolated temperature at time t.
    Requires at least two samples and assumes times are roughly sorted.
    Returns None if t lies outside the sample time range or values are invalid.
    """
    if not times or not temps or len(times) != len(temps):
        return None
    ts = np.array([ti.timestamp() for ti in times], dtype=float)
    vs = np.array(temps, dtype=float)
    if ts.size == 0:
        return None
    if ts.size == 1:
        try:
            return float(vs[0])
        except Exception:
            return None
    tt = float(t.timestamp())
    # Find right bracketing index
    idx = int(np.searchsorted(ts, tt))
    # If outside range, do not extrapolate
    if idx == 0 or idx >= ts.size:
        return None
    t0, t1 = ts[idx - 1], ts[idx]
    v0, v1 = vs[idx - 1], vs[idx]
    if not (np.isfinite(t0) and np.isfinite(t1) and np.isfinite(v0) and np.isfinite(v1)):
        return None
    if t1 <= t0:
        return None
    w = (tt - t0) / (t1 - t0)
    v = v0 + w * (v1 - v0)
    try:
        v = float(v)
    except Exception:
        return None
    return v if np.isfinite(v) else None


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

    X_T = []           # temperature (C) for plotting
    Y_R = []           # displayed ratio = (m3_long/m3_short) / scaling_factor
    T_ratio = []       # timestamps corresponding to the ratio points (for ratio vs time)
    T_for_fit = []     # temperatures used for ln(R) vs T fit
    R_raw_list = []    # original ratios (no scaling) for correction
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
        # Skip known excluded intervals (50 us readout window)
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
        R_raw = (m3l / m3s)
        if not (np.isfinite(R_raw) and R_raw > 0):
            continue

        X_T.append(float(Tval))
        Y_R.append(float(R_raw / scaling_factor))
        T_ratio.append(t)
        T_for_fit.append(float(Tval))
        R_raw_list.append(float(R_raw))
        if directory:
            seen_dirs.add(directory)

    if not X_T:
        print("[INFO] No valid points for ratio vs temperature in March 17-31.")
    else:
        X = np.asarray(X_T, dtype=float)
        Y = np.asarray(Y_R, dtype=float)

        # Compute ln(R) vs T slope and corrected ratio at T_ref
        R_corr_scaled = None
        # Initialize fit variables so later blocks can reference them
        a_ln = b_ln = sb_ln = sa_ln = None
        cov_ab = None
        T_ref = 17.8
        if len(T_for_fit) >= 2:
            T_arr = np.asarray(T_for_fit, dtype=float)
            R_arr = np.asarray(R_raw_list, dtype=float)
            lnR = np.log(R_arr)
            dT = T_arr - T_ref
            try:
                (b_ln, a_ln), _cov_ln = np.polyfit(dT, lnR, 1, cov=True)
            except Exception:
              b_ln = np.nan
              a_ln = np.nan
              _cov_ln = None
            if np.isfinite(b_ln):
                # Temperature-corrected ratio in ln-space
                lnR_corr = lnR - b_ln * dT
                R_corr = np.exp(lnR_corr)
                R_corr_scaled = (R_corr / float(scaling_factor)).tolist()
                # Uncertainties on slope/intercept
                if _cov_ln is not None and _cov_ln.size >= 4:
                    sb_ln = float(np.sqrt(_cov_ln[0, 0]))
                    sa_ln = float(np.sqrt(_cov_ln[1, 1]))
                    cov_ab = float(_cov_ln[0, 1])
                else:
                    sb_ln = float('nan')
                    sa_ln = float('nan')
                print(f"[INFO] ln(R) vs T fit (Mar 17 - 31): b = {b_ln:.6g} ± {sb_ln:.2g}; T_ref = {T_ref:.3f} C")
                # New figure: ln(R) vs Temperature with linear fit
                try:
                    plt.figure(figsize=(8, 5))
                    plt.plot(T_arr, lnR, 'o', alpha=0.85, label='data')
                    xfitT = np.linspace(float(T_arr.min()), float(T_arr.max()), 200)
                    yfit_ln = a_ln + b_ln * (xfitT - T_ref)
                    # Uncertainty band disabled per request
                    plt.plot(xfitT, yfit_ln, 'r--', label=f'fit: ln R = ({b_ln:.4g}±{sb_ln:.2g})·(T - {T_ref:.2f}) + ({a_ln:.4g}±{sa_ln:.2g})')
                    plt.xlabel(r"Temperature ($^\circ$C)")
                    plt.ylabel(r'ln R = ln($Q_l/Q_s$)')
                    plt.title(r'ln($Q_l/Q_s$) vs Temperature — Mar 17-31, 2025')
                    plt.legend(loc='best')
                    plt.tight_layout()
                    try:
                        os.makedirs(PLOTS_DIR, exist_ok=True)
                        out_ln = os.path.join(PLOTS_DIR, 'lnR_vs_temperature_march_window.png')
                        plt.savefig(out_ln, dpi=500)
                        print(f"Saved plot: {out_ln}")
                    except Exception:
                        pass
                except Exception:
                    pass
                print(f"[INFO] ln(R) vs T fit (Mar 17 - 31): b = {b_ln:.6g} ± {sb_ln:.2g}; T_ref = {T_ref:.3f} C")

                # (Removed combined ratio and ln(R) twin-axis plot by request.)

        # Original vs Corrected Ratio vs Temperature overlay (model fit only; no ±1σ bands)
        try:
            os.makedirs(PLOTS_DIR, exist_ok=True)
        except Exception:
            pass
        try:
            plt.figure(figsize=(8, 5))
            # Original series
            plt.plot(X, Y, 'o', alpha=0.6, label='Original ratio')
            # Corrected series
            if R_corr_scaled is not None:
                Yc = np.asarray(R_corr_scaled, dtype=float)
                plt.plot(X, Yc, 'o', alpha=0.6, label=fr'Corrected ratio @ $T_{{ref}}={T_ref:.2f}\,^{{\circ}} C$')
            # Model-derived curves using the single ln-space fit
            try:
                if (a_ln is not None) and (b_ln is not None) and np.isfinite(b_ln):
                    xfit = np.linspace(float(np.min(X)), float(np.max(X)), 200)
                    ln_mu = a_ln + b_ln * (xfit - T_ref)
                    yfit_orig = np.exp(ln_mu) / float(scaling_factor)
                    plt.plot(xfit, yfit_orig, 'b--', label='Original fit')
                    # Corrected predicted is constant at T_ref
                    yfit_corr = np.exp(a_ln) / float(scaling_factor)
                    plt.plot(xfit, np.full_like(xfit, yfit_corr), 'g--', label=f'Corrected ratio fit')
                    # Uncertainty bands disabled per request
            except Exception:
                pass
            plt.xlabel(r"Temperature ($^\circ$C)")
            plt.ylabel(r"$Q_l/Q_s$")
            plt.title('Original vs Corrected Ratio vs Temperature — Mar 17-31, 2025')
            plt.legend(loc='best')
            plt.tight_layout()
            out_temp_overlay = os.path.join(PLOTS_DIR, 'ratio_vs_temperature_march_window_overlay.png')
            plt.savefig(out_temp_overlay, dpi=500)
            print(f"Saved plot: {out_temp_overlay}")
        except Exception:
            pass

        # Original vs Corrected Ratio vs Time with linear fits; add corrected min/max using slope uncertainty
        if T_ratio:
            plt.figure(figsize=(10, 4))
            ord_idx = np.argsort([ti.timestamp() for ti in T_ratio])
            times_sorted = [T_ratio[i] for i in ord_idx]
            ratios_orig_sorted = [Y_R[i] for i in ord_idx]
            plt.plot(times_sorted, ratios_orig_sorted, 'o', alpha=0.75, label='Original ratio')
            if R_corr_scaled is not None:
                ratios_corr_sorted = [R_corr_scaled[i] for i in ord_idx]
                plt.plot(times_sorted, ratios_corr_sorted, 'o', alpha=0.75, label=fr'Corrected ratio @ $T_{{ref}}={T_ref:.2f}\,^{{\circ}} C$')
            # Fit original and corrected vs time
            tnum = np.asarray([ts.timestamp() for ts in times_sorted], dtype=float)
            t0 = float(np.median(tnum))
            tx = tnum - t0
            # Original fit
            try:
                (m_o, b_o), cov_o = np.polyfit(tx, np.asarray(ratios_orig_sorted, dtype=float), 1, cov=True)
                sm_o = float(np.sqrt(cov_o[0, 0])) if cov_o.size >= 4 else np.nan
                sb_o = float(np.sqrt(cov_o[1, 1])) if cov_o.size >= 4 else np.nan
                yfit_o = m_o * tx + b_o
                plt.plot(times_sorted, yfit_o, 'b--', alpha=0.9)
            except Exception:
                pass
            # Corrected fit
            if R_corr_scaled is not None:
                try:
                    (m_c, b_c), cov_c = np.polyfit(tx, np.asarray(ratios_corr_sorted, dtype=float), 1, cov=True)
                    sm_c = float(np.sqrt(cov_c[0, 0])) if cov_c.size >= 4 else np.nan
                    sb_c = float(np.sqrt(cov_c[1, 1])) if cov_c.size >= 4 else np.nan
                    yfit_c = m_c * tx + b_c
                    plt.plot(times_sorted, yfit_c, 'g--', alpha=0.9)
                except Exception:
                    pass
            plt.gcf().autofmt_xdate()
            plt.xlabel('Time')
            plt.ylabel(r"$Q_l/Q_s$")
            plt.title('Original vs Corrected Ratio vs Time — Mar 17 - 31, 2025')
            plt.legend(loc='best')
            plt.tight_layout()
        
            try:
                os.makedirs(PLOTS_DIR, exist_ok=True)
                out_time = os.path.join(PLOTS_DIR, 'ratio_time_original_vs_corrected_march_window.png')
                plt.savefig(out_time, dpi=500)
                print(f"Saved plot: {out_time}")
            except Exception:
                pass

        # Show both figures
        plt.show()
