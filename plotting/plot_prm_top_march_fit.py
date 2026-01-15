import os
import argparse
from datetime import datetime
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

PLOTS_DIR = 'plots'


def load_prm_csv(path: str) -> Tuple[List[datetime], List[float]]:
    """Load PRM Top lifetime CSV. Expects columns with a datetime and a numeric lifetime.
    Tries pandas first; falls back to a simple CSV parser.
    Returns (times, taus)."""
    times: List[datetime] = []
    taus: List[float] = []
    if not os.path.exists(path):
        return times, taus
    # Try pandas for flexible parsing
    try:
        import pandas as pd
        df = pd.read_csv(path)
        if df.empty:
            return times, taus
        # Heuristic: find a datetime-like column
        time_col = None
        for c in df.columns:
            
            s = pd.to_datetime(df[c], errors='coerce')
            if s.notna().sum() >= max(3, len(df)//4):
                time_col = c
                times = list(s.dropna().astype('datetime64[ns]'))
                idx = s.notna()
                df = df[idx]
                break
        if time_col is None:
            return [], []
        # Find a numeric tau column distinct from time_col
        tau_col = None
        for c in df.columns:
            if c == time_col:
                continue
            vals = pd.to_numeric(df[c], errors='coerce')
            if vals.notna().sum() >= max(3, len(df)//4):
                tau_col = c
                taus = list(vals.dropna().astype(float))
                break
        if tau_col is None:
            return [], []
        # Align lengths
        n = min(len(times), len(taus))
        return list(times)[:n], list(taus)[:n]
    except Exception:
        pass

    # Fallback: assume two-column CSV with 'Timestamp,Value'
    import csv as _csv
    fmts = [
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
        '%Y/%m/%d %H:%M:%S',
        '%Y-%m-%d',
        '%Y/%m/%d',
    ]
    try:
        with open(path, 'r') as f:
            r = _csv.reader(f)
            header = next(r, None)
            for row in r:
                if not row or len(row) < 2:
                    continue
                tstr = row[0].strip()
                vstr = row[1].strip()
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
                taus.append(v)
    except Exception:
        return [], []
    return times, taus


def fit_line_time(times: List[datetime], ys: List[float]):
    """Fit y = m * (t_days) + b, where t_days is days since first sample in 'times'.
    Returns (m_per_day, b, sm, sb, t0)."""
    if len(times) < 2:
        return None
    t0 = times[0].timestamp()
    t_days = np.array([(t.timestamp() - t0) / 86400.0 for t in times], dtype=float)
    y = np.array(ys, dtype=float)
    try:
        (m, b), cov = np.polyfit(t_days, y, 1, cov=True)
        sm = float(np.sqrt(cov[0, 0])) if cov.size >= 4 else np.nan
        sb = float(np.sqrt(cov[1, 1])) if cov.size >= 4 else np.nan
        return m, b, sm, sb, t0
    except Exception:
        return None


def plot_prm_window(csv_path: str,
                    start: datetime,
                    end: datetime,
                    out_name: str = 'prm_top_march17_end_fit.png'):
    times, taus = load_prm_csv(csv_path)
    if not times:
        print(f"[INFO] No data in {csv_path}")
        return None
    # Filter window
    rows = [(t, v) for t, v in zip(times, taus) if (t >= start and t <= end and np.isfinite(v) and v > 0)]
    if not rows:
        print(f"[INFO] No points in window {start} to {end}.")
        return None
    rows.sort(key=lambda r: r[0])
    t_win = [r[0] for r in rows]
    y_win = [float(r[1]) for r in rows]

    # Fit
    fit = fit_line_time(t_win, y_win)
    # Build plot
    plt.figure(figsize=(10, 5))
    plt.plot(t_win, y_win, 'o', color='tab:blue', alpha=0.8, label='PRM Top lifetime (ms)')
    title = f'PRM Top Lifetime — March 17–31, 2025 (linear fit)'
    if fit is not None:
        m, b, sm, sb, t0 = fit
        # Construct fit line over window
        t0_dt = datetime.fromtimestamp(t0)
        xfit = np.linspace(t_win[0].timestamp(), t_win[-1].timestamp(), 200)
        xfit_days = (xfit - t0) / 86400.0
        yfit = m * xfit_days + b
        xfit_dt = [datetime.fromtimestamp(x) for x in xfit]
        plt.plot(xfit_dt, yfit, 'r--', linewidth=2.0,
                 label=f'fit: tau = ({m:.3g}±{sm:.2g})·days + ({b:.3g}±{sb:.2g})')
    plt.yscale('log')
    plt.xlabel('Time')
    plt.ylabel('Electron Lifetime (ms)')
    plt.title(title)
    plt.gcf().autofmt_xdate()
    plt.legend(loc='best')
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, out_name)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot: {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description='Plot PRM Top lifetime for a March 17–31 window with linear fit.')
    np02data_dir = os.environ.get('NP02DATA_DIR', '../np02data')
    ap.add_argument('--csv', default=os.environ.get('NP02_PRM_TOP_CSV', os.path.join(np02data_dir, 'prm_Top_lifetime_data_GainOnly.csv')),
                    help='Path to prm_Top_lifetime_data_GainOnly.csv')
    ap.add_argument('--year', type=int, default=2025, help='Year of March window')
    ap.add_argument('--start-day', type=int, default=17, help='Start day in March')
    ap.add_argument('--end-day', type=int, default=31, help='End day in March')
    ap.add_argument('--out', default='prm_top_march17_end_fit.png', help='Output PNG filename (in plots/)')
    args = ap.parse_args()

    start = datetime(args.year, 3, args.start_day, 0, 0, 0)
    end = datetime(args.year, 3, args.end_day, 23, 59, 59)
    plot_prm_window(args.csv, start, end, out_name=args.out)


if __name__ == '__main__':
    main()

