import os
import glob
from datetime import datetime
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NP02DATA_DIR = os.environ.get('NP02DATA_DIR', '../np02data')
ROOT_DIR = NP02DATA_DIR
PLOTS_DIR = 'plots'

# Window: June 26 15:06 to July 02 10:00 (year 2025)
START = datetime(2025, 5, 14, 00, 00)
END   = datetime(2025, 5, 15, 23, 59)

MONTH_MAP = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
}


def parse_dir_time(directory: str) -> Optional[datetime]:
    parts = directory.strip('/').split('/')
    if len(parts) < 5:
        return None
    ym = parts[-4]  # e.g., '2025_Jun'
    day = parts[-3]
    hour = parts[-2]
    minute = parts[-1]
    try:
        year_str, month_word = ym.split('_')
        month_str = MONTH_MAP.get(month_word, '01')
        ts = f"{year_str}-{month_str}-{day} {hour}:{minute}"
        return datetime.strptime(ts, '%Y-%m-%d %H:%M')
    except Exception:
        return None


def load_f_file(path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, usecols=['BinCenter', 'Population'])
    except Exception:
        return None
    try:
        x = pd.to_numeric(df['BinCenter'], errors='coerce').to_numpy(dtype=float)
        y = pd.to_numeric(df['Population'], errors='coerce').to_numpy(dtype=float)
    except Exception:
        return None
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size == 0 or y.size == 0:
        return None
    return x, y


def main():
    # Find candidate measurement directories (presence of F1.txt)
    pattern = f"{ROOT_DIR}/2025_*/**/F1.txt"
    dirs = []
    for f1 in glob.iglob(pattern, recursive=True):
        d = os.path.dirname(f1)
        t = parse_dir_time(d)
        if t is None:
            continue
        if t < START or t > END:
            continue
        dirs.append((t, d))
    if not dirs:
        print('[INFO] No measurement directories found in the requested window.')
        return
    dirs.sort(key=lambda r: r[0])

    # Prepare figure with 2x2 subplots: F1, F2, F3, F4 overlays
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    ax_f1, ax_f2 = axes[0, 0], axes[0, 1]
    ax_f3, ax_f4 = axes[1, 0], axes[1, 1]

    # Color map across time span
    times = [t for t, _ in dirs]
    t0 = min(times).timestamp()
    t1 = max(times).timestamp()
    def time_color(tt: datetime):
        if t1 == t0:
            u = 0.5
        else:
            u = (tt.timestamp() - t0) / (t1 - t0)
        return plt.cm.viridis(u)

    n_plotted = { 'F1': 0, 'F2': 0, 'F3': 0, 'F4': 0 }
    for t, d in dirs:
        c = time_color(t)
        # F1
        f1 = os.path.join(d, 'F1.txt')
        dat = load_f_file(f1)
        if dat is not None:
            x, y = dat
            ax_f1.step(x, y, where='mid', color=c, alpha=0.35)
            n_plotted['F1'] += 1
        # F2
        f2 = os.path.join(d, 'F2.txt')
        dat = load_f_file(f2)
        if dat is not None:
            x, y = dat
            ax_f2.step(x, y, where='mid', color=c, alpha=0.35)
            n_plotted['F2'] += 1
        # F3
        f3 = os.path.join(d, 'F3.txt')
        dat = load_f_file(f3)
        if dat is not None:
            x, y = dat
            ax_f3.step(x, y, where='mid', color=c, alpha=0.35)
            n_plotted['F3'] += 1
        # F4
        f4 = os.path.join(d, 'F4.txt')
        dat = load_f_file(f4)
        if dat is not None:
            x, y = dat
            ax_f4.step(x, y, where='mid', color=c, alpha=0.35)
            n_plotted['F4'] += 1

    # Labeling
    ax_f1.set_title('F1 (Short PM Data)')
    ax_f2.set_title('F2 (Long PM Data)')
    ax_f3.set_title('F3 (Short PM Background)')
    ax_f4.set_title('F4 (Long PM Background)')
    for ax in (ax_f1, ax_f2, ax_f3, ax_f4):
        ax.set_xlabel('BinCenter [V]')
        ax.set_ylabel('Population')
        ax.grid(True, alpha=0.25)

    fig.suptitle(f'F1/F2/F3/F4 overlays — {START.strftime("%b %d %H:%M")} to {END.strftime("%b %d %H:%M")}')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, 'F1234_overlay_2025-06-26_15-06_to_2025-07-02_10-00.png')
    fig.savefig(out, dpi=150)
    print(f'Saved overlay plot: {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()

