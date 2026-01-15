import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime

# Configuration
NP02DATA_DIR = os.environ.get('NP02DATA_DIR', '../np02data')
ROOT_DIR = NP02DATA_DIR
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# Date window (inclusive)
START_DATE = datetime(2025, 6, 27, 15, 0, 0)
END_DATE: datetime | None = datetime(2025, 7, 2, 11, 0, 0)

# Month map used in folder names like 2025_Jun, 2025_Jul, etc.
MONTH_MAP = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
}


def iter_measurement_dirs(root_dir: str):
    """Yield directories that look like measurement folders by presence of F1.txt."""
    seen = set()
    pattern = f"{root_dir}/202*/**/F1.txt"
    for f1 in glob.iglob(pattern, recursive=True):
        d = os.path.dirname(f1)
        if d not in seen:
            seen.add(d)
            yield d


def parse_timestamp_from_dir(directory: str) -> datetime | None:
    """Parse a timestamp from a directory path like .../2025_Jul/01/12/34/."""
    parts = directory.strip('/').split('/')
    if len(parts) < 5:
        return None
    year_month = parts[-4]
    day = parts[-3]
    hour = parts[-2]
    minute = parts[-1]
    try:
        year_str, month_word = year_month.split('_')
        month_str = MONTH_MAP.get(month_word, '01')
        tstr = f"{year_str}-{month_str}-{day} {hour}:{minute}"
        return datetime.strptime(tstr, '%Y-%m-%d %H:%M')
    except Exception:
        return None


def load_signal(path: str):
    """Load a histogram file with columns BinCenter, Population. Returns (x, y) or (None, None)."""
    try:
        df = pd.read_csv(path, usecols=['BinCenter', 'Population'])
        x = pd.to_numeric(df['BinCenter'], errors='coerce').to_numpy(dtype=float)
        y = pd.to_numeric(df['Population'], errors='coerce').to_numpy(dtype=float)
        if np.isnan(x).any() or np.isnan(y).any() or x.size == 0 or y.size == 0:
            return None, None
        return x, y
    except Exception:
        return None, None


def plot_overlay_for_directory(directory: str, timestamp: datetime):
    """Plot F1, F2, F3, F4 signals overlaid with no scaling, save a PNG."""
    files = {
        'F1 (Short Data)': os.path.join(directory, 'F1.txt'),
        'F2 (Long Data)':  os.path.join(directory, 'F2.txt'),
        'F3 (Short BG)':   os.path.join(directory, 'F3.txt'),
        'F4 (Long BG)':    os.path.join(directory, 'F4.txt'),
    }
    # Require all 4 to exist; if not, skip quietly
    if not all(os.path.exists(p) for p in files.values()):
        return

    plt.figure(figsize=(10, 7))
    colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']

    ok_any = False
    for (label, fpath), color in zip(files.items(), colors):
        x, y = load_signal(fpath)
        if x is None:
            continue
        plt.step(x, y, where='mid', label=label, color=color, alpha=0.9)
        ok_any = True

        # Apply a Gaussian fit to F3 (Short BG) around mean ~0.8
        if label.startswith('F3'):
            try:
                # Focus fit to a window around 0.8 to stabilize
                mu0 = 0.8
                x_min, x_max = mu0 - 0.2, mu0 + 0.2
                m = (x >= x_min) & (x <= x_max) & np.isfinite(x) & np.isfinite(y)
                if np.count_nonzero(m) >= 8:
                    xw = x[m]
                    yw = y[m]
                    # Basic Gaussian model
                    def gauss(xx, A, mu, sigma):
                        return A * np.exp(-0.5 * ((xx - mu) / sigma) ** 2)
                    A0 = float(np.max(yw)) if yw.size else 1.0
                    p0 = [A0, mu0, 0.05]
                    bounds = ([0.0, x_min, 1e-3], [np.inf, x_max, 0.5])
                    popt, _ = curve_fit(gauss, xw, yw, p0=p0, bounds=bounds, maxfev=20000)
                    A, mu, sigma = popt
                    # Plot the fitted Gaussian over a smooth axis in the window
                    xfit = np.linspace(x_min, x_max, 400)
                    yfit = gauss(xfit, A, mu, sigma)
                    plt.plot(xfit, yfit, '-', color='black', linewidth=1.5,
                             label=f"F3 Gaussian fit (mu={mu:.3f}, sigma={sigma:.3f})")
            except Exception:
                # If fit fails, just skip overlay
                pass

    if not ok_any:
        plt.close()
        return

    title = timestamp.strftime('%Y-%b-%d %H:%M')
    plt.title(f'Overlayed signals (no scaling) — {title}')
    plt.xlabel('BinCenter [V]')
    plt.ylabel('Population [a.u.]')
    plt.legend(loc='best')
    plt.tight_layout()

    out_name = f"overlay_{timestamp.strftime('%Y_%b_%d_%H_%M')}.png"
    out_path = os.path.join(PLOTS_DIR, out_name)
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


def main():
    for d in iter_measurement_dirs(ROOT_DIR):
        ts = parse_timestamp_from_dir(d)
        if ts is None:
            continue
        if ts < START_DATE:
            continue
        if END_DATE is not None and ts > END_DATE:
            continue
        plot_overlay_for_directory(d, ts)


if __name__ == '__main__':
    main()
