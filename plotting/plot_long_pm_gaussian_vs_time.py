import os
import glob
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Root of the NP02 data tree
NP02DATA_DIR = os.environ.get('NP02DATA_DIR', '../np02data')
ROOT_DIR = NP02DATA_DIR
PLOTS_DIR = 'plots'
INDIVIDUAL_PLOTS_SUBDIR = 'long_pm_individual_fits'
TEMPERATURE_FILE = os.environ.get('NP02_LONG_PM_TEMPERATURE_FILE', os.path.join(NP02DATA_DIR, 'May14_20.csv'))
REDO_LONG_PM_PLOTS = os.environ.get('REDO_LONG_PM_PLOTS', 'Yes')

# Time window (inclusive). Adjust as needed.
START = datetime(2025, 5, 14, 16, 00)
END   = datetime(2025, 5, 19, 14, 00)

# Long PM files (data/background)
DATA_FILE = 'F2.txt'
BG_FILE   = 'F4.txt'

MONTH_MAP = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
}


def gauss(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def should_regenerate_plots() -> bool:
    return str(REDO_LONG_PM_PLOTS).strip().lower() == 'yes'


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


def load_xy(path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
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
    x, y = x[m], y[m]
    if x.size == 0 or y.size == 0:
        return None
    return x, y


def load_temperature_series(path: str,
                            start: datetime,
                            end: datetime) -> Tuple[np.ndarray, np.ndarray]:
    """Load temperature timestamps/values within the requested window."""
    if not os.path.exists(path):
        return np.array([]), np.array([])
    try:
        df = pd.read_csv(
            path,
            header=None,
            names=['timestamp', 'temperature_c'],
            parse_dates=['timestamp'],
            sep=',',
            engine='python',
            skipinitialspace=True,
        )
    except Exception:
        return np.array([]), np.array([])
    if df.empty:
        return np.array([]), np.array([])
    mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
    if not mask.any():
        return np.array([]), np.array([])
    subset = df.loc[mask]
    return subset['timestamp'].to_numpy(), subset['temperature_c'].to_numpy(dtype=float)


def fit_long_gaussian(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Fit a single Gaussian to y(x) using the full data range."""
    # Initial guesses
    try:
        A0 = float(np.max(y))
        mu0 = float(x[np.argmax(y)])
    except Exception:
        return None
    sigma0 = 0.05
    try:
        popt, _ = curve_fit(gauss, x, y, p0=[A0, mu0, sigma0], maxfev=5000)
        A, mu, sigma = map(float, popt)
        return A, mu, sigma
    except Exception:
        return None


def main():
    # Discover directories that contain the Long PM files and fall within the window
    pattern = f"{ROOT_DIR}/2025_*/**/{DATA_FILE}"
    rows = []  # (time, A, mu, sigma, dir) for background-subtracted signal (kept for ref)
    data_fits = []         # (time, xfit, yfit) for F2
    background_fits = []   # (time, xfit, yfit) for F4
    data_curves = []       # (time, x, y) data (full range)
    background_curves = [] # (time, x, y) background (full range)
    data_mu_times = []     # times for data (F2) mu series
    data_mus = []          # mu from data (F2)
    bkg_mu_times = []      # times for background (F4) mu series
    bkg_mus = []           # mu from background (F4)
    individual_plots = []  # per timestamp plotting payload
    for f2 in glob.iglob(pattern, recursive=True):
        d = os.path.dirname(f2)
        t = parse_dir_time(d)
        if t is None:
            continue
        if t < START or t > END:
            continue
        # Load data and background
        xy_data = load_xy(os.path.join(d, DATA_FILE))
        xy_bg   = load_xy(os.path.join(d, BG_FILE))
        if xy_data is None or xy_bg is None:
            continue
        x_d, y_d = xy_data
        x_b, y_b = xy_bg
        if x_d.size == 0 or x_b.size == 0:
            continue
        # Prepare plotting payload holders
        inner_payload = {'curve': (x_d, y_d)}
        outer_payload = {'curve': (x_b, y_b)}
        # Fit Gaussian to DATA (F2) on the full range
        fit_d = fit_long_gaussian(x_d, y_d)
        if fit_d is not None:
            Ad, mud, sigd = fit_d
            xfit_d = np.linspace(float(x_d.min()), float(x_d.max()), 600)
            yfit_d = gauss(xfit_d, Ad, mud, sigd)
            data_fits.append((t, xfit_d, yfit_d))
            data_curves.append((t, x_d, y_d))
            data_mu_times.append(t)
            data_mus.append(mud)
            inner_payload['fit'] = (xfit_d, yfit_d)
            inner_payload['params'] = (Ad, mud, sigd)
        # Fit Gaussian to BACKGROUND (F4) on the full range
        fit_b = fit_long_gaussian(x_b, y_b)
        if fit_b is not None:
            Ab, mub, sigb = fit_b
            xfit_b = np.linspace(float(x_b.min()), float(x_b.max()), 600)
            yfit_b = gauss(xfit_b, Ab, mub, sigb)
            background_fits.append((t, xfit_b, yfit_b))
            background_curves.append((t, x_b, y_b))
            bkg_mu_times.append(t)
            bkg_mus.append(mub)
            outer_payload['fit'] = (xfit_b, yfit_b)
            outer_payload['params'] = (Ab, mub, sigb)
        # Interpolate background to data x-grid (full range)
        order = np.argsort(x_b)
        x_b_sorted = x_b[order]
        y_b_sorted = y_b[order]
        try:
            y_b_on_d = np.interp(x_d, x_b_sorted, y_b_sorted, left=0.0, right=0.0)
        except Exception:
            continue
        # Least-squares scaling of background (full range)
        denom = float(np.dot(y_b_on_d, y_b_on_d))
        scale = float(np.dot(y_b_on_d, y_d)) / denom if denom > 0 else 0.0
        y_sig = y_d - scale * y_b_on_d
        # Fit Gaussian to background-subtracted signal (full range)
        fit = fit_long_gaussian(x_d, y_sig)
        if fit is None:
            continue
        A, mu, sigma = fit
        rows.append((t, A, mu, sigma, d))
        individual_plots.append({'time': t, 'inner': inner_payload, 'outer': outer_payload})

    if not rows and (not data_fits and not background_fits):
        print('[INFO] No fits found in the requested window.')
        return
    rows.sort(key=lambda r: r[0])
    times = [r[0] for r in rows]
    amps  = [r[1] for r in rows]
    mus   = [r[2] for r in rows]
    sigs  = [r[3] for r in rows]

    os.makedirs(PLOTS_DIR, exist_ok=True)

    temp_times, temp_values = load_temperature_series(TEMPERATURE_FILE, START, END)
    temp_seconds_sorted = None
    temp_values_sorted = None
    if temp_times.size and temp_values.size:
        order = np.argsort(temp_times)
        temp_times_sorted = pd.to_datetime(temp_times[order])
        temp_values_sorted = temp_values[order]
        temp_seconds_sorted = temp_times_sorted.view('int64').astype(np.float64) / 1e9

    # (Removed) amplitude vs time plot per request

    # Plot peak position (mu) vs time; inner/out separate canvases
    if data_mu_times or bkg_mu_times:
        num_axes = int(bool(data_mu_times)) + int(bool(bkg_mu_times))
        fig, axes = plt.subplots(num_axes, 1, figsize=(10, 3.5 * num_axes), sharex=True)
        if num_axes == 1:
            axes = [axes]
        ax_idx = 0

        def overlay_temperature(ax):
            if temp_times.size and temp_values.size:
                ax_temp = ax.twinx()
                temp_line, = ax_temp.plot(temp_times, temp_values, '-', color='tab:red', label='Temperature (°C)')
                ax_temp.set_ylabel('Temperature (°C)')
                handles, labels = ax.get_legend_handles_labels()
                handles_t, labels_t = ax_temp.get_legend_handles_labels()
                ax.legend(handles + handles_t, labels + labels_t, loc='best')
            else:
                ax.legend(loc='best')

        if data_mu_times:
            ax_inner = axes[ax_idx]
            ax_idx += 1
            ax_inner.plot(data_mu_times, data_mus, 'o', alpha=0.85, label=r'F2 $\mu$ (inner anode)')
            ax_inner.set_ylabel('Gaussian mean (V)')
            ax_inner.set_title(f'F2 (inner anode)')
            ax_inner.grid(True, alpha=0.25)
            overlay_temperature(ax_inner)

        if bkg_mu_times:
            ax_outer = axes[ax_idx]
            ax_idx += 1
            ax_outer.plot(bkg_mu_times, bkg_mus, 'o', alpha=0.85, label=r'F4 $\mu$ (outer anode)')
            ax_outer.set_ylabel('Gaussian mean (V)')
            ax_outer.set_title(f'F4 (outer anode)')
            ax_outer.grid(True, alpha=0.25)
            overlay_temperature(ax_outer)

        axes[-1].set_xlabel('Time')
        fig.autofmt_xdate()
        fig.tight_layout()
        plt.show()
        out2 = os.path.join(PLOTS_DIR, 'long_pm_gaussian_mean_vs_time.png')
        fig.savefig(out2, dpi=150)
        print(f'Saved plot: {out2}')
        plt.close(fig)

    def temps_for(mu_times_list, mu_values_list):
        if (
            not mu_times_list
            or temp_seconds_sorted is None
            or temp_values_sorted is None
            or temp_seconds_sorted.size == 0
        ):
            return np.array([]), np.array([])
        mu_seconds = np.array([t.timestamp() for t in mu_times_list], dtype=float)
        mu_values_arr = np.array(mu_values_list, dtype=float)
        mask = (mu_seconds >= temp_seconds_sorted[0]) & (mu_seconds <= temp_seconds_sorted[-1])
        if not np.any(mask):
            return np.array([]), np.array([])
        mu_seconds = mu_seconds[mask]
        mu_values_arr = mu_values_arr[mask]
        temps_interp = np.interp(mu_seconds, temp_seconds_sorted, temp_values_sorted)
        return temps_interp, mu_values_arr

    inner_temps, inner_mus = temps_for(data_mu_times, data_mus)

    redo_plots = should_regenerate_plots()
    if not redo_plots:
        print("[INFO] REDO_LONG_PM_PLOTS is not 'Yes'; skipping temperature correlation plots.")
    else:
        inner_fit_params = None
        inner_fit_fn = None
        inner_cov = None
        if inner_temps.size >= 2:
            try:
                coeffs, inner_cov = np.polyfit(inner_temps, inner_mus, 1, cov=True)
                inner_fit_params = coeffs
                inner_fit_fn = np.poly1d(coeffs)
            except Exception as exc:
                print(f"[WARN] Failed to fit inner μ vs T: {exc}")

        if inner_temps.size:
            fig, ax_inner = plt.subplots(figsize=(8, 3.2))
            ax_inner.scatter(inner_temps, inner_mus, alpha=0.85, label=r'F2 $\mu$ vs T (inner anode)')
            if inner_fit_fn is not None and inner_cov is not None:
                temp_grid = np.linspace(float(inner_temps.min()), float(inner_temps.max()), 200)
                y_fit = inner_fit_fn(temp_grid)
                ax_inner.plot(temp_grid, y_fit, color='tab:blue', lw=1.5, label='Linear fit')
                var_fit = (
                    inner_cov[0, 0] * temp_grid ** 2
                    + 2 * inner_cov[0, 1] * temp_grid
                    + inner_cov[1, 1]
                )
                var_fit = np.clip(var_fit, a_min=0.0, a_max=None)
                sigma_fit = np.sqrt(var_fit)
                ax_inner.fill_between(
                    temp_grid,
                    y_fit - sigma_fit,
                    y_fit + sigma_fit,
                    color='tab:blue',
                    alpha=0.18,
                    label='Fit ±1σ (param)'
                )
                slope, intercept = inner_fit_params
                slope_err = float(np.sqrt(inner_cov[0, 0])) if inner_cov is not None else float('nan')
                intercept_err = float(np.sqrt(inner_cov[1, 1])) if inner_cov is not None else float('nan')
                fitted = inner_fit_fn(inner_temps)
                resid = inner_mus - fitted
                sigma_resid = np.std(resid, ddof=1) if inner_mus.size > 1 else 1.0
                chi2 = float(np.sum((resid / sigma_resid) ** 2)) if sigma_resid > 0 else float('nan')
                dof = max(inner_mus.size - 2, 1)
                chi2_red = chi2 / dof if dof > 0 else float('nan')
                stats_lines = [
                    rf'$N$ = {inner_mus.size}',
                    rf'$\mu = ({slope:.4g} \pm {slope_err:.2g})\,T + ({intercept:.4g} \pm {intercept_err:.2g})$',
                    rf'$\chi^2_{{\nu}}$ = {chi2_red:.3g}',
                ]
                ax_inner.text(
                    0.02,
                    0.05,
                    '\n'.join(stats_lines),
                    transform=ax_inner.transAxes,
                    fontsize=9,
                    ha='left',
                    va='bottom',
                    bbox={'facecolor': 'white', 'alpha': 0.75, 'edgecolor': 'none'}
                )
            ax_inner.set_xlabel('Temperature (°C)')
            ax_inner.set_ylabel('Gaussian mean (V)')
            ax_inner.set_title(r'F2 (inner anode) $\mu$ vs temperature')
            ax_inner.grid(True, alpha=0.25)
            ax_inner.legend(loc='best')
            fig.tight_layout()
            out_temp_inner = os.path.join(PLOTS_DIR, 'long_pm_mu_vs_temperature_inner.png')
            fig.savefig(out_temp_inner, dpi=150)
            print(f'Saved plot: {out_temp_inner}')
            plt.close(fig)
        else:
            out_temp_inner = os.path.join(PLOTS_DIR, 'long_pm_mu_vs_temperature_inner.png')
            if os.path.exists(out_temp_inner):
                os.remove(out_temp_inner)
                print(f"[INFO] Removed stale plot with no inner μ data: {out_temp_inner}")


    if individual_plots:
        out_dir = os.path.join(PLOTS_DIR, INDIVIDUAL_PLOTS_SUBDIR)
        os.makedirs(out_dir, exist_ok=True)
        for payload in individual_plots:
            tt = payload['time']
            fig, ax = plt.subplots(figsize=(8, 4))
            legend_handles = []
            legend_labels = []
            inner = payload.get('inner')
            outer = payload.get('outer')
            if inner:
                x_curve, y_curve = inner['curve']
                curve_line, = ax.step(x_curve, y_curve, where='mid', color='tab:blue', alpha=0.35, label='F2 data (inner anode)')
                legend_handles.append(curve_line)
                legend_labels.append('F2 data (inner anode)')
                fit = inner.get('fit')
                if fit is not None:
                    x_fit, y_fit = fit
                    fit_line, = ax.plot(x_fit, y_fit, color='tab:blue', lw=1.5, label='F2 fit (inner anode)')
                    legend_handles.append(fit_line)
                    legend_labels.append('F2 fit (inner anode)')
            if outer:
                x_curve, y_curve = outer['curve']
                curve_line, = ax.step(x_curve, y_curve, where='mid', color='tab:orange', alpha=0.35, label='F4 data (outer anode)')
                legend_handles.append(curve_line)
                legend_labels.append('F4 data (outer anode)')
                fit = outer.get('fit')
                if fit is not None:
                    x_fit, y_fit = fit
                    fit_line, = ax.plot(x_fit, y_fit, color='tab:orange', lw=1.5, label='F4 fit (outer anode)')
                    legend_handles.append(fit_line)
                    legend_labels.append('F4 fit (outer anode)')
            fit_lines = []
            params_inner = inner.get('params') if inner else None
            if params_inner is not None:
                Ai, mui, sigi = params_inner
                fit_lines.append(rf"F2 fit: A={Ai:.3g}, $\mu$={mui:.4f} V, σ={sigi:.4f} V")
            params_outer = outer.get('params') if outer else None
            if params_outer is not None:
                Ao, muo, sigo = params_outer
                fit_lines.append(rf"F4 fit: A={Ao:.3g}, $\mu$={muo:.4f} V, σ={sigo:.4f} V")
            if fit_lines:
                ax.text(
                    0.02,
                    0.98,
                    "\n".join(fit_lines),
                    transform=ax.transAxes,
                    ha='left',
                    va='top',
                    fontsize=9,
                    bbox={'facecolor': 'white', 'alpha': 0.75, 'edgecolor': 'none'}
                )
            ax.set_title(tt.strftime('%Y-%m-%d %H:%M'))
            ax.set_xlabel('BinCenter [V]')
            ax.set_ylabel('Population')
            ax.grid(True, alpha=0.25)
            if legend_handles:
                ax.legend(legend_handles, legend_labels, loc='best')
            fig.tight_layout()
            fname = f"long_pm_fits_{tt.strftime('%Y%m%d_%H%M')}.png"
            fpath = os.path.join(out_dir, fname)
            fig.savefig(fpath, dpi=150)
            print(f'Saved individual plot: {fpath}')
            plt.close(fig)

    # Overlay all fitted Gaussians and the underlying data (fit window) for DATA (F2) and BACKGROUND (F4)
    if data_fits or background_fits:
        # Build a time-based colormap
        all_times = [tt for (tt, *_rest) in data_fits] + [tt for (tt, *_rest) in background_fits]
        if all_times:
            tmin = min(all_times).timestamp()
            tmax = max(all_times).timestamp()
        else:
            tmin = tmax = START.timestamp()
        def color_for(tt: datetime):
            if tmax == tmin:
                u = 0.5
            else:
                u = (tt.timestamp() - tmin) / (tmax - tmin)
            return plt.cm.viridis(u)

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
        ax1, ax2 = axes[0], axes[1]
        # Data curves and fits
        for (tt, xd, yd), (tt2, xf, yf) in zip(data_curves, data_fits):
            c = color_for(tt)
            ax1.step(xd, yd, where='mid', color=c, alpha=0.15)
            ax1.plot(xf, yf, color=c, alpha=0.7)
        ax1.set_title('Long PM DATA (F2 inner anode): Gaussian fits overlay')
        ax1.set_xlabel('BinCenter [V]')
        ax1.set_ylabel('Population (fit)')
        ax1.grid(True, alpha=0.25)
        # Background curves and fits
        for (tt, xb, yb), (tt2, xf, yf) in zip(background_curves, background_fits):
            c = color_for(tt)
            ax2.step(xb, yb, where='mid', color=c, alpha=0.15)
            ax2.plot(xf, yf, color=c, alpha=0.7)
        ax2.set_title('Long PM BACKGROUND (F4 outer anode): Gaussian fits overlay')
        ax2.set_xlabel('BinCenter [V]')
        ax2.set_ylabel('Population (fit)')
        ax2.grid(True, alpha=0.25)
        fig.suptitle(f'Gaussian fits overlay — {START.strftime("%b %d %H:%M")} to {END.strftime("%b %d %H:%M")}')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_overlay = os.path.join(PLOTS_DIR, 'long_pm_data_background_gaussian_fits_overlay.png')
        fig.savefig(out_overlay, dpi=150)
        print(f'Saved overlay plot: {out_overlay}')
        plt.close(fig)


if __name__ == '__main__':
    main()
