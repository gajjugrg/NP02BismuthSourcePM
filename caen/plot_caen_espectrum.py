#!/usr/bin/env python3
"""Plot a CAEN EspectrumR .txt3 file.

This is the canonical location for the CAEN spectrum tooling (under `src/`).

Example:
  python caen/plot_caen_espectrum.py \
        "../data/caen/run" \
        --logy --save spectrum.png
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


@dataclass
class Spectrum:
    adc: np.ndarray
    counts: np.ndarray
    realtime: Optional[str]
    livetime: Optional[str]


_TIME_RE = re.compile(r"^(RealTime|LiveTime)\s*=\s*(.*)\s*$")
_CH_FILE_RE = re.compile(r"^CH(\d+)@(.+)\.txt3$")
_TS_RE = re.compile(r"(\d{8}_\d{6})")

# Default peak guesses (user-provided): channel 1 ~3200, channel 2 ~4100, channel 3 ~325.
# In CAEN naming we usually have CH0/CH1/CH2, so we map:
#   CH0 -> 3200, CH1 -> 4100, CH2 -> 325.
_DEFAULT_PEAK_ADC: dict[int, float] = {0: 3200.0, 1: 4100.0, 2: 325.0}
_DEFAULT_FIT_HALF_WINDOW: dict[int, float] = {0: 250.0, 1: 250.0, 2: 150.0}

_CHANNEL_LABELS: dict[int, str] = {
    0: "Inner Long",
    1: "Outer Long",
    2: "Test Pulse",
}

_FIT_CURVE_POINTS = 2000


def _gaussian_with_offset(x: np.ndarray, amplitude: float, mean: float, sigma: float, offset: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2) + offset


def _crystal_ball(x: np.ndarray, amplitude: float, mean: float, sigma: float, alpha: float, n: float) -> np.ndarray:
    """Crystal Ball lineshape: Gaussian core + low-side power-law tail."""
    t = (x - mean) / sigma
    abs_alpha = np.abs(alpha)
    A = (n / abs_alpha) ** n * np.exp(-0.5 * abs_alpha * abs_alpha)
    B = n / abs_alpha - abs_alpha
    out = np.empty_like(t, dtype=float)
    core = t > -abs_alpha
    tail = ~core
    out[core] = np.exp(-0.5 * t[core] * t[core])
    if np.any(tail):
        denom = np.maximum(B - t[tail], 1e-12)
        out[tail] = A * denom ** (-n)
    return amplitude * out


def _crystal_ball_with_offset(
    x: np.ndarray,
    amplitude: float,
    mean: float,
    sigma: float,
    alpha: float,
    n: float,
    offset: float,
) -> np.ndarray:
    return _crystal_ball(x, amplitude, mean, sigma, alpha, n) + offset


def _fit_gaussian_peak(
    x: np.ndarray,
    y: np.ndarray,
    peak_guess: float,
    half_window: float,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Fit a Gaussian + constant offset around a peak guess.

    Returns (params, covariance) where params = [A, mu, sigma, C].
    """

    if x.size == 0:
        return None

    lo = peak_guess - half_window
    hi = peak_guess + half_window
    mask = (x >= lo) & (x <= hi) & np.isfinite(y)
    xw = x[mask]
    yw = y[mask]
    if xw.size < 8:
        return None

    y_min = float(np.nanmin(yw))
    y_max = float(np.nanmax(yw))
    amplitude0 = max(1.0, y_max - y_min)
    offset0 = max(0.0, y_min)

    w = np.clip(yw - offset0, 0.0, None)
    if float(np.sum(w)) > 0:
        mean0 = float(np.sum(xw * w) / np.sum(w))
    else:
        mean0 = float(peak_guess)
    sigma0 = max(5.0, float(half_window) / 5.0)

    p0 = np.array([amplitude0, mean0, sigma0, offset0], dtype=float)
    bounds = (
        np.array([0.0, lo, 1e-3, 0.0], dtype=float),
        np.array([np.inf, hi, np.inf, np.inf], dtype=float),
    )

    try:
        popt, pcov = curve_fit(
            _gaussian_with_offset,
            xw,
            yw,
            p0=p0,
            bounds=bounds,
            maxfev=20000,
        )
    except Exception:
        return None

    return popt, pcov


def _fit_crystal_ball_peak(
    x: np.ndarray,
    y: np.ndarray,
    peak_guess: float,
    half_window: float,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Fit a Crystal Ball + constant offset around a peak guess.

    Returns (params, covariance) where params = [A, mu, sigma, alpha, n, C].
    """

    if x.size == 0:
        return None

    lo = peak_guess - half_window
    hi = peak_guess + half_window
    mask = (x >= lo) & (x <= hi) & np.isfinite(y)
    xw = x[mask]
    yw = y[mask]
    if xw.size < 10:
        return None

    y_min = float(np.nanmin(yw))
    y_max = float(np.nanmax(yw))
    amplitude0 = max(1.0, y_max - y_min)
    offset0 = max(0.0, y_min)

    w = np.clip(yw - offset0, 0.0, None)
    if float(np.sum(w)) > 0:
        mean0 = float(np.sum(xw * w) / np.sum(w))
    else:
        mean0 = float(peak_guess)

    sigma0 = max(5.0, float(half_window) / 6.0)
    alpha0 = 1.5
    n0 = 3.0

    p0 = np.array([amplitude0, mean0, sigma0, alpha0, n0, offset0], dtype=float)
    bounds = (
        np.array([0.0, lo, 1e-3, 0.05, 0.5, 0.0], dtype=float),
        np.array([np.inf, hi, np.inf, 10.0, 50.0, np.inf], dtype=float),
    )

    try:
        popt, pcov = curve_fit(
            _crystal_ball_with_offset,
            xw,
            yw,
            p0=p0,
            bounds=bounds,
            maxfev=40000,
        )
    except Exception:
        return None

    return popt, pcov


def _iter_lines(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            yield line.rstrip("\n")


def _parse_spectrum(path: str) -> Spectrum:
    realtime: Optional[str] = None
    livetime: Optional[str] = None

    adc_bins: List[int] = []
    counts: List[int] = []

    for raw_line in _iter_lines(path):
        line = raw_line.strip()
        if not line:
            continue

        m = _TIME_RE.match(line)
        if m:
            if m.group(1) == "RealTime":
                realtime = m.group(2)
            else:
                livetime = m.group(2)
            continue

        parts = line.split()
        if len(parts) < 2:
            continue
        if "=" in line:
            continue

        try:
            adc_bin = int(float(parts[0]))
            ct = int(float(parts[1]))
        except ValueError:
            continue

        adc_bins.append(adc_bin)
        counts.append(ct)

    adc_arr = np.asarray(adc_bins, dtype=int)
    counts_arr = np.asarray(counts, dtype=float)
    if adc_arr.size == 0:
        raise ValueError(f"No numeric spectrum rows found in: {path}")

    return Spectrum(adc=adc_arr, counts=counts_arr, realtime=realtime, livetime=livetime)


def _extract_timestamp_from_path(path: str) -> Optional[str]:
    matches = _TS_RE.findall(os.path.basename(path))
    if not matches:
        return None
    return matches[-1]


def _build_title(path: str, spectrum: Spectrum, channel: Optional[int] = None) -> str:
    # Keep per-panel titles minimal (no LiveTime/RealTime; time is shown once as a figure title).
    if channel is not None:
        label = _CHANNEL_LABELS.get(channel)
        return f"CH{channel} -- {label}" if label else f"CH{channel}"
    return os.path.basename(path)


def _extract_common_timestamp(paths: List[str]) -> Optional[str]:
    """Return a common YYYYmmdd_HHMMSS timestamp if present, else best-effort from first path."""

    if not paths:
        return None

    sets = []
    for p in paths:
        matches = _TS_RE.findall(os.path.basename(p))
        sets.append(set(matches))

    common = set.intersection(*sets) if sets else set()
    if common:
        # If multiple, pick the lexicographically last (usually latest).
        return sorted(common)[-1]

    return _extract_timestamp_from_path(paths[0])


def _format_time_label(ts: str) -> str:
    dt = _parse_timestamp(ts)
    if dt is None:
        return ts
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _infer_channel_paths(path: str, channels: List[int]) -> List[tuple[int, str]]:
    directory = os.path.dirname(path)
    base = os.path.basename(path)
    m = re.search(r"\bCH(\d+)(@.*)$", base)
    if not m:
        raise ValueError(
            "Input filename must contain 'CH<n>@...' so sibling channels can be inferred. "
            f"Got: {base}"
        )
    ch_str = m.group(1)
    suffix = m.group(2)
    width = len(ch_str)

    results: List[tuple[int, str]] = []
    for ch in channels:
        ch_formatted = str(ch).zfill(width)
        sibling = os.path.join(directory, f"CH{ch_formatted}{suffix}")
        if os.path.exists(sibling):
            results.append((ch, sibling))
    return results


def _parse_timestamp(ts: str) -> Optional[datetime]:
    try:
        return datetime.strptime(ts, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def _find_channel_paths_in_run_dir(run_dir: str, channels: List[int], timestamp: Optional[str]) -> List[tuple[int, str]]:
    raw_dir = os.path.join(run_dir, "RAW")
    if not os.path.isdir(raw_dir):
        raise ValueError(f"Run directory does not contain RAW/: {run_dir}")

    groups: dict[str, dict[int, str]] = {}
    group_ts: dict[str, Optional[datetime]] = {}
    group_mtime: dict[str, float] = {}

    for name in os.listdir(raw_dir):
        m = _CH_FILE_RE.match(name)
        if not m:
            continue
        ch = int(m.group(1))
        suffix = m.group(2)
        if not name.endswith(".txt3"):
            continue
        full_path = os.path.join(raw_dir, name)
        groups.setdefault(suffix, {})[ch] = full_path

        ts_match = _TS_RE.findall(suffix)
        ts_dt = _parse_timestamp(ts_match[-1]) if ts_match else None
        if suffix not in group_ts or (group_ts[suffix] is None and ts_dt is not None):
            group_ts[suffix] = ts_dt

        try:
            mtime = os.path.getmtime(full_path)
        except OSError:
            mtime = 0.0
        group_mtime[suffix] = max(group_mtime.get(suffix, 0.0), mtime)

    required = set(channels)

    candidates: List[str] = []
    for suffix, ch_map in groups.items():
        if not required.issubset(ch_map.keys()):
            continue
        if timestamp is not None and timestamp not in suffix:
            continue
        candidates.append(suffix)

    if not candidates:
        want = ", ".join(f"CH{c}" for c in channels)
        hint = f" (timestamp filter: {timestamp})" if timestamp is not None else ""
        raise ValueError(f"No complete channel set found in {raw_dir} for {want}{hint}.")

    def sort_key(suffix: str) -> tuple[int, float, float]:
        ts = group_ts.get(suffix)
        if ts is None:
            return (0, group_mtime.get(suffix, 0.0), 0.0)
        return (1, ts.timestamp(), group_mtime.get(suffix, 0.0))

    best_suffix = max(candidates, key=sort_key)
    ch_map = groups[best_suffix]
    return [(ch, ch_map[ch]) for ch in channels]


def _run_dir_has_complete_channel_set(run_dir: str, channels: List[int]) -> bool:
    raw_dir = os.path.join(run_dir, "RAW")
    if not os.path.isdir(raw_dir):
        return False

    required = set(channels)
    groups: dict[str, set[int]] = {}
    for name in os.listdir(raw_dir):
        m = _CH_FILE_RE.match(name)
        if not m:
            continue
        ch = int(m.group(1))
        suffix = m.group(2)
        if not name.endswith(".txt3"):
            continue
        groups.setdefault(suffix, set()).add(ch)

    return any(required.issubset(chs) for chs in groups.values())


def _choose_default_run_dir(caen_root: str, channels: List[int]) -> Optional[str]:
    """Pick a default run dir under CAEN root that contains a complete channel set."""

    preferred_names = [
        "NewTP_200mVpp_100Hz_PosRamp",
        "NewTP_200mVpp_100Hz_PosRamp_1",
        "NewTP_200mVpp_100Hz_PosRamp_2",
        "NewTP_200mVpp_100Hz_PosRamp_3",
        "NewTP_200mVpp_100Hz_PosRamp_4",
    ]

    for name in preferred_names:
        candidate = os.path.join(caen_root, name)
        if os.path.isdir(candidate) and _run_dir_has_complete_channel_set(candidate, channels):
            return candidate

    # Fallback: first directory under caen_root with a complete set.
    try:
        entries = sorted(os.listdir(caen_root))
    except OSError:
        return None

    for name in entries:
        candidate = os.path.join(caen_root, name)
        if os.path.isdir(candidate) and _run_dir_has_complete_channel_set(candidate, channels):
            return candidate

    return None


def main() -> int:
    caen_root = os.environ.get("CAEN_DATA_DIR")
    if caen_root is None:
        repo_root = Path(__file__).resolve().parents[2]
        caen_root = str((repo_root / "data" / "caen").resolve())
    default_path = _choose_default_run_dir(caen_root, [0, 1, 2]) or os.path.join(caen_root, "NewTP_200mVpp_100Hz_PosRamp_1")

    parser = argparse.ArgumentParser(description="Read a CAEN EspectrumR .txt3 file and plot it.")
    parser.add_argument(
        "path",
        nargs="?",
        default=default_path,
        help="Path to a run directory (containing RAW/) or a single CH*.txt3 file",
    )
    parser.add_argument("--channels", nargs="*", type=int, default=[0, 1, 2], help="Channels to plot")
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Optional timestamp filter like 20260121_133853 (if omitted, uses latest set found)",
    )
    parser.add_argument("--no-fit", action="store_true", help="Disable peak fitting/annotations")
    parser.add_argument(
        "--fit-model",
        choices=["gaussian", "crystalball"],
        default="crystalball",
        help="Peak model for peak fitting. Default: crystalball",
    )
    parser.add_argument("--logy", action="store_true", help="Use log scale on Y axis")
    parser.add_argument("--keep-adc0", action="store_true", help="Keep the ADC=0 bin (ignored by default)")
    parser.add_argument("--xmin", type=float, default=None, help="Minimum x to display")
    parser.add_argument("--xmax", type=float, default=5000, help="Maximum x to display")
    parser.add_argument("--save", default=None, help="Save figure to this path (e.g. spectrum.png)")

    args = parser.parse_args()

    if os.path.isdir(args.path):
        channel_paths = _find_channel_paths_in_run_dir(args.path, args.channels, args.timestamp)
    else:
        if args.timestamp is not None:
            print("Note: --timestamp is only used when providing a run directory.")
        channel_paths = _infer_channel_paths(args.path, args.channels)

    if len(channel_paths) == 0:
        base = os.path.basename(args.path)
        raise ValueError(
            "No channel files found. Check the input path and/or pass different --channels values. "
            f"(input: {base})"
        )

    fig, axes = plt.subplots(
        nrows=len(channel_paths),
        ncols=1,
        sharex=False,
        figsize=(10, max(3, 3 * len(channel_paths))),
    )
    # Normalize axes to a plain list for simpler downstream code.
    if isinstance(axes, np.ndarray):
        axes = axes.ravel().tolist()
    else:
        axes = [axes]

    # Put TIME once on the figure title.
    time_label = _extract_common_timestamp([p for _, p in channel_paths])
    if time_label:
        fig.suptitle(f"TIME {_format_time_label(time_label)}")

    for ax, (ch, path) in zip(axes, channel_paths):
        spectrum = _parse_spectrum(path)
        x_adc = spectrum.adc.astype(float)
        mask = np.ones_like(spectrum.adc, dtype=bool) if args.keep_adc0 else (spectrum.adc != 0)
        x_adc = x_adc[mask]
        y = spectrum.counts[mask]

        x = x_adc

        ax.step(x, y, where="mid", linewidth=1.0)
        ax.set_ylabel("Counts")
        ax.set_title(_build_title(path, spectrum, channel=ch))
        ax.set_xlabel("ADC")
        ax.grid(True, which="both", alpha=0.25)

        # Peak fit + annotation (fit always performed in ADC space).
        if not args.no_fit and ch in _DEFAULT_PEAK_ADC:
            peak_guess = _DEFAULT_PEAK_ADC[ch]
            half_window = _DEFAULT_FIT_HALF_WINDOW.get(ch, 200.0)
            fit_model = "gaussian" if ch == 1 or ch == 2 else args.fit_model
            if fit_model == "crystalball":
                fit = _fit_crystal_ball_peak(x_adc, y, peak_guess=peak_guess, half_window=half_window)
                if fit is not None:
                    popt, pcov = fit
                    amplitude, mean, sigma, alpha, n, offset = popt
                    sigma = abs(float(sigma))
                    alpha = abs(float(alpha))
                    n = abs(float(n))

                    mean_err = sigma_err = alpha_err = n_err = None
                    try:
                        perr = np.sqrt(np.diag(pcov))
                        mean_err = float(perr[1])
                        sigma_err = float(perr[2])
                        alpha_err = float(perr[3])
                        n_err = float(perr[4])
                    except Exception:
                        pass

                    x_fit_adc = np.linspace(
                        max(float(np.min(x_adc)), float(mean) - half_window),
                        min(float(np.max(x_adc)), float(mean) + half_window),
                        _FIT_CURVE_POINTS,
                    )
                    y_fit = _crystal_ball_with_offset(x_fit_adc, amplitude, mean, sigma, alpha, n, offset)
                    x_fit = x_fit_adc
                    ax.plot(x_fit, y_fit, linewidth=1.5)

                    mean_display = float(mean)
                    sigma_display = float(sigma)
                    mean_err_display = mean_err
                    sigma_err_display = sigma_err

                    if mean_err_display is not None and sigma_err_display is not None:
                        text = (
                            f"$\\mu$ = {mean_display:.1f} ± {mean_err_display:.1f}\n"
                            f"$\\sigma$ = {sigma_display:.1f} ± {sigma_err_display:.1f}\n"
                        )
                    else:
                        text = f"$\\mu$ = {mean_display:.1f}\n$\\sigma$ = {sigma_display:.1f}\n$\\alpha$ = {alpha:.2f}\n$n$ = {n:.1f}"

                    ax.text(
                        0.98,
                        0.95,
                        text,
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=10,
                        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                    )
            else:
                fit = _fit_gaussian_peak(x_adc, y, peak_guess=peak_guess, half_window=half_window)
                if fit is not None:
                    popt, pcov = fit
                    amplitude, mean, sigma, offset = popt
                    sigma = abs(float(sigma))

                    mean_err = sigma_err = None
                    try:
                        perr = np.sqrt(np.diag(pcov))
                        mean_err = float(perr[1])
                        sigma_err = float(perr[2])
                    except Exception:
                        pass

                    x_fit_adc = np.linspace(
                        max(float(np.min(x_adc)), float(mean) - half_window),
                        min(float(np.max(x_adc)), float(mean) + half_window),
                        _FIT_CURVE_POINTS,
                    )
                    y_fit = _gaussian_with_offset(x_fit_adc, amplitude, mean, sigma, offset)
                    x_fit = x_fit_adc
                    ax.plot(x_fit, y_fit, linewidth=1.5)

                    mean_display = float(mean)
                    sigma_display = float(sigma)
                    mean_err_display = mean_err
                    sigma_err_display = sigma_err

                    if mean_err_display is not None and sigma_err_display is not None:
                        text = f"$\\mu$ = {mean_display:.1f} ± {mean_err_display:.1f}\n$\\sigma$ = {sigma_display:.1f} ± {sigma_err_display:.1f}"
                    else:
                        text = f"$\\mu$ = {mean_display:.1f}\n$\\sigma$ = {sigma_display:.1f}"

                    ax.text(
                        0.98,
                        0.95,
                        text,
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=10,
                        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                    )

        if args.logy:
            ax.set_yscale("log")

        if args.xmin is not None or args.xmax is not None:
            xmin = args.xmin if args.xmin is not None else float(np.nanmin(x))
            xmax = args.xmax if args.xmax is not None else float(np.nanmax(x))
            ax.set_xlim(xmin, xmax)

    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved: {args.save}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
