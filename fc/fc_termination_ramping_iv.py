"""
Build IV curves for FC termination ramping data in the new “flat list” format.

The file (or the inline string below) is read as repeating blocks of 14 numbers:
  1) High-voltage setting   [V]
  2) Supply current         [µA]
  3-14) Termination currents for channels 1-12 [A]

Channels 1-6 are TCO and 7-12 are BEAM. Currents are already absolute values per
step, so no cumulative sum is applied.
You can paste the data directly into INLINE_DATA below to avoid reading a file.
The supply-current IV (column 2) is plotted separately with a fitted resistance.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

NP02DATA_DIR = os.environ.get("NP02DATA_DIR", "../np02data")
#DATA_FILE = Path(os.path.join(NP02DATA_DIR, "fc_termination_ramping.txt"))
DATA_FILE = Path(os.environ.get("FC_TERMINATION_RAMPING_FILE", os.path.join(NP02DATA_DIR, "fc_termination_raming_NewCable.txt")))
INLINE_DATA = ""  # paste data here to override the file input
BLOCK_SIZE = 14  # 1 voltage + 1 supply current + 12 channel currents
CHANNEL_GROUPS = {ch: ("TCO" if ch <= 6 else "BEAM") for ch in range(1, 13)}


def _collect_values(path: Path, inline_data: str) -> List[float]:
    if inline_data.strip():
        lines = inline_data.strip().splitlines()
    else:
        if not path.exists():
            raise FileNotFoundError(f"Missing data file: {path}")
        with open(path, "r") as f:
            lines = f.readlines()
    values: List[float] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for token in stripped.split():
            values.append(float(token))
    return values


def load_ramping_data(path: Path, inline_data: str = "") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns voltages (N), channels (12), channel currents (12 x N), supply current (N).
    """
    values = _collect_values(path, inline_data)
    if len(values) % BLOCK_SIZE != 0:
        raise ValueError(f"Expected blocks of {BLOCK_SIZE} values, got {len(values)}")

    data = np.array(values, dtype=float).reshape(-1, BLOCK_SIZE)
    voltages = data[:, 0]
    supply_current_uA = data[:, 1]
    channel_currents = data[:, 2:]  # (N, 12)

    channels = np.arange(1, channel_currents.shape[1] + 1, dtype=int)
    # transpose so shape is (12, N) to match plotting loop
    return voltages, channels, channel_currents.T, supply_current_uA


def fit_iv_curve(currents: np.ndarray, voltages: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit V = slope * I + intercept, returning slope (Ohm), intercept (V), slope error.
    """
    (slope, intercept), cov = np.polyfit(currents, voltages, 1, cov=True)
    slope_err = float(np.sqrt(cov[0, 0]))
    return slope, intercept, slope_err


def plot_iv(
    voltages: np.ndarray,
    currents: np.ndarray,
    channels: np.ndarray,
    slopes: np.ndarray,
    slope_errs: np.ndarray,
    intercepts: np.ndarray,
    out_dir: Path,
):
    plt.figure(figsize=(10, 6))
    lines = []
    labels = []
    annotations = []
    voltage_kV = voltages * 1e-3
    group_colors = {"TCO": "tab:orange", "BEAM": "tab:blue"}
    for ch, I, slope, slope_err, intercept in zip(channels, currents, slopes, slope_errs, intercepts):
        current_uA = I * 1e6
        group = CHANNEL_GROUPS.get(ch, "TCO")
        color = group_colors.get(group, "tab:gray")
        # scatter points only
        line, = plt.plot(current_uA, voltage_kV, marker="o", linestyle="", label=f"{group} Ch {ch}", color=color)
        # fit line
        x_fit_uA = np.linspace(np.min(current_uA), np.max(current_uA), 50)
        x_fit_A = x_fit_uA * 1e-6
        y_fit_kV = (slope * x_fit_A + intercept) * 1e-3
        plt.plot(x_fit_uA, y_fit_kV, linestyle="-", color=color, alpha=0.7)
        lines.append(line)
        labels.append(f"{group} Ch {ch}")
        annotations.append(
            (f"Ch {ch}", f"V = ({slope * 1e-9:.2f} ± {slope_err * 1e-9:.2f}) GΩ · I + {intercept * 1e-3:.2f} kV")
        )
    plt.xlabel("Current [µA]")
    plt.ylabel("Voltage [kV]")
    plt.title("FC termination ramping IV curves")
    legend = plt.legend(lines, labels, ncol=2, fontsize=8, title="Channels")
    legend_text = "\n".join(f"{name}: {text}" for name, text in annotations)
    plt.gca().text(
        1.02,
        0.95,
        legend_text,
        transform=plt.gca().transAxes,
        fontsize=8,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fc_termination_ramping_iv.png"
    plt.tight_layout()
    plt.show()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved IV plot: {out_path}")


def plot_resistances(channels: np.ndarray, slopes: np.ndarray, slope_errs: np.ndarray, out_dir: Path):
    plt.figure(figsize=(8, 4))
    colors = ["tab:orange" if ch <= 6 else "tab:blue" for ch in channels]
    plt.errorbar(
        channels,
        slopes * 1e-9,
        yerr=slope_errs * 1e-9,
        fmt="o",
        color="tab:gray",
        ecolor="tab:gray",
        capsize=3,
    )
    for ch, slope, err, color in zip(channels, slopes, slope_errs, colors):
        y = (slope + err) * 1e-9
        plt.text(ch, y + 0.02, f"{slope * 1e-9:.2f}±{err * 1e-9:.2f}", fontsize=8, ha="center", va="bottom", color=color)
    plt.xlabel("Channel")
    plt.ylabel("Resistance [GOhm]")
    plt.title("FC termination ramping resistance per channel (TCO=orange, BEAM=blue)")
    plt.grid(True, alpha=0.3)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fc_termination_ramping_resistance.png"
    plt.tight_layout()
    plt.show()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved resistance plot: {out_path}")


def plot_supply_iv(
    voltages: np.ndarray,
    supply_current_uA: np.ndarray,
    slope_ohm: float | None,
    slope_err_ohm: float | None,
    intercept_v: float | None,
    out_dir: Path,
):
    """
    Plot the supply current vs voltage.
    """
    if supply_current_uA is None or supply_current_uA.size != voltages.size:
        return
    plt.figure(figsize=(6, 4))
    voltage_kV = voltages * 1e-3
    # scatter points only (no connecting line)
    plt.plot(supply_current_uA, voltage_kV, marker="o", linestyle="", color="black", label="Power Supply")
    if slope_ohm is not None and intercept_v is not None:
        # build fit line
        x_fit = np.linspace(np.min(supply_current_uA), np.max(supply_current_uA), 50)
        x_fit_A = x_fit * 1e-6
        y_fit_kV = (slope_ohm * x_fit_A + intercept_v) * 1e-3
        plt.plot(x_fit, y_fit_kV, color="tab:red", linestyle="-", label="Fit")
        if slope_err_ohm is not None:
            plt.text(
                0.98,
                0.02,
                f"R = {np.abs(slope_ohm) * 1e-9:.2f} ± {slope_err_ohm * 1e-9:.2f} GΩ",
                transform=plt.gca().transAxes,
                fontsize=8,
                ha="right",
                va="bottom",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )
    plt.xlabel("Supply current [µA]")
    plt.ylabel("Voltage [kV]")
    plt.title("Supply current IV")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fc_termination_ramping_supply_iv.png"
    plt.tight_layout()
    plt.show()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved supply IV plot: {out_path}")


def main():
    voltages, channels, channel_currents, supply_current_uA = load_ramping_data(DATA_FILE, INLINE_DATA)
    # Keep signed channel currents; display positive resistances via abs(slopes)
    currents = channel_currents
    supply_fit = None  # (slope, intercept, slope_err)
    if supply_current_uA is not None and supply_current_uA.size == voltages.size:
        supply_currents_A = supply_current_uA * 1e-6
        supply_fit = fit_iv_curve(supply_currents_A, voltages)

    slopes = []
    slope_errs = []
    intercepts = []
    for I in currents:
        slope, intercept, slope_err = fit_iv_curve(I, voltages)
        slopes.append(slope)
        slope_errs.append(slope_err)
        intercepts.append(intercept)

    slopes = np.array(slopes)
    slope_errs = np.array(slope_errs)
    intercepts = np.array(intercepts)
    slopes_abs = np.abs(slopes)

    print("Channel | Group | Resistance (GOhm) | Intercept (V)")
    for ch, slope_abs, intercept in zip(channels, slopes_abs, intercepts):
        group = CHANNEL_GROUPS.get(int(ch), "TCO")
        print(f"{ch:7d} | {group:5s} | {slope_abs * 1e-9:.4e} | {intercept:.3e}")
    if supply_fit is not None:
        supply_slope, supply_intercept, supply_slope_err = supply_fit
        print(f"{'Supply':>7s} | {'SUP':5s} | {np.abs(supply_slope) * 1e-9:.4e} | {supply_intercept:.3e}")

    finite_mask = np.isfinite(slopes_abs) & (slopes_abs != 0)
    if np.any(finite_mask):
        effective_resistance = 1.0 / np.sum(1.0 / slopes_abs[finite_mask])
        print(f"Effective parallel resistance: {effective_resistance * 1e-9} GOhm")
    else:
        print("Effective resistance: undefined (no finite resistances)")

    out_dir = DATA_FILE.parent / "plots"
    plot_iv(voltages, currents, channels, slopes_abs, slope_errs, intercepts, out_dir)
    plot_resistances(channels, slopes_abs, slope_errs, out_dir)
    if supply_fit is not None:
        plot_supply_iv(
            voltages,
            supply_current_uA,
            slope_ohm=np.abs(supply_fit[0]),
            slope_err_ohm=supply_fit[2],
            intercept_v=supply_fit[1],
            out_dir=out_dir,
        )
    else:
        plot_supply_iv(voltages, supply_current_uA, slope_ohm=None, slope_err_ohm=None, intercept_v=None, out_dir=out_dir)


if __name__ == "__main__":
    main()
