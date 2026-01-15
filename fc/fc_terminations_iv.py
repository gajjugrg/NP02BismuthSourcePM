"""
Build IV curves for FC terminations using incremental current data.

The input file contains per-channel current increases at each voltage step.
Columns diff_k represent the incremental increase relative to the previous step.
To obtain the total current at each voltage, take the cumulative sum of the increments.

Currents are provided in 1e-7 A units.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# File and constants ---------------------------------------------------------
NP02DATA_DIR = os.environ.get("NP02DATA_DIR", "../np02data")
DATA_FILE = Path(os.environ.get("FC_TERMINATIONS_FILE", os.path.join(NP02DATA_DIR, "fc_terminations.txt")))
VOLTAGES = np.array([29608, 59682, 89760, 119835, 153918], dtype=float)  # volts
CURRENT_STEP_SCALE = 1e-7  # each diff column is in 1e-7 A
CHANNEL_GROUPS = {ch: ("TCO" if ch <= 6 else "BEAM") for ch in range(1, 13)}


def load_increment_table(path: Path) -> pd.DataFrame:
    """
    Load fc_terminations table and return a DataFrame with columns:
      channel, diff1, diff2, ...
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    n_steps = VOLTAGES.size
    column_names = ["channel"] + [f"diff_{i+1}" for i in range(n_steps)]
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=column_names,
        engine="python",
    )
    df = df.dropna(how="all")
    df["channel"] = df["channel"].astype(int)
    for name in column_names[1:]:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    df = df.dropna()
    return df


def compute_cumulative_currents(diff_values: np.ndarray) -> np.ndarray:
    """
    Convert incremental currents (diff rows) into cumulative currents for each voltage.
    """
    # Convert to amps and take cumulative sum along axis=1
    return np.cumsum(diff_values * CURRENT_STEP_SCALE, axis=1)


def fit_iv_curve(voltages: np.ndarray, currents: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit voltage as a function of current: V = slope * I + intercept.
    Returns slope (Ohms), intercept (V), and slope uncertainty.
    """
    (slope, intercept), cov = np.polyfit(currents, voltages, 1, cov=True)
    slope_err = float(np.sqrt(cov[0, 0]))
    return slope, intercept, slope_err


def plot_all_channels(
    voltages: np.ndarray,
    currents: np.ndarray,
    channels: np.ndarray,
    slopes: np.ndarray,
    slope_errs: np.ndarray,
    intercepts: np.ndarray,
    output_dir: Path,
):
    """
    Plot IV curves for all channels on a single figure.
    """
    plt.figure(figsize=(10, 6))
    lines = []
    labels = []
    annotations = []
    group_colors = {"TCO": "tab:orange", "BEAM": "tab:blue"}
    for chan, I, slope, slope_err, intercept in zip(channels, currents, slopes, slope_errs, intercepts):
        current_uA = I * 1e6
        voltage_kV = voltages * 1e-3
        group = CHANNEL_GROUPS.get(int(chan), "TCO")
        color = group_colors.get(group, None)
        # scatter points only
        line, = plt.plot(current_uA, voltage_kV, marker="o", linestyle="", label=f"{group} Ch {chan}", color=color)
        # fitted line
        x_fit_uA = np.linspace(np.min(current_uA), np.max(current_uA), 50)
        x_fit_A = x_fit_uA * 1e-6
        y_fit_kV = (slope * x_fit_A + intercept) * 1e-3
        plt.plot(x_fit_uA, y_fit_kV, linestyle="-", color=line.get_color(), alpha=0.7)
        annotations.append(
            (
                f"{group} Ch {chan}",
                f"V = ({slope * 1e-9:.2f} ± {slope_err * 1e-9:.2f}) GΩ · I + {intercept * 1e-3:.2f} kV",
            )
        )
        lines.append(line)
        labels.append(f"{group} Ch {chan}")
    plt.xlabel("Current [µA]")
    plt.ylabel("Voltage [kV]")
    plt.title("FC termination IV curves")
    ax = plt.gca()
    legend = plt.legend(lines, labels, ncol=2, fontsize=8, title="Channels")
    legend_text = "\n".join([f"{name}: {text}" for name, text in annotations])
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
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "fc_terminations_iv.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()
    plt.close()
    print(f"Saved IV plot: {out_path}")


def plot_resistances(channels: np.ndarray, resistances: np.ndarray, res_errors: np.ndarray, output_dir: Path):
    """
    Plot resistance vs channel number.
    """
    plt.figure(figsize=(8, 4))
    colors = ["tab:orange" if ch <= 6 else "tab:blue" for ch in channels]
    plt.errorbar(
        channels,
        resistances * 1e-9,
        yerr=res_errors * 1e-9,
        fmt="o",
        color="tab:gray",
        ecolor="tab:gray",
        capsize=3,
    )
    for ch, res, err, color in zip(channels, resistances, res_errors, colors):
        y = (res + err) * 1e-9
        plt.text(ch, y + 0.02, f"{res * 1e-9:.2f}±{err * 1e-9:.2f}", fontsize=8, ha="center", va="bottom", color=color)
    plt.xlabel("Channel")
    plt.ylabel("Resistance [GΩ]")
    plt.title("FC termination resistance per channel (TCO=orange, BEAM=blue)")
    plt.grid(True, alpha=0.3)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "fc_terminations_resistance.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()
    plt.close()
    print(f"Saved resistance plot: {out_path}")


def main():
    df = load_increment_table(DATA_FILE)
    channels = df["channel"].to_numpy(dtype=int)
    diff_cols = [f"diff_{i+1}" for i in range(VOLTAGES.size)]
    increments = df[diff_cols].to_numpy(dtype=float)

    cumulative_currents = compute_cumulative_currents(increments)

    results: Dict[int, Dict[str, float]] = {}
    resistances = []
    resistance_errs = []
    slopes = []
    slope_errs = []
    intercepts = []
    for chan, currents in zip(channels, cumulative_currents):
        slope, intercept, slope_err = fit_iv_curve(VOLTAGES, currents)
        resistance = slope
        resistance_err = slope_err
        results[chan] = {
            "slope_A_per_V": slope,
            "slope_err_A_per_V": slope_err,
            "intercept_A": intercept,
            "resistance_ohm": resistance,
            "resistance_err_ohm": resistance_err,
        }
        resistances.append(resistance)
        resistance_errs.append(resistance_err)
        slopes.append(slope)
        slope_errs.append(slope_err)
        intercepts.append(intercept)

    # Print summary table
    print("Channel | Resistance (GOhm) ± err | Intercept (V)")
    for chan in sorted(results):
        res = results[chan]
        print(
            f"{chan:7d} | {res['resistance_ohm'] * 1e-9:.4e} ± {res['resistance_err_ohm'] * 1e-9:.2e} | {res['intercept_A']:.3e}"
        )

    finite_mask = np.isfinite(resistances) & (np.array(resistances) != 0)
    if np.any(finite_mask):
        R = np.array(resistances)[finite_mask]
        dR = np.array(resistance_errs)[finite_mask]
        inv_R = 1.0 / R
        effective_resistance = 1.0 / np.sum(inv_R)
        # propagate error: d(Reff) = Reff^2 * sqrt(sum((dR_i / R_i^2)^2))
        eff_err = effective_resistance**2 * np.sqrt(np.sum((dR / (R**2)) ** 2))
        print(f"Effective parallel resistance: {effective_resistance * 1e-9:.4e} ± {eff_err * 1e-9:.2e} GOhm")
    else:
        print("Effective resistance: undefined (no finite resistances)")

    # Plot IV curves
    output_dir = DATA_FILE.parent / "plots"
    plot_all_channels(
        VOLTAGES,
        cumulative_currents,
        channels,
        np.array(slopes),
        np.array(slope_errs),
        np.array(intercepts),
        output_dir,
    )
    plot_resistances(channels, np.array(resistances), np.array(resistance_errs), output_dir)


if __name__ == "__main__":
    main()
