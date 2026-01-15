"""TP CrystalBall plotting from JSON (no refits).

Reads the JSON produced by TP_fit_CB_to_json.py and:
- Produces the time-series plot (fit_means_vs_time.png)
- Writes the derived CSV (fit_means_data_low_cb.csv)

Usage examples:
  python TP_plot_CB_from_json.py
  python TP_plot_CB_from_json.py --json plots/tp_cb_fit_results.json
  python TP_plot_CB_from_json.py --start "2025-12-01 00:00" --end "2025-12-03 23:59" --no-show
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


PLOTS_DIR_DEFAULT = "plots"
JSON_IN_DEFAULT = os.path.join(PLOTS_DIR_DEFAULT, "tp_cb_fit_results.json")
_NP02DATA_DIR_DEFAULT = os.environ.get("NP02DATA_DIR", "../np02data")
TEMP_CSV_PATH_DEFAULT = os.environ.get("NP02_TEMP_CSV", os.path.join(_NP02DATA_DIR_DEFAULT, "Temp.csv"))

PLOT_START = datetime(2025, 11, 21, 16, 00)
MV_SCALE = 1000.0

DATA_LABELS = {
    "F1_high": "TP_Response_Inner_long_PM",
    "F2_high": "TP_Response_Outer_long_PM",
    "F1_low": "Inner_low_CB",
    "F2_low": "Outer_low_CB",
    "F3": "Test_pulse",
}


def _parse_datetime(dt_str: str) -> Optional[datetime]:
    if not dt_str:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(dt_str.strip(), fmt)
        except ValueError:
            continue
    return None


def _to_millivolts_scalar(value_v: Optional[float]) -> Optional[float]:
    if value_v is None:
        return None
    if not np.isfinite(value_v):
        return None
    return float(value_v) * MV_SCALE


def _load_temperature_series(csv_path: str, start_time: Optional[datetime]) -> Tuple[List[datetime], List[float]]:
    if not csv_path or not os.path.exists(csv_path):
        return [], []
    try:
        df = pd.read_csv(csv_path, comment="#", header=None)
    except Exception:
        return [], []
    if df.shape[1] < 2:
        return [], []

    ts_raw = df.iloc[:, 0]
    temp_raw = df.iloc[:, 1]
    ts_parsed = pd.to_datetime(ts_raw.astype(str).str.strip(), format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
    temp_vals = pd.to_numeric(temp_raw, errors="coerce")
    df_clean = pd.DataFrame({"timestamp": ts_parsed, "temperature": temp_vals}).dropna(subset=["timestamp", "temperature"])

    times: List[datetime] = []
    temps: List[float] = []
    for _, row in df_clean.iterrows():
        ts = row["timestamp"].to_pydatetime()
        if start_time and ts < start_time:
            continue
        times.append(ts)
        temps.append(float(row["temperature"]))
    return times, temps


def _overlay_temperature(ax, temp_times, temp_vals, label_tracker: Dict[str, bool]):
    if not temp_times:
        return
    twin = ax.twinx()
    twin.plot(
        temp_times,
        [-v for v in temp_vals],
        color="tab:red",
        alpha=0.6,
        linewidth=1.0,
        label="Ambient temperature (inverted)",
    )
    if not label_tracker.get("temp_label"):
        twin.set_ylabel("Ambient temp [°C] (inverted)", color="tab:red")
        label_tracker["temp_label"] = True
    twin.tick_params(axis="y", labelcolor="tab:red")
    twin.grid(False)


def plot_fit_means(
    plots_dir: str,
    mean_records: Dict[str, List[Tuple[datetime, float, Optional[float]]]],
    start: Optional[datetime],
    end: Optional[datetime],
    temp_csv_path: str,
    show: bool,
) -> None:
    keys_present = any(mean_records.get(k) for k in DATA_LABELS.keys())
    if not keys_present:
        print("No data to plot.")
        return

    filtered_records: Dict[str, List[Tuple[datetime, float, Optional[float]]]] = {}
    for key, samples in mean_records.items():
        if samples:
            filtered_records[key] = [
                (t, v, e)
                for t, v, e in samples
                if ((start is None or t >= start) and (end is None or t <= end))
            ]
        else:
            filtered_records[key] = []

    fig, axes = plt.subplots(5, 1, figsize=(10, 13), sharex=True)
    ax_f1_high, ax_f2_high, ax_f1_low, ax_f2_low, ax_f3 = axes

    start_time_candidates: List[datetime] = []
    for samples in filtered_records.values():
        if samples:
            start_time_candidates.append(min(samples, key=lambda x: x[0])[0])
    if start:
        start_time_candidates.append(start)
    start_time = min(start_time_candidates) if start_time_candidates else None

    temp_times, temp_vals = _load_temperature_series(temp_csv_path, start_time)
    temp_label_tracker: Dict[str, bool] = {}

    def _plot_series(ax, samples, color, ylim=None):
        if not samples:
            ax.set_visible(False)
            return
        samples = sorted(samples, key=lambda x: x[0])
        times = [t for t, _, _ in samples]
        values = [v * MV_SCALE for _, v, _ in samples]
        errs = np.array([e * MV_SCALE if e is not None else np.nan for _, _, e in samples], dtype=float)
        ax.plot(times, values, marker="o", markersize=3.5, linestyle="none", color=color)
        finite_mask = np.isfinite(errs)
        if finite_mask.any():
            vals_arr = np.array(values, dtype=float)
            band = 2.0 * errs
            ax.fill_between(times, vals_arr - band, vals_arr + band, color=color, alpha=0.12, linewidth=0)
        ax.set_ylabel("Mean [mV]")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        _overlay_temperature(ax, temp_times, temp_vals, temp_label_tracker)

    _plot_series(ax_f1_high, filtered_records.get("F1_high", []), "tab:blue")
    _plot_series(ax_f2_high, filtered_records.get("F2_high", []), "tab:orange")
    _plot_series(ax_f1_low, filtered_records.get("F1_low", []), "tab:blue")
    _plot_series(ax_f2_low, filtered_records.get("F2_low", []), "tab:orange")
    _plot_series(ax_f3, filtered_records.get("F3", []), "tab:purple", ylim=(181.2, 183))

    axes[-1].set_xlabel("Measurement time")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, "fit_means_vs_time.png")
    if show:
        plt.show()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved mean-vs-time plot: {out_path}")


def write_plot_data(plots_dir: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return

    def _row_ts(row: Dict[str, object]) -> datetime:
        ts = row.get("timestamp")
        return ts if isinstance(ts, datetime) else datetime.min

    rows = sorted(rows, key=_row_ts)

    os.makedirs(plots_dir, exist_ok=True)
    data_path = os.path.join(plots_dir, "fit_means_data_low_cb.csv")
    with open(data_path, "w", newline="") as f:
        f.write("# Derived with low-region CB fits (no scaling)\n")
        f.write(f"# Generated at {datetime.now(timezone.utc).isoformat()}\n")
        writer = csv.writer(f)
        header = ["timestamp_iso"]
        for k in DATA_LABELS.keys():
            header.append(DATA_LABELS[k] + "_mV")
            header.append(DATA_LABELS[k] + "_err_mV")
        writer.writerow(header)

        for row in rows:
            ts = row.get("timestamp")
            if not isinstance(ts, datetime):
                continue
            values: List[str] = []
            for key in DATA_LABELS.keys():
                v = row.get(key)
                e = row.get(f"{key}_err")
                mv_val = _to_millivolts_scalar(float(v) if isinstance(v, (int, float, np.floating)) else None)
                mv_err = _to_millivolts_scalar(float(e) if isinstance(e, (int, float, np.floating)) else None)
                values.append(f"{mv_val:.3f}" if mv_val is not None else "")
                values.append(f"{mv_err:.3f}" if mv_err is not None else "")
            writer.writerow([ts.isoformat()] + values)

    print(f"Wrote plot data: {data_path}")


def _load_json(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TP CrystalBall results from JSON (no fitting).")
    parser.add_argument("--plots-dir", default=PLOTS_DIR_DEFAULT)
    parser.add_argument("--json", default=JSON_IN_DEFAULT)
    parser.add_argument("--start", default=None, help="YYYY-MM-DD HH:MM[:SS]")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD HH:MM[:SS]")
    parser.add_argument("--temp-csv", default=TEMP_CSV_PATH_DEFAULT)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    start = _parse_datetime(args.start) if args.start else PLOT_START
    end = _parse_datetime(args.end) if args.end else None

    # Ensure JSON default follows plots-dir if user changed plots-dir.
    json_in = args.json
    if json_in == JSON_IN_DEFAULT:
        json_in = os.path.join(args.plots_dir, "tp_cb_fit_results.json")

    data = _load_json(json_in)
    measurements = data.get("measurements")
    if not isinstance(measurements, dict):
        print(f"No 'measurements' found in: {json_in}")
        return

    # If the JSON contains only older data than PLOT_START and user did not
    # explicitly provide --start, don't silently plot nothing.
    if args.start is None and start == PLOT_START:
        ts_list: List[datetime] = []
        for ts_iso in measurements.keys():
            if not isinstance(ts_iso, str):
                continue
            try:
                ts_list.append(datetime.fromisoformat(ts_iso))
            except ValueError:
                continue
        if ts_list and start is not None and max(ts_list) < start:
            print(
                f"Note: JSON max timestamp ({max(ts_list)}) is before PLOT_START ({PLOT_START}); "
                "plotting full available range."
            )
            start = None

    mean_records: Dict[str, List[Tuple[datetime, float, Optional[float]]]] = {}
    rows: List[Dict[str, object]] = []

    for ts_iso, result in measurements.items():
        if not isinstance(ts_iso, str) or not isinstance(result, dict):
            continue
        try:
            ts = datetime.fromisoformat(ts_iso)
        except ValueError:
            continue
        if start and ts < start:
            continue
        if end and ts > end:
            continue

        row: Dict[str, object] = {"timestamp": ts}
        for key in DATA_LABELS.keys():
            item = result.get(key)
            if not isinstance(item, dict):
                row[key] = None
                row[f"{key}_err"] = None
                continue
            mean_v = item.get("mean_v")
            err_v = item.get("err_v")
            mean_f = float(mean_v) if isinstance(mean_v, (int, float)) and np.isfinite(mean_v) else None
            err_f = float(err_v) if isinstance(err_v, (int, float)) and np.isfinite(err_v) else None
            if mean_f is not None:
                mean_records.setdefault(key, []).append((ts, mean_f, err_f))
            row[key] = mean_f
            row[f"{key}_err"] = err_f
        rows.append(row)

    # Plot + CSV
    plot_fit_means(
        plots_dir=args.plots_dir,
        mean_records=mean_records,
        start=start,
        end=end,
        temp_csv_path=args.temp_csv,
        show=not args.no_show,
    )
    write_plot_data(args.plots_dir, rows)


if __name__ == "__main__":
    main()
