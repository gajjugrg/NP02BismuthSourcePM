"""Open a waveform file and plot the waveform.

- Supports `.wfm` via `tm_data_types.read_file` if available.
- Supports CSV files with common time/voltage columns.
- If no file is provided, offers a GUI file picker (when available),
  otherwise falls back to the first `.wfm`/`.csv` found.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Tuple
import sys

import numpy as np
import matplotlib.pyplot as plt


def _import_tm_reader():
    try:
        from tm_data_types import read_file  # type: ignore
        return read_file
    except Exception:
        return None


def _interactive_pick_file() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        fname = filedialog.askopenfilename(
            title="Select waveform file",
            filetypes=[
                ("Waveform/CSV", "*.wfm *.csv"),
                ("Tektronix WFM", "*.wfm"),
                ("CSV", "*.csv"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return Path(fname) if fname else None
    except Exception:
        return None


def _coerce_to_xy(obj: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a waveform-like object to (t, y) arrays.

    Tries pandas.DataFrame, dicts, tuples, numpy arrays, and common
    tm_data_types objects (e.g., Normalized or AnalogWaveform).
    """
    # pandas DataFrame
    try:
        import pandas as pd  # type: ignore
        if isinstance(obj, pd.DataFrame):
            df = obj
            # Prefer common time/voltage column names
            t_candidates = [c for c in df.columns if c.lower() in {"time", "t", "x"}]
            y_candidates = [
                c for c in df.columns if c.lower() in {"voltage", "v", "y", "ch1", "ch", "value", "data"}
            ]
            if not t_candidates:
                t_candidates = list(df.columns[:1])
            if not y_candidates:
                y_candidates = list(df.columns[1:2]) or list(df.columns[:1])
            t = np.asarray(df[t_candidates[0]].to_numpy())
            y = np.asarray(df[y_candidates[0]].to_numpy())
            return t, y
    except Exception:
        pass

    # dict-like
    if isinstance(obj, dict):
        keys = {k.lower(): k for k in obj.keys()}
        def _pick(ls):
            for k in ls:
                if k in keys:
                    return keys[k]
            return None
        t_key = _pick(["time", "t", "x"])  # type: ignore
        y_key = _pick(["voltage", "v", "y", "ch1", "data", "value"])  # type: ignore
        if t_key is not None and y_key is not None:
            return np.asarray(obj[t_key]), np.asarray(obj[y_key])
        if "x" in keys and "y" in keys:
            return np.asarray(obj[keys["x"]]), np.asarray(obj[keys["y"]])

    # tuple/list
    if isinstance(obj, (list, tuple)):
        if len(obj) >= 2:
            return np.asarray(obj[0]), np.asarray(obj[1])
        if len(obj) == 1:
            y = np.asarray(obj[0])
            t = np.arange(y.size)
            return t, y

    # tm_data_types: Normalized-like (has spacing/offset and array interface)
    spacing = getattr(obj, "spacing", None)
    if spacing is not None:
        try:
            y = np.asarray(obj)
            offset = float(getattr(obj, "offset", 0.0))
            t = offset + float(spacing) * np.arange(y.size)
            return t, y
        except Exception:
            pass

    # tm_data_types: AnalogWaveform-like (has y_axis_values and maybe meta_info)
    y_vals = getattr(obj, "y_axis_values", None)
    if y_vals is not None:
        try:
            y = np.asarray(y_vals)
        except Exception:
            try:
                y = np.asarray(getattr(y_vals, "data", y_vals))
            except Exception:
                y = np.array(y_vals)
        # Attempt to derive time from meta info
        mi = getattr(obj, "meta_info", None)
        dt = None
        for name in ("x_axis_spacing", "x_increment", "dt", "x_spacing"):
            dt = getattr(mi, name, None) if mi is not None else None
            if dt is not None:
                break
        t0 = 0.0
        for name in ("x_axis_offset", "x_origin", "t0"):
            val = getattr(mi, name, None) if mi is not None else None
            if val is not None:
                t0 = float(val)
                break
        if dt is not None:
            t = t0 + float(dt) * np.arange(len(y))
        else:
            t = np.arange(len(y))
        return t, y

    # numpy array
    arr = np.asarray(obj)
    if arr.ndim == 1:
        t = np.arange(arr.size)
        return t, arr
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 0], arr[:, 1]

    raise ValueError("Unrecognized waveform format; cannot infer (t, y)")


def _load_csv(path: Path):
    import pandas as pd  # type: ignore
    return pd.read_csv(path)


def load_waveform(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ext = path.suffix.lower()
    tm_reader = _import_tm_reader()
    if ext == ".wfm":
        if tm_reader is None:
            raise RuntimeError("tm_data_types.read_file not available; cannot read .wfm")
        obj = tm_reader(str(path))
        return _coerce_to_xy(obj)
    if ext == ".csv":
        df = _load_csv(path)
        return _coerce_to_xy(df)
    # Unknown extension: try CSV load as a best-effort
    df = _load_csv(path)
    return _coerce_to_xy(df)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Open a waveform file and plot it")
    parser.add_argument("file", type=str, nargs="?", help="Path to .wfm or .csv file")
    parser.add_argument("--save", type=str, default=None, help="Optional PNG output path")
    args = parser.parse_args(argv)

    # Resolve path or interactively pick one
    if args.file:
        path = Path(args.file).expanduser()
    else:
        path = _interactive_pick_file()
        if path is None:
            # Fallback to first match in CWD or example folder
            candidates = list(Path.cwd().glob("*.wfm")) + list(Path.cwd().glob("*.csv"))
            candidates += list((Path.cwd() / "example_waveforms").glob("*.wfm"))
            candidates += list((Path.cwd() / "example_waveforms").glob("*.csv"))
            path = candidates[0] if candidates else None

    if path is None:
        print("No file selected or found. Provide a path or place a .wfm/.csv here.")
        return 2

    if not path.exists():
        print(f"File not found: {path}")
        return 2

    try:
        t, y = load_waveform(path)
    except Exception as e:
        print(f"Failed to load waveform: {e}")
        return 1

    if t.size == 0 or y.size == 0:
        print("Loaded waveform is empty; nothing to plot.")
        return 3

    plt.figure(figsize=(10, 4))
    plt.plot(t, y, lw=1.0)
    plt.xlabel("Time (s)" if np.any(np.diff(t)) else "Sample Index")
    plt.ylabel("Amplitude")
    plt.title(path.name)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if args.save:
        out = Path(args.save)
        plt.savefig(out, dpi=200)
        print(f"Saved plot to: {out}")

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
