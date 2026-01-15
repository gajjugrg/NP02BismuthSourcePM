import argparse
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _extract_entry_fields(key: str, entry: Dict[str, Any], prefer_temp: bool) -> Optional[Tuple[datetime, float, float, str]]:
    """Extract (timestamp, m3_short, m3_long, directory_id) from a cache entry.
    - prefer_temp: if True, prefer temperature-corrected fields when present.
    Returns None if required fields unavailable.
    """
    if not isinstance(entry, dict):
        return None
    meta = entry.get("meta") if isinstance(entry, dict) else None

    # Timestamp
    ts = None
    if isinstance(meta, dict):
        ts = meta.get("timestamp")
    if ts is None:
        ts = entry.get("timestamp")
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(str(ts))
    except Exception:
        return None

    # Directory id for dedupe
    dir_id = None
    if isinstance(meta, dict):
        dir_id = meta.get("directory")
    if not dir_id and isinstance(key, str):
        dir_id = key.split("|")[0]
    if not dir_id:
        dir_id = str(key)

    # m3 values (prefer meta or temp-corrected if requested)
    m3s = None
    m3l = None
    if isinstance(meta, dict):
        if prefer_temp:
            m3s = meta.get("m3_short_temp_corr", meta.get("m3_short"))
            m3l = meta.get("m3_long_temp_corr", meta.get("m3_long"))
        else:
            m3s = meta.get("m3_short")
            m3l = meta.get("m3_long")
    if (m3s is None or m3l is None) and isinstance(entry, dict):
        # Fallback to short/long groups from fit runs
        short = entry.get("short")
        long  = entry.get("long")
        if isinstance(short, dict) and m3s is None:
            m3s = short.get("m3")
        if isinstance(long, dict) and m3l is None:
            m3l = long.get("m3")

    if m3s is None or m3l is None:
        return None
    try:
        return t, float(m3s), float(m3l), dir_id
    except Exception:
        return None


def load_cache(cache_file: str) -> Dict[str, Any]:
    with open(cache_file, "rb") as f:
        d = pickle.load(f)
    if not isinstance(d, dict):
        raise RuntimeError(f"Cache object is not a dict: {type(d)}")
    return d


def collect_series(cache: Dict[str, Any], prefer_temp: bool, m3_short_min: float, m3_short_max: float):
    rows: List[Tuple[datetime, float, float, str]] = []
    for k, v in cache.items():
        rec = _extract_entry_fields(k, v, prefer_temp)
        if rec is None:
            continue
        t, m3s, m3l, did = rec
        if not (m3_short_min < m3s < m3_short_max):
            continue
        rows.append(rec)
    # Deduplicate by directory_id, keep earliest timestamp
    rows.sort(key=lambda r: r[0])
    seen = set()
    uniq: List[Tuple[datetime, float, float, str]] = []
    for r in rows:
        if r[3] in seen:
            continue
        seen.add(r[3])
        uniq.append(r)
    if not uniq:
        return [], [], []
    ts = [r[0] for r in uniq]
    svals = [r[1] for r in uniq]
    lvals = [r[2] for r in uniq]
    return ts, svals, lvals


def plot_m3_from_cache(
    cache_file: str,
    out_png: str,
    prefer_temp: bool,
    m3_short_min: float,
    m3_short_max: float,
):
    cache = load_cache(cache_file)
    ts, svals, lvals = collect_series(cache, prefer_temp, m3_short_min, m3_short_max)
    if not ts:
        print("No valid m3 points found to plot (after cuts).")
        return 2

    plt.figure(figsize=(10, 5))
    plt.plot(ts, svals, marker='o', linestyle='none', label='4 cm drift (short)', color='tab:blue')
    plt.plot(ts, lvals, marker='o', linestyle='none', label='20 cm drift (long)', color='tab:green')
    plt.gcf().autofmt_xdate()
    title = 'm3 vs time'
    if prefer_temp:
        title += ' (temp-corrected)'
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Pulse height [V]')
    plt.legend(loc='best')
    plt.tight_layout()

    out_dir = os.path.dirname(out_png) or '.'
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot: {out_png}")
    plt.close()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Plot short/long m3 series from fit_cache.pkl")
    p.add_argument('--cache', default='fit_cache.pkl', help='Path to fit_cache.pkl')
    p.add_argument('--out', default='plots/m3_short_long_from_cache.png', help='Output PNG path')
    p.add_argument('--use-temp', action='store_true', help='Prefer temperature-corrected m3 if present')
    p.add_argument('--min-m3-short', type=float, default=0.550, help='Lower cut on m3_short')
    p.add_argument('--max-m3-short', type=float, default=1.38, help='Upper cut on m3_short')
    args = p.parse_args(argv)

    try:
        return plot_m3_from_cache(
            cache_file=args.cache,
            out_png=args.out,
            prefer_temp=args.use_temp,
            m3_short_min=args.min_m3_short,
            m3_short_max=args.max_m3_short,
        )
    except FileNotFoundError:
        print(f"Cache file not found: {args.cache}")
        return 2
    except Exception as e:
        print(f"Failed to plot from cache: {e}")
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

