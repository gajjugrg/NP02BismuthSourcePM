import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from datetime import datetime
import pickle
import json

SAVE_FILES = True # set to False to skip writing plots to disk

# Debug flag: when True, emit [INFO] and [SKIP] messages
debug = False

# Force rerun: when True, recompute and overwrite cache even if present
FORCE_RERUN = True

# Directory to save plots
PLOTS_DIR = 'plots_october'
if SAVE_FILES:
    os.makedirs(PLOTS_DIR, exist_ok=True)

# Date window filter (inclusive)
# Restrict processing to October 2025 onward measurements.
START_DATE = datetime(2025, 10, 1, 0, 0)
END_DATE = None
MIN_BIN_CENTER = 0  # Cut: drop all bins with BinCenter < 0.35 V for data and background

# File to cache fit parameters so we don't re-fit on reruns
FIT_CACHE_FILE = 'fit_cache_october.pkl'
FIT_CACHE_INDEX_FILE = 'fit_cache_october_index.json'
ERROR_CACHE_INDEX_FILE = 'fit_cache_october_errors.json'

# Temperature overlay configuration
NP02DATA_DIR = os.environ.get('NP02DATA_DIR', '../np02data')
TEMP_OVERLAY_CSV = os.environ.get('NP02_TEMP_OVERLAY_CSV', os.path.join(NP02DATA_DIR, 'Oct15_Nov5.csv'))
TEMP_OVERLAY_PLOT = 'temp_m3_long_oct15_nov5_overlay.png'

if FORCE_RERUN:
    for cache_path in (FIT_CACHE_FILE, FIT_CACHE_INDEX_FILE, ERROR_CACHE_INDEX_FILE):
        try:
            if os.path.exists(cache_path):
                os.remove(cache_path)
                if debug:
                    print(f"[INFO] Removed cache file due to FORCE_RERUN: {cache_path}")
        except OSError as e:
            print(f"[WARN] Failed to remove cache file {cache_path}: {e}")

# Lazy-loaded cache to avoid startup penalty when only skipping
fit_cache = {}
_fit_cache_loaded = False

def load_fit_cache():
    global fit_cache, _fit_cache_loaded
    if _fit_cache_loaded:
        return
    if os.path.exists(FIT_CACHE_FILE):
        try:
            with open(FIT_CACHE_FILE, 'rb') as f:
                fit_cache = pickle.load(f)
            if not isinstance(fit_cache, dict):
                print(f"[WARN] {FIT_CACHE_FILE} is not a dict. Resetting cache.")
                fit_cache = {}
        except Exception as e:
            print(f"[WARN] Failed to load {FIT_CACHE_FILE}: {e}. Starting with empty cache.")
            fit_cache = {}
    _fit_cache_loaded = True

def save_fit_cache():
    # Ensure cache loaded before saving updates
    if not _fit_cache_loaded:
        # nothing to save yet
        return
    try:
        with open(FIT_CACHE_FILE, 'wb') as f:
            pickle.dump(fit_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"[WARN] Failed to save {FIT_CACHE_FILE}: {e}")

# Lightweight index of processed directories for fast skip without loading pickle
processed_dirs = set()
if os.path.exists(FIT_CACHE_INDEX_FILE):
    try:
        with open(FIT_CACHE_INDEX_FILE, 'r') as f:
            processed_dirs = set(json.load(f))
    except Exception as e:
        print(f"[WARN] Failed to load {FIT_CACHE_INDEX_FILE}: {e}. Rebuilding index later.")
        processed_dirs = set()

def save_cache_index():
    try:
        with open(FIT_CACHE_INDEX_FILE, 'w') as f:
            json.dump(sorted(processed_dirs), f)
    except Exception as e:
        print(f"[WARN] Failed to save {FIT_CACHE_INDEX_FILE}: {e}")

# Lightweight index of directories that failed processing (with reason)
# so they can be skipped on subsequent runs unless FORCE_RERUN is True.
error_index = {}
if os.path.exists(ERROR_CACHE_INDEX_FILE):
    try:
        with open(ERROR_CACHE_INDEX_FILE, 'r') as f:
            error_index = json.load(f)
            if not isinstance(error_index, dict):
                error_index = {}
    except Exception as e:
        print(f"[WARN] Failed to load {ERROR_CACHE_INDEX_FILE}: {e}. Starting empty errors index.")
        error_index = {}

def save_error_index():
    try:
        with open(ERROR_CACHE_INDEX_FILE, 'w') as f:
            json.dump(error_index, f)
    except Exception as e:
        print(f"[WARN] Failed to save {ERROR_CACHE_INDEX_FILE}: {e}")

# Map textual month to a numeric string
MONTH_MAP = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
}

def gaussian(x, m1, m2, m3):
    """
    Custom Gaussian function with three components:
      G(x) = m1 * [ exp( -0.5 * ( (x - m3)^2 / (m2^2 ) ) )
                    + 0.262 * exp( -0.5 * ( (x - m3*1.0747)^2 / (m2^2 ) ) )
                    + 0.077 * exp( -0.5 * ( (x - m3*1.0861)^2 / (m2^2 ) ) ) ]
    """
    return m1 * (
        np.exp(-0.5 * ((x - m3) ** 2) / (m2 ** 2))
        + 0.262 * np.exp(-0.5 * ((x - m3 * 1.0747) ** 2) / (m2 ** 2))
        + 0.077 * np.exp(-0.5 * ((x - m3 * 1.0861) ** 2) / (m2 ** 2))
    )


def get_delta_t_and_scaling_factor(measurement_time):
    """
    Return the DELTA_T value and scaling factor for the October 2025 onward runs.
    All data for this period was taken at 520 V/cm, so the values are fixed.
    """
    delta_t = 16.0 / 0.1635
    scaling_factor = 0.92
    return delta_t, scaling_factor

def process_monitor(data_file, bg_file, x_scale=1.0, y_scale=1.0, cached_params=None):
    """
    Process a single monitor (e.g., short or long PM).
    Uses fixed x_scale and y_scale (no fitting).
    Uses cached Gaussian fit parameters if provided.
    Returns a dictionary with results or None if something fails.
    """
    try:
        # Read only required columns to reduce memory
        data = pd.read_csv(data_file, usecols=['BinCenter', 'Population'])
        bg_data = pd.read_csv(bg_file,   usecols=['BinCenter', 'Population'])
    except Exception as e:
        msg = f"Error reading files: {e} (data_file: {data_file}, bg_file: {bg_file})"
        if debug:
            print(msg)
        return None, msg

    if 'BinCenter' not in data.columns or 'Population' not in data.columns:
        msg = f"Missing columns in data file: {data_file}"
        if debug:
            print(msg)
        return None, msg
    if 'BinCenter' not in bg_data.columns or 'Population' not in bg_data.columns:
        msg = f"Missing columns in background file: {bg_file}"
        if debug:
            print(msg)
        return None, msg

    # Main data (force numeric; coerce non-numeric to NaN)
    voltage = pd.to_numeric(data['BinCenter'], errors='coerce').to_numpy(dtype=np.float32)
    count   = pd.to_numeric(data['Population'], errors='coerce').to_numpy(dtype=np.float32)

    # Background data (force numeric)
    voltage_bg = pd.to_numeric(bg_data['BinCenter'], errors='coerce').to_numpy(dtype=np.float32)
    bg_count   = pd.to_numeric(bg_data['Population'], errors='coerce').to_numpy(dtype=np.float32)

    # Validate arrays
    if np.isnan(voltage).any() or np.isnan(count).any():
        msg = f"[SKIP] Non-numeric values in data file: {data_file} — skipping."
        if debug:
            print(msg)
        return None, msg
    if np.isnan(voltage_bg).any() or np.isnan(bg_count).any():
        msg = f"[SKIP] Non-numeric values in background file: {bg_file} — skipping."
        if debug:
            print(msg)
        return None, msg

    # Empty-array guards
    if voltage.size == 0 or count.size == 0:
        msg = f"[SKIP] Empty data arrays in data file: {data_file} — skipping."
        if debug:
            print(msg)
        return None, msg
    if voltage_bg.size == 0 or bg_count.size == 0:
        msg = f"[SKIP] Empty arrays in background file: {bg_file} — skipping."
        if debug:
            print(msg)
        return None, msg

    # Apply lower BinCenter cut before any processing
    if MIN_BIN_CENTER is not None:
        mask_data = (voltage >= MIN_BIN_CENTER) & np.isfinite(voltage) & np.isfinite(count)
        mask_bg   = (voltage_bg >= MIN_BIN_CENTER) & np.isfinite(voltage_bg) & np.isfinite(bg_count)
        voltage    = voltage[mask_data]
        count      = count[mask_data]
        voltage_bg = voltage_bg[mask_bg]
        bg_count   = bg_count[mask_bg]
        if voltage.size == 0 or voltage_bg.size == 0:
            msg = f"[SKIP] All bins below cut {MIN_BIN_CENTER} V for data={data_file} or bg={bg_file}."
            if debug:
                print(msg)
            return None, msg

    # Fixed x-scaling to the background
    voltage_bg_scaled = voltage_bg * x_scale

    # Fixed y-scaling to the background
    scaled_bg = bg_count * y_scale

    # Pre-check to avoid ValueError: array of sample points is empty
    if voltage_bg_scaled.size == 0 or scaled_bg.size == 0:
        msg = f"[SKIP] Background sample points empty for data={data_file}, bg={bg_file} — skipping."
        if debug:
            print(msg)
        return None, msg

    # Ensure xp is monotonic increasing for np.interp
    order = np.argsort(voltage_bg_scaled)
    voltage_bg_scaled = voltage_bg_scaled[order]
    scaled_bg = scaled_bg[order]

    # Interpolate with safety guard
    try:
        bg_on_data_axis = np.interp(voltage, voltage_bg_scaled, scaled_bg, left=0, right=0)
    except (TypeError, ValueError) as e:
        msg = f"[SKIP] Interp error for files data={data_file}, bg={bg_file}: {e}"
        if debug:
            print(msg)
        return None, msg

    # Subtract background from data
    signal_subtracted = count - bg_on_data_axis

    # Fit the custom Gaussian to the subtracted signal (or reuse cached params)
    if cached_params is not None and all(k in cached_params for k in ('m1', 'm2', 'm3')):
        m1, m2, m3 = cached_params['m1'], cached_params['m2'], cached_params['m3']
    else:
        # p0 = [guess for m1, guess for m2, guess for m3]
        initial_guess = [
            np.max(signal_subtracted),  # m1
            0.05,                       # m2
            voltage[np.argmax(signal_subtracted)]  # m3
        ]
        try:
            popt, _ = curve_fit(gaussian, voltage, signal_subtracted, p0=initial_guess)
            m1, m2, m3 = popt
        except Exception as e:
            msg = f"Gaussian fit failed: {e} (data_file: {data_file}, bg_file: {bg_file})"
            if debug:
                print(msg)
            return None, msg

    # Generate the fitted Gaussian
    voltage_fit = np.linspace(float(voltage.min()), float(voltage.max()), 1000, dtype=np.float32)
    gaussian_fit = gaussian(voltage_fit, m1, m2, m3)

    return {
        'voltage': voltage,
        'bg_voltage_scaled': voltage_bg_scaled,
        'count': count,
        'scaled_bg': bg_on_data_axis,
        'signal_subtracted': signal_subtracted,
        'voltage_fit': voltage_fit,
        'gaussian_fit': gaussian_fit,
        'm1': m1,
        'm2': m2,
        'm3': m3,
        'x_scale': x_scale,
        'y_scale': y_scale,
    }, None

def plot_histograms_and_signals(directory, year_month, day, hour, minute,
                                second, display_label, measurement_time,
                                short_results, long_results):
    """
    Plot histograms, signals, and Gaussian fits for short (F1-F3) and long (F2-F4) monitors.
    Returns (m3_short, m3_long) for computing tau if possible.
    """
    plt.figure(figsize=(12, 12))

    # ----------------- Subplot 1: F1 & F3 (Short) ----------------- #
    plt.subplot(2, 1, 1)
    m3_short = None
    if short_results is not None:
        plt.step(short_results['voltage'],
                 short_results['count'],
                 where='mid',
                 label='F1 (Short PM Data)',
                 color='blue')
        plt.step(short_results['voltage'],
                 short_results['scaled_bg'],
                 where='mid',
                 label='F3 (Short PM BG, Scaled)',
                 color='green')
        plt.step(short_results['voltage'],
                 short_results['signal_subtracted'],
                 where='mid',
                 label='F1 - F3',
                 color='red')

        # Overplot the Gaussian fit
        plt.plot(short_results['voltage_fit'],
                 short_results['gaussian_fit'],
                 label=(
                     f'Gaussian Fit\n'
                     f'(m1={short_results["m1"]:.3f}, '
                     f'm2={short_results["m2"]:.3f}, '
                     f'm3={short_results["m3"]:.3f})'
                 ),
                 color='orange')

        m3_short = short_results['m3']

        # Display scale factors
        plt.text(
            0.05, 0.9,
            f'x-scale: {short_results["x_scale"]:.3f}\n'
            f'y-scale: {short_results["y_scale"]:.3f}',
            transform=plt.gca().transAxes,
            fontsize=10,
            color='black',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
        )

    plt.title(f'Short PM ({display_label})')
    plt.xlabel('bin center [V]')
    plt.ylabel('count')
    plt.legend(loc='best')
    

    # ----------------- Subplot 2: F2 & F4 (Long) ----------------- #
    plt.subplot(2, 1, 2)
    m3_long = None
    if long_results is not None:
        plt.step(long_results['voltage'],
                 long_results['count'],
                 where='mid',
                 label='F2 (Long PM Data)',
                 color='blue')
        plt.step(long_results['voltage'],
                 long_results['scaled_bg'],
                 where='mid',
                 label='F4 (Long PM BG, Scaled)',
                 color='green')
        plt.step(long_results['voltage'],
                 long_results['signal_subtracted'],
                 where='mid',
                 label='F2 - F4',
                 color='red')

        plt.plot(long_results['voltage_fit'],
                 long_results['gaussian_fit'],
                 label=(
                     f'Gaussian Fit\n'
                     f'(m1={long_results["m1"]:.3f}, '
                     f'm2={long_results["m2"]:.3f}, '
                     f'm3={long_results["m3"]:.3f})'
                 ),
                 color='brown')

        m3_long = long_results['m3']

        # Display scale factors
        plt.text(
            0.05, 0.9,
            f'x-scale: {long_results["x_scale"]:.3f}\n'
            f'y-scale: {long_results["y_scale"]:.3f}',
            transform=plt.gca().transAxes,
            fontsize=10,
            color='black',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
        )

    plt.title(f'Long PM ({display_label})')
    plt.xlabel('bin center [V]')
    plt.ylabel('count')
    plt.legend(loc='best')
    plt.tight_layout()

    minute_key = minute if second == "00" else f"{minute}_{second}"
    plot_filename = f"{PLOTS_DIR}/plot_{year_month}_{day}_{hour}_{minute_key}.png"
    if SAVE_FILES:
        plt.savefig(plot_filename)
        action = "Plot saved"
    else:
        plt.show()
        action = "Plot generated (not saved)"
    plt.close()
    print(f"{action}: {plot_filename}")

    return m3_short, m3_long

def process_directory(directory):
    """
    Process a single directory for short and long monitors.
    Applies constant scaling factors and stores tau in unified fit_cache.
    """
    if debug:
        print(f"[INFO] Processing directory: {directory}")
    # Fast skip via lightweight index (no pickle load)
    if (not FORCE_RERUN) and (directory in processed_dirs or directory in error_index):
        if debug:
            if directory in processed_dirs:
                print(f"[SKIP] Skipping already processed: {directory}")
            else:
                print(f"[SKIP] Skipping previously errored: {directory}")
        return

    # For directories not in the index, fall back to full cache (lazy-loaded)
    load_fit_cache()
    existing = fit_cache.get(directory)
    if (not FORCE_RERUN) and isinstance(existing, dict) and 'meta' in existing and 'tau' in existing['meta']:
        processed_dirs.add(directory)
        save_cache_index()
        if debug:
            print(f"[SKIP] Skipping already processed: {directory}")
        return
    # If forcing rerun, clear previous entries for a clean recompute
    if FORCE_RERUN:
        if debug:
            print(f"[INFO] Force rerun enabled. Recomputing: {directory}")
        fit_cache.pop(directory, None)

    path_parts = directory.strip("/").split("/")
    year_month_idx = None
    for idx, part in enumerate(path_parts):
        if re.fullmatch(r"\d{4}_[A-Za-z]{3}", part):
            year_month_idx = idx
            break
    if year_month_idx is None:
        if debug:
            print(f"[SKIP] Unable to locate year_month in path: {directory}")
        return

    try:
        year_month = path_parts[year_month_idx]
        day_raw = path_parts[year_month_idx + 1]
        hour_raw = path_parts[year_month_idx + 2]
        minute_raw = path_parts[year_month_idx + 3]
    except IndexError:
        if debug:
            print(f"[SKIP] Incomplete timestamp components in path: {directory}")
        return

    second_raw = path_parts[year_month_idx + 4] if len(path_parts) > year_month_idx + 4 else '00'

    # Parse date/time
    try:
        year_str, month_word = year_month.split('_')
        month_str = MONTH_MAP.get(month_word, '01')
        day = f"{int(day_raw):02d}"
        hour = f"{int(hour_raw):02d}"
        minute = f"{int(minute_raw):02d}"
        second = f"{int(second_raw):02d}"
        timestamp_str = f"{year_str}-{month_str}-{day} {hour}:{minute}:{second}"
        measurement_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
    except (ValueError, IndexError):
        if debug:
            print(f"[SKIP] Failed to parse timestamp from path: {directory}")
        return

    # Skip measurements outside the configured date window
    if START_DATE is not None and measurement_time < START_DATE:
        if debug:
            print(f"[SKIP] Before cutoff {START_DATE}: {directory}")
        return
    if END_DATE is not None and measurement_time > END_DATE:
        if debug:
            print(f"[SKIP] After cutoff {END_DATE}: {directory}")
        return

    short_data = os.path.join(directory, 'F1.txt')
    short_bg   = os.path.join(directory, 'F3.txt')
    long_data  = os.path.join(directory, 'F2.txt')
    long_bg    = os.path.join(directory, 'F4.txt')

    # ------------------------------------------------
    # Adjust these constants to desired scale factors:
    SHORT_X_SCALE = 1
    SHORT_Y_SCALE = 4.5
    LONG_X_SCALE  = 0.8
    LONG_Y_SCALE  = 2.7
    # ------------------------------------------------

    # Build a cache key per directory and scaling configuration
    cache_key = f"{directory}|sx={SHORT_X_SCALE:.5f}|sy={SHORT_Y_SCALE:.5f}|lx={LONG_X_SCALE:.5f}|ly={LONG_Y_SCALE:.5f}"
    cached_entry = {} if FORCE_RERUN else fit_cache.get(cache_key, {})
    cached_short = cached_entry.get('short') if isinstance(cached_entry, dict) else None
    cached_long  = cached_entry.get('long') if isinstance(cached_entry, dict) else None

    short_results = None
    short_err = None
    if os.path.exists(short_data) and os.path.exists(short_bg):
        short_results, short_err = process_monitor(short_data,
                                                   short_bg,
                                                   x_scale=SHORT_X_SCALE,
                                                   y_scale=SHORT_Y_SCALE,
                                                   cached_params=cached_short)

    long_results = None
    long_err = None
    if os.path.exists(long_data) and os.path.exists(long_bg):
        long_results, long_err = process_monitor(long_data,
                                                 long_bg,
                                                 x_scale=LONG_X_SCALE,
                                                 y_scale=LONG_Y_SCALE,
                                                 cached_params=cached_long)

    if not short_results or not long_results:
        if debug:
            reasons = []
            if not short_results:
                r = short_err or f"Short pair failed (data={short_data}, bg={short_bg})"
                reasons.append(f"short: {r}")
            if not long_results:
                r = long_err or f"Long pair failed (data={long_data}, bg={long_bg})"
                reasons.append(f"long: {r}")
            print(f"Error processing directory: {directory}. Reasons: {', '.join(reasons)}")
        else:
            print(f"Error processing directory: {directory}")
        # Tag directory as errored to skip next runs (unless FORCE_RERUN)
        try:
            error_index[directory] = {
                'timestamp': measurement_time.isoformat() if isinstance(measurement_time, datetime) else None,
                'reasons': reasons if debug else None
            }
            save_error_index()
            if debug:
                print(f"[TAG] Marked as errored: {directory}. Will skip next runs (use FORCE_RERUN to retry).")
        except Exception:
            pass
        return

    
    # If forcing rerun, drop any stale cache for this key before writing
    if FORCE_RERUN:
        fit_cache.pop(cache_key, None)
    fit_cache[cache_key] = {
        'timestamp': measurement_time.isoformat(),
        'short': {
            'm1': short_results['m1'],
            'm2': short_results['m2'],
            'm3': short_results['m3'],
            'x_scale': short_results['x_scale'],
            'y_scale': short_results['y_scale']
        },
        'long': {
            'm1': long_results['m1'],
            'm2': long_results['m2'],
            'm3': long_results['m3'],
            'x_scale': long_results['x_scale'],
            'y_scale': long_results['y_scale']
        }
    }
    # Delay saving until after possible tau computation to reduce writes

    display_label = (measurement_time.strftime('%b %d %H:%M')
                     if measurement_time.second == 0
                     else measurement_time.strftime('%b %d %H:%M:%S'))

    m3_short, m3_long = plot_histograms_and_signals(
        directory, year_month, day, hour, minute, second, display_label,
        measurement_time, short_results, long_results
    )

    # Compute tau if both m3 values exist and pass cuts; store in unified cache with directory key

    if (m3_short is not None) and (m3_long is not None):
        delta_t, scaling_factor = get_delta_t_and_scaling_factor(measurement_time)
        tau = -delta_t / np.log(m3_long / (scaling_factor*m3_short))/1e3 # in milliseconds
        dir_entry = fit_cache.setdefault(directory, {})
        dir_entry['meta'] = {
            'timestamp': measurement_time.isoformat(),
            'tau': tau,
            'm3_short': m3_short,
            'm3_long': m3_long,
            'directory': directory,
        }
        fit_cache[directory] = dir_entry
        processed_dirs.add(directory)
        save_cache_index()
    # Save cache once per directory processed
        #print(f"{measurement_time}: tau = {tau:.4e} ms")
    save_fit_cache()

def plot_m3_vs_time():
    """
    Plot m3 values (from short and long PM) over time using unified cache.
    """
    # Ensure cache is available
    load_fit_cache()
    # Collect and sort by time from cache; de-duplicate by directory
    time_m3 = []
    seen = set()
    for key, entry in fit_cache.items():
        if not isinstance(entry, dict):
            continue
        meta = entry.get('meta', {})
        ts = meta.get('timestamp') or entry.get('timestamp')
        if not ts:
            continue
        try:
            t = datetime.fromisoformat(ts)
        except Exception:
            continue
        directory = meta.get('directory') or (key if isinstance(key, str) else None)
        if directory in seen:
            continue
        m3_short = meta.get('m3_short')
        m3_long = meta.get('m3_long')
        if m3_short is None and 'short' in entry:
            m3_short = entry['short'].get('m3')
        if m3_long is None and 'long' in entry:
            m3_long = entry['long'].get('m3')
        # Apply cuts before plotting (only on m3_short)
        if m3_short is not None and m3_long is not None:
            time_m3.append((t, m3_short, m3_long))
            if directory:
                seen.add(directory)

    if not time_m3:
        return []

    time_m3.sort(key=lambda x: x[0])

    print("m3 values used for plotting:")
    for t, short_val, long_val in time_m3:
        print(f"  {t.strftime('%Y-%m-%d %H:%M:%S')} | short={short_val:.5f} | long={long_val:.5f}")

    times, m3_shorts, m3_longs = zip(*time_m3)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, m3_shorts, marker='o', linestyle='none',
            label='4 cm drift', color='blue')
    ax.plot(times, m3_longs,  marker='o', linestyle='none',
            label='20 cm drift', color='green')

    ax.set_title('Charge attenuation trend at 520 V/cm')
    ax.set_xlabel('Measurement Time')
    ax.set_ylabel('Pulse height [V]')
    ax.legend(loc='best')
    #ax.set_yscale('log')
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()
    fig.tight_layout()

    m3_plot_filename = f"{PLOTS_DIR}/m3_over_time.png"
    if SAVE_FILES:
        fig.savefig(m3_plot_filename)
        print(f"m3 vs Time plot saved: {m3_plot_filename}")
    else:
        plt.show()

    # Plot m3_short / m3_long ratio over time
    ratio_times = []
    ratios = []
    for t, short_val, long_val in time_m3:
        if np.isfinite(short_val) and np.isfinite(long_val) and long_val != 0:
            ratio_times.append(t)
            ratios.append(short_val / long_val)

    if ratio_times:
        fig_ratio, ax_ratio = plt.subplots(figsize=(10, 5))
        ax_ratio.plot(ratio_times, ratios, marker='o', linestyle='none', color='purple')
        ax_ratio.axhline(1.0, color='gray', linestyle='--', linewidth=1.0)
        ax_ratio.set_title('m3_short / m3_long vs Time (Oct 2025 onward)')
        ax_ratio.set_xlabel('Measurement Time')
        ax_ratio.set_ylabel('m3_short / m3_long')
        ax_ratio.grid(True, alpha=0.25)
        ax_ratio.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        fig_ratio.autofmt_xdate()
        fig_ratio.tight_layout()

        ratio_plot_filename = f"{PLOTS_DIR}/m3_ratio_over_time.png"
        if SAVE_FILES:
            fig_ratio.savefig(ratio_plot_filename)
            print(f"m3 ratio vs Time plot saved: {ratio_plot_filename}")
        else:
            plt.show()
        plt.close(fig_ratio)
    else:
        print("No valid m3_long values available to compute m3_short/m3_long ratio.")

    # Plot long m3 values within [0.75, 0.80]
    filtered_long = [(t, long_val) for t, _, long_val in time_m3
                     if np.isfinite(long_val) and 0.76 <= long_val <= 0.80]
    if filtered_long:
        times_long, long_values = zip(*filtered_long)
        fig_long, ax_long = plt.subplots(figsize=(10, 5))
        ax_long.plot(times_long, long_values, marker='o', linestyle='none', color='darkorange')
        ax_long.set_title('20 cm drift PM')
        ax_long.set_xlabel('Measurement Time')
        ax_long.set_ylabel('Peak [V]')
        ax_long.grid(True, alpha=0.25)
        ax_long.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax_long.set_ylim(0.76, 0.78)
        fig_long.autofmt_xdate()
        fig_long.tight_layout()

        filtered_plot_filename = f"{PLOTS_DIR}/m3_long_076_to_078.png"
        if SAVE_FILES:
            fig_long.savefig(filtered_plot_filename)
            print(f"Filtered long m3 plot saved: {filtered_plot_filename}")
        else:
            plt.show()
        plt.close(fig_long)
    else:
        print("No long m3 values found in the range 0.75-0.80.")

    plt.close(fig)
    return time_m3

def _load_temperature_series(path):
    times = []
    temps = []
    if not path or not os.path.exists(path):
        return times, temps
    fmts = ('%Y/%m/%d %H:%M:%S', '%Y-%m-%d %H:%M:%S')
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if ',' not in line:
                    continue
                t_str, val_str = line.split(',', 1)
                t_str = t_str.strip()
                val_str = val_str.strip()
                timestamp = None
                for fmt in fmts:
                    try:
                        timestamp = datetime.strptime(t_str, fmt)
                        break
                    except ValueError:
                        continue
                if timestamp is None:
                    continue
                try:
                    value = float(val_str)
                except ValueError:
                    continue
                times.append(timestamp)
                temps.append(value)
    except OSError as e:
        print(f"[WARN] Failed to read temperature series {path}: {e}")
    return times, temps

def _interpolate_temperature_at_times(temp_times, temp_vals, target_times):
    if not temp_times or not temp_vals or not target_times:
        return [None] * len(target_times)
    ts = np.array([t.timestamp() for t in temp_times], dtype=float)
    vs = np.array(temp_vals, dtype=float)
    if ts.size == 0 or vs.size == 0:
        return [None] * len(target_times)
    results = []
    for t in target_times:
        tt = float(t.timestamp())
        if tt < ts[0] or tt > ts[-1]:
            results.append(None)
        else:
            results.append(float(np.interp(tt, ts, vs)))
    return results

def plot_temp_and_m3_long_overlay(time_m3, temp_csv_path):
    if not time_m3 or not temp_csv_path:
        return
    temp_times, temp_vals = _load_temperature_series(temp_csv_path)
    if not temp_times:
        if debug:
            print(f"[INFO] No temperature samples found at {temp_csv_path}; skipping overlay.")
        return
    order_temp = np.argsort(np.array([t.timestamp() for t in temp_times]))
    temp_times = [temp_times[i] for i in order_temp]
    temp_vals = [temp_vals[i] for i in order_temp]

    long_points = [(t, long_val) for t, _, long_val in time_m3 if np.isfinite(long_val)]
    if not long_points:
        if debug:
            print("[INFO] No m3_long points available for temperature overlay.")
        return
    long_points.sort(key=lambda x: x[0])
    long_times = [p[0] for p in long_points]
    long_vals = [p[1] for p in long_points]

    start_time = long_times[0]
    filtered_temp = [(t, v) for t, v in zip(temp_times, temp_vals) if t >= start_time]
    if filtered_temp:
        temp_times, temp_vals = zip(*filtered_temp)
        temp_times = list(temp_times)
        temp_vals = list(temp_vals)
    else:
        if debug:
            print("[INFO] No temperature samples at or after first m3_long point; skipping overlay.")
        return

    window_start = min(min(temp_times), long_times[0])
    window_end = max(max(temp_times), long_times[-1])
    if window_start.date() == window_end.date():
        window_label = window_start.strftime('%b %d, %Y')
    elif window_start.year == window_end.year:
        window_label = f"{window_start.strftime('%b %d')}–{window_end.strftime('%b %d, %Y')}"
    else:
        window_label = f"{window_start.strftime('%b %d, %Y')}–{window_end.strftime('%b %d, %Y')}"

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(long_times, long_vals, 'o', label='Long PM fit value', color='tab:green')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Long PM fit value [V]')
    ax1.grid(True, alpha=0.2)

    ax2 = ax1.twinx()
    temp_vals_inverted = [-v for v in temp_vals]
    ax2.plot(temp_times, temp_vals_inverted, '-', label='Temperature (C, inverted)', color='tab:red', alpha=0.7)
    ax2.set_ylabel('Temperature (C, inverted)')

    fig.autofmt_xdate()
    fig.tight_layout()
    title = f'Temperature and m3_long overlay ({window_label})'
    fig.suptitle(title)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='best')
    plt.show()
    if SAVE_FILES:
        out_path = os.path.join(PLOTS_DIR, TEMP_OVERLAY_PLOT)
        fig.savefig(out_path, dpi=150)
        print(f"Temperature vs m3_long overlay saved: {out_path}")
    else:
        plt.show()
    plt.close(fig)

    interpolated_temps = _interpolate_temperature_at_times(temp_times, temp_vals, long_times)
    long_vals_for_corr = []
    inverted_temps_for_corr = []
    for m3_long_val, temp_val in zip(long_vals, interpolated_temps):
        if temp_val is None or not np.isfinite(temp_val):
            continue
        long_vals_for_corr.append(m3_long_val)
        inverted_temps_for_corr.append(-temp_val)
    if len(long_vals_for_corr) >= 2:
        corr_matrix = np.corrcoef(long_vals_for_corr, inverted_temps_for_corr)
        corr = corr_matrix[0, 1]
        if np.isfinite(corr):
            print(f"Inverted temperature vs long PM fit correlation: {corr:.4f}")
        else:
            print("Correlation between inverted temperature and long PM fit is undefined (constant series).")
    else:
        print("Not enough data to compute correlation between inverted temperature and long PM fit.")


def plot_tau():
    """
    Plot tau over time for the October 2025 onward runs (520 V/cm) using the unified cache.
    """
    # Ensure cache is available
    load_fit_cache()
    # Collect and sort tau data from cache; de-duplicate by directory
    time_taus = []
    seen = set()
    for key, entry in fit_cache.items():
        if not isinstance(entry, dict):
            continue
        meta = entry.get('meta', {})
        if 'timestamp' in meta and 'tau' in meta:
            try:
                t = datetime.fromisoformat(meta['timestamp'])
            except Exception:
                continue
            directory = meta.get('directory') or (key if isinstance(key, str) else None)
            if directory in seen:
                continue
            # Retrieve m3_short to apply cuts
            m3_short = meta.get('m3_short')
            if m3_short is None:
                ent = fit_cache.get(key)
                if ent and 'short' in ent:
                    m3_short = ent['short'].get('m3')
            if m3_short is None:
                continue
            time_taus.append((t, meta['tau']))
            if directory:
                seen.add(directory)

    if not time_taus:
        return

    time_taus.sort(key=lambda x: x[0])
    times, taus = zip(*time_taus)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, taus, 'o', label='520 V/cm', color='blue')
    ax.set_title('Electron Lifetime (tau) Over Time at 520 V/cm')
    ax.set_xlabel('Time')
    ax.set_ylabel('Electron Lifetime (ms)')
    ax.set_yscale('log')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()

    tau_plot_filename = f"{PLOTS_DIR}/tau_over_time.png"
    fig.tight_layout()
    if SAVE_FILES:
        fig.savefig(tau_plot_filename)
        print(f"Tau over time plot saved: {tau_plot_filename}")
    else:
        plt.show()
    plt.close(fig)
# Process all directories
ROOT_DIR = NP02DATA_DIR

def iter_measurement_dirs(root_dir: str):
    """Yield directories that look like measurement folders by presence of F1.txt.
    This avoids traversing parent directories that lack F1-F4 and reduces noise.
    """
    seen = set()
    pattern = f"{root_dir}/20??_[A-Za-z][a-z][a-z]/**/F1.txt"
    if START_DATE is not None:
        month_threshold = (START_DATE.year, START_DATE.month)
    else:
        month_threshold = None

    for f1 in glob.iglob(pattern, recursive=True):
        directory = os.path.dirname(f1)
        if directory in seen:
            continue

        year_month_part = next(
            (part for part in directory.split(os.sep) if re.fullmatch(r"\d{4}_[A-Za-z]{3}", part)),
            None
        )
        if not year_month_part:
            continue

        year_str, month_word = year_month_part.split('_', 1)
        month_num = MONTH_MAP.get(month_word)
        if month_num is None:
            continue

        try:
            year = int(year_str)
            month = int(month_num)
        except ValueError:
            continue

        if month_threshold is not None and (year, month) < month_threshold:
            continue

        seen.add(directory)
        yield directory

for dir_path in iter_measurement_dirs(ROOT_DIR):
    process_directory(dir_path)


save_fit_cache()
time_m3_points = plot_m3_vs_time()
plot_tau()
if TEMP_OVERLAY_CSV and time_m3_points:
    plot_temp_and_m3_long_overlay(time_m3_points, TEMP_OVERLAY_CSV)
