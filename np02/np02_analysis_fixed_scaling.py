import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime
import pickle
import json

SAVE_FILES = False # set to False to skip writing plots to disk

# Debug flag: when True, emit [INFO] and [SKIP] messages
debug = False

# Force rerun: when True, recompute and overwrite cache even if present
FORCE_RERUN = False

# Directory to save plots
PLOTS_DIR = 'plots'
if SAVE_FILES:
    os.makedirs(PLOTS_DIR, exist_ok=True)

# Cuts for valid points
M3_SHORT_MIN = 0.550
M3_SHORT_MAX = 2

# Date window filter (inclusive)
# - Set START_DATE to the earliest measurement time to keep
# - Set END_DATE to the latest measurement time to keep, or None to keep everything after START_DATE
# Default: June 15 through July 2
START_DATE = datetime(2024, 12, 27, 15, 0)
END_DATE = None
MIN_BIN_CENTER = 0  # Cut: drop all bins with BinCenter < 0.35 V for data and background

# File to cache fit parameters so we don't re-fit on reruns
FIT_CACHE_FILE = 'fit_cache.pkl'
FIT_CACHE_INDEX_FILE = 'fit_cache_index.json'
ERROR_CACHE_INDEX_FILE = 'fit_cache_errors.json'

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
    Determine the correct DELTA_T value and scaling factor based on the measurement time.
    Returns a tuple (DELTA_T, scaling_factor).
    """
    # Define the cutoff timestamps
    start_260_vcm = datetime(2025, 1, 1, 0, 0)
    end_260_vcm = datetime(2025, 1, 10, 9, 0)

    if measurement_time < start_260_vcm or measurement_time > end_260_vcm:
        # 520 V/cm
        delta_t = 16.0 / 0.1635
        scaling_factor = 0.92
    else:
        # 260 V/cm
        delta_t = 16.0 / 0.1104
        scaling_factor = 0.93

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

    plt.title(f'Short PM ({year_month}_{day} {hour}:{minute})')
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

    plt.title(f'Long PM ({year_month}_{day} {hour}:{minute})')
    plt.xlabel('bin center [V]')
    plt.ylabel('count')
    plt.legend(loc='best')
    plt.tight_layout()

    plot_filename = f"{PLOTS_DIR}/plot_{year_month}_{day}_{hour}_{minute}.png"
    if SAVE_FILES:
        plt.savefig(plot_filename)
        print(f"Plot saved: {plot_filename}")
    plt.close()

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
    if len(path_parts) < 5:
        return

    # For example: /.../2024_Dec/20/22/55/
    # path_parts[-4] = "2024_Dec", path_parts[-3] = "20", ...
    year_month = path_parts[-4]
    day = path_parts[-3]
    hour = path_parts[-2]
    minute = path_parts[-1]

    # Parse date/time
    try:
        year_str, month_word = year_month.split('_')
        if month_word in MONTH_MAP:
            month_str = MONTH_MAP[month_word]
        else:
            month_str = '01'
        timestamp_str = f"{year_str}-{month_str}-{day} {hour}:{minute}"
        measurement_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M')
    except (ValueError, IndexError):
        measurement_time = datetime.now()

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
    SHORT_X_SCALE = 0.97
    SHORT_Y_SCALE = 2.00
    LONG_X_SCALE  = 0.91
    LONG_Y_SCALE  = 1.20
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

    m3_short, m3_long = plot_histograms_and_signals(
        directory, year_month, day, hour, minute,
        short_results, long_results
    )

    # Compute tau if both m3 values exist and pass cuts; store in unified cache with directory key

    if (m3_short is not None) and (m3_long is not None):
        # Apply cuts: 0.550 < m3_short < 1.50
        if not (m3_short > M3_SHORT_MIN and m3_short < M3_SHORT_MAX):
            if debug:
                print(f"[SKIP] Cuts failed for {directory}: m3_short={m3_short:.3f}")
            return
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
        if m3_short is not None and m3_long is not None and (m3_short > M3_SHORT_MIN and m3_short < M3_SHORT_MAX):
            time_m3.append((t, m3_short, m3_long))
            if directory:
                seen.add(directory)

    if not time_m3:
        return

    time_m3.sort(key=lambda x: x[0])
    times, m3_shorts, m3_longs = zip(*time_m3)

    plt.figure(figsize=(10, 6))
    plt.plot(times, m3_shorts, marker='o', linestyle='none',
             label='4 cm drift', color='blue')
    plt.plot(times, m3_longs,  marker='o', linestyle='none',
             label='20 cm drift', color='green')

    plt.gcf().autofmt_xdate()
    plt.title('Charge attenuation trend at 500 V/cm')
    plt.xlabel('Measurement Time')
    plt.ylabel('Pulse height [V]')
    plt.legend(loc='best')
    plt.tight_layout()

    m3_plot_filename = f"{PLOTS_DIR}/m3_over_time.png"
    if SAVE_FILES:
        plt.savefig(m3_plot_filename)
        print(f"m3 vs Time plot saved: {m3_plot_filename}")
    else:
        plt.show()
    plt.close()


def plot_tau():
    """
    Plot tau over time with different colors for different electric fields using unified cache.
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
            if not (m3_short > M3_SHORT_MIN and m3_short < M3_SHORT_MAX):
                continue
            time_taus.append((t, meta['tau']))
            if directory:
                seen.add(directory)

    if not time_taus:
        return

    time_taus.sort(key=lambda x: x[0])
    times, taus = zip(*time_taus)

    # Split the data into two categories based on the electric field
    times_520 = []
    taus_520 = []
    times_260 = []
    taus_260 = []

    cutoff_time_start = datetime(2025, 1, 1, 0, 0)
    cutoff_time_end = datetime(2025, 1, 10, 8, 0)

    for t, tau in zip(times, taus):
        if t < cutoff_time_start or t > cutoff_time_end:
            times_520.append(t)
            taus_520.append(tau)
        else:
            times_260.append(t)
            taus_260.append(tau)

    # Plot the tau values with different colors
    plt.figure(figsize=(10, 6))
    plt.plot(times_520, taus_520, 'o', label='520 V/cm', color='blue')
    plt.plot(times_260, taus_260, 'o', label='260 V/cm', color='red')

    plt.title('Electron Lifetime (tau) Over Time')
    plt.xlabel('Time')
    plt.ylabel('Electron Lifetime (ms)')
    plt.yscale('log')
    plt.legend()

    # Save the plot
    tau_plot_filename = f"{PLOTS_DIR}/tau_over_time.png"
    plt.tight_layout()
    if SAVE_FILES:
        plt.savefig(tau_plot_filename)
        print(f"Tau over time plot saved: {tau_plot_filename}")
    else:
        plt.show()
    plt.close()
# Process all directories
ROOT_DIR = os.environ.get('NP02DATA_DIR', '../np02data')

def iter_measurement_dirs(root_dir: str):
    """Yield directories that look like measurement folders by presence of F1.txt.
    This avoids traversing parent directories that lack F1-F4 and reduces noise.
    """
    seen = set()
    pattern = f"{root_dir}/202*/**/F1.txt"
    for f1 in glob.iglob(pattern, recursive=True):
        d = os.path.dirname(f1)
        if d not in seen:
            seen.add(d)
            yield d

for dir_path in iter_measurement_dirs(ROOT_DIR):
    process_directory(dir_path)


save_fit_cache()
plot_m3_vs_time()
plot_tau()
