import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from datetime import datetime

# Directory to save plots
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# File to store tau data
TAU_DATA_FILE = 'tau_data.json'

DELTA_T = 16 / 0.1601

# Load tau data from file if it exists
if os.path.exists(TAU_DATA_FILE):
    with open(TAU_DATA_FILE, 'r') as f:
        tau_data = json.load(f)
else:
    tau_data = {}

def save_tau_data():
    """Save tau data to file."""
    with open(TAU_DATA_FILE, 'w') as f:
        json.dump(tau_data, f)

# Gaussian function
def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def process_monitor(data_file, bg_file):
    """Process a single monitor (short or long) and return results."""
    try:
        data = pd.read_csv(data_file)
        bg_data = pd.read_csv(bg_file)
    except Exception as e:
        print(f"Error reading files: {e}")
        return None

    if 'BinCenter' not in data.columns or 'Population' not in data.columns:
        return None
    if 'BinCenter' not in bg_data.columns or 'Population' not in bg_data.columns:
        return None

    voltage = data['BinCenter']
    count = data['Population']
    bg_count = bg_data['Population']

    first_derivative = np.gradient(count, voltage)
    minima_indices = argrelextrema(first_derivative, np.less)[0]

    if len(minima_indices) == 0:
        return None

    shoulder_index = minima_indices[0]
    signal_shoulder = count.iloc[shoulder_index:]
    bg_shoulder = bg_count.iloc[shoulder_index:]

    if np.mean(bg_shoulder) == 0:
        return None

    #scaling_factor = 1.0 
    scaling_factor = np.mean(signal_shoulder) / np.mean(bg_shoulder)
    scaled_bg = bg_count * scaling_factor
    signal_subtracted = count - scaled_bg

    initial_guess = [max(signal_subtracted), voltage.iloc[np.argmax(signal_subtracted)], 0.05]

    try:
        popt, _ = curve_fit(gaussian, voltage, signal_subtracted, p0=initial_guess)
    except Exception as e:
        print(f"Gaussian fit failed: {e}")
        return None

    a_fit, x0_fit, sigma_fit = popt
    voltage_fit = np.linspace(min(voltage), max(voltage), 1000)
    gaussian_fit = gaussian(voltage_fit, a_fit, x0_fit, sigma_fit)

    return {
        'voltage': voltage,
        'count': count,
        'scaled_bg': scaled_bg,
        'signal_subtracted': signal_subtracted,
        'voltage_fit': voltage_fit,
        'gaussian_fit': gaussian_fit,
        'x0_fit': x0_fit,
        'sigma_fit': sigma_fit,
        'scaling_factor': scaling_factor
    }
def plot_histograms_and_signals(directory, year_month, day, hour, minute, short_results, long_results):
    """Plot histograms, signals, and Gaussian fits."""
    plt.figure(figsize=(12, 12))

    # Histograms for F1, F2, F3, F4
    plt.subplot(3, 1, 1)
    if short_results is not None:
        plt.step(short_results['voltage'], short_results['count'], where='mid', label='F1 (Short PM Data)', color='blue')
        plt.step(short_results['voltage'], short_results['scaled_bg'], where='mid', label='F3 (Short PM Background)', color='green')
    if long_results is not None:
        plt.step(long_results['voltage'], long_results['count'], where='mid', label='F2 (Long PM Data)', color='purple')
        plt.step(long_results['voltage'], long_results['scaled_bg'], where='mid', label='F4 (Long PM Background)', color='orange')

    plt.title(f'Histograms for F1, F2, F3, and F4 on {year_month}_{day} at {hour}_{minute}')
    plt.xlabel('Bin Center (V)')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True)

    # Background-subtracted signals and Gaussian fits
    plt.subplot(3, 1, 2)
    if short_results is not None:
        n_bins = len(short_results['voltage'])  # Number of bins matches voltage length
        plt.hist(
            short_results['voltage'], bins=n_bins, 
            weights=short_results['signal_subtracted'], 
            histtype='step', stacked=True, fill=False, 
            label='Short PM BG-Subtracted', color='red'
        )
        plt.plot(short_results['voltage_fit'], short_results['gaussian_fit'], label=f'Short PM Fit (mean={short_results["x0_fit"]:.4f})', color='orange')
    if long_results is not None:
        n_bins = len(long_results['voltage'])
        plt.hist(
            long_results['voltage'], bins=n_bins, 
            weights=long_results['signal_subtracted'], 
            histtype='step', stacked=True, fill=False, 
            label='Long PM BG-Subtracted', color='magenta'
        )
        plt.plot(long_results['voltage_fit'], long_results['gaussian_fit'], label=f'Long PM Fit (mean={long_results["x0_fit"]:.4f})', color='brown')

    plt.title('Background-Subtracted Signals and Gaussian Fits')
    plt.xlabel('Peak Voltage (V)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_filename = f"{PLOTS_DIR}/plot_{year_month}_{day}_{hour}_{minute}.png"
    plt.savefig(plot_filename)
    plt.close()
    print(f"Plot saved: {plot_filename}")

# File to save tau time sequence and lifetimes
TAU_SEQUENCE_FILE = 'NP02_purity.txt'

def process_directory(directory):
    """Process a single directory for short and long monitors."""
    if directory in tau_data:
        print(f"Skipping already processed directory: {directory}")
        return

    path_parts = directory.strip("/").split("/")
    if len(path_parts) < 5:
        return

    year_month = path_parts[-4]
    day, hour, minute = path_parts[-3], path_parts[-2], path_parts[-1]

    # Convert year_month to datetime
    year_str, month_str = year_month.split("_")
    year = int(year_str)
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    month = month_map[month_str]
    dt = datetime(year, month, int(day), int(hour), int(minute))

    short_data = os.path.join(directory, 'F1.txt')
    short_bg = os.path.join(directory, 'F3.txt')
    long_data = os.path.join(directory, 'F2.txt')
    long_bg = os.path.join(directory, 'F4.txt')

    short_results = process_monitor(short_data, short_bg) if os.path.exists(short_data) and os.path.exists(short_bg) else None
    long_results = process_monitor(long_data, long_bg) if os.path.exists(long_data) and os.path.exists(long_bg) else None

    if not short_results or not long_results:
        return

    plot_histograms_and_signals(directory, year_month, day, hour, minute, short_results, long_results)

    # Compute and store tau
    tau = -DELTA_T / np.log(1.07*long_results['x0_fit'] / short_results['x0_fit'])
    tau_data[directory] = {'timestamp': dt.isoformat(), 'tau': tau}
    print(f"{dt}: tau = {tau:.4e} us")

    with open(TAU_SEQUENCE_FILE, 'a') as f:
        f.write(f"{dt.isoformat()}\t{tau:.4e} us\n")

# Process all directories
ROOT_DIR = os.environ.get('NP02DATA_DIR', '../np02data')
for dir_path in glob.glob(f"{ROOT_DIR}/2024_*/**/**/**/"):
    process_directory(dir_path)

# Save tau data
save_tau_data()

# Plot tau over time
if tau_data:
    time_taus = [(datetime.fromisoformat(v['timestamp']), v['tau']) for v in tau_data.values()]
    time_taus.sort(key=lambda x: x[0])
    times, taus = zip(*time_taus)
    plt.figure(figsize=(10, 4))
    plt.plot(times, taus, marker='o', linestyle='none')
    plt.title('Tau Over Time')
    plt.xlabel('Time')
    plt.ylabel('Electron Lifetime (us)')
    plt.yscale('log')
    plt.grid(True)
    plt.show()
    tau_plot_filename = f"{PLOTS_DIR}/tau_over_time.png"
    plt.savefig(tau_plot_filename)
    plt.close()
    print(f"Tau over time plot saved: {tau_plot_filename}")

def write_chronological_tau():
    """Write the tau data to a text file in chronological order."""
    if tau_data:
        time_taus = [(datetime.fromisoformat(v['timestamp']), v['tau']) for v in tau_data.values()]
        time_taus.sort(key=lambda x: x[0])  # Sort by time

        with open(TAU_SEQUENCE_FILE, 'w') as f:
            for time, tau in time_taus:
                f.write(f"{time.isoformat()}\t{tau:.4e} us\n")
        print(f"Tau data written to {TAU_SEQUENCE_FILE} in chronological order.")

# Save tau data in chronological order
write_chronological_tau()
