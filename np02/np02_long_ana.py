import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
import os
import glob

# Define the directory where plots will be saved
plots_directory = 'plots'

# Create the plots directory if it doesn't exist
if not os.path.exists(plots_directory):
    os.makedirs(plots_directory)

# Define Gaussian function for fitting
def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

# Function to process the files and generate plots
def process_directory(directory):
    # Expected directory format:
    # np02data/2024_Dec/07/13/56/F*.txt
    path_parts = directory.strip("/").split("/")
    if len(path_parts) < 5:
        print(f"Invalid directory structure: {directory}")
        return

    year_month = path_parts[-5]  # e.g., "2024_Dec"
    day = path_parts[-4]         # e.g., "07"
    hour = path_parts[-3]        # e.g., "13"
    minute = path_parts[-2]      # e.g., "56"

    date_part = f"{year_month}_{day}"
    time_part = f"{hour}_{minute}"

    # Find F1.txt (data) and F3.txt (background) files
    data_file_path = os.path.join(directory, 'F1.txt')
    background_file_path = os.path.join(directory, 'F3.txt')

    if not os.path.exists(data_file_path) or not os.path.exists(background_file_path):
        print(f"Missing required files in directory: {directory}")
        return

    # Read the data and background
    try:
        data = pd.read_csv(data_file_path)
        background_data = pd.read_csv(background_file_path)
    except Exception as e:
        print(f"Error reading files in {directory}: {e}")
        return

    # Validate required columns
    if 'BinCenter' not in data.columns or 'Population' not in data.columns:
        print(f"Required columns 'BinCenter' and 'Population' not found in {data_file_path}")
        return
    if 'BinCenter' not in background_data.columns or 'Population' not in background_data.columns:
        print(f"Required columns 'BinCenter' and 'Population' not found in {background_file_path}")
        return

    voltage = data['BinCenter']
    count = data['Population']
    background_voltage = background_data['BinCenter']
    background_count = background_data['Population']

    # Compute the first derivative of the signal to identify changes in slope
    first_derivative = np.gradient(count, voltage)

    # Find local minima in the signal's first derivative
    minima_indices = argrelextrema(first_derivative, np.less)[0]

    if len(minima_indices) == 0:
        print(f"No minima found in first derivative for directory: {directory}")
        return

    shoulder_start_index = minima_indices[0]
    shoulder_start_voltage = voltage.iloc[shoulder_start_index]

    print(f"Shoulder start detected at: {shoulder_start_voltage:.4f} V")

    # Extract the shoulder region for signal and background
    shoulder_region_signal = count.iloc[shoulder_start_index:]
    shoulder_region_background = background_count.iloc[shoulder_start_index:]

    # Calculate the scaling factor for the background
    if np.mean(shoulder_region_background) == 0:
        print("Background shoulder mean is zero, skipping scaling.")
        return
    scaling_factor = np.mean(shoulder_region_signal) / np.mean(shoulder_region_background)

    # Scale the background
    scaled_background = background_count * scaling_factor

    # Subtract the scaled background from the signal
    signal_subtracted = count - scaled_background

    # Initial guess for Gaussian fit
    initial_guess = [max(signal_subtracted), voltage.iloc[np.argmax(signal_subtracted)], 0.05]

    try:
        popt, _ = curve_fit(gaussian, voltage, signal_subtracted, p0=initial_guess)
    except Exception as e:
        print(f"Gaussian fit failed for directory: {directory}, error: {e}")
        return

    a_fit, x0_fit, sigma_fit = popt

    # Generate fitted data for plotting
    voltage_fit = np.linspace(min(voltage), max(voltage), 1000)
    gaussian_fit = gaussian(voltage_fit, a_fit, x0_fit, sigma_fit)

    # Plot the results
    plt.figure(figsize=(12, 8))

    # Plot 1: Original data and scaled background
    plt.subplot(2, 1, 1)
    plt.plot(voltage, count, label='Signal (Original Data)', color='blue')
    plt.plot(voltage, scaled_background, label=f'Scaled Background (scale={scaling_factor:.2f})', color='green')
    plt.title(f'Signal and Scaled Background on {date_part} at {time_part}')
    plt.xlabel('Peak Voltage (V)')
    plt.ylabel('Count (Number)')
    plt.legend()
    plt.grid(True)

    # Plot 2: Background-subtracted data and Gaussian fit
    plt.subplot(2, 1, 2)
    plt.plot(voltage, signal_subtracted, label='Background-Subtracted Signal', color='red')
    plt.plot(voltage_fit, gaussian_fit, label=f'Gaussian Fit (mean={x0_fit:.4f}, std={sigma_fit:.4f})', color='orange')
    plt.title('Background-Subtracted Signal and Gaussian Fit')
    plt.xlabel('Peak Voltage (V)')
    plt.ylabel('Count (Number)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Save the plot
    plot_filename = f"{plots_directory}/plot_{date_part}_{time_part}_long.png"
    plt.savefig(plot_filename)
    plt.close()

    # Log the Gaussian fit mean
    log_filename = f"{plots_directory}/gaussian_mean_log_long.txt"
    with open(log_filename, 'a') as log_file:
        log_file.write(f"{date_part} {time_part}, {x0_fit:.4f} V\n")

    print(f"Plot saved: {plot_filename}")
    print(f"Gaussian mean: {x0_fit:.4f} V")

# Adjust the root directory to match the new structure
root_directory = os.environ.get('NP02DATA_DIR', '../np02data')  # set NP02DATA_DIR to point at your np02data folder

# Traverse directories and process files
for directory in glob.glob(f"{root_directory}/2024_*/**/**/**/"):
    process_directory(directory)
