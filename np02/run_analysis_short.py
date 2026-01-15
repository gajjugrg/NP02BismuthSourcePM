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
    # Extract the date and time from the directory path
    path_parts = directory.strip("/").split("/")
    
    # Assuming the date is the second last and time is the last part of the path
    date_part = path_parts[-2]  # Extract date (e.g., 2024-08-31)
    time_part = path_parts[-1]  # Extract time (e.g., 16h24)
    
    # Find files starting with "F1" and "F3" in the directory
    file_path = glob.glob(os.path.join(directory, 'F1--*.txt'))[0]  # Dynamically find F1 files
    background_file_path = glob.glob(os.path.join(directory, 'F3--*.txt'))[0]  # Dynamically find F3 files
    
    # Read the data and background
    data = pd.read_csv(file_path, skiprows=4)  # Adjust skiprows based on your file
    background_data = pd.read_csv(background_file_path, skiprows=4)

    # Extract voltage and count (signal and background)
    voltage = data['Time']
    count = data['Ampl']
    background_count = background_data['Ampl']

    # Compute the first derivative (numerical gradient) of the signal to identify changes in slope
    first_derivative = np.gradient(count, voltage)

    # Find local minima in the signal's first derivative to identify shoulder start
    minima_indices = argrelextrema(first_derivative, np.less)[0]
    shoulder_start_index = minima_indices[0]  # First minimum after the main peak
    shoulder_start_voltage = voltage[shoulder_start_index]

    print(f"Shoulder start detected at: {shoulder_start_voltage} V")

    # Scale the background to match the shoulder region in the signal
    # Extract the shoulder region for signal and background
    shoulder_region_signal = count[shoulder_start_index:]  # From shoulder start to the end
    shoulder_region_background = background_count[shoulder_start_index:]

    # Calculate the scaling factor that aligns the background shoulder with the signal shoulder
    scaling_factor = np.mean(shoulder_region_signal) / np.mean(shoulder_region_background)

    # Scale the background using the calculated scaling factor
    scaled_background = background_count * scaling_factor

    # Subtract the scaled background from the signal
    signal_subtracted = count - scaled_background

    # Initial guess for the Gaussian fit (amplitude, mean, std deviation)
    initial_guess = [max(signal_subtracted), voltage[np.argmax(signal_subtracted)], 0.05]

    # Fit the Gaussian to the background-subtracted signal
    popt_corrected, pcov_corrected = curve_fit(gaussian, voltage, signal_subtracted, p0=initial_guess)

    # Extract the fit parameters
    a_fit_corrected, x0_fit_corrected, sigma_fit_corrected = popt_corrected

    # Generate fitted data for plotting
    voltage_fit = np.linspace(min(voltage), max(voltage), 1000)
    gaussian_fit = gaussian(voltage_fit, a_fit_corrected, x0_fit_corrected, sigma_fit_corrected)

    # Plot the original data, scaled background, and background-subtracted data with Gaussian fit
    plt.figure(figsize=(12, 8))

    # Plot 1: Original signal data and background
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
    plt.plot(voltage_fit, gaussian_fit, label=f'Gaussian Fit (mean={x0_fit_corrected:.4f}, std={sigma_fit_corrected:.4f})', color='orange')
    plt.title('Background-Subtracted Signal and Gaussian Fit')
    plt.xlabel('Peak Voltage (V)')
    plt.ylabel('Count (Number)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Save the plot in the plots directory with the date and time in the filename
    plot_filename = f"{plots_directory}/plot_{date_part}_{time_part}_short.png"
    plt.savefig(plot_filename)
    plt.close()

    # Save the Gaussian fit mean to a text file with the combined date and time in one column
    log_filename = f"{plots_directory}/gaussian_mean_log_short.txt"
    with open(log_filename, 'a') as log_file:
        log_file.write(f"{date_part} {time_part}, {x0_fit_corrected:.4f} mV\n")

    print(f"Plot saved: {plot_filename}")
    print(f"Gaussian mean: {x0_fit_corrected} V")


# Traverse through all subdirectories and run the script in each
root_directory = '/Users/Gajju/NP02_activities/purityMonitor/data'  # Replace with the correct root directory path

for directory in glob.glob(f"{root_directory}/2024-*/**/"):  # Traverse directories matching /date/time structure
    process_directory(directory)
