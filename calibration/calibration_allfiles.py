import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Simple Gaussian Function
def gaussian(x, mean, sigma, amplitude):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

# Recursive search for files in a specific directory
def find_files_in_directory(directory, target_files):
    file_paths = []
    for file in target_files:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            file_paths.append(file_path)
    return file_paths

# Main function for processing and plotting
def process_calibration_data(root_dir, zoom_range=(1.0, 1.7)):
    # Target files
    target_files = ['F1.txt', 'F2.txt', 'F3.txt', 'F4.txt']
    
    # Directory to save all plots
    plots_dir = os.path.join(root_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Iterate over each date directory
    for date_dir in sorted(os.listdir(root_dir)):
        date_path = os.path.join(root_dir, date_dir)
        if not os.path.isdir(date_path) or date_dir == "plots":
            continue

        date_gaussian_means = []

        # Process hour directories
        for hour_dir in sorted(os.listdir(date_path)):
            hour_path = os.path.join(date_path, hour_dir)
            if not os.path.isdir(hour_path):
                continue

            # Process minute directories
            for minute_dir in sorted(os.listdir(hour_path)):
                minute_path = os.path.join(hour_path, minute_dir)
                if not os.path.isdir(minute_path):
                    continue

                # Find target files in the minute directory
                file_paths = find_files_in_directory(minute_path, target_files)
                if not file_paths:
                    print(f"No target files found in {minute_path}")
                    continue

                # Plot overlayed histograms and fits
                plt.figure(figsize=(10, 6))
                minute_gaussian_means = []

                for file_path in file_paths:
                    file_name = os.path.basename(file_path).split(".")[0]

                    # Read data
                    try:
                        df = pd.read_csv(file_path, skiprows=1, header=None, names=["BinCenter", "Population"], delimiter=",")
                        df["BinCenter"] = pd.to_numeric(df["BinCenter"], errors="coerce")  # Ensure numeric conversion
                        df["Population"] = pd.to_numeric(df["Population"], errors="coerce")  # Ensure numeric conversion

                        bin_centers = df["BinCenter"].dropna().values  # Remove NaN entries
                        populations = df["Population"].dropna().values  # Remove NaN entries

                        if bin_centers.size == 0 or populations.size == 0:
                            print(f"File {file_path} contains no valid data.")
                            continue

                        # Filter by zoom range
                        zoom_mask = (bin_centers >= zoom_range[0]) & (bin_centers <= zoom_range[1])
                        bin_centers_zoom = bin_centers[zoom_mask]
                        populations_zoom = populations[zoom_mask]

                        if bin_centers_zoom.size == 0 or populations_zoom.size == 0:
                            print(f"File {file_path} contains no data in zoom range.")
                            continue

                        # Fit Gaussian
                        initial_guess = [np.mean(bin_centers_zoom), np.std(bin_centers_zoom), np.max(populations_zoom)]
                        params, _ = curve_fit(gaussian, bin_centers_zoom, populations_zoom, p0=initial_guess)
                        gaussian_mean = params[0]

                    except Exception as e:
                        print(f"Fit failed for file {file_name}: {e}")
                        continue

                    # Store Gaussian mean
                    minute_gaussian_means.append((minute_dir, file_name, gaussian_mean))
                    date_gaussian_means.append((f"{hour_dir}:{minute_dir}", file_name, gaussian_mean))

                    # Overlay histogram and fit
                    plt.hist(bin_centers_zoom, bins=len(bin_centers_zoom), weights=populations_zoom,
                             histtype='step', label=f"{file_name} Data")
                    plt.plot(bin_centers_zoom, gaussian(bin_centers_zoom, *params), '-', 
                             label=f"{file_name} Fit (mean={gaussian_mean:.3f})")

                plt.xlabel("Bin Center")
                plt.ylabel("Population")
                plt.title(f"Overlayed Plots and Fits (Date: {date_dir}, Hour: {hour_dir}, Minute: {minute_dir})")
                plt.legend()
                plt.grid()

                # Save overlayed plot for the minute
                date_plots_dir = os.path.join(plots_dir, date_dir)
                os.makedirs(date_plots_dir, exist_ok=True)
                save_path = os.path.join(date_plots_dir, f"overlay_plot_{hour_dir}_{minute_dir}.png")
                plt.savefig(save_path)
                plt.close()

        # Plot Gaussian means for the current date
        if date_gaussian_means:
            date_gaussian_means.sort(key=lambda x: x[0])  # Sort by time
            times = [x[0] for x in date_gaussian_means]
            gaussian_means = [x[2] for x in date_gaussian_means]
            file_labels = [x[1] for x in date_gaussian_means]

            plt.figure(figsize=(10, 6))
            for file_label in set(file_labels):
                file_mask = [file_label == lbl for lbl in file_labels]
                plt.plot(np.array(times)[file_mask], np.array(gaussian_means)[file_mask], 'o-', label=f"{file_label}")

            plt.xlabel("Time (hh:mm)")
            plt.ylabel("Gaussian Mean")
            plt.title(f"Gaussian Means vs Time for Date: {date_dir}")
            plt.legend()
            plt.grid()

            summary_path = os.path.join(date_plots_dir, "gaussian_means_summary.png")
            plt.show()
            plt.savefig(summary_path)
            plt.close()

# Example usage
if __name__ == "__main__":
    np02data_dir = os.environ.get("NP02DATA_DIR", "../np02data")
    root_directory = os.environ.get("NP02_CALIBRATION_DIR", os.path.join(np02data_dir, "calibration"))
    process_calibration_data(root_directory, zoom_range=(1.0, 1.7))
