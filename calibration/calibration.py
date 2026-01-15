import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erfc
import numpy as np

# Gaussian + Exponential Convolution Function
def gauss_exp_convolution(x, mean, sigma, amplitude, tau):
    z = (sigma**2 - tau * (x - mean)) / (np.sqrt(2) * sigma * tau)
    return (amplitude / (2 * tau)) * np.exp((sigma**2 / (2 * tau**2)) - (x - mean) / tau) * erfc(z)

# Recursive search for files
def find_files(root_dir, file_names):
    file_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file in file_names:
                file_paths.append(os.path.join(root, file))
    return file_paths

# Main Function for Overlayed Plot and Fit
def process_files_with_overlay(root_dir, zoom_range=(1.0, 1.7)):
    # Target files
    target_files = ['F1.txt', 'F2.txt', 'F3.txt', 'F4.txt']
    file_paths = find_files(root_dir, target_files)
    
    data = {}
    for path in file_paths:
        name = os.path.basename(path).split(".")[0]
        df = pd.read_csv(path, skiprows=1, header=None, names=["BinCenter", "Population"], delimiter=",")
        #df = pd.read_csv(path, comment='#', skiprows=5, names=["Time", "Ampl"])
        data[name] = df
    
    # Initialize plot
    plt.figure(figsize=(10, 6))
    results = {}
    
    for name, df in data.items():
        # Extract data
        bin_centers = df["BinCenter"].values
        weights = df["Population"].values
        mean_data = np.average(bin_centers, weights=weights)  # Weighted mean
        
        # Zoom range filtering
        zoom_mask = (bin_centers >= zoom_range[0]) & (bin_centers <= zoom_range[1])
        bin_centers_zoom = bin_centers[zoom_mask]
        weights_zoom = weights[zoom_mask]
        
        # Plot unfilled histogram
        plt.hist(bin_centers_zoom, bins=len(bin_centers_zoom), weights=weights_zoom, 
                 histtype='step', stacked=False, fill=False, label=f"{name} Data (mean={mean_data:.3f})")
        
        # Fit Gaussian + Exponential Convolution
        try:
            non_zero_indices = weights_zoom > 0
            bin_centers_fit = bin_centers_zoom[non_zero_indices]
            weights_fit = weights_zoom[non_zero_indices]
            p0 = [np.mean(bin_centers_fit), np.std(bin_centers_fit), max(weights_fit), 0.1]  # Initial guess
            
            popt, _ = curve_fit(gauss_exp_convolution, bin_centers_fit, weights_fit, p0=p0, maxfev=10000)
            
            # Compute fit and chi-squared
            y_fit = gauss_exp_convolution(bin_centers_fit, *popt)
            chi_sq = np.sum((weights_fit - y_fit) ** 2 / y_fit)
            reduced_chi_sq = chi_sq / (len(y_fit) - len(popt))
            
            results[name] = {"mean": popt[0], "sigma": popt[1], "tau": popt[3], "chi_squared": reduced_chi_sq}
            
            # Overlay fit
            plt.plot(bin_centers_fit, y_fit, '--', label=f"{name} Fit (χ²={reduced_chi_sq:.3f})")
        except Exception as e:
            results[name] = {"error": str(e)}
            print(f"Fit failed for {name}: {e}")

    # Finalize plot
    plt.title("Overlayed Data with Gaussian + Exponential Decay Fits")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Counts")
    plt.legend()
    plt.xlim(zoom_range)
    plt.show()

    # Print results
    for name, res in results.items():
        if "error" in res:
            print(f"{name}: Fit Failed - {res['error']}")
        else:
            print(f"{name}: μ={res['mean']:.3f}, σ={res['sigma']:.3f}, τ={res['tau']:.3f}, χ²={res['chi_squared']:.3f}")

# Run the function
if __name__ == "__main__":
    root_directory = "./"  # Current directory where files are located
    process_files_with_overlay(root_directory, zoom_range=(1.2, 1.5))
