import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Define the paths to the log files
log_file_path = 'plots/gaussian_mean_log_short.txt'
log_file_long_path = 'plots/gaussian_mean_log_long.txt'

# Function to read the log file
def read_log_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                # Split the line into the date-time and Gaussian mean parts
                date_time_str, gaussian_mean_str = line.split(',')
                # Convert the date-time string to a datetime object
                date_time = datetime.strptime(date_time_str.strip(), '%Y-%m-%d %Hh%M')
                # Convert the Gaussian mean to a float
                gaussian_mean = float(gaussian_mean_str.strip().split()[0])
                # Append the parsed data to the list
                data.append((date_time, gaussian_mean))
            except ValueError as e:
                print(f"Error parsing line: {line.strip()} - {e}")
                continue
    return pd.DataFrame(data, columns=['DateTime', 'GaussianMean'])

# Read the two log files
df_mean = read_log_file(log_file_path)
df_mean_long = read_log_file(log_file_long_path)

# Merge the two dataframes on the DateTime column
df = pd.merge(df_mean, df_mean_long, on='DateTime', suffixes=('_mean', '_mean_long'))

# Sort the merged dataframe by DateTime in chronological order
df = df.sort_values(by='DateTime')

# Calculate the electron lifetime using the given formula
df['ElectronLifetime'] = 16 / (0.2175 * np.log(df['GaussianMean_mean'] / df['GaussianMean_mean_long']))

# Plot electron lifetime vs time
plt.figure(figsize=(10, 6))
plt.plot(df['DateTime'], df['ElectronLifetime'], marker='o', linestyle='-', color='g', label='Electron Lifetime')
plt.title('Electron Lifetime vs Time')
plt.xlabel('Time')
plt.ylabel('Electron Lifetime (τ in us)')
plt.grid(True)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()

# Save and display the plot
plt.savefig('plots/electron_lifetime_vs_time.png')
plt.show()

# Optional: Print the first few rows of the calculated data
print(df[['DateTime', 'ElectronLifetime']])
