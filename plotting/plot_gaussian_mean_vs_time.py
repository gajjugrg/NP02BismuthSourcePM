import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Define the path to the log file
log_file_path = 'plots/gaussian_mean_log_short.txt'

# Read the log file
data = []
with open(log_file_path, 'r') as f:
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
            # Log or print an error message if there is an issue with parsing
            print(f"Error parsing line: {line.strip()} - {e}")
            continue

# Check if there is any valid data to plot
if len(data) > 0:
    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data, columns=['DateTime', 'GaussianMean'])

    # Sort the DataFrame by the DateTime column in chronological order
    df = df.sort_values(by='DateTime')

    # Plot Gaussian mean vs time
    plt.figure(figsize=(10, 6))
    plt.plot(df['DateTime'], df['GaussianMean'], marker='o', color='b')
    plt.title('Gaussian Mean vs Time (Inner short)')
    plt.xlabel('Time')
    plt.ylabel('Gaussian Mean (mV)')
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()

    # Save and display the plot
    plt.savefig('plots/gaussian_mean_vs_time.png')
    plt.show()
else:
    print("No valid data to plot.")
