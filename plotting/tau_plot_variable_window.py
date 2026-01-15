import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Running average function that resets at each segment
def calculate_segmented_running_average(data, condition, window_size):
    running_avg = []
    segment_data = data[condition]
    for i in range(1, len(segment_data) + 1):
        if i < window_size:
            avg = np.mean(segment_data[:i])  # Take average of available points
        else:
            avg = np.mean(segment_data[i - window_size:i])  # Sliding window average
        running_avg.append(avg)
    return running_avg

# Load Bi source based purity monitor data
tau_data_file = 'tau_data.txt'
tau_data_df = pd.read_csv(tau_data_file, sep=r'\s+')

# Convert timestamp to datetime and sort
tau_data_df['timestamp'] = pd.to_datetime(tau_data_df['timestamp'])
tau_data_df.sort_values(by='timestamp', inplace=True)

# Load UV lamp based purity monitor data
uv_lamp_file = 'UV_lamp.csv'
uv_lamp_df = pd.read_csv(uv_lamp_file)
uv_lamp_df.columns = ['timestamp', 'tau_ms']
uv_lamp_df['tau_us'] = uv_lamp_df['tau_ms'] * 1000
uv_lamp_df['timestamp'] = pd.to_datetime(uv_lamp_df['timestamp'], format='%Y/%m/%d:%H')

# Constants
DELTA_T_520 = 16.0 / 0.1635
DELTA_T_260 = 16.0 / 0.1104
SCALING_FACTOR_520 = 0.92
SCALING_FACTOR_260 = 0.93

# Define timestamp-based conditions
before_jan1 = tau_data_df['timestamp'] < pd.Timestamp('2025-01-01 00:00')
between_jan1_and_jan10 = (tau_data_df['timestamp'] >= pd.Timestamp('2025-01-01 00:00')) & (tau_data_df['timestamp'] <= pd.Timestamp('2025-01-10 09:00'))
after_jan10 = tau_data_df['timestamp'] > pd.Timestamp('2025-01-10 09:00')

# Apply running average within each segment
window_size = 12
tau_data_df['m3_long_avg'] = np.nan
tau_data_df['m3_short_avg'] = np.nan

# Calculate running averages for each segment
tau_data_df.loc[before_jan1, 'm3_long_avg'] = calculate_segmented_running_average(tau_data_df['m3_long'], before_jan1, window_size)
tau_data_df.loc[between_jan1_and_jan10, 'm3_long_avg'] = calculate_segmented_running_average(tau_data_df['m3_long'], between_jan1_and_jan10, window_size)
tau_data_df.loc[after_jan10, 'm3_long_avg'] = calculate_segmented_running_average(tau_data_df['m3_long'], after_jan10, window_size)

tau_data_df.loc[before_jan1, 'm3_short_avg'] = calculate_segmented_running_average(tau_data_df['m3_short'], before_jan1, window_size)
tau_data_df.loc[between_jan1_and_jan10, 'm3_short_avg'] = calculate_segmented_running_average(tau_data_df['m3_short'], between_jan1_and_jan10, window_size)
tau_data_df.loc[after_jan10, 'm3_short_avg'] = calculate_segmented_running_average(tau_data_df['m3_short'], after_jan10, window_size)

# Recalculate tau using the updated running averages
updated_tau_list = []
for idx, row in tau_data_df.iterrows():
    if row['timestamp'] < pd.Timestamp('2025-01-01 00:00') or row['timestamp'] > pd.Timestamp('2025-01-10 09:00'):
        delta_t = DELTA_T_520
        scaling_factor = SCALING_FACTOR_520
    else:
        delta_t = DELTA_T_260
        scaling_factor = SCALING_FACTOR_260

    if not np.isnan(row['m3_long_avg']) and not np.isnan(row['m3_short_avg']):
        tau = -delta_t / np.log(row['m3_long_avg'] / (scaling_factor * row['m3_short_avg']))
        updated_tau_list.append(tau)
    else:
        updated_tau_list.append(None)

# Add recalculated tau to the DataFrame
tau_data_df['tau_moving_avg_updated'] = updated_tau_list

# Filter to remove negative tau values
tau_data_df_filtered = tau_data_df[tau_data_df['tau_moving_avg_updated'] > 0]



new_tau_list = []
for idx, row in tau_data_df.iterrows():
    if row['timestamp'] < pd.Timestamp('2025-01-01 00:00') or row['timestamp'] > pd.Timestamp('2025-01-10 09:00'):
        delta_t = DELTA_T_520
        #long_max = 0.737524266675
        long_max = 0.738
    else:
        delta_t = DELTA_T_260
        long_max = 0.567
        #long_max = 0.566112936505

    if not np.isnan(row['m3_long']) and row['m3_long_avg'] > 0:
            # Apply the new method
        tau = -(20/0.1635) / np.log(row['m3_long_avg'] / long_max)
        new_tau_list.append(tau)
    else:
        new_tau_list.append(None)

# Add new tau values to the DataFrame
tau_data_df['tau_new_method'] = new_tau_list

# Filter for valid values of tau_new_method
tau_data_df_filtered_new = tau_data_df[tau_data_df['tau_new_method'] > 0]



# Plot the results
plt.figure(figsize=(12, 6))

plt.plot(
    tau_data_df_filtered['timestamp'],
    tau_data_df_filtered['tau_moving_avg_updated'],
    'o',
    label=f'Updated Tau (Variable Window Size = {window_size})',
    color='blue',
    alpha=0.7
)
plt.plot(tau_data_df_filtered['timestamp'], tau_data_df_filtered['tau'], label='Original Tau', marker='o', linestyle='none', alpha=0.7)
plt.plot(uv_lamp_df['timestamp'], uv_lamp_df['tau_us'], label='UV Lamp Data', marker='o', linestyle='none', color='red', alpha=0.7)

plt.plot(
    tau_data_df_filtered_new['timestamp'],
    tau_data_df_filtered_new['tau_new_method'],
    'o',
    label=f'Long Purity Monitor (Variable Window Size = {window_size})',
    color='green',
    alpha=0.7
)

plt.yscale('log')
plt.title(f'Tau vs Time (with Variable Running Average, Window Size={window_size})')
plt.xlabel('Timestamp')
plt.ylabel('Electron Lifetime (µs)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
plt.close()


plt.figure(figsize=(10,6))
plt.plot(tau_data_df_filtered['timestamp'], tau_data_df_filtered['m3_short_avg'], label='short average', marker='o', linestyle='none', color ='blue', alpha=0.7)
plt.plot(tau_data_df_filtered['timestamp'], tau_data_df_filtered['m3_short'], label='short original', marker='o', linestyle='none', color ='red', alpha=0.7)
plt.plot(tau_data_df_filtered['timestamp'], tau_data_df_filtered['m3_long_avg'], label='long average', marker='o', linestyle='none', color='blue', alpha=0.7)
plt.plot(tau_data_df_filtered['timestamp'], tau_data_df_filtered['m3_long'], label='long original', marker='o', linestyle='none', color='red', alpha=0.7)
plt.title('Charge attenuation trend')
plt.xlabel('Measurement Time')
plt.ylabel('Pulse height [V]')
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()
plt.show()
