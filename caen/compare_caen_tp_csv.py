import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# File paths for CAEN and TP mean data
caen_csv = "plots_caen/caen_fit_means_data.csv"
tp_csv = "plots_scope/fit_means_data_low_cb.csv"
plot_dir = "plots_compare"
os.makedirs(plot_dir, exist_ok=True)
PLOT_START = datetime(2026, 1, 24, 7, 0)

# Read CAEN CSV (ADC units)
df_caen = pd.read_csv(caen_csv)
# Read TP CSV (mV units, skip comment lines)
df_tp = pd.read_csv(tp_csv, comment='#')

# Parse timestamps
if 'timestamp_iso' in df_caen.columns:
    df_caen['timestamp'] = pd.to_datetime(df_caen['timestamp_iso'], errors='coerce')
if 'timestamp_iso' in df_tp.columns:
    df_tp['timestamp'] = pd.to_datetime(df_tp['timestamp_iso'], errors='coerce')
if 'timestamp' in df_caen.columns:
    df_caen = df_caen[df_caen['timestamp'] >= PLOT_START]
if 'timestamp' in df_tp.columns:
    df_tp = df_tp[df_tp['timestamp'] >= PLOT_START]

# Channel mapping: CAEN col <-> TP col, plus pretty names for titles
channel_map = [
    ('CH0_InnerLong_high_mean_adc', 'TP_Response_Inner_long_PM_mV', 'Inner Long PM (High)'),
    ('CH1_OuterLong_high_mean_adc', 'TP_Response_Outer_long_PM_mV', 'Outer Long PM (High)'),
    ('CH0_InnerLong_low_mean_adc', 'Inner_low_CB_mV', 'Inner Long PM (Low)'),
    ('CH1_OuterLong_low_mean_adc', 'Outer_low_CB_mV', 'Outer Long PM (Low)'),
    ('CH2_Test Pulse_mean_adc', 'Test_pulse_mV', 'Test Pulse'),
]

saved_files = []
for caen_col, tp_col, label in channel_map:
    if caen_col not in df_caen.columns or tp_col not in df_tp.columns:
        print(f"Skipping {caen_col} vs {tp_col}: column missing in one of the files.")
        continue
    fig, ax1 = plt.subplots(figsize=(10,5))
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    caen_plot = df_caen[['timestamp', caen_col]].dropna(subset=['timestamp'])
    tp_plot = df_tp[['timestamp', tp_col]].dropna(subset=['timestamp'])
    caen_plot = caen_plot.sort_values('timestamp')
    tp_plot = tp_plot.sort_values('timestamp')
    # CAEN: left y-axis (scatter only)
    ax1.plot(
        caen_plot['timestamp'],
        caen_plot[caen_col],
        label='CAEN [ADC]',
        color=color1,
        marker='o',
        linestyle='-',
        linewidth=1,
        markersize=4,
    )
    ax1.set_ylabel(f'CAEN [ADC]', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    # TP: right y-axis (scatter only)
    ax2 = ax1.twinx()
    ax2.plot(
        tp_plot['timestamp'],
        tp_plot[tp_col],
        label='TP [mV]',
        color=color2,
        marker='x',
        linestyle='-',
        linewidth=1,
        markersize=4,
    )
    ax2.set_ylabel(f'TP [mV]', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    plt.title(f'{label}')
    ax1.set_xlabel('Time')
    fig.tight_layout()
    fig.autofmt_xdate()
    ax1.grid(True, alpha=0.3)
    # Custom legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles1 + handles2, labels1 + labels2, loc='upper left')
    outname = os.path.join(plot_dir, f'overlay_{caen_col}_vs_{tp_col}.png')
    plt.savefig(outname)
    saved_files.append(outname)
    plt.show()
    plt.close(fig)

print('Overlay comparison plots with dual y-axes saved:')
for fname in saved_files:
    print('  ', fname)
