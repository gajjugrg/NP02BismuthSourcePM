import os
from datetime import datetime
from typing import List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

NP02DATA_DIR = os.environ.get('NP02DATA_DIR', '../np02data')

CSV_FILES = [
    ('Nov07-Nov11', os.path.join(NP02DATA_DIR, 'Nov07_Nov11.csv')),
    ('Nov07-Nov14', os.path.join(NP02DATA_DIR, 'Nov07_Nov14.csv')),
]

PLOTS_DIR = 'plots'
OUT_NAME = 'temperature_overlay.png'


def load_temperature_series(path: str) -> Tuple[List[datetime], List[float]]:
    if not os.path.exists(path):
        return [], []
    try:
        df = pd.read_csv(path, header=None, names=['timestamp', 'temperature'])
    except Exception:
        return [], []
    df = df.dropna(subset=['timestamp', 'temperature'])
    times: List[datetime] = []
    temps: List[float] = []
    for _, row in df.iterrows():
        try:
            ts = datetime.strptime(str(row['timestamp']).strip(), '%Y/%m/%d %H:%M:%S')
            times.append(ts)
            temps.append(float(row['temperature']))
        except Exception:
            continue
    return times, temps


def plot_overlay():
    plt.figure(figsize=(10, 5))
    for label, path in CSV_FILES:
        times, temps = load_temperature_series(path)
        if not times:
            continue
        plt.plot(times, temps, marker='o', linestyle='-', label=f'{label} ({os.path.basename(path)})')
    plt.title('Temperature Overlay: Nov 7–Nov 14')
    plt.xlabel('Time')
    plt.ylabel('Temperature [°C]')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.gcf().autofmt_xdate()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, OUT_NAME)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Saved temperature overlay plot: {out_path}')


if __name__ == '__main__':
    plot_overlay()
