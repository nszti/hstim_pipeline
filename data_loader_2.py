import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from tkinter import filedialog, Tk
import os

# === Parameters ===
num_channels = 32
sampling_rate = 20000  # Hz
dtype = np.int16
gain = 0.195  # μV per bit
offset = 0
filter_band = (500, 5000)  # Bandpass in Hz
time_range = (60, 120)  # seconds to load and plot
channels_to_plot = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]  # change as needed

# === Select .dat file via dialog ===
Tk().withdraw()
dat_path = filedialog.askopenfilename(title="Select .dat file", filetypes=[("DAT files", "*.dat")])
if not dat_path:
    raise RuntimeError("No .dat file selected.")

# === Calculate number of samples ===
file_size_bytes = os.path.getsize(dat_path)
num_total_samples = file_size_bytes // np.dtype(dtype).itemsize // num_channels

# === Load data (time_range only) ===
start_sample = int(time_range[0] * sampling_rate)
end_sample = int(time_range[1] * sampling_rate)
num_samples_to_load = end_sample - start_sample

with open(dat_path, 'rb') as f: #only load the dataset in the time
    f.seek(start_sample * num_channels * np.dtype(dtype).itemsize)
    data = np.fromfile(f, dtype=dtype, count=num_samples_to_load * num_channels)

#data = np.fromfile(dat_path, dtype=dtype) #load the whole dataset

# === Reshape to (samples, channels) ===
data = data.reshape((-1, num_channels))

# === Convert to μV ===
data = data * gain + offset
#print(data)

# === Subtract row-wise mean ===
row_means = np.mean(data, axis=1, keepdims=True)
data = data - row_means

# === RMS of unfiltered data ===
rms_unfiltered = np.sqrt(np.mean(data**2, axis=0))
mean_unfiltered = np.mean(rms_unfiltered)
print("RMS of unfiltered data (μV):")
for i, rms in enumerate(rms_unfiltered):
    print(f"Ch {i+1}: {rms:.2f} μV")
print(f'Mean RMS of unfiltered data: {mean_unfiltered:.2f} μV\n')

# === Apply zero-phase filter ===
def bandpass_filtfilt(traces, fs, fmin, fmax, order=3):
    nyq = 0.5 * fs
    low, high = fmin / nyq, fmax / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, traces, axis=0)
filtered_data = bandpass_filtfilt(data, sampling_rate, filter_band[0], filter_band[1])

# === RMS of filtered data ===
rms_per_ch = np.sqrt(np.mean(filtered_data**2, axis =0))
mean_rms = np.mean(rms_per_ch)
print("\nRMS of filtered data (μV):")
for i,rms in enumerate(rms_per_ch):
    print(f'ch{i+1}: {rms}uV')
print(f'Mean RMS of filtered data: {mean_rms:.2f} μV\n')

import pandas as pd

# === Save filtered RMS of selected channels to csv ===
filtered_rms_dict = {
    'Channel': [f'Ch{ch}' for ch in channels_to_plot],
    'RMS (µV)': [rms_per_ch[ch] for ch in channels_to_plot]
}
rms_df = pd.DataFrame(filtered_rms_dict)

output_csv_path = os.path.splitext(dat_path)[0] + '_filtered_rms.csv'
rms_df.to_csv(output_csv_path, index=False)
print(f"\nFiltered RMS values saved to: {output_csv_path}")

# === Plot a few channels ===
plt.figure(figsize=(15, 10))
time_axis = np.arange(filtered_data.shape[0]) / sampling_rate

for i, ch in enumerate(channels_to_plot):
    plt.plot(time_axis, filtered_data[:, ch] + i * 500, label=f"Ch {ch}")  # offset each channel vertically

plt.xlabel("Time (s)")
plt.ylabel("Voltage (μV, offset for visibility)")
plt.title("Filtered Traces")
plt.legend()
plt.tight_layout()
plt.show()
