import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load Excel data
df = pd.read_excel('/mnt/data/nanoz_imp.xlsx')

# Extract and clean frequencies
frequencies = []
impedance_data = []

for col in df.columns:
    try:
        freq = float(col.replace('Hz', '').replace('E', 'e'))
        values = df[col].dropna().values
        frequencies.append(freq)
        impedance_data.append(values)
    except:
        continue

# Sort frequencies
sorted_idx = np.argsort(frequencies)
frequencies = np.array(frequencies)[sorted_idx]
impedance_data = [impedance_data[i] for i in sorted_idx]

# Plot each boxplot manually with log-scale positioning
plt.figure(figsize=(12, 6))
for freq, data in zip(frequencies, impedance_data):
    log_pos = np.log10(freq)
    log_width = 0.05  # adjust this to control box spacing in log scale
    left = 10**(log_pos - log_width / 2)
    right = 10**(log_pos + log_width / 2)
    center = 10**log_pos
    plt.boxplot(data, positions=[center], widths=[right - left],
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='skyblue', alpha=0.7))

# Plot mean points
means = [np.mean(vals) for vals in impedance_data]
plt.scatter(frequencies, means, color='red', s=60, zorder=3, label='Mean')

# Log-log scale
plt.xscale('log')
plt.yscale('log')
plt.xlim(1, 1.5e4)
plt.ylim(1, 1e6)

# Labeling
plt.xticks(frequencies, [f"{f:.0f}Hz" if f >= 10 else f"{f:.3f}Hz" for f in frequencies], rotation=45)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Impedance (kΩ)')
plt.title('NanoZ Impedance Boxplots per Frequency (Log–Log)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()
