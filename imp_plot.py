import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter
# Reload the Excel file
file_path = 'c:/Hyperstim/2025_07_01/intan_p7_m.xlsx'
df = pd.read_excel(file_path)

# Extract frequencies and impedance data
frequencies = []
impedance_data = []

for col in df.columns:
    try:
        freq = float(col.replace('Hz', '').replace('E', 'e'))
        values = df[col].dropna().values
        frequencies.append(freq)
        impedance_data.append(values/1000)
    except:
        continue

# Sort by frequency
sorted_idx = np.argsort(frequencies)
frequencies = np.array(frequencies)[sorted_idx]
impedance_data = [impedance_data[i] for i in sorted_idx]

# Convert frequencies to log10 for plotting on a linear axis
log_freqs = np.log10(frequencies)
labels = [f"{f:.0f}Hz" if f >= 10 else f"{f:.1f}Hz" for f in frequencies]

# Plot using linear scale (manual log x-axis)
plt.figure(figsize=(10, 6))
positions = log_freqs

# Boxplots at log-scaled x positions
for pos, data in zip(positions, impedance_data):
    plt.boxplot(data, positions=[pos], widths=0.12,
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='black', alpha=0.7))

# Overlay means
means = [np.mean(d) for d in impedance_data]
plt.scatter(positions, means, color='red', label='Mean', zorder=3, s=15)

# Manual log x-ticks
plt.xticks(positions, labels, rotation=45)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Impedance (kΩ)')
plt.title('Intan N17_P7')
plt.yscale('log')

#plt.xscale('log')
#plt.yticks([1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
#plt.gca().invert_yaxis()
'''plt.gca().yaxis.set_major_locator(LogLocator(base = 10.0, numticks = 10))
plt.gca().yaxis.set_minor_formatter(NullFormatter())'''
#plt.grid(True, which='both', linestyle='-', linewidth=0.5)
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()
