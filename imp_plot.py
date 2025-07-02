import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter
# Reload the Excel file
file_path = 'c:/Hyperstim/2025_07_01/nanoz_imp.xlsx'
df = pd.read_excel(file_path)

# Extract frequencies and impedance data
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

# Sort by frequency
sorted_idx = np.argsort(frequencies)
frequencies = np.array(frequencies)[sorted_idx]
impedance_data = [impedance_data[i] for i in sorted_idx]

# Convert frequencies to log10 for plotting on a linear axis
log_freqs = np.log10(frequencies)
labels = [f"{f:.0f}Hz" if f >= 10 else f"{f:.3f}Hz" for f in frequencies]

# Plot using linear scale (manual log x-axis)
plt.figure(figsize=(10, 6))
positions = log_freqs

# Boxplots at log-scaled x positions
for pos, data in zip(positions, impedance_data):
    plt.boxplot(data, positions=[pos], widths=0.12,
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='skyblue', alpha=0.7))

# Overlay means
means = [np.mean(d) for d in impedance_data]
plt.scatter(positions, means, color='red', label='Mean', zorder=3, s=15)

# Manual log x-ticks
plt.xticks(positions, labels, rotation=45)
plt.xlabel('Frequency (Hz)')
plt.gca().invert_yaxis()
plt.ylabel('Impedance (kΩ)')
plt.title('NanoZ N17_P3')

ax = plt.gca()
# Logarithmic scale for Y-axis
ax.set_yscale('log')
# Major ticks at 10^0 to 10^6
ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=7))
ax.yaxis.set_major_formatter(FormatStrFormatter("10^%d"))
# Optional: hide minor ticks completely
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[], numticks=0))
ax.yaxis.set_minor_formatter(NullFormatter())
# Grid only at major ticks
ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.6)

# Correct Y-limits
ax.set_ylim(1e0, 1e6)
plt.legend()
plt.tight_layout()
plt.show()
