import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import LogLocator, NullFormatter

# Reload the Excel file
file_path = 'c:/Hyperstim/2025_07_01/nanoz_m_imp.xlsx'
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
labels = [f"{f:.0f}Hz" if f >= 10 else f"{f:.1f}Hz" for f in frequencies]

# Plot using linear scale (manual log x-axis)
#plt.figure(figsize=(10, 6))
fig, ax = plt.subplots()
positions = log_freqs

# Boxplots at log-scaled x positions
for pos, data in zip(positions, impedance_data):
    ax.boxplot(data, positions=[pos], widths=0.12,
                patch_artist=True, showfliers=True, whis = [0, 100],
                boxprops=dict(facecolor='lightblue', alpha=0.7), medianprops=dict(color = 'red')
    )

# Overlay means
means = [np.mean(d) for d in impedance_data]
plt.scatter(positions, means, color='black', marker = 'x', label='Mean', zorder=3, s=15)

# Manual log x-ticks
plt.xticks(positions, labels, rotation=45)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Impedance (KΩ)')
plt.title('Nanoz magnitude values')
plt.yscale('log')
plt.ylim(1e0, 1e4)
#ax.set_yscale('log')
#plt.grid(True, which='both', linestyle='-', linewidth=0.5)
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()

# ======= rms boxplot =======
'''
df = pd.read_excel(file_path)

# Create the plot
plt.figure(figsize=(6, 6))
ax = sns.boxplot(x='1-500', y='Noise (μV RMS)', data=df, showfliers=False, width=0.6)
sns.stripplot(x='1-500', y='Noise (μV RMS)', data=df, color='black', size=4, jitter=True, alpha=0.5)


grouped = df.groupby('1-500')['Noise (μV RMS)']
positions = range(len(grouped))
for pos, (label, values) in zip(positions, grouped):
    mean = values.mean()
    median = values.median()
    plt.plot(pos, mean, 'ko', markersize=8)
    plt.plot([pos - 0.2, pos + 0.2], [median, median], color='red', lw=2)

plt.ylabel("Noise (μV$_{RMS}$)", fontsize=12)
plt.title("B", loc='left', fontsize=14, weight='bold')
sns.despine()
plt.tight_layout()
plt.show()'''
