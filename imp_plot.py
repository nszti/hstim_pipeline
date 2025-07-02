'''import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the Excel file
file_path = 'nanoz_data.xlsx'  # <-- update with your filename
df = pd.read_excel(file_path)

# Convert wide format to long format
long_df = df.melt(var_name='Frequency_Hz', value_name='Impedance_kOhm')

# Clean frequency labels and convert to numeric
long_df['Frequency_Hz'] = long_df['Frequency_Hz'].str.replace('Hz', '', regex=False)
long_df['Frequency_Hz'] = long_df['Frequency_Hz'].replace('1E4', '10000')  # if needed
long_df['Frequency_Hz'] = pd.to_numeric(long_df['Frequency_Hz'])

# Drop NaNs
long_df = long_df.dropna()

# Set up Seaborn style
sns.set(style="whitegrid")

# Create boxplot
plt.figure(figsize=(12, 6))
ax = sns.boxplot(
    x='Frequency_Hz',
    y='Impedance_kOhm',
    data=long_df,
    showfliers=False
)

# Overlay mean impedance per frequency
means = long_df.groupby('Frequency_Hz')['Impedance_kOhm'].mean().reset_index()
sns.scatterplot(
    x='Frequency_Hz',
    y='Impedance_kOhm',
    data=means,
    color='red',
    label='Mean',
    zorder=10,
    marker='o',
    s=70,
    ax=ax
)

# Logarithmic axes
plt.yscale('log')
plt.xscale('log')
plt.ylim(1e0, 1e6)
plt.xlim(1, 10000)

# Labels and styling
plt.title('Impedance Boxplots by Frequency (log-log)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Impedance (kΩ)')
plt.legend()
plt.tight_layout()
plt.show()
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load Excel file
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your Excel data
file_path = '/mnt/data/nanoz_imp.xlsx'
df = pd.read_excel(file_path)

# Extract frequency values and impedance lists
frequencies = []
impedance_lists = []

for col in df.columns:
    freq_str = col.replace('Hz', '').replace('E', 'e')
    try:
        freq = float(freq_str)
        values = df[col].dropna().values
        frequencies.append(freq)
        impedance_lists.append(values)
    except ValueError:
        continue

# Sort by frequency
freqs = np.array(frequencies)
sorted_indices = np.argsort(freqs)
freqs = freqs[sorted_indices]
impedance_lists = [impedance_lists[i] for i in sorted_indices]

# Create plot
plt.figure(figsize=(12, 6))

# Plot each box manually at the correct log-scaled x-position
for i, (freq, imp) in enumerate(zip(freqs, impedance_lists)):
    # Shift box width in log space
    box = plt.boxplot(imp,
                      positions=[freq],
                      widths=[freq * 0.2],  # scale width to freq
                      patch_artist=True,
                      showfliers=False)

    # Style the boxes
    for patch in box['boxes']:
        patch.set(facecolor='skyblue', alpha=0.7)

# Overlay mean values
means = [np.mean(x) for x in impedance_lists]
plt.scatter(freqs, means, color='red', s=80, label='Mean', zorder=3)

# Set log-log scale
plt.xscale('log')
plt.yscale('log')
plt.xlim(1, 1e4)
plt.ylim(1, 1e6)

# Labels and formatting
plt.xlabel('Frequency (Hz)')
plt.ylabel('Impedance (kΩ)')
plt.title('NanoZ Impedance Boxplots (Log–Log)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()

