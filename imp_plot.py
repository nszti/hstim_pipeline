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
file_path = '/mnt/data/nanoz_imp.xlsx'
df = pd.read_excel(file_path)

# Extract numeric frequencies from column names
frequencies = []
impedance_lists = []

for col in df.columns:
    freq_str = col.replace('Hz', '').replace('E', 'e')  # handle '1E4Hz'
    try:
        freq = float(freq_str)
        values = df[col].dropna().values
        frequencies.append(freq)
        impedance_lists.append(values)
    except ValueError:
        continue  # skip non-numeric columns

# Sort by frequency for plotting
sorted_indices = np.argsort(frequencies)
frequencies = np.array(frequencies)[sorted_indices]
impedance_lists = [impedance_lists[i] for i in sorted_indices]

# Plot
plt.figure(figsize=(12, 6))

# Create boxplots at actual frequency positions
positions = frequencies
bp = plt.boxplot(impedance_lists, positions=positions, widths=0.1 * positions,
                 patch_artist=True, showfliers=False)

# Overlay mean values
means = [np.mean(vals) for vals in impedance_lists]
plt.scatter(positions, means, color='red', label='Mean', zorder=3)

# Set log-log scale
plt.xscale('log')
plt.yscale('log')
plt.xlim(1, 1e4)
plt.ylim(1, 1e6)

# Labels
plt.xlabel('Frequency (Hz)')
plt.ylabel('Impedance (kΩ)')
plt.title('NanoZ Impedance Boxplots')
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.show()
