import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import LogLocator, NullFormatter

def rms_load(file_path):

    # Reload the Excel file
    file_path = file_path
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
    #plt.ylabel('Phase (°)')
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
def plot_single_boxplot_lfp(file_path_1,file_path_2, file_path_3, label='Condition A', label_2 = 'Condition A', label_3 = 'Condition A'):
    # Load Excel file (only one column of numbers)
    df = pd.read_excel(file_path_1, header=None)
    df.columns = ['Noise (μV RMS)']
    #df['Condition'] = condition_label

    plt.figure(figsize=(4, 6))
    sns.color_palette('pastel')
    sns.boxplot(
        #x=' ',
        y='Noise (μV RMS)',
        data=df,
        width=0.12,
        showfliers=True,
        whis=(0, 100),
        color='white'
    )
    sns.stripplot(
        #x=' ',
        y='Noise (μV RMS)',
        data=df,
        color = 'orange',
        size = 4,
        jitter = 0.03,
        alpha = 0.7

    )
    # Add mean and median
    mean_val = df['Noise (μV RMS)'].mean()
    median_val = df['Noise (μV RMS)'].median()
    plt.plot(0, mean_val, 'ko', markersize=5, zorder=10)
    plt.plot([-0.06, 0.06], [median_val, median_val], color='red', linewidth=2, zorder=3)
    plt.xlabel(label)

    plt.ylim(1.2, 4.2)
    plt.ylabel("Noise (μV$_{RMS}$)", fontsize=12)
    plt.title("lfp")
    sns.despine()
    plt.tight_layout()


    # ==== ua ===

    df = pd.read_excel(file_path_2, header=None)
    df.columns = ['Noise (μV RMS)']
    #df['Condition'] = condition_label

    plt.figure(figsize=(4, 6))
    sns.color_palette('pastel')
    sns.boxplot(
        #x=' ',
        y='Noise (μV RMS)',
        data=df,
        width=0.12,
        showfliers=True,
        whis= (0,100),
        color='white'
    )
    sns.stripplot(
        #x=' ',
        y='Noise (μV RMS)',
        data=df,
        color = 'blue',
        size = 4,
        jitter = 0.03,
        alpha = 0.7

    )
    # Add mean and median
    mean_val = df['Noise (μV RMS)'].mean()
    median_val = df['Noise (μV RMS)'].median()
    plt.plot(0, mean_val, 'ko', markersize=5, zorder=10)
    plt.plot([-0.06, 0.06], [median_val, median_val], color='red', linewidth=2, zorder=3)
    plt.xlabel(label_2)

    plt.ylim(1.2, 4.2)
    plt.ylabel("Noise (μV$_{RMS}$)", fontsize=12)
    plt.title("ua")
    sns.despine()
    plt.tight_layout()


    # ==== in vivo ===

    df = pd.read_excel(file_path_3, header=None)
    df.columns = ['Noise (μV RMS)']
    # df['Condition'] = condition_label

    plt.figure(figsize=(4, 6))
    sns.color_palette('pastel')
    sns.boxplot(
        # x=' ',
        y='Noise (μV RMS)',
        data=df,
        width=0.12,
        showfliers=True,
        whis=(0, 100),
        color='white'
    )
    sns.stripplot(
        # x=' ',
        y='Noise (μV RMS)',
        data=df,
        color='green',
        size=4,
        jitter=0.03,
        alpha=0.7

    )
    # Add mean and median
    mean_val = df['Noise (μV RMS)'].mean()
    median_val = df['Noise (μV RMS)'].median()
    plt.plot(0, mean_val, 'ko', markersize=5, zorder=10)
    plt.plot([-0.06, 0.06], [median_val, median_val], color='red', linewidth=2, zorder=3)
    plt.xlabel(label_3)

    plt.ylim(1.2, 4.2)
    plt.ylabel("Noise (μV$_{RMS}$)", fontsize=12)
    plt.title("In vivo")
    sns.despine()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    rms_load("c:/Hyperstim/2025_07_01/nanoz_m_imp.xlsx")
    #plot_single_boxplot_lfp("c:/Hyperstim/2025_07_01/collect/rms_lfp.xlsx", "c:/Hyperstim/2025_07_01/collect/rms_ua.xlsx", "d:/kell/Fig2_in_vivo_RMS_noise/in_vivo_RMS.xlsx", label ='In vitro \n (1-500 Hz)' , label_2='In vitro \n (500-5000 Hz)', label_3= 'In vivo')


