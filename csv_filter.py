import pandas as pd
import os

'''# New files uploaded to process
new_input_files = [

    "c:/Hyperstim/2025_07_01/N17_P10/N17_P10_7500Hz.csv",
    "c:/Hyperstim/2025_07_01/N17_P10/N17_P10_5000Hz.csv",
    "c:/Hyperstim/2025_07_01/N17_P10/N17_P10_2000Hz.csv",
"c:/Hyperstim/2025_07_01/N17_P10/N17_P10_1000Hz.csv",
"c:/Hyperstim/2025_07_01/N17_P10/N17_P10_500Hz.csv",
"c:/Hyperstim/2025_07_01/N17_P10/N17_P10_200Hz.csv",
"c:/Hyperstim/2025_07_01/N17_P10/N17_P10_100Hz.csv",
"c:/Hyperstim/2025_07_01/N17_P10/N17_P10_50Hz.csv",
"c:/Hyperstim/2025_07_01/N17_P10/N17_P10_20Hz.csv",
"c:/Hyperstim/2025_07_01/N17_P10/N17_P10_10Hz.csv",
"c:/Hyperstim/2025_07_01/N17_P10/N17_P10_5Hz.csv",
"c:/Hyperstim/2025_07_01/N17_P10/N17_P10_2Hz.csv",
#"c:/Hyperstim/2025_07_01/N17_P10/N17_P10_1Hz.csv"
]

# Process and save filtered files for the new uploads
new_filtered_file_paths = []
for file_path in new_input_files:
    df = pd.read_csv(file_path)
    # Identify the correct frequency in the column names
    available_columns = df.columns
    freq = [col for col in available_columns if "Impedance Magnitude at" in col][0].split(" ")[-3]
    magnitude_col = f'Impedance Magnitude at {freq} Hz (ohms)'
    phase_col = f'Impedance Phase at {freq} Hz (degrees)'

    filtered_df = df[['Channel Number', magnitude_col, phase_col]]

    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    filtered_file_path = f"c:/Hyperstim/2025_07_01/N17_P10/filtered/{name}_filtered{ext}"
    filtered_df.to_csv(filtered_file_path, index=False)
    new_filtered_file_paths.append(filtered_file_path)

new_filtered_file_paths

"c:/Hyperstim/2025_07_01/N17_P10/N17_P10_7500Hz.csv",
    "c:/Hyperstim/2025_07_01/N17_P10/N17_P10_5000Hz.csv",
    "c:/Hyperstim/2025_07_01/N17_P10/N17_P10_2000Hz.csv",
"c:/Hyperstim/2025_07_01/N17_P10/N17_P10_1000Hz.csv",
"c:/Hyperstim/2025_07_01/N17_P10/N17_P10_500Hz.csv",
"c:/Hyperstim/2025_07_01/N17_P10/N17_P10_200Hz.csv",
"c:/Hyperstim/2025_07_01/N17_P10/N17_P10_100Hz.csv",
"c:/Hyperstim/2025_07_01/N17_P10/N17_P10_50Hz.csv",
"c:/Hyperstim/2025_07_01/N17_P10/N17_P10_20Hz.csv",
"c:/Hyperstim/2025_07_01/N17_P10/N17_P10_10Hz.csv",
"c:/Hyperstim/2025_07_01/N17_P10/N17_P10_5Hz.csv",
"c:/Hyperstim/2025_07_01/N17_P10/N17_P10_2Hz.csv",
'''

##=================


# Define the threshold for high impedance: 200,000 ohms (200KΩ)
impedance_threshold = 1_000_000

# Combine all previously filtered file paths
all_filtered_files = [
"c:/Hyperstim/2025_07_01/N17_P7/N17_P7_1Hz_nanoz.xlsx"
]

# Process each file and filter out rows with impedance > 200KΩ
filtered_low_impedance_paths = []

for file_path in all_filtered_files:
    #df = pd.read_csv(file_path)
    df = pd.read_excel
    # Identify the impedance magnitude column dynamically
    impedance_col = [col for col in df.columns if "Impedance Magnitude at" in col][0]
    filtered_df = df[df[impedance_col] <= impedance_threshold]

    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    filtered_file_path = f"c:/Hyperstim/2025_07_01/N17_P7/low_impedance/{name}_lowImpedance{ext}"
    filtered_df.to_csv(filtered_file_path, index=False)
    filtered_low_impedance_paths.append(filtered_file_path)

filtered_low_impedance_paths
