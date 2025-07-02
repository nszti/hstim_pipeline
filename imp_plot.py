import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file
file_path = 'nanoz_data.xlsx'  # Change if needed
df = pd.read_excel(file_path)

# Prepare long-format data with Frequency + Type (M/P)
long_data = []

for col in df.columns:
    if '_M' in col or '_P' in col:
        freq = col.split('Hz')[0]
        kind = 'Magnitude' if '_M' in col else 'Phase'
        for val in df[col].dropna():
            long_data.append({
                'Frequency': freq,
                'Type': kind,
                'Value': val,
                'Label': f"{freq}Hz_{kind[0]}"  # e.g., "1Hz_M" or "1Hz_P"
            })

# Create DataFrame
long_df = pd.DataFrame(long_data)
long_df['Label'] = long_df['Label'].astype(str)


# Create a custom x-label order: each freq appears twice (M then P)
# This helps with ordering in the plot
frequency_order = ['1E4','7500','5000','2000','1000', '500','200', '100', '50', '20','10','2',  '5',  '1']

x_order = [f"{str(f)}Hz_M" for f in frequency_order] + [f"{str(f)}Hz_P" for f in frequency_order]


# Set up plot
plt.figure(figsize=(16, 6))
sns.set(style="whitegrid")
sns.boxplot(x='Label', y='Value', data=long_df, order=x_order, palette='Set2')

# Style
plt.title('NanoZ Impedance: Magnitude and Phase per Frequency')
plt.xlabel('Frequency and Type')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
