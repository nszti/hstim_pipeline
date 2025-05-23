import numpy as np
import matplotlib.pyplot as plt
# Given coordinates of the ROIs
import os
'''
coords = np.array([
    [273, 142],
    [235, 368],
    [191, 88],
    [247, 178],
    [401, 166],
    [437, 88],
    [310, 107],
    [358, 270],
    [194, 23],
    [389, 331],
    [483, 114],
    [199, 72],
    [391, 152]


])

    #36
    [273, 142],
    [191, 88],
    [268, 446],
    [247, 178],
    [401, 166],
    [437, 88],
    [310, 107],
    [194, 23],
    [203, 75],
    [321, 198],
    [317, 258],
    [483, 114],
    [199, 72]
    #37
    [193, 354],
    [216, 170],
    [273, 142],
    [235, 368],
    [225, 254],
    [114, 379],
    [313, 317],
    [408, 152],
    [273, 300],
    [131, 310],
    [292, 176],
    [40, 487],
    [203, 361],
    [323, 221],
    [321, 342],
    [103, 432]
    #38
[235, 368],
    [191, 88],
    [268, 446],
    [247, 178],
    [401, 166],
    [437, 88],
    [310, 107],
    [194, 23],
    [321, 198],
    [483, 114],
    [391, 152],
    #39
    [273, 142],
    [235, 368],
    [191, 88],
    [247, 178],
    [401, 166],
    [358, 270],
    [194, 23],
    [321, 342],
    [483, 114],
    [391, 152],
    #40
    [216, 170],
    [139, 174],
    [191, 88],
    [268, 446],
    [247, 178],
    [401, 166],
    [437, 88],
    [310, 107],
    [358, 270],
    [194, 23],
    [321, 198],
    [40, 487],
    [483, 114],
    [245, 48],
    [391, 152],
    #41
    [193, 354],
    [216, 170],
    [273, 142],
    [235, 368],
    [225, 254],
    [114, 379],
    [313, 317],
    [408, 152],
    [131, 310],
    [292, 176],
    [40, 487],
    [389, 331],
    [323, 221],
    [321, 342]
    #42
     [273, 142],
    [139, 174],
    [191, 88],
    [268, 446],
    [247, 178],
    [401, 166],
    [437, 88],
    [310, 107],
    [358, 270],
    [194, 23],
    [321, 198],
    [203, 105],
    [483, 114],
    [139, 170],
    [199, 72],
    [245, 48],
    [391, 152]
    #43
    [193, 354],
    [216, 170],
    [273, 142],
    [235, 368],
    [191, 88],
    [225, 254],
    [247, 178],
    [194, 23],
    [131, 310],
    [292, 176],
    [321, 342]
    #44
    [193, 354],
    [216, 170],
    [273, 142],
    [225, 254],
    [114, 379],
    [313, 317],
    [408, 152],
    [131, 310],
    [292, 176],
    [40, 487],
    [389, 331],
    [203, 361],
    [323, 221],
    [182, 180],
    [321, 342],
    [103, 432]
    #45
    [193, 354],
    [216, 170],
    [273, 142],
    [191, 88],
    [225, 254],
    [247, 178],
    [114, 379],
    [313, 317],
    [408, 152],
    [131, 310],
    [292, 176],
    [389, 331],
    [182, 180],
    [321, 342],
    [103, 432]
    #46




mean_center = np.mean(coords, axis=0)
mean_x, mean_y = mean_center
print(f"2: Mean center of mass (average of coordinates): ({mean_x}, {mean_y})")
# Define a function for inverse distance weighting
def inverse_distance_weighted_center_of_mass(coords, alpha=2, center_of_mass=(0, 0)):
    x0, y0 = center_of_mass
    # Calculate distances from each point to the center
    distances = np.sqrt((coords[:, 0] - x0) ** 2 + (coords[:, 1] - y0) ** 2)

    # Calculate weights as the inverse of the distances raised to the power of alpha
    weights = 1 / (distances ** alpha)

    # Normalize the weights
    weights /= np.sum(weights)

    # Calculate the weighted center of mass
    x_cm = np.sum(weights * coords[:, 0])
    y_cm = np.sum(weights * coords[:, 1])

    return x_cm, y_cm


# Calculate the weighted center of mass using the inverse distance weighting
x_cm, y_cm = inverse_distance_weighted_center_of_mass(coords)
print(f"1: Weighted Center of Mass w center(0,0): ({x_cm}, {y_cm})")

#================

# 1: Calculate the mean center of the given coordinates (simple average of X and Y coordinates)
mean_center = np.mean(coords, axis=0)
mean_x, mean_y = mean_center
#print(f"2: Mean center of mass (average of coordinates): ({mean_x}, {mean_y})")


# 2: Calculate the inverse distance weighted center of mass using the mean center as reference
def inverse_distance_weighted_center_of_mass(coords, center_of_mass, alpha=2):
    x0, y0 = center_of_mass
    # Calculate distances from each point to the center (mean center in this case)
    distances = np.sqrt((coords[:, 0] - x0) ** 2 + (coords[:, 1] - y0) ** 2)

    # Handle division by zero (if a point coincides with the center, set weight to a large value)
    distances = np.where(distances == 0, 1e-6, distances)  # Avoid zero distance

    # Calculate weights as the inverse of the distances raised to the power of alpha
    weights = 1 / (distances ** alpha)

    # Normalize the weights so that they sum to 1
    weights /= np.sum(weights)

    # Calculate the weighted center of mass
    x_cm = np.sum(weights * coords[:, 0])
    y_cm = np.sum(weights * coords[:, 1])

    return x_cm, y_cm


# 3: Use the mean center to calculate the weighted center of mass
x_cm, y_cm = inverse_distance_weighted_center_of_mass(coords, mean_center)
print(f"2: Weighted Center of Mass: ({x_cm}, {y_cm})")


#================

'''

'''files_data = [
    {'mean': [295.692307692307, 150.384615384615], 'type1': [246.6300,	107.5317891], 'type2': [283.560931,	144.6394872]},  # File 36
{'mean': [282.2941176,	149.4705882], 'type1': [230.8154,	119.1702], 'type2': [273.9856,	142.3939]},  # File 42
{'mean': [282.7333333,	180.6], 'type1': [241.5932,	133.3360], 'type2': [281.1639,	171.8521]},  # File 40
{'mean': [316.1818182,	175.2727273], 'type1': [266.2077,	125.0842], 'type2': [322.2270,	181.6071]},  # File 38
{'mean': [316.0, 161.46153846153845], 'type1': [259.6339300691869, 116.29589902260454],'type2': [309.90374383921676, 146.45366662440972]},  # File 46
{'mean': [309.4,	184.3], 'type1': [257.3940,	130.5072], 'type2': [303.9162,	173.3317]},  # File 39
{'mean': [228.9090909090909, 218.63636363636363], 'type1': [217.6865775689308, 162.46127186374488], 'type2': [233.95092565028241, 211.08493761258342]},  # File 43
{'mean': [239.86666666666667, 253.66666666666666], 'type1': [224.2307561290624, 214.08267863967583],'type2': [228.50063902899498, 249.87156922669328]}, # File 45
{'mean': [248.0714286,	285.9285714], 'type1': [241.6830,	261.9804], 'type2': [248.8401,	280.1693]},  # File 41
    {'mean': [228.9375,	297.8125], 'type1': [229.8641,	274.7318], 'type2': [237.5335,	301.8662]},  # File 37
    {'mean': [232.875, 288.0], 'type1': [226.39342150598156, 258.8626434215348],'type2': [231.8473226253097, 278.11331512394787]}  # File 44/11


]'''
in_um = 1.07
files_data = [
    {'type2': [283.560931*in_um,	144.6394872*in_um]},
{'type2': [273.9856*in_um,	142.3939*in_um]},
{'type2': [281.1639*in_um,	171.8521*in_um]},
{'type2': [322.2270*in_um,	181.6071*in_um]},
{'type2': [309.90374383921676*in_um, 146.45366662440972*in_um]},
{'type2': [303.9162*in_um,	173.3317*in_um]},
{'type2': [233.95092565028241*in_um, 211.08493761258342*in_um]},
{'type2': [228.50063902899498*in_um, 249.87156922669328*in_um]},
{'type2': [248.8401*in_um,	280.1693*in_um]},
    {'type2': [237.5335*in_um,	301.8662*in_um]},
    {'type2': [231.8473226253097*in_um, 278.11331512394787*in_um]}
]
# Create a figure to hold the plot
plt.figure(figsize=(8, 8))

# Loop through each file data and plot mean, type1 CoM, and type2 CoM
for i, data in enumerate(files_data):
    # Extract mean, type1, type2 CoM coordinates for each file
    #mean_x, mean_y = data['mean']
    #type1_x, type1_y = data['type1']
    type2_x, type2_y = data['type2']

    '''plt.scatter(mean_x, mean_y, color='blue', marker='o' )
    plt.scatter(type1_x, type1_y, color='green', marker='x')
    plt.scatter(type2_x, type2_y, color='red', marker='^')'''
    #plt.scatter(mean_x, mean_y, color='blue', marker='o', label=f'File {i + 1} - Mean')
    #plt.scatter(type1_x, type1_y, color='green', marker='x', label=f'File {i + 1} - Type 1 CoM')
    plt.scatter(type2_x, type2_y, color='red', marker='^', label=f'File {i + 1}')

    # Label the file number above each marker
    #plt.text(mean_x, mean_y + 5, f'File {i + 1}', color='blue', fontsize=9, ha='center')
    #plt.text(type1_x, type1_y + 5, f'File {i + 1}', color='green', fontsize=9, ha='center')
    plt.text(type2_x, type2_y + 5, f'File {i + 1}', color='red', fontsize=9, ha='center')

# Set plot limits (assuming 512x512 FOV)
plt.xlim(0, 512*in_um)
plt.ylim(0, 512*in_um)

# Add title and labels
plt.title("Plotting Mean and CoM Coordinates from 6 Files on 512x512 FOV")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")

# Add grid and legend
plt.grid(True)
plt.gca().invert_yaxis()  # Invert y-axis to match image coordinate system (upper left is (0, 0))
plt.legend()

# Show plot

plt.savefig(os.path.join('c:/Hyperstim/data_analysis/2025-04-15-Amouse-invivo-GCaMP6f/merged_tiffs/cellreg_files/36_37_38_39_40_41_42_43_44_45_46_ordered/', f'CoM_mean_weighted.svg'))
plt.show()