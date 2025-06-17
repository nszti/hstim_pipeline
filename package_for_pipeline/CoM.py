import numpy as np
import matplotlib.pyplot as plt
import os


def inverse_distance_weighted_center_of_mass(coords, alpha=2, center_of_mass=(0, 0)):
    x0, y0 = center_of_mass
    # Calculate distances from each point to the predefined center
    distances = np.sqrt((coords[:, 0] - x0) ** 2 + (coords[:, 1] - y0) ** 2)
    weights = 1 / (distances ** alpha)
    weights /= np.sum(weights)
    x_cm = np.sum(weights * coords[:, 0])
    y_cm = np.sum(weights * coords[:, 1])
    return x_cm, y_cm

def plot_weighted_com(coords_list, plot_order, output_path, in_um=1.07):
    """
    Plots the inverse-distance weighted center of mass (CoM) for a list of coordinate arrays.


    Parameters:
    - coords_list: List of np.array, each containing (x, y) coordinates for one file.
    - plot_order: List of integers indicating the order in which to plot files.
        Example [2, 0, 1] means: plot File 3 first, then File 1, then File 2 --> give the index of the file in the coord_list
    - output_path: Path where the plot will be saved.
    - in_um: Scaling factor from pixels to micrometers.
    """
    # Calculate weighted CoMs
    com_list = [inverse_distance_weighted_center_of_mass(coords) for coords in coords_list]
    com_list_um = np.array(com_list) * in_um

    # Plot
    plt.figure(figsize=(8, 8))

    for order_position, file_idx in enumerate(plot_order):
        x_um, y_um = com_list_um[file_idx]
        plt.scatter(x_um, y_um, color='red', marker='^')
        plt.text(x_um, y_um - 10, f'File {order_position + 1}', color='black', fontsize=7, ha='center')

    # Plot limits (for 512x512 FOV)
    plt.xlim(0, 512 * in_um)
    plt.ylim(0, 512 * in_um)
    plt.title("CoM Coordinates")
    plt.xlabel("X Coordinate (\µm)")
    plt.ylabel("Y Coordinate (\µm)")
    plt.grid(True)
    plt.gca().invert_yaxis()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

