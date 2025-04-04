import scipy.io as sio
import numpy as np

# === Load data ===
mat = sio.loadmat('cell_registered_struct.mat')  # Path to your CellReg output
cell_to_index_map = mat['cell_registered_struct']['cell_to_index_map'][0, 0]

num_registered_cells, num_sessions = cell_to_index_map.shape

#Ttotal number of detected cells in each session
# max cell index in each session
total_cells_per_session = np.max(cell_to_index_map, axis=0)

print("Total number of detected cells:")
for i, count in enumerate(total_cells_per_session):
    print(f"  Session {i+1}: {int(count)} cells")

#number of overlapping cells between session pairs
print("\nNumber of overlapping cells between session pairs:")
for i in range(num_sessions):
    for j in range(i + 1, num_sessions):
        overlap_mask = np.logical_and(cell_to_index_map[:, i] > 0,
                                      cell_to_index_map[:, j] > 0)
        num_overlap = np.sum(overlap_mask)
        print(f"  Sessions {i+1} & {j+1}: {num_overlap} overlapping cells")
