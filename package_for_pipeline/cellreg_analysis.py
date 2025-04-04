import scipy.io as sio
import hdf5storage
import numpy as np

data = hdf5storage.loadmat('cell_registered_struct.mat')
cell_to_index_map = data['cell_registered_struct']['cell_to_index_map']
cell_to_index_map = np.array(cell_to_index_map, dtype=int)
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

