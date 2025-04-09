import numpy as np
import h5py
from scipy import io

# === Load data ===
'''mat = loadmat('c:/Hyperstim/data_analysis/2023_09_25_GCAMP6F/merged_tiffs/cellreg_files/cellRegistered_20250404_115715.mat')  # Path to your CellReg output
cell_to_index_map = mat['cell_registered_struct']['cell_to_index_map'][0, 0]'''
#f = h5py.File('c:/Hyperstim/data_analysis/2023_09_25_GCAMP6F/merged_tiffs/cellreg_files/cellRegistered_20250404_115715.mat','r')
with h5py.File('x:/cellRegistered_20250404_115825.mat', 'r') as file:
    # List all top-level keys in the .mat file
    #print(list(file.keys()))
    # Access a specific dataset
    data = file['cell_registered_struct']['cell_to_index_map'][:][:]
    print(data)
'''data = hdf5storage.loadmat('cell_registered_struct.mat')
cell_to_index_map = data['cell_registered_struct']['cell_to_index_map']
cell_to_index_map = np.array(cell_to_index_map, dtype=int)'''
#mat = io.loadmat('c:/Hyperstim/data_analysis/2023_09_25_GCAMP6F/merged_tiffs/cellreg_files/cellRegistered_20250404_115715.mat')
#cell_to_index_map = np.array(data['cell_registered_struct']['cell_to_index_map'], dtype=int)

num_sessions, num_cells  = data.shape
print(num_cells, num_sessions)
#Ttotal number of detected cells in each session
# max cell index in each session
total_cells_per_session = np.max(data, axis=0)

#print("Total number of detected cells:")
#for i, count in enumerate(total_cells_per_session):
    #print(f"  Session {i+1}: {int(count)} cells")

#number of overlapping cells between session pairs
#print("\nNumber of overlapping cells between session pairs:")
'''for i in range(num_sessions):
    for j in range(i + 1, num_registered_cells):
        overlap_mask = np.logical_and(data[:, i] > 0, data[:, j] > 0)
        print(overlap_mask)
        num_overlap = np.sum(overlap_mask)
        #print(f"  Sessions {i} & {i+1}: {num_overlap} overlapping cells")'''

'''for column in data[:,]:
    #print(column)
    for i in range(len(column)):
        #print(f'i:{i+1}')
        print(column[i])
        for j in range(data[:,]):
            #print(f'j:{j + 1}')
            #print(column)
            if column[i] > 0 and column[j] > 0:
                #print(row[i], row[j])
                session_pair_counts[i,j] += 1
for i in range(num_sessions):
    for j in range(i+1, num_sessions):
        print(f'overlapping cells in session {i+1} & {j+1}: {session_pair_counts[i,j]}  ')'''




for i in range(num_sessions):
    for j in range(i + 1, num_sessions):
        holder = []
        #holder = 0

        for row in range(num_cells):
            if data[i][row] > 0 and data[j][row] > 0:
                holder.append(True)  # or append row itself if you want to store matched neurons
                #holder += 1
        # Count how many matches were found for this session pair
        num_matches = len(holder)
        print(f"Sessions {i+1} & {j+1}: {num_matches} overlapping cells")




