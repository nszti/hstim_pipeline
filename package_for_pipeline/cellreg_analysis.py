import numpy as np
import h5py
from scipy import io

def cellreg_analysis(expDir, mat_file):
    # === Load data ===
    cell_reg_path = output_folder = os.path.join(expDir, 'cellreg_files/')
    input_file = os.path.join(cell_reg_path, mat_file)
    with h5py.File(input_file, 'r') as file:
        # List all top-level keys in the .mat file
        #print(list(file.keys()))
        # nested list of cell_to_index_map
        data = file['cell_registered_struct']['cell_to_index_map'][:][:]
    num_sessions, num_cells  = data.shape
    print(num_cells, num_sessions)
    total_cells_per_session = np.max(data, axis=0)

    result_mtx = np.zeros((num_sessions, num_cells))
    for i in range(num_sessions):
        for j in range(i + 1, num_sessions):
            holder = []
            for row in range(num_cells):
                if data[i][row] > 0 and data[j][row] > 0:
                    holder.append(True)
            result_mtx[i][j] = len(holder)
            result_mtx[j][i] = len(holder)
            # Count how many matches were found for this session pair
            num_matches = len(holder)
            # Print the number of matches for this session pair
            print(f"Sessions {i+1} & {j+1}: {num_matches} overlapping cells")
    df = pd.DataFrame(result_mtx, columns=[f'Session {i+1}' for i in range(num_sessions)])
    csv_path = os.path.join(cell_reg_path, 'overlap_matrix.csv')
    df.to_csv(csv_path)
print("Overlap matrix saved as overlap_matrix.csv")




