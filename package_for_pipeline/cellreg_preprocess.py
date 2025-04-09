import os.path
import numpy as np
from scipy.io import savemat
from pathlib import Path
import h5py
import pandas as pd

def suite2p_to_cellreg_masks(expDir, list_of_file_nums):
    base_dir = Path(expDir)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]
    for numbers_to_merge in list_of_file_nums:
        suffix = '_'.join(map(str, numbers_to_merge))
        num_to_search = []
        for dir in filenames:
            num_to_search_split = dir.split('MUnit_')
            # print(num_to_search_split)
            if len(num_to_search_split) > 1:
                file_suffix = num_to_search_split[1].rsplit('.', 1)[0]
                if file_suffix == suffix:
                    matched_file = dir
                    print(f'Found file: {matched_file}')
                    # print(matched_file)
                    break
        else:
            continue

        if matched_file:
            stat_path = expDir + dir + '/suite2p/plane0/stat.npy'
            ops_path = expDir + dir + '/suite2p/plane0/ops.npy'
            iscell_path = expDir + dir + '/suite2p/plane0/iscell.npy'
            stat = np.load(stat_path, allow_pickle=True)
            ops = np.load(ops_path, allow_pickle=True).item()
            iscell = np.load(iscell_path, allow_pickle=True)

            cell_indices = np.where(iscell[:, 0] == 1)[0]
            stat = [stat[i] for i in cell_indices]


            filtered_stat = []
            for roi in stat:
                y, x = roi['med']
                if x > 1 and y > 1:  # Exclude ROIs too close to the image edge (which may lead to 0 distances)
                    filtered_stat.append(roi)
                else:
                    continue

            Ly, Lx = ops['Ly'], ops['Lx']
            masks = []
            for roi in filtered_stat:
                mask = np.zeros((Ly, Lx), dtype=np.uint8)
                ypix = roi['ypix']
                xpix = roi['xpix']
                mask[ypix, xpix] = 1
                masks.append(mask)
            if masks:
                mask_stack = np.stack(masks, axis=-0).astype(np.double)   # [nROIs, Ly, Lx]
                output_folder = os.path.join(expDir, 'cellreg_files')
                os.makedirs(output_folder, exist_ok=True)
                out_name = f'{matched_file}.mat'
                out_path = os.path.join(output_folder, out_name)
                savemat(out_path, {'cells_map': mask_stack})
                print(f" Saved: {out_path} with shape {mask_stack.shape}")

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
            # Count how many matches were found for this session pair
            num_matches = len(holder)
            # Print the number of matches for this session pair
            print(f"Sessions {i+1} & {j+1}: {num_matches} overlapping cells")

    print("Overlap matrix saved as overlap_matrix.csv")