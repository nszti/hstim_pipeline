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
                print(mask_stack.shape)
                output_folder = os.path.join(expDir, 'cellreg_files')
                os.makedirs(output_folder, exist_ok=True)
                out_name = f'{matched_file}.mat'
                out_path = os.path.join(output_folder, out_name)
                savemat(out_path, {'cells_map': mask_stack})
                print(f" Saved: {out_path} with shape {mask_stack.shape}")

def cellreg_analysis(expDir, mat_file, list_of_file_nums, postfix):
    # === Load data ===
    stat_paths = [os.path.join(expDir, sess_dir, 'suite2p', 'plane0', 'stat.npy') for sess_dir in list_of_file_nums]
    all_stats = [np.load(p, allow_pickle=True) for p in stat_paths]
    cell_reg_path = output_folder = os.path.join(expDir, 'cellreg_files/')
    cell_reg_path_input = cell_reg_path + postfix
    input_file = os.path.join(cell_reg_path_input, mat_file)
    with h5py.File(input_file, 'r') as file:
        # List all top-level keys in the .mat file
        #print(list(file.keys()))
        # nested list of cell_to_index_map
        data = file['cell_registered_struct']['cell_to_index_map'][:][:]
    num_sessions, num_cells  = data.shape
    print(num_cells, num_sessions)
    total_cells_per_session = np.max(data, axis=0)

    '''
    result_rows = []
    for i in range(num_sessions):
        for j in range(i + 1, num_sessions):
            holder = []
            for row in range(num_cells):
                if data[i][row] > 0 and data[j][row] > 0:
                    holder.append(True)
            # Count how many matches were found for this session pair
            num_matches = len(holder)
            percent_overlap = (num_matches / num_cells) * 100
            # Print the number of matches for this session pair
            print(f"Sessions {i+1} & {j+1}: {num_matches} overlapping cells which is {percent_overlap:.2f}% of total cells")
            result_rows.append([f"Session {i + 1}", f"Session {j + 1}", num_matches, f"{percent_overlap:.2f}%"])
    result_rows.append(["Total cells registered", num_cells])
    csv_path = os.path.join(cell_reg_path, 'session_pair_overlap.csv')
    df = pd.DataFrame(result_rows, columns=['Session A', 'Session B', 'Number of Overlapping Cells', 'Overlap %'])
    df.to_csv(csv_path, index=False)
    print("Overlap matrix saved as overlap_matrix.csv")
    '''

    match_pairs = []

    for cellreg_idx in range(data.shape[0]):
        for i in range(data.shape[1]):
            roi_i = int(data[cellreg_idx, i])
            if roi_i <= 0:
                continue
            for j in range(i + 1, data.shape[1]):
                roi_j = int(data[cellreg_idx, j])
                if roi_j <= 0:
                    continue

                # med info from stat
                stat_i = all_stats[i][roi_i]
                stat_j = all_stats[j][roi_j]
                y_i, x_i = stat_i['med']
                y_j, x_j = stat_j['med']
                print(y_i, x_i)
                match_pairs.append({
                    'CellReg Index': cellreg_idx,
                    'Session A': i + 1,
                    'Cr ROI A': roi_i,
                    'med A': [float(y_i), float(x_i)], #(y,x) og suite2p format
                    'Session B': j + 1,
                    'Cr ROI B': roi_j,
                    'med B': [float(y_j), float(x_j)]
                })

    match_df = pd.DataFrame(match_pairs)
    csv_out = os.path.join(cell_reg_path, 'cellreg_matched_rois.csv')
    #match_df.to_csv(csv_out, index=False)
    #print(f"Matched ROI pairs saved to: {csv_out}")

def single_block_activation(expDir, postfix, mat_file, frame_rate, num_stims_per_repeat, list_of_file_nums, start_btw_stim, stim_dur, threshold_value):
    base_dir = Path(expDir)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]
    cell_reg_path = os.path.join(expDir, 'cellreg_files/')
    cell_reg_path_input = cell_reg_path + postfix
    input_file = os.path.join(cell_reg_path_input, mat_file)
    with h5py.File(input_file, 'r') as file:
        data = file['cell_registered_struct']['cell_to_index_map'][:][:]
        data = data.T  # transpose to [cell_reg_idx, session]
    num_cells, num_sessions = data.shape
    print(f"{num_cells} registered cells across {num_sessions} sessions")

    all_stats = {}
    session_counter = 0

    for numbers_to_merge in list_of_file_nums:
        suffix = '_'.join(map(str, numbers_to_merge))
        for dir in filenames:
            num_to_search_split = dir.split('MUnit_')
            if len(num_to_search_split) > 1:
                file_suffix = num_to_search_split[1].rsplit('.', 1)[0]
                if file_suffix == suffix:
                    matched_file = dir
                    stat_path = os.path.join(expDir, matched_file, 'suite2p', 'plane0', 'stat.npy')
                    stat_data = np.load(stat_path, allow_pickle=True)
                    all_stats[session_counter] = stat_data
                    session_counter += 1
                    print(f"Session {session_counter} -> {matched_file}")
                    break
        else:
            continue
        activated_roi_indices = []
        if matched_file:
            print(f"\nAnalyzing directory: {dir}")
            # Load required data
            F_path = expDir + dir + '/suite2p/plane0/F0.npy'
            iscelll_path = expDir + dir + '/suite2p/plane0/iscell.npy'
            stim_start_times_path = expDir + dir + '/stimTimes.npy'
            stat_path = expDir + dir + '/suite2p/plane0/stat.npy'
            ops_path = expDir + dir + '/suite2p/plane0/ops.npy'
            print(f"Loading data from: {F_path}")
            print(f"Loading stim times from: {stim_start_times_path}")

            F = np.load(F_path, allow_pickle=True)
            iscell = np.load(iscelll_path, allow_pickle=True)
            stim_start_times = np.load(stim_start_times_path, allow_pickle=True)
            stat = np.load(stat_path, allow_pickle=True)
            ops = np.load(ops_path, allow_pickle=True).item()
            # print(stat)

            # --------CALCULATIONS--------
            # Extract the ROI indexes for cells
            cell_indices = np.where(iscell[:, 0] == 1)[0]  # Get indices of valid ROIs
            stimulation_duration_frames = int(round((stim_dur / 1000) * frame_rate, 0))
            num_cells = len(cell_indices)
            start_btw_stim_frames = int(start_btw_stim * frame_rate)
            stim_start = int(stim_start_times[0][0])
            start_timepoints = [stim_start + i * start_btw_stim_frames for i in range(num_stims_per_repeat)]
            baseline_duration = int(stim_start_times[0]) - 1
            activation_results = {roi_idx: [] for roi_idx in cell_indices}
            activation_count = 0
            for roi_idx in cell_indices:
                F_index_act = np.where(cell_indices == roi_idx)[0][0]
                baseline_data = F[F_index_act, :max(1, stim_start - 1)]
                baseline_avg = np.mean(baseline_data)
                baseline_std = np.std(baseline_data)
                threshold = baseline_std * threshold_value + baseline_avg
                roi_activation = []
                is_active = False  # whether the roi is active
                for start_time in start_timepoints:
                    stim_end_time = start_time + stimulation_duration_frames
                    stim_segment = F[F_index_act, start_time:stim_end_time]
                    avg_stim_resp = np.mean(stim_segment)
                    activation = 1 if avg_stim_resp > threshold else 0
                    roi_activation.append(activation)
                    if activation == 1:
                        is_active = True
                activation_results[roi_idx] = roi_activation
                if is_active:
                    activation_count += 1
            # print(activation_results)
            print(f"Number of activated neurons: {activation_count} out of {num_cells} cells")
            #===
            Ly, Lx = ops['Ly'], ops['Lx']
            masks = []
            activated_roi_indices = []
            med_values = []
            for roi_data, pattern in activation_results.items():
                if any(pattern):
                    roi = stat[roi_data]
                    xpix = roi['xpix']
                    ypix = roi['ypix']
                    mask = np.zeros((Ly, Lx), dtype=np.uint8)
                    mask[ypix, xpix] = 1
                    masks.append(mask)
                    activated_roi_indices.append(roi_data)  # for cellreg_to_suite2p bc activated_roi_indices[i] will contain the original stat index
                    med_values.append(roi['med'])
            if masks:
                mask_stack = np.stack(masks, axis=-0).astype(np.double)  # [nROIs, Ly, Lx]
                print(mask_stack.shape)
                output_folder = os.path.join(expDir, 'cellreg_files')
                os.makedirs(output_folder, exist_ok=True)
                out_name = f'{matched_file}.mat'
                out_path = os.path.join(output_folder, out_name)
                savemat(out_path, {'cells_map': mask_stack})
                print(f" Saved: {out_path} with shape {mask_stack.shape}")

            activated_roi_df  = pd.DataFrame({
                'ROI_Index': activated_roi_indices,
                'Med_Values': med_values
            })
            activated_roi_df.to_csv(os.path.join(cell_reg_path_input, 'activation_summary.csv'), index=False)
            print(f"Activation summary saved to: {os.path.join(cell_reg_path_input, 'activation_summary.csv')}")

            column_names = [f"Stim {i + 1}" for i in range(num_stims_per_repeat)]
            activation_df = pd.DataFrame.from_dict(activation_results, orient='index', columns=column_names)
            activation_df.insert(0, "ROI", activation_df.index)
            csv_path = os.path.join(expDir, dir, f'activation_results_file{file_suffix}.csv')
            # activation_df.to_csv(csv_path, index=False)
            # print(f"Results saved to {csv_path}")

            # === Match activated ROIs with cellreg data ===
            print(all_stats[0][1])
            matched_results = []
            for i in range(num_sessions-1):
                for j in range(i+1, num_sessions):
                    for row in range(data.shape[0]):
                        print(row)
                        roi_i= int(data[row,i])
                        roi_j = int(data[row, j])
                        print(i, j, roi_j)
                        if roi_i >0 and roi_j > 0:
                            stat_i = all_stats[i][roi_i]
                            stat_j = all_stats[j][roi_j]
                            med_i = tuple(stat_i['med'])
                            med_j = tuple(stat_j['med'])
                            #print(roi_i, med_i, roi_j, med_j)

            #for cellreg_idx in data[:,]:
            '''
                for i in range(data.shape[cellreg_idx]):
                    roi_i = int(data[cellreg_idx, i])
                    if roi_i <= 0:
                        continue
                    for j in range(i + 1, data.shape[1]):
                        roi_j = int(data[cellreg_idx, j])
                        if roi_j <= 0 or j >= len(all_stats) or roi_j >= len(all_stats[j]):
                            continue
                        # med info from stat
                        stat_i = all_stats[i][roi_i]
                        stat_j = all_stats[j][roi_j]
                        y_i, x_i = stat_i['med']
                        y_j, x_j = stat_j['med']
                        #print(f'{data[cellreg_idx]},({y_i, x_i}), ({y_j, x_j})')
                        
                        for idx, row in activated_roi_df.iterrows():
                            print(y_i, x_i,row['Med_Values'])
                            if (y_i, x_i) == row['Med_Values']:
                                matched_results.append({
                                    'CellReg_Index': cellreg_idx,
                                    'Session_A': i + 1,
                                    'Session_B': j + 1,
                                    'Suite2p_ROI_A': roi_i,
                                    'Suite2p_ROI_B': roi_j,
                                    'Match_Med': (y_i, x_i)
                                })
            '''


            #print(matched_results)
            results_df = pd.DataFrame(matched_results)
            out_path = os.path.join(cell_reg_path_input, 'matched_cellreg_to_suite2p.csv')
            #results_df.to_csv(out_path, index=False)
            #print(f"matched cellreg to suite2p saved to {out_path}")

