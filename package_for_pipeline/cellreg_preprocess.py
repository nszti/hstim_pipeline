import os.path
import numpy as np
from scipy.io import savemat
from pathlib import Path
import h5py
import pandas as pd


def suite2p_to_cellreg_masks(expDir, list_of_file_nums):
    '''

    Parameters
    ----------
    expDir: root_directory
    list_of_file_nums

    Returns
    -------
    saves result .mat files into new cellreg_files directory
    '''
    base_dir = Path(expDir)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]
    print(filenames)
    for numbers_to_merge in list_of_file_nums:
        #print(numbers_to_merge)
        suffix = '_'.join(map(str, numbers_to_merge))
        num_to_search = []
        for dir in filenames:
            print(dir)
            num_to_search_split = dir.split('MUnit_')
            if len(num_to_search_split) > 1:
                file_suffix = num_to_search_split[1].rsplit('.', 1)[0]
                if file_suffix == suffix:
                    matched_file = dir
                    print(f'Found file: {matched_file}')
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
                if x > 1 and y > 1:
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
                mask_stack = np.stack(masks, axis=-0).astype(np.double)  # [nROIs, Ly, Lx]
                print(mask_stack.shape)
                output_folder = os.path.join(expDir, 'cellreg_files')
                os.makedirs(output_folder, exist_ok=True)
                print(matched_file)
                out_name = f'{matched_file}.mat'
                out_path = os.path.join(output_folder, out_name)
                savemat(out_path, {'cells_map': mask_stack})
                print(f" Saved: {out_path} with shape {mask_stack.shape}")


def cellreg_analysis_overlap(expDir, mat_file, list_of_file_nums, postfix):
    '''

    Parameters
    ----------
    expDir: root_directory
    mat_file: cell_Registered[date_num].mat saved after running the cellreg process
    list_of_file_nums
    postfix: if there is a separate directory for the cellreg files, you can add a postfix so it finds the path

    Returns
    -------
    session_pair_overlap.csv: contains how many activated cells are the same between each 2 sessions
    also prints the results to the console
    '''
    base_dir = Path(expDir)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]
    for numbers_to_merge in list_of_file_nums:
        suffix = '_'.join(map(str, numbers_to_merge))
        num_to_search = []
        for dir in filenames:
            num_to_search_split = dir.split('MUnit_')
            if len(num_to_search_split) > 1:
                file_suffix = num_to_search_split[1].rsplit('.', 1)[0]
                if file_suffix == suffix:
                    matched_file = dir
                    print(f'Found file: {matched_file}')
                    break
        else:
            continue
        matched_file = matched_file + '/'
        postfix = 'cellreg_files'
        found_file = os.path.join(expDir, matched_file)
        cell_reg_path = found_file
        cell_reg_path_input = os.path.join(cell_reg_path,postfix)
        input_file = os.path.join(cell_reg_path_input, mat_file)
        print(input_file)
        with h5py.File(input_file, 'r') as file:
            data = file['cell_registered_struct']['cell_to_index_map'][:][:]
        num_sessions, num_cells = data.shape
        print(num_cells, num_sessions)
        total_cells_per_session = np.max(data, axis=0)

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
                print(
                    f"Sessions {i + 1} & {j + 1}: {num_matches} overlapping cells which is {percent_overlap:.2f}% of total cells")
                result_rows.append([f"Session {i + 1}", f"Session {j + 1}", num_matches, f"{percent_overlap:.2f}%"])

        cumulative_overlap = np.logical_and(data[0] > 0, data[1] > 0)
        for session in range(2, num_sessions):
            cumulative_overlap = np.logical_and(cumulative_overlap, data[session] > 0)

        num_cumulative = np.sum(cumulative_overlap)
        percent_cumulative = (num_cumulative / num_cells) * 100
        print(f"Cumulative overlap across sessions: {num_cumulative} cells ({percent_cumulative:.2f}%)")
        result_rows.append(["Sum overlap", f"Sessions 1 to {num_sessions}", num_cumulative, f"{percent_cumulative:.2f}%"])

        result_rows.append(["Total cells registered", num_cells])
        csv_path = os.path.join(cell_reg_path_input, 'session_pair_overlap.csv')
        df = pd.DataFrame(result_rows, columns=['Session A', 'Session B', 'Number of Overlapping Cells', 'Overlap %'])
        df.to_csv(csv_path, index=False)
        print("Overlap matrix saved as session_pair_overlap.csv")


def single_block_activation(expDir, postfix, mat_file, frame_rate, num_stims_per_repeat, list_of_file_nums, start_btw_stim, stim_dur, threshold_value):
    '''

    Parameters
    ----------
    expDir: root_directory
    postfix: if needed
    mat_file
    frame_rate
    num_stims_per_repeat
    list_of_file_nums
    start_btw_stim: trial_delay *from the experiment records*
    stim_dur: duration *from the experiment records*
    threshold_value: usually 3

    Returns
    -------

    '''
    base_dir = Path(expDir)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]

    all_stats = {}
    session_idx = 0

    for numbers_to_merge in list_of_file_nums:
        suffix = '_'.join(map(str, numbers_to_merge))
        for dir in filenames:
            if dir.startswith('merged') and f'MUnit_{suffix}' in dir:
                matched_file = dir
                stat_path = os.path.join(expDir, matched_file, 'suite2p', 'plane0', 'stat.npy')
                if os.path.exists(stat_path):
                    stat_data = np.load(stat_path, allow_pickle=True)
                    all_stats[session_idx] = stat_data
                    print(f"Session {session_idx}: Loaded from {matched_file}")
                    session_idx += 1  # increment only after successful match
                    print(session_idx)
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
            print(f"Number of activated neurons: {activation_count} out of {num_cells} cells")
            # ===
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
                    activated_roi_indices.append(
                        roi_data)  # for cellreg_to_suite2p bc activated_roi_indices[i] will contain the original stat index
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

            activated_roi_df = pd.DataFrame({
                'ROI_Index': activated_roi_indices,
                'Med_Values': med_values
            })
            cell_reg_path_input  =  expDir + '/cellreg_files'
            activated_roi_df.to_csv(os.path.join(cell_reg_path_input, 'activation_summary.csv'), index=False)
            print(f"Activation summary saved to: {os.path.join(cell_reg_path_input, 'activation_summary.csv')}")

            column_names = [f"Stim {i + 1}" for i in range(num_stims_per_repeat)]
            activation_df = pd.DataFrame.from_dict(activation_results, orient='index', columns=column_names)
            activation_df.insert(0, "ROI", activation_df.index)
            csv_path = os.path.join(expDir, dir, f'activation_results_file{suffix}.csv')
            # activation_df.to_csv(csv_path, index=False)

            # === Match activated ROIs with cellreg data ===
            matched_results = []
            num_sessions = 1
            for i in range(num_sessions - 1):
                for j in range(i + 1, num_sessions):
                    for row in range(data.shape[0]):
                        print(row)
                        roi_i = int(data[row, i])
                        roi_j = int(data[row, j])
                        print(i, j, roi_j)
                        if roi_i > 0 and roi_j > 0:
                            stat_i = all_stats[i][roi_i]
                            stat_j = all_stats[j][roi_j]
                            med_i = tuple(stat_i['med'])
                            med_j = tuple(stat_j['med'])

            # print(matched_results)
            results_df = pd.DataFrame(matched_results)
            out_path = os.path.join(cell_reg_path_input, 'matched_cellreg_to_suite2p.csv')
            # results_df.to_csv(out_path, index=False)
            # print(f"matched cellreg to suite2p saved to {out_path}")

