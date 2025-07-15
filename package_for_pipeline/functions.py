import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
from pathlib import Path
from scipy.io import savemat
import h5py
import re
import ast
from pprint import pprint

from sklearn.metrics import euclidean_distances
from suite2p.gui.io import load_files
import subprocess

#stim_dur
def stim_dur_val(root_directory, tiff_dir, list_of_file_nums):
    '''

    Parameters
    ----------
    tiff_dir: path to 'merged_tiffs' directory

    Returns:
    -------
    stimDurations.npy calculated from frequencies
    '''

    fileid_txt_path = os.path.join(root_directory, 'fileID.txt')
    print(fileid_txt_path)

    file_ids_txt = []
    with open(fileid_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("MUnit_"):
                line = line.replace("MUnit_", "")
            if line:
                file_ids_txt.append(int(line))
    fileid_to_index = {}
    for idx, fid in enumerate(file_ids_txt):
        fileid_to_index[fid] = idx

    base_dir = Path(tiff_dir)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]
    #print(filenames)

    for numbers_to_merge in list_of_file_nums:
        suffix = '_'.join(map(str, numbers_to_merge))
        num_to_search = []
        for dir in filenames:
            num_to_search_split = dir.split('MUnit_')
            #print(num_to_search_split)
            if len(num_to_search_split) > 1:
                file_suffix = num_to_search_split[1].rsplit('.', 1)[0]
                if file_suffix == suffix:
                    matched_file = dir
                    print(matched_file)
                    break
        else:
            continue

        if matched_file:
            dir_path = base_dir / matched_file
            frequency_path = os.path.join(tiff_dir, 'frequencies.npy')
            frequency = np.load(frequency_path, allow_pickle=True)
            print(frequency)
            '''if not os.path.exists(frequency_path):
                print(f"Frequency file not found")
                exit(1)
            else:
                print(f"Stimulation frequencies: {frequency}")
                stim_duration = []
                for freq in frequency:
                    ref_freq = 100
                    ref_dur = 1
                    stim_dur = ref_freq / (ref_dur * freq)  # float
                    stim_duration.append(stim_dur)
                print(stim_duration)
                print(f"Stimulation durations for {matched_file}: {stim_duration}")
                np.save(dir_path /'stimDurations.npy', stim_duration)'''
            stim_duration = []
            for file_num in numbers_to_merge:
                if file_num not in fileid_to_index:
                    print(f"FileID {file_num} not found in fileID.txt. Skipping.")
                    stim_duration.append(np.nan)
                    continue
                idx = fileid_to_index[file_num]
                idx = fileid_to_index[file_num]
                freq = frequency[idx]
                ref_freq = 100
                ref_dur = 1
                stim_dur = ref_freq / (ref_dur * freq)  # float
                stim_duration.append(stim_dur)
            print(stim_duration)
            print(f'Stimulation durations for {matched_file}: {stim_duration}')
            np.save(dir_path / 'stimDurations.npy', stim_duration)




#electrodeROI
def electROI_val(tiff_dir,list_of_file_nums):
    '''
    Parameters
    ----------
    Returns:
    -------
    saves electrodeROI.npy (spec. elec roi num) from 'selected_elec_ROI.npy'
    '''
    base_dir = Path(tiff_dir)
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
                    print(matched_file)
                    break
        else:
            continue

        if matched_file:
            dir_path = base_dir / matched_file
            print(dir_path)
            elec_roi_path = dir_path / 'selected_elec_ROI.npy'
            print(elec_roi_path)
            selected_elec_ROI = np.load(elec_roi_path, allow_pickle=True)
            if not os.path.exists(elec_roi_path):
                print(f" File not found")
                exit(1)
            else:
                print(f"ROIs of used electrodes: {selected_elec_ROI}")
                electrodeROI = []
                for roi in selected_elec_ROI:
                    electrodeROI.append(roi)
                print(f"Used electrode ROI for {dir}: {electrodeROI}")
                np.save(dir_path / 'electrodeROI.npy', electrodeROI)

#distance
def dist_vals (tiff_dir, list_of_file_nums):
    '''

    Parameters
    ----------
    tiff_dir
    list_of_file_nums

    Returns
    -------
    'roi_num' contains the roi number of detected cells--> gets saved into 'ROI_numbers.npy'
    euclidean distance from electrode roi--> 'distances.npy'

    '''
    #new_elec_med_value = (None, None)
    base_dir = Path(tiff_dir)

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
                    print(matched_file)
                    break
        else:
            continue

        if matched_file:
            iscell_path = tiff_dir  + matched_file + '/suite2p/plane0/iscell.npy'
            stat_path = tiff_dir  + matched_file + '/suite2p/plane0/stat.npy'
            electrodeROI_path = tiff_dir + matched_file + '/selected_elec_ROI.npy'
            csv_file_path = tiff_dir + matched_file + '/elec_roi_info.csv'
            print(csv_file_path)
            stat = np.load(stat_path, allow_pickle=True)
            iscell = np.load(iscell_path, allow_pickle=True)
            electrodeROI = np.load(electrodeROI_path, allow_pickle=True)
            print(electrodeROI)
            distances = []
            # extract cell roi info
            first_column = iscell[:, 0]
            tempforcells = []
            for index, value in enumerate(first_column):
                if value == 1:
                    roi_info = f"{index}, Value: {value}"
                    tempforcells.append([roi_info.split(',')[0]])

            # extract all roi med info
            med_values = [roi['med'] for roi in stat]
            tempforallmed = []
            tempforallroi = []
            for roi_number, med_value in enumerate(med_values):
                tempforallroi.append(roi_number)
                tempforallmed.append(med_value)

            # dataframes for cells & all roi
            dfcell_roi = pd.DataFrame(tempforcells, columns=['roi_num'])
            mergedallmedinfo = list(zip(tempforallroi, tempforallmed))
            dfallmedinfo = pd.DataFrame(mergedallmedinfo, columns=['roi_num', 'med'])

            # matching
            matched_roi_med = []
            for roi_num in tempforcells:
                roi_num = int(roi_num[0])  # extracting roi nums from tempforcells
                if roi_num in tempforallroi:
                    med_value = dfallmedinfo.loc[dfallmedinfo['roi_num'] == roi_num, 'med'].values
                    if len(med_value) > 0:
                        matched_roi_med.append((roi_num, med_value[0]))
            # df for matched info
            dfmatched = pd.DataFrame(matched_roi_med, columns=['roi_num', 'med_value'])



            # Distance calc w dfmatched-------------------------------------------
            def euclidean_distance(point1, point2):
                return np.linalg.norm(np.array(point1) - np.array(point2))

            # fv minimum distance search
            def minimum_distance_search(med_values, start_roi):
                start_point = None  # spec starting point(ha kell)
                for roi, coords in zip(roi_numbers, med_values):
                    print(med_values)
                    if roi == electrodeROI.any():
                        start_point = coords
                        break

                if start_point is None:
                    raise ValueError(f"ROI {start_roi} not found in the dataset.")
                distances = [euclidean_distance(start_point, coords) for coords in med_values]
                return distances


            roi_numbers = dfmatched['roi_num'] #df 0tol szamol!!
            med_values = dfmatched['med_value']
            distances = minimum_distance_search(med_values, electrodeROI)
            # print(distances)
            # extracting electrode info
            #electrode_med = []
            if isinstance(electrodeROI, list) or isinstance(electrodeROI, np.ndarray):
                for roi in electrodeROI:
                    electrode_i = dfmatched[dfmatched['roi_num'] == roi].index

            else:
                electrode_i = dfmatched[dfmatched['roi_num'] == electrodeROI].index
            electrode_med = dfmatched.loc[electrode_i, 'med_value'].iloc[0]
            x_value, y_value = electrode_med
            # Distance calc w dfmatched-------------------------------------------

            # df for electrode med info
            electrode_df = pd.DataFrame({'electrode med x': [x_value], 'electrode med y': [y_value]})

            # Results
            result_df = pd.DataFrame({
                'ROI_Number': roi_numbers,
                'Med_Values': med_values,
                'distance': distances
            })

            # concatenate elec info & roi info
            finaldf = pd.concat([electrode_df, result_df], ignore_index=True)
            finaldf.to_csv(csv_file_path, index=False)

            print(f"Results saved to {csv_file_path}")
            # print(result_df)
            # print(result_df.shape)
            dir_path = os.path.join(base_dir, dir)
            np.save(dir_path + '/suite2p/plane0/distances.npy', result_df)
            np.save(dir_path + '/suite2p/plane0/ROI_numbers.npy', roi_numbers)
            print(
                f"distances.npy saved to {dir_path + '/suite2p/plane0/distances.npy'}, ROI_numbers.npy saved to {dir_path + '/suite2p/plane0/ROI_numbers.npy'}")
            # save output as .npy file
            # np.save(expDir + '/' + dir + '/suite2p/plane0/distances.npy', result_df)
            # np.save(expDir + '/' + dir + '/suite2p/plane0/distances.npy', distances)
            np.save(tiff_dir + '/' + dir + '/suite2p/plane0/ROI_numbers.npy', roi_numbers)


def spontaneous_baseline(tiff_dir, list_of_file_nums, list_of_roi_nums, frame_rate = 30.97, baseline_frame=3, plot_start_frame = 0, plot_end_frame=None ):
    '''

    Parameters
    ----------
    tiff_dir
    list_of_file_nums
    list_of_roi_nums: int list, defines which ROIs look at
    frame_rate
    baseline_frame: frame number to consider as baseline
    plot_start_frame,plot_end_frame: to define concrete frame fragment to save

    Returns
    -------
    saves F0.npy corrected fluorescent trace
    roi[roi_index].svg ca traces of all the ROIs specified in list_of_roi_nums

    '''
    base_dir = Path(tiff_dir)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]
    print(filenames)
    for numbers_to_merge in list_of_file_nums:
        # print(numbers_to_merge)
        suffix = '_'.join(map(str, numbers_to_merge))
        for dir in filenames:
            print(dir)
            num_to_search_split = dir.split('MUnit_')
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
            print(matched_file)
            F_path = tiff_dir + dir + '/suite2p/plane0/F.npy'
            F = np.load(F_path, allow_pickle=True)  # shape: (n_rois, n_timepoints)
            dir_path = os.path.join(tiff_dir, dir)

            baseline_frames = max(baseline_frame, int(10 / 1000 * frame_rate))  # ~10ms

            all_norm_traces = []

            for roi_trace in F:
                baseline_value = np.mean(roi_trace[:baseline_frames])
                norm_trace = (roi_trace - baseline_value) / baseline_value
                all_norm_traces.append(norm_trace)

            all_norm_traces = np.array(all_norm_traces)

            # Save normalised trace
            s2p_path = tiff_dir + dir + '/suite2p/plane0'
            out_path = os.path.join(s2p_path, 'F0.npy')
            np.save(out_path, all_norm_traces)
            print(f"Normalized traces saved to: {out_path}")
            print(f"Shape of normalized trace array: {all_norm_traces.shape}")

            # Plot selected ROI traces
            for roi_index in list_of_roi_nums:
                plot_path = os.path.join(dir_path, f'roi_{roi_index}.svg')
                if roi_index < len(all_norm_traces):
                    trace = all_norm_traces[roi_index]
                    trace = trace[plot_start_frame:]
                    if plot_end_frame is not None:
                        print(plot_start_frame, plot_end_frame)
                        trace = trace[plot_start_frame:plot_end_frame]
                    #trace = (trace- np.min(trace)) / (np.max(trace) - np.min(trace))
                    plt.figure()
                    plt.plot(trace)
                    plt.title(f'Normalized Trace for ROI {roi_index}')
                    plt.xlabel('Time (frames)')
                    plt.ylim(-0.3, 1.8)
                    plt.ylabel('ΔF/F0')
                    plt.tight_layout()
                    print(plot_path)
                    plt.savefig(plot_path)
                    #plt.show()
                else:
                    print(f"ROI {roi_index} is out of bounds. Max index: {len(all_norm_traces) - 1}")

def baseline_val(root_directory,tiff_dir, list_of_file_nums ):
    '''

    Parameters
    ----------
    root_directory
    tiff_dir
    list_of_file_nums

    Returns
    -------
    saves all_norm_traces, prints shape of all_norm_traces, output: F0.npy (baseline corrected fluorescence trace)
    '''
    base_dir = Path(tiff_dir)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]
    matching_tiff = []
    file_nums_to_search = list_of_file_nums[0]

    for file in os.listdir(root_directory):
        #print(file)
        if file.endswith('.tif') and 'MUnit_' in file:
            start_index = file.find('MUnit_') + len('MUnit_')
            end_index = file.rfind('.tif')

            if start_index != -1 and end_index != -1 and start_index < end_index:
                number_str = file[start_index:end_index]

                if number_str.isdigit():
                    file_number = int(number_str)
                    if file_number in file_nums_to_search:
                        matching_tiff.append(file)

    for numbers_to_merge in list_of_file_nums:
        suffix = '_'.join(map(str, numbers_to_merge))
        matched_dir = None
        num_to_search = []
        for dir in filenames:
            num_to_search_split = dir.split('MUnit_')
            # print(num_to_search_split)
            if len(num_to_search_split) > 1:
                file_suffix = num_to_search_split[1].rsplit('.', 1)[0]
                if file_suffix == suffix:
                    matched_dir = dir
                    #print(matched_dir)
                    break
        else:
            continue
        if matched_dir:

            F_path = tiff_dir + matched_dir + '/suite2p/plane0/F.npy'
            iscell_path = tiff_dir + matched_dir + '/suite2p/plane0/iscell.npy'
            stim_start_times_path = tiff_dir + matched_dir + '/stimTimes.npy'

            F = np.load(F_path, allow_pickle=True)
            iscell = np.load(iscell_path, allow_pickle=True)
            stim_start_times = np.load(stim_start_times_path, allow_pickle=True)
            print(stim_start_times)


            all_norm_traces = []
            cellcount = 0
            # Iterate through all rois
            for cell_index, (fluorescence_trace, (iscell_value, _)) in enumerate(zip(F, iscell)):
                # Check iscell==1
                if iscell_value == 1:
                    cellcount += 1
                    baseline_duration = int(stim_start_times[0]) - 1
                    if baseline_duration is not None:
                        baseline_value = np.mean(fluorescence_trace[:baseline_duration])
                        normalized_trace = (fluorescence_trace - baseline_value) / baseline_value
                        all_norm_traces.append(normalized_trace)
            # convert the list of baseline_diffs to a npy array
            all_norm_traces = np.array(all_norm_traces)
            dir_path = os.path.join(base_dir, dir)
            np.save(dir_path + '/suite2p/plane0/F0.npy', all_norm_traces)
            print(f"F0.npy saved to {dir_path + '/suite2p/plane0/F0.npy'}")

#activated_neurons
def activated_neurons_val(root_directory, tiff_dir, list_of_file_nums, threshold_value):
    '''
    :param input_file_path: 'D:/2P/E/test/merged_GCaMP6f_23_09_25_3-6_pos_amp/'
    :param time_block: type: number, time block duration in frames, example: 1085
    :return: saves distance results as 'result_df', can print sum of roi_num-med_val distances, output: activated_neurons.npy
    '''
    base_dir = Path(tiff_dir)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]
    file_nums_to_search = list_of_file_nums[0]
    matching_tiff = []
    for file in os.listdir(root_directory):
    # print(file)
        if file.endswith('.tif') and 'MUnit_' in file:
            start_index = file.find('MUnit_') + len('MUnit_')
            end_index = file.rfind('.tif')
            if start_index != -1 and end_index != -1 and start_index < end_index:
                number_str = file[start_index:end_index]
                if number_str.isdigit():
                    file_number = int(number_str)
                    if file_number in file_nums_to_search:
                        matching_tiff.append(file)

    for numbers_to_merge in list_of_file_nums:
        suffix = '_'.join(map(str, numbers_to_merge))
        for dir in filenames:
            num_to_search_split = dir.split('MUnit_')
            # print(num_to_search_split)
            if len(num_to_search_split) > 1:
                file_suffix = num_to_search_split[1].rsplit('.', 1)[0]
                if file_suffix == suffix:
                    matched_dir = dir
                    #print(matched_dir)
                    break
        else:
            continue

        file_dir = Path(base_dir / matched_dir)
        baseline_durations = []
        for num_id in file_nums_to_search:
            stim_times_path = os.path.join(file_dir, f'sti'
                                                     f''
                                                     f'mTime_{num_id}.npy')
            if os.path.exists(stim_times_path):
                stim_times = np.load(stim_times_path, allow_pickle=True)
                if stim_times.size > 0:
                    baseline_duration = int(stim_times[0]) - 1
                    print(baseline_duration)
                    baseline_durations.append(baseline_duration)
        if matched_dir:
            # Load the fluorescence traces and iscell array
            F0_path = tiff_dir + matched_dir + '/suite2p/plane0/F0.npy'
            iscell_path = tiff_dir + matched_dir + '/suite2p/plane0/iscell.npy'
            ROI_numbers_path = tiff_dir + matched_dir + '/suite2p/plane0/ROI_numbers.npy'
            stim_start_times_path = tiff_dir + matched_dir + '/stimTimes.npy'
            frame_numbers_path = tiff_dir + matched_dir + '/frameNum.npy'
            stat_path = tiff_dir + matched_dir + '/suite2p/plane0/stat.npy'
            ops_path = tiff_dir + matched_dir + '/suite2p/plane0/ops.npy'

            F0 = np.load(F0_path, allow_pickle=True)
            iscell = np.load(iscell_path, allow_pickle=True)
            ROI_numbers = np.load(ROI_numbers_path, allow_pickle=True)
            stim_start_times = np.load(stim_start_times_path, allow_pickle=True)
            frame_numbers = np.load(frame_numbers_path, allow_pickle=True)
            stat = np.load(stat_path, allow_pickle=True)
            ops = np.load(ops_path, allow_pickle=True).item()

            time_block = 1
            if len(frame_numbers) > 0:
                time_block = int(frame_numbers[0]) #1085
                print(time_block)
            # Calculate TIFF trigger start and end tuples
            num_tif_triggers = int(np.round(len(F0[0]) / time_block))
            print(len(F0[0]) / time_block)
            print(num_tif_triggers)
            tif_triggers = []
            for i in range(num_tif_triggers):
                start_time = i * time_block
                end_time = start_time + time_block
                tif_triggers.append((start_time, end_time))
            ROI_numbers=[]
            threshold_list = []
            results_list = []
            # Iterate through all ROIs
            print(len(baseline_durations))
            for i in range(len(F0)):
                #print(i)
                roi_thresholds = []
                roi_results = []
                for baseline_duration, (start_time, end_time) in zip(baseline_durations, tif_triggers):
                    baseline_dur = F0[i, start_time:start_time + baseline_duration]
                    baseline_avg = np.mean(baseline_dur)
                    baseline_std = np.std(baseline_dur)
                    threshold = baseline_std * threshold_value + baseline_avg
                    roi_thresholds.append(threshold)
                    # Check if fluorescence exceeds threshold for the current tuple
                    stim_avg = np.mean(F0[i, (start_time + baseline_duration):(start_time + baseline_duration + 465)])
                    if stim_avg > threshold:
                        exceed_threshold = 1
                    else:
                        exceed_threshold = 0
                        # Append result (1 or 0) to the list for the current ROI
                    roi_results.append(int(exceed_threshold))
                # Append threshold values and results for the current ROI to the overall lists
                threshold_list.append(roi_thresholds)
                results_list.append(roi_results)
                ROI_numbers.append(i)
            print(results_list)
            # for cellreg .mat file===========
            Ly, Lx = ops['Ly'], ops['Lx']
            masks = []
            activated_roi_indices = []
            for i, result_row in enumerate(results_list):
                if any(result_row):
                    roi = stat[i]
                    xpix = roi['xpix']
                    ypix = roi['ypix']
                    mask = np.zeros((Ly, Lx), dtype=np.uint8)
                    mask[ypix, xpix] = 1
                    masks.append(mask)
                    activated_roi_indices.append(i)
            if masks:
                mask_stack = np.stack(masks, axis=0).astype(np.double)  # [nROIs, Ly, Lx]
                print(mask_stack.shape)
                output_folder = os.path.join(base_dir, 'cellreg_files')
                os.makedirs(output_folder, exist_ok=True)
                mat_name = matched_dir + '_cellreg_input.mat'
                mat_path = os.path.join(output_folder, mat_name)
                savemat(mat_path, {'cells_map': mask_stack})
                print(f"Saved CellReg input file to {mat_path} with shape {mask_stack.shape}")
            # for cellreg .mat file end===========
            result_df = pd.DataFrame({
                'ROI_number': ROI_numbers,
                'thresholds': threshold_list,
                'activated_neurons': results_list
            })
            pd.set_option('display.max_rows', None)

        dir_path = os.path.join(base_dir, dir)
        np.save(dir_path + '/suite2p/plane0/activated_neurons.npy', result_df)
        print(f"activated_neurons.npy saved to {dir_path + '/suite2p/plane0/activated_neurons.npy'}")
        #np.save(expDir + '/' + dir + '/suite2p/plane0/activated_neurons.npy', result_df)

#timecourse
def timecourse_vals(tiff_dir, list_of_file_nums, num_trials):
    '''
    :param expDir:
    :param frame_rate: 31Hz
    :param num_trials: 5, number of stimulation in one repeat
    :return: saves 'results.npz'
    '''
    base_dir = Path(tiff_dir)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]
    frame_rate = 30.97

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
                    break
        else:
            continue

        if matched_file:

            F_path = tiff_dir + matched_file + '/suite2p/plane0/F0.npy'
            stim_start_times_path = tiff_dir + matched_file + '/stimTimes.npy'
            stim_duration_path = tiff_dir + matched_file + '/stimDurations.npy'
            block_frames_path = tiff_dir + matched_file + '/frameNum.npy'
            roi_number_path = tiff_dir + matched_file + '/suite2p/plane0/ROI_numbers.npy'

            F = np.load(F_path, allow_pickle=True)
            stim_start = np.load(stim_start_times_path, allow_pickle=True)
            block_frames = np.load(block_frames_path, allow_pickle=True)
            print(block_frames)
            stim_duration = np.load(stim_duration_path, allow_pickle=True)
            roi_num = np.load(roi_number_path, allow_pickle=True)
            #stim_duration = [1.0, 1.0, 1.0, 1.0, 1.0] #amp
            start_timepoints = []
            for i in stim_start:
                start_timepoints.append(i)

            time_block = []
            for b in block_frames:
                time_block.append(b)
            stimulation_duration = []
            for s in stim_duration:
                stimulation_duration.append(s)
            num_blocks = len(time_block)
            resting_period = 3
            rest_dur_f = resting_period * frame_rate
            stim_dur_f = []
            end_f = []

            for s in stimulation_duration:
                frameNo = math.floor(s * frame_rate)
                stim_dur_f.append(frameNo)
                print(stim_dur_f)
                end_f.append(frameNo + rest_dur_f)
                print(end_f)

            blocks_start = []
            for i in range(len(time_block)):
                prev_blocks_duration = sum(time_block[0:i])
                start_time = prev_blocks_duration
                end_time = start_time + time_block[i] - 1
                blocks_start.append(start_time)
            stimResults = np.empty([len(F), num_blocks, num_trials], 'int')
            restResults = np.empty([len(F), num_blocks, num_trials], 'int')
            stimAvgs = np.empty([len(F), num_blocks, num_trials])
            restAvgs = np.empty([len(F), num_blocks, num_trials])
            baselineAvgs = np.empty([len(F), num_blocks])
            full_trial_traces = np.zeros((len(F), num_blocks, num_trials, 217), dtype=object)

            for iTrace in range(len(F)):
                for iBlock in range(num_blocks):
                    baseline_dur = F[iTrace, int(blocks_start[iBlock]): int(blocks_start[iBlock]) + (
                                int(start_timepoints[iBlock]) - 1)]
                    baseline_avg = np.mean(baseline_dur)
                    baselineAvgs[iTrace, iBlock] = baseline_avg
                    baseline_std = np.std(baseline_dur)
                    threshold = baseline_std * 3 + baseline_avg
                    for iTrial in range(num_trials):
                        trial_start = blocks_start[iBlock] + (int(start_timepoints[iBlock]) + iTrial * int(end_f[iBlock]))
                        trial_end = int(trial_start) + stim_dur_f[iBlock]
                        stim_trace_plot = F[iTrace, int(trial_start)-31:int(trial_end)]
                        stim_trace = F[iTrace, int(trial_start):int(trial_end)]
                        avg_stim = np.mean(stim_trace)
                        stimAvgs[iTrace][iBlock][iTrial] = avg_stim

                        if avg_stim > threshold:
                            stim_above_thr = True
                        else:
                            stim_above_thr = False

                        stimResults[iTrace][iBlock][iTrial] = stim_above_thr

                        rest_trace_start = blocks_start[iBlock] + (int(start_timepoints[iBlock]) + (
                                    (iTrial + 1) * (int(stim_dur_f[iBlock])) + (int(iTrial * rest_dur_f))))
                        rest_trace_end = int(rest_trace_start) + int(rest_dur_f)
                        rest_trace = F[iTrace, int(rest_trace_start):int(rest_trace_end)]
                        avg_rest = np.mean(rest_trace)
                        restAvgs[iTrace][iBlock][iTrial] = avg_rest

                        if avg_rest > threshold:
                            rest_above_thr = True

                        else:
                            rest_above_thr = False
                        restResults[iTrace, iBlock, iTrial] = rest_above_thr

                        full_trial = np.concatenate((stim_trace_plot, rest_trace))
                        trial_length = len(full_trial)
                        full_trial_traces[iTrace, iBlock, iTrial, :trial_length] = full_trial[:trial_length]


            numRows = math.ceil(math.sqrt(len(F)))
            fig, axs = plt.subplots(numRows, numRows, squeeze=False)
            for i in range(numRows):
                #print(range(numRows))
                #print(i)
                for j in range(numRows):
                    #print(j)
                    if i * numRows + j < len(F):
                        axs[i][j].imshow(stimResults[i * numRows + j, :, :])
                        axs[i][j].set_title('ROI' + str(roi_num[i * numRows + j]))
                    else:
                        print()
            #plt.show()
            np.savez(tiff_dir + dir + '/results.npz', stimResults=stimResults, restResults=restResults,
                     stimAvgs=stimAvgs, restAvgs=restAvgs, baselineAvgs=baselineAvgs,full_trial_traces=full_trial_traces)
            print(f"results.npz saved to {tiff_dir + dir + '/' + 'results.npz'}")


#data_analysis
def data_analysis_values (stim_type, tiff_dir, list_of_file_nums):
    '''
    :param stim_type: 4 type: 'amp','freq','pulse_no','dur'
    :return:
    '''
    base_dir = Path(tiff_dir)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]

    for numbers_to_merge in list_of_file_nums:
        suffix = '_'.join(map(str, numbers_to_merge))
        for dir in filenames:
            num_to_search_split = dir.split('MUnit_')
            if len(num_to_search_split) > 1:
                file_suffix = num_to_search_split[1].rsplit('.', 1)[0]
                if file_suffix == suffix:
                    matched_file = dir
                    break
        else:
            continue

        if matched_file:
            output_dir = tiff_dir + matched_file
            container = np.load(tiff_dir + matched_file + '/results.npz', allow_pickle=True)
            distances = np.load(tiff_dir +  matched_file + '/suite2p/plane0/distances.npy', allow_pickle=True)
            ROI_IDs = np.load(tiff_dir + matched_file + '/suite2p/plane0/ROI_numbers.npy', allow_pickle=True)
            electrode_ROI = np.load(tiff_dir +  matched_file + '/electrodeROI.npy', allow_pickle=True)

            distanceFromElectrode = distances[:, 2]
            stimResults = container["stimResults"]
            restResults = container["restResults"]
            stimAvgs = container["stimAvgs"]
            restAvgs = container["restAvgs"]
            baselineAvgs = container["baselineAvgs"]
            baselineAvgs = container["baselineAvgs"]
            full_trial_traces = container["full_trial_traces"]

            # remove electrode ROI from data
            for i in ROI_IDs:
                if i == electrode_ROI[0]:
                    electrode_ROI_index = i
            distanceFromElectrode = np.delete(distanceFromElectrode, electrode_ROI_index, axis=0)
            stimResults = np.delete(stimResults, electrode_ROI_index, axis=0)
            restResults = np.delete(restResults, electrode_ROI_index, axis=0)
            stimAvgs = np.delete(stimAvgs, electrode_ROI_index, axis=0)
            restAvgs = np.delete(restAvgs, electrode_ROI_index, axis=0)
            baselineAvgs = np.delete(baselineAvgs, electrode_ROI_index, axis=0)
            full_trial_traces = np.delete(full_trial_traces, electrode_ROI_index, axis=0)

            # collect ROI, block and trial numbers
            ROI_No = stimResults.shape[0]
            block_No = stimResults.shape[1]
            trial_No = stimResults.shape[2]

            if stim_type == 'amp':
                legend = ['10', '20', '30', '15', '25']
            elif stim_type == 'freq':
                legend = ['50', '100', '200']
            elif stim_type == 'pulse_dur':
                legend = ['50', '100', '200', '400']
            else:
                legend = ['20', '50', '100', '200']
            trialLabels = ['1', '2', '3', '4', '5']

            # collect neurons activated during a block
            activatedNeurons = np.empty([ROI_No, block_No], 'int')
            for iROI in range(ROI_No):
                for iBlock in range(block_No):
                    sumTrials = sum(stimResults[iROI, iBlock, :])
                    if sumTrials > 0:
                        activatedNeurons[iROI][iBlock] = 1
                    else:
                        activatedNeurons[iROI][iBlock] = 0

            # compute the number and fraction of neurons activated (or silent) during a block
            activeNeuronsPerBlock = np.empty(block_No, 'int')
            silentNeuronsPerBlock = np.empty(block_No, 'int')
            activeNeuronsPerBlockFraction = np.empty(block_No)
            silentNeuronsPerBlockFraction = np.empty(block_No)

            for iBlock in range(block_No):
                activeNeuronsPerBlock[iBlock] = sum(activatedNeurons[:, iBlock])
                activeNeuronsPerBlockFraction[iBlock] = activeNeuronsPerBlock[iBlock] / ROI_No
                silentNeuronsPerBlock[iBlock] = stimResults.shape[0] - activeNeuronsPerBlock[iBlock]
                silentNeuronsPerBlockFraction[iBlock] = silentNeuronsPerBlock[iBlock] / ROI_No

            # plot the number and fraction of neurons activated (or silent) during a block
            fig, axs = plt.subplots(2, 2, figsize = (12,8))
            axs[0, 0].plot(legend, activeNeuronsPerBlock, marker="o")
            axs[0, 0].set_xlabel('Stimulation current(uA)')
            axs[0, 0].set_ylabel('Number of active neurons')

            axs[0, 1].plot(legend, activeNeuronsPerBlockFraction, marker="o")
            axs[0, 1].set_xlabel('Stimulation current(uA)')
            axs[0, 1].set_ylabel('Fraction of active neurons')


            # compute the number and fraction of neurons activated during trials of a block
            activeNeuronsPerBlockPerTrial = np.empty([trial_No, block_No], 'int')
            activeNeuronsPerBlockPerTrialFraction = np.empty([trial_No, block_No])

            for iBlock in range(block_No):
                for iTrial in range(trial_No):
                    activeNeuronsPerBlockPerTrial[iTrial][iBlock] = sum(stimResults[:, iBlock, iTrial])
                    activeNeuronsPerBlockPerTrialFraction[iTrial][iBlock] = sum(stimResults[:, iBlock, iTrial]) / ROI_No

            # plot the number and fraction of neurons activated during trials of a block
            axs[1, 0].plot(trialLabels, activeNeuronsPerBlockPerTrial, marker="o")
            axs[1, 0].legend(legend)
            axs[1, 0].set_xlabel('Trial number')
            axs[1, 0].set_ylabel('Number of active neurons')

            axs[1, 1].plot(trialLabels, activeNeuronsPerBlockPerTrialFraction, marker="o")
            axs[1, 1].legend(legend)
            axs[1, 1].set_xlabel('Trial number')
            axs[1, 1].set_ylabel('Fraction of active neurons')

            # calculate and plot the mean amplitudes during stimulation trials and blocks
            avgCA = np.empty([block_No, trial_No])
            avgCAperBlock = np.empty([block_No])
            for iBlock in range(block_No):
                if stim_type == 'freq' and iBlock == 0:
                    for iTrial in range(trial_No):
                        avgCA[iBlock][iTrial] = np.mean(stimAvgs[:, iBlock, iTrial])
                    avgCAperBlock[iBlock] = np.mean(avgCA[iBlock, :3])
                else:
                    for iTrial in range(trial_No):
                        avgCA[iBlock][iTrial] = np.mean(stimAvgs[:, iBlock, iTrial])
                    avgCAperBlock[iBlock] = np.mean(avgCA[iBlock, :])
            print(avgCA)
            avgCAperTrial = np.mean(avgCA, axis=0)
            plt.savefig(output_dir + '/09_17_amp_fig1_2.svg')


            fig2, axs = plt.subplots(2, 2, figsize = (12,8))
            axs[0, 0].plot(legend, avgCAperBlock, marker="o")
            axs[0, 0].set_ylabel('Mean dF/F0')
            axs[0, 0].set_xlabel('Stimulation amplitude (uA)')
            axs[0, 1].plot(trialLabels, avgCAperTrial, marker="o")
            axs[0, 1].set_ylabel('Mean dF/F0')
            axs[0, 1].set_xlabel('Trial number')
            axs[1, 0].set_ylabel('Mean dF/F0')
            axs[1, 0].set_xlabel('Trial number')
            axs[1, 1].set_ylabel('Mean dF/F0')
            axs[1, 1].set_xlabel('Trial number')

            # calculate and plot the mean amplitudes during stimulation trials of a block
            avgCAduringTrials = np.empty([block_No, trial_No])
            for iBlock in range(block_No):
                if stim_type == 'freq' and iBlock == 0:
                    for iTrial in range(trial_No):
                        avgCAduringTrials[iBlock][iTrial] = np.mean(stimAvgs[:,iBlock, :3 ])
                else:
                    for iTrial in range(trial_No):
                        avgCAduringTrials[iBlock][iTrial] = np.mean(stimAvgs[:, iBlock, iTrial])

                axs[1, 0].plot(trialLabels, avgCAduringTrials[iBlock, :])
            axs[1, 0].legend(legend)

            # calculate and plot the mean amplitudes during rest periods of a block
            avgCAduringRest = np.empty([block_No, trial_No])
            for iBlock in range(block_No):
                for iTrial in range(trial_No):
                    avgCAduringRest[iBlock][iTrial] = np.mean(restAvgs[:, iBlock, iTrial])

                axs[1, 1].plot(trialLabels, avgCAduringRest[iBlock, :])
            axs[1, 1].legend(legend)
            plt.savefig(output_dir + '/09_17_amp_fig2_2.svg')

            # plot calcium traces during stimulation
            tracesPerBlock = np.empty([ROI_No, 217])
            avgTracePerBlock = np.empty([block_No, trial_No, 217])
            for iBlock in range(block_No):
                for iTrial in range(trial_No):
                    tracesPerBlock = full_trial_traces[:, iBlock, iTrial, :]
                    avgTracePerBlock[iBlock, iTrial, :] = np.mean(tracesPerBlock, axis=0)  #

            plot_dur = 7 * 31
            ymin = -0.01
            ymax = 0.35

            fig3, axs = plt.subplots(2, 5, figsize = (12,8))
            for iBlock in range(block_No):
                for iTrial in range(trial_No):
                    axs[0, iBlock].plot(avgTracePerBlock[iBlock, iTrial, 0:plot_dur])
                    axs[0, iBlock].set_title(legend[iBlock])
                    axs[0, iBlock].set_ylim([ymin, ymax])
                    axs[0, iBlock].legend(trialLabels)
                    axs[1, iTrial].plot(avgTracePerBlock[iBlock, iTrial, 0:plot_dur])
                    axs[1, iTrial].set_title(trialLabels[iTrial])
                    axs[1, iTrial].set_ylim([ymin, ymax])
                    axs[1, iTrial].legend(legend)
            axs[0, 0].set_ylabel('Mean dF/F0')
            axs[1, 0].set_ylabel('Mean dF/F0')

            # distance calculation and plot
            binSize = 50
            maxDistance = 600
            bin_numbers = int(maxDistance / binSize)
            CAduringStim = [[[] for _ in range(bin_numbers)] for _ in range(stimResults.shape[1])]
            activatedNeuronsDuringStim = np.zeros([block_No, bin_numbers])

            for iROI in range(stimResults.shape[0]):
                for iBlock in range(stimResults.shape[1]):
                    binNo = math.floor((distanceFromElectrode[iROI] / maxDistance) / (1 / bin_numbers))
                    CAduringStim[iBlock][binNo].append(np.mean(stimAvgs[iROI, iBlock, :]))
                    if activatedNeurons[iROI][iBlock] == 1:
                        activatedNeuronsDuringStim[iBlock][binNo] += 1


            distanceMeans = np.empty([stimResults.shape[1], bin_numbers])
            plt.savefig(output_dir + '/09_17_amp_fig3_2.svg')

            # plot distance vs. mean calcium activity and distance vs. activated neurons
            fig4, axs = plt.subplots(2, 2, figsize = (12,8))
            x_axis = ['0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350-400', '400-450',
                      '450-500', '500-550', '550-600']

            for iBlock in range(stimResults.shape[1]):
                for iBin in range(bin_numbers):
                    distanceMeans[iBlock][iBin] = np.mean(CAduringStim[iBlock][iBin])
                    print()

                axs[0, 0].plot(x_axis, distanceMeans[iBlock, :])
                axs[1, 0].plot(x_axis, activatedNeuronsDuringStim[iBlock, :])
            axs[0, 0].legend(legend)
            axs[0, 0].set_ylabel('Mean dF/F0')
            axs[0, 0].set_xlabel('Distance from electrode (um)')
            axs[1, 0].legend(legend)
            axs[1, 0].set_ylabel('Activated neurons')
            axs[1, 0].set_xlabel('Distance from electrode (um)')


            # calculate and plot mean distance of activated neurons vs. blocks
            distancesPerBlock = [[] for _ in range(block_No)]
            for iROI in range(ROI_No):
                for iBlock in range(block_No):
                    if activatedNeurons[iROI][iBlock] == 1:
                        distancesPerBlock[iBlock].append(distanceFromElectrode[iROI])

            meanDistancesPerBlock = np.empty([block_No])
            for iBlock in range(block_No):
                meanDistancesPerBlock[iBlock] = np.mean(distancesPerBlock[iBlock], axis=0)
            axs[0, 1].plot(legend, meanDistancesPerBlock)
            axs[0, 1].set_ylabel('Mean distance of activated neurons (um)')
            axs[0, 1].set_xlabel('Stimulation amplitudes')
            plt.savefig(output_dir + '/09_18_amp_fig4_2.svg')

            #norm
            fig5, axs = plt.subplots(2, 2, figsize=(12, 8))
            x_axis = ['0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350-400', '400-450',
                      '450-500', '500-550', '550-600']
            # print(bin_numbers)
            bin_areas = [(math.pi * ((i+1)*binSize)** 2-math.pi * (i*binSize)**2)for i in range(bin_numbers)] #pi(r^2_outer-r^2_inner), outer: (i+1)*binSize, inner: i*binSize

            for iBlock in range(stimResults.shape[1]):
                for iBin in range(bin_numbers):
                    distanceMeans[iBlock][iBin] = np.mean(CAduringStim[iBlock][iBin])/bin_areas[iBin]
                    print()
                    activatedNeuronsDuringStim[iBlock][iBin] /= bin_areas[iBin]

                axs[0, 0].plot(x_axis, distanceMeans[iBlock, :])
                axs[1, 0].plot(x_axis, activatedNeuronsDuringStim[iBlock, :])
            axs[0, 0].legend(legend)
            axs[0, 0].set_ylabel('Mean dF/F0')
            axs[0, 0].set_xlabel('Normalised distance from electrode (um)')
            axs[1, 0].legend(legend)
            axs[1, 0].set_ylabel('Activated neurons')
            axs[1, 0].set_xlabel('Normalised distance from electrode (um)')

            # calculate and plot mean distance of activated neurons vs. blocks
            distancesPerBlock = [[] for _ in range(block_No)]
            for iROI in range(ROI_No):
                for iBlock in range(block_No):
                    if activatedNeurons[iROI][iBlock] == 1:
                        distancesPerBlock[iBlock].append(distanceFromElectrode[iROI])

            meanDistancesPerBlock = np.empty([block_No])
            # print(meanDistancesPerBlock.shape)
            for iBlock in range(block_No):
                meanDistancesPerBlock[iBlock] = np.mean(distancesPerBlock[iBlock], axis=0)
            axs[0, 1].plot(legend, meanDistancesPerBlock)
            axs[0, 1].set_ylabel('Mean distance of activated neurons (um)')
            axs[0, 1].set_xlabel('Stimulation amplitudes')
            plt.savefig(output_dir + '/09_17_amp_fig5_2.svg')
            np.save(output_dir + '/active_neurons', activeNeuronsPerBlock)
            np.save(output_dir + '/avgCAPerBlock', avgCAperBlock)
            np.save(output_dir + '/avgCaPerTrial', avgCAperTrial)
            np.savez(output_dir + '/results_GCaMP6s_09_17_freq.npz', activeNeuronsPerBlock=activeNeuronsPerBlock, avgCAperBlock=avgCAperBlock,
                     avgCAperTrial=avgCAperTrial)

            plt.show()


def plot_stim_traces(expDir, frame_rate, num_repeats, num_stims_per_repeat, list_of_file_nums, start_btw_stim, trial_delay, roi_idx, stim_dur=200, threshold_value = 3):
    '''

    Parameters
    ----------
    expDir
    frame_rate
    num_repeats
    num_stims_per_repeat
    list_of_file_nums
    start_btw_stim
    trial_delay
    roi_idx
    stim_dur
    threshold_value

    Returns
    -------

    '''
    roi_idx_og = roi_idx
    base_dir = Path(expDir)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]

    for numbers_to_merge in list_of_file_nums:
        suffix = '_'.join(map(str, numbers_to_merge))
        for dir in filenames:
            num_to_search_split = dir.split('MUnit_')
            if len(num_to_search_split) > 1:
                file_suffix = num_to_search_split[1].rsplit('.', 1)[0]
                if file_suffix == suffix:
                    matched_file = dir
                    print(matched_file)
                    break
        else:
            continue

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
            #print(stat)

    #--------CALCULATIONS--------
            # Extract the ROI indexes for cells
            # NB! AMPLITUDOKAT MODOSITSD HOGY A PLOTOKNAL RENDEZVE LEGYENEK
            stimulation_amplitudes = [10, 20, 30, 15, 25]
            cell_indices = np.where(iscell[:, 0] == 1)[0]  # Get indices of valid ROIs
            print(f'cells{cell_indices}')
            num_cells = len(cell_indices)
            stimulation_duration_frames = int(round((stim_dur / 1000) * frame_rate,0))
            num_cells = len(cell_indices)
            if roi_idx_og  not in cell_indices:
                raise ValueError

            F_index = np.where(cell_indices == roi_idx_og)[0][0]
            # Calculate time windows (1s before, 3s after)
            pre_frames = int(np.round(frame_rate,0))  # 1 second before
            post_frames = int(np.round((frame_rate * 3),0))  # 3 seconds after
            total_frames = int(np.round((pre_frames + post_frames), 0))
            start_btw_stim_frames = start_btw_stim * frame_rate
            trial_delay_frames = trial_delay * frame_rate

            # Storage for traces: Shape (ROIs, repeats, stimulations, frames)
            all_traces = np.zeros((num_repeats, num_stims_per_repeat, total_frames))
            start_timepoints = []
            for repeat in range(num_repeats):
                for stim_idx in range(num_stims_per_repeat):
                    if stim_idx == 0 and repeat == 0:
                        start_stim = int(stim_start_times[0][0])  # First stimulation from stim_start_times
                        start_timepoints.append(start_stim)
                    elif repeat == 0:

                        start_stim = int(np.round((stim_start_times[0][0] + stim_idx * start_btw_stim_frames),0))
                        start_timepoints.append(start_stim)
                    else:
                        start_stim = int(np.round((stim_start_times[0][0] + (stim_idx * start_btw_stim_frames) + (repeat * (((num_stims_per_repeat-1) * start_btw_stim_frames)+ trial_delay_frames))),0))
                        start_timepoints.append(start_stim)

                    # Define time window (1 sec before, 3 sec after)
                    pre_start = max(0, start_stim  - pre_frames)
                    post_end = min(F.shape[1], start_stim  + post_frames)
                    trace_segment = F[F_index, pre_start:post_end]
                    # store trace for roi repeat & stimulation index
                    if trace_segment.shape[0] == 0:
                        trace_segment = np.zeros(total_frames)
                    all_traces[repeat, stim_idx] = trace_segment

            min_trace_value = np.min(all_traces)
            max_trace_value = np.max(all_traces)
            #np.save(expDir + dir + '/start_timepoints.npy', start_timepoints)

        #---CALUCALTE ACTIVATED NEURONS PER REPEAT---
            # Activation calc
            baseline_duration = int(stim_start_times[0]) - 1
            activation_results = {roi_id: [] for roi_id in cell_indices}
            activation_count = 0
            stat = np.load(stat_path, allow_pickle=True)
            Ly, Lx = ops['Ly'], ops['Lx']
            for roi_id in cell_indices:
                F_index_act = np.where(cell_indices == roi_id)[0][0]
                baseline_data = F[F_index_act, :max(1, int(stim_start_times[0]) - 1)]
                baseline_avg = np.mean(baseline_data) if baseline_data.size > 0 else 0
                baseline_std = np.std(baseline_data) if baseline_data.size > 0 else 0
                threshold = baseline_std * threshold_value + baseline_avg
                roi_activation = []
                activated_rois = []
                is_active = False # whether the roi is active
                for repeat in range(num_repeats):
                    repeat_activation = []
                    for stim_idx in range(num_stims_per_repeat):
                        stim_idx_global = repeat * num_stims_per_repeat + stim_idx
                        start_time = start_timepoints[stim_idx_global]
                        stim_end_time = start_time + stimulation_duration_frames
                        stim_segment = F[F_index_act, start_time:stim_end_time]
                        avg_stim_resp = np.mean(stim_segment)
                        activation = 1 if avg_stim_resp > threshold else 0
                        repeat_activation.append(activation)
                        if activation == 1:
                            is_active = True
                    roi_activation.append(repeat_activation)
                activation_results[roi_id] = roi_activation
                if is_active:
                    activation_count += 1

            Ly, Lx = ops['Ly'], ops['Lx']
            masks = []
            activated_roi_indices = []
            for roi_data, pattern in activation_results.items():
                flat = [stim for repeat in pattern for stim in repeat] #flattens nested activation list so we can check if any activation
                if any(flat):
                    roi = stat[roi_data]
                    xpix = roi['xpix']
                    ypix = roi['ypix']
                    mask = np.zeros((Ly, Lx), dtype=np.uint8)
                    mask[ypix, xpix] = 1
                    masks.append(mask)
                    activated_roi_indices.append(roi_data)  # for cellreg_to_suite2p bc activated_roi_indices[i] will contain the original stat index

            if masks:
                mask_stack = np.stack(masks, axis=-0).astype(np.double)  # [nROIs, Ly, Lx]
                print(mask_stack.shape)
                output_folder = os.path.join(expDir, 'cellreg_files')
                os.makedirs(output_folder, exist_ok=True)
                out_name = f'{matched_file}.mat'
                out_path = os.path.join(output_folder, out_name)
                savemat(out_path, {'cells_map': mask_stack})
                print(f" Saved: {out_path} with shape {mask_stack.shape}")

            '''csv_path = os.path.join(expDir, dir, f'og_stat_idx.csv')
            statidx_df = pd.DataFrame()'''
            print(activated_roi_indices)
            # print(activation_results)
            print(f"Number of activated neurons: {activation_count} out of {num_cells} cells")
            column_names = [f"Repeat {i + 1}" for i in range(num_repeats)]
            activation_df = pd.DataFrame.from_dict(activation_results, orient='index', columns=column_names)
            activation_df.insert(0, "ROI", activation_df.index)
            csv_path = os.path.join(expDir, dir, f'activation_results_file{file_suffix}.csv')

        #=======Average x coordinates calculation=======
            x_coords_per_repeat_stim = [[[] for _ in range(num_stims_per_repeat)] for _ in range(num_repeats)]
            y_coords_per_repeat_stim = [[[] for _ in range(num_stims_per_repeat)] for _ in range(num_repeats)]
            for repeat in range(num_repeats):
                for stim_idx in range(num_stims_per_repeat):
                    x_coords = []
                    y_coords = []
                    activated_rois_csv = []
                    for roi_id in activation_results.keys():
                        act = activation_results[roi_id]
                        if act[repeat][stim_idx] == 1:  # any() missing but integare is not iterable
                            if 'med' in stat[roi_id]:
                                x_coords.append(stat[roi_id]['med'][1])
                                y_coords.append(stat[roi_id]['med'][0])
                                activated_rois_csv.append(roi_id)
                    if x_coords:

                        avg_x = np.mean(x_coords)
                        avg_y = np.mean(y_coords)
                    else:
                        avg_x = np.nan
                        avg_y = np.nan
                    x_coords_per_repeat_stim[repeat][stim_idx] = avg_x
                    y_coords_per_repeat_stim[repeat][stim_idx] = avg_y


                    med_df = pd.DataFrame({
                        'ROI': activated_rois_csv,
                        'X_coord': x_coords,
                        'Y_coord': y_coords
                    })
                    med_csv_path = os.path.join(expDir, dir, f'med_of_act_ns_{file_suffix}.csv')
                    med_df.to_csv(med_csv_path, index=False)
                    print(f"Saved centroid coordinates to {med_csv_path}")

            # Avg coords for each repeat
            data = []
            for stim_idx in range(num_stims_per_repeat):
                row = {'Stimulation': f'{stimulation_amplitudes[stim_idx]}uA'}
                for repeat in range(num_repeats):
                    row[f'Avg_X_Repeat_{repeat + 1}'] = x_coords_per_repeat_stim[repeat][stim_idx]
                    row[f'Avg_Y_Repeat_{repeat + 1}'] = y_coords_per_repeat_stim[repeat][stim_idx]
                data.append(row)
            # Calculate overall averages and standard deviations
            all_x_vals = [x for repeat_vals in x_coords_per_repeat_stim for x in repeat_vals]
            all_y_vals = [y for repeat_vals in y_coords_per_repeat_stim for y in repeat_vals]
            overall_avg_x = np.mean(all_x_vals)
            overall_avg_y = np.mean(all_y_vals)
            x_std = np.std(all_x_vals)
            y_std = np.std(all_y_vals)

            data.append({'Stimulation': 'Overall_Avg', 'Avg_X_Repeat_1': overall_avg_x, 'Avg_Y_Repeat_1': overall_avg_y})
            data.append({'Stimulation': 'Std_Dev_all', 'Avg_X_Repeat_1': x_std, 'Avg_Y_Repeat_1': y_std })
            data.append({'Stimulation': 'Sum_cells', 'Avg_X_Repeat_1': num_cells, **{f'Avg_X_Repeat_{i + 2}': '' for i in range(num_repeats - 1)}, **{f'Avg_Y_Repeat_{i + 1}': '' for i in range(num_repeats)}})

            # Sort data by ascending order of stimulation amplitudes
            data = sorted(data, key=lambda x: int(x['Stimulation'].replace('uA', '')) if 'uA' in x['Stimulation'] else float('inf'))

            df = pd.DataFrame(data)
            #calc avg & std per amplitude
            amplitude_groups = df[df['Stimulation'].str.contains('uA')].groupby('Stimulation')
            avg_std_data = []
            for name, group in amplitude_groups:
                avg_x = group[[f'Avg_X_Repeat_{i + 1}' for i in range(num_repeats)]].mean(axis=1).mean()
                avg_y = group[[f'Avg_Y_Repeat_{i + 1}' for i in range(num_repeats)]].mean(axis=1).mean()
                std_x = group[[f'Avg_X_Repeat_{i + 1}' for i in range(num_repeats)]].std(axis=1).mean()
                std_y = group[[f'Avg_Y_Repeat_{i + 1}' for i in range(num_repeats)]].std(axis=1).mean()
                avg_std_data.append({'Stimulation': name, 'Avg_X': avg_x, 'Avg_Y': avg_y, 'Std_X': std_x, 'Std_Y': std_y})

            for row in data:
                if 'uA' in row['Stimulation']:
                    stim = row['Stimulation']
                    avg_std_row = next(item for item in avg_std_data if item['Stimulation'] == stim)
                    row['Avg_X'] = avg_std_row['Avg_X']
                    row['Avg_Y'] = avg_std_row['Avg_Y']
                    row['Std_X'] = avg_std_row['Std_X']
                    row['Std_Y'] = avg_std_row['Std_Y']

            # DF for final data
            final_df = pd.DataFrame(data)

            csv_path = os.path.join(expDir, dir, f'avg_x_y_per_repeat_stim_file_{file_suffix}.csv')
            final_df.to_csv(csv_path, index=False)
            print(f"Avg med x,y values saved to {csv_path}")

        # =======Average x coordinates calculation END=======

        #Stimulation counts
            stim_activation_counts = []
            sorted_indices = np.argsort(stimulation_amplitudes)
            print(f's:{sorted_indices}')
            sorted_amplitudes = np.array(stimulation_amplitudes)[sorted_indices]
            for repeat in range(num_repeats):
                for stim_idx in range(num_stims_per_repeat):
                    sorted_stim_idx = sorted_indices[stim_idx]
                    stim_start = start_timepoints[repeat * num_stims_per_repeat + sorted_stim_idx]
                    stim_end = stim_start + stimulation_duration_frames
                    activated_rois = []
                    for roi_id in cell_indices:
                        F_index_act = np.where(cell_indices == roi_id)[0][0]
                        stim_data = F[F_index_act, stim_start:stim_end]
                        avg_stim_data = np.mean(stim_data)
                        baseline_data = F[F_index_act, :max(1, int(stim_start_times[0]) - 1)]
                        baseline_avg = np.mean(baseline_data) if baseline_data.size > 0 else 0
                        baseline_std = np.std(baseline_data) if baseline_data.size > 0 else 0
                        threshold = baseline_std * threshold_value + baseline_avg
                        if np.any(avg_stim_data > threshold):
                            activated_rois.append(roi_id)

                    stim_activation_counts.append({
                        'Repeat': repeat + 1,
                        'Stimulation': stim_idx + 1,
                        'Activated_ROIs': activated_rois,
                        'Sum_Activated_ROIs': len(activated_rois)
                    })
            # dataframe for csv
            data = {'stim ampl': [f'{amp}ua' for amp in sorted_amplitudes]}
            for repeat in range(num_repeats):
                data[f'Repeat {repeat + 1}'] = [', '.join(map(str, stim_activation_counts[repeat * num_stims_per_repeat + stim_idx]['Activated_ROIs'])) for stim_idx in range(num_stims_per_repeat)]
                data[f'Sum_Repeat {repeat + 1}'] = [stim_activation_counts[repeat * num_stims_per_repeat + stim_idx]['Sum_Activated_ROIs'] for stim_idx in range(num_stims_per_repeat)]
            stim_activation_df = pd.DataFrame(data)
            stim_activation_csv_path = os.path.join(expDir, dir, f'stim_activation_counts_file{file_suffix}.csv')
            stim_activation_df.to_csv(stim_activation_csv_path, index=False)
            print(f"Stimulation activation counts saved to {stim_activation_csv_path}")

            # ------------PLOTTING------------

            #---grid of subplots of activated rois---
            fig, axs = plt.subplots(num_repeats, num_stims_per_repeat, figsize=(15, 3 * num_repeats))
            #loop through each repeat and stimulation
            for repeat in range(num_repeats):
                for stim_idx in range(num_stims_per_repeat):
                    sorted_stim_idx = sorted_indices[stim_idx]

                    ax = axs[repeat, stim_idx]
                    im = np.zeros((ops['Ly'], ops['Lx']))

                    # plot activated rois for current repeat & stimulation
                    for roi_id in cell_indices:
                        if activation_results[roi_id][repeat][sorted_stim_idx] == 1:
                            ypix = stat[roi_id]['ypix'][~stat[roi_id]['overlap']]
                            xpix = stat[roi_id]['xpix'][~stat[roi_id]['overlap']]
                            im[ypix, xpix] = 1

                    ax.imshow(im *5, cmap='gray',vmin = 0, vmax =1, interpolation='nearest')
                    ax.set_title(f'Repeat {repeat + 1}, Stim {sorted_amplitudes[stim_idx]}uA')
                    ax.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(expDir, dir, 'roi_map_per_stim.svg'))
            plt.show()

    #----plot1 : Trace jele egy roinak stim num(novekvo ampl ertekben) & repeat num szerint sorban

            time = np.linspace(-1, 3, total_frames)
            # Create grid plot
            fig, axes = plt.subplots(num_repeats, num_stims_per_repeat, figsize=(5 * num_stims_per_repeat, 4 * num_repeats))
            #fig.suptitle('Calcium Traces Around Stimulation', fontsize=16)

            for repeat in range(num_repeats):

                for stim_idx in range(num_stims_per_repeat):
                    ax = axes[repeat, stim_idx]
                    ax.plot(time, all_traces[repeat, stim_idx], label=f"Repeat {repeat}, Stim {stim_idx}")
                    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)  # Mark stimulation onset
                    ax.set_title(f'Repeat {repeat + 1}, Stim {stim_idx + 1}')
                    ax.set_xlim(-1, 3)
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('ΔF/F')
                    ax.set_ylim(min_trace_value, max_trace_value)
                    ax.grid(True)

            plt.tight_layout()
            savepath = os.path.join(expDir, dir, 'stim_traces_grid.svg')
            plt.savefig(savepath)
            plt.show()

    #--------plot 2 Trace#1 overlapped minden trialre
            # Create figure of overlapped traces with one subplot per repeat
            amplitude_values = sorted([10, 20, 30, 15, 25])  # Adjust if necessary
            print(amplitude_values)
            amplitude_colors = {10: 'blue', 20: 'orange', 30: 'green', 15: 'red', 25: 'purple'}

            # Create figure with one subplot per repeat
            fig, axes = plt.subplots(1, num_repeats, figsize=(4 * num_repeats, 4), sharey=True)
            fig.suptitle(f'Overlapping Stimulations for ROI {roi_idx_og}', fontsize=16)

            # Define stimulation period for shading (e.g., 1s to 2s after onset)
            stim_start_sec = 1  # Relative to onset (adjust if needed)
            stim_end_sec = 2
            colors = ['blue', 'red', 'purple', 'brown', 'green']
            for repeat in range(num_repeats):
                ax = axes[repeat] if num_repeats > 1 else axes  # Handle case when only 1 repeat
                for stim_idx, amplitude in enumerate(amplitude_values):
                    # Assign color based on predefined mapping
                    color = amplitude_colors.get(amplitude, 'black')  # Default to black if missing

                    # Plot trace with defined color
                    ax.plot(time, all_traces[repeat, stim_idx], color=color, label=f"{amplitude} μA")

                avg_trace = np.mean(all_traces[:, stim_idx, :], axis=0)
                ax.plot(time, avg_trace, color='black', linewidth=1, label="Avg Response")

                # Add shaded region to indicate stimulation period
                #ax.axvspan(stim_start_sec, stim_end_sec, color='gray', alpha=0.3)

                # Formatting
                ax.set_xlabel('Time (s)')
                if repeat == 0:
                    ax.set_ylabel('Mean ΔF/F₀')
                ax.set_title(f'Trial {repeat + 1}')
                ax.set_ylim(min_trace_value, max_trace_value)
                ax.grid(True)

            # Add a single legend outside the subplots
            legend_handles = [plt.Line2D([0], [0], color=amplitude_colors[amplitude], lw=2, label=f"{amplitude} μA") for amplitude in amplitude_values]
            legend_handles.append(plt.Line2D([0], [0], color='black', lw=2, linestyle='dashed', label="Avg Response"))  # Add Avg Response
            fig.legend(handles=legend_handles, loc='upper right', fontsize=8, title="Legend", bbox_to_anchor=(0.98, 1))

            plt.tight_layout()
            plt.savefig(os.path.join(expDir, dir, f'overlapping_per_trial_for_roi0_05_tif17.png'))
            plt.show()

    # --------plot 3 Trace#2 overlapped minden amplitudora

            #overlap trials by amplitude
            trial_values = [1,2,3,4,5]  # Adjust as needed
            trial_colors = {1: 'blue', 2: 'orange', 3: 'green', 4: 'red', 5: 'purple'}

            # Create figure with one subplot per amplitude
            fig, axes = plt.subplots(1, len(amplitude_values), figsize=(4 * len(amplitude_values), 4), sharey=True)
            fig.suptitle(f'Overlapping Trials for Each Amplitude', fontsize=16)

            # Define stimulation period for shading (e.g., 1s to 2s after onset)
            stim_start_sec = 1  # Relative to onset (adjust if needed)
            stim_end_sec = 2
            for stim_idx, amplitude in enumerate(amplitude_values):
                ax = axes[stim_idx] if len(amplitude_values) > 1 else axes  # Handle case when only 1 amplitude
                #color = trial_colors.get(trial, 'black')  # Assign color based on amplitude
                for repeat, trial in enumerate(trial_values):
                    # Plot each trial for this amplitude
                    color = trial_colors.get(trial + 1, 'black')  # Assign color based on trial
                    ax.plot(time, all_traces[repeat, stim_idx], color=color, alpha=0.5, label=f"Trial {stim_idx + 1}")

                # Add a bold average trace for this amplitude
                avg_trace = np.mean(all_traces[:, stim_idx, :], axis=0)
                ax.plot(time, avg_trace, color='black', linewidth=2, label="Avg Response")

                # Add shaded region to indicate stimulation period
               # ax.axvspan(stim_start_sec, stim_end_sec, color='gray', alpha=0.3)

                # Formatting
                ax.set_xlabel('Time (s)')
                if stim_idx == 0:
                    ax.set_ylabel('Mean ΔF/F₀')
                #ax.set_ylabel('Mean ΔF/F₀')
                ax.set_title(f'{amplitude} μA')
                ax.set_ylim(min_trace_value, max_trace_value)

                # Remove duplicate legend entries
                handles, labels = ax.get_legend_handles_labels()
                unique_legend = dict(zip(labels, handles))  # Remove duplicates
                ax.grid(True)

            # single legend outside subplots
            legend_handles = [plt.Line2D([0], [0], color=color, lw=1, label=f"Trial {trial}") for trial, color in trial_colors.items()]
            legend_handles.append(plt.Line2D([0], [0], color='black', lw=2, linestyle='dashed', label="Avg Response"))  # Add Avg Response
            fig.legend(handles=legend_handles, loc='upper left', fontsize=8, title="Legend", bbox_to_anchor=(0.95, 1))

            plt.tight_layout()
            plt.savefig(os.path.join(expDir, dir, f'overlapping_per_param_for_roi0_05_tif17.png'))
            plt.show()

    # plot 4.2: ROIadik F_index & trace az osszes iscell==1 ROIra + .mat file save PER AMPLITUDE for cellreg
            num_rois = len(cell_indices)
            all_traces_grand_avg = np.zeros((num_rois, num_repeats, num_stims_per_repeat, total_frames))
            start_timepoints = []
            print(f'shape1: {all_traces_grand_avg.shape}')
            for roi_array_idx, roi_id in enumerate(cell_indices):
                F_index_for_all = np.where(cell_indices == roi_id)[0][0]
                for repeat in range(num_repeats):
                    for stim_idx in range(num_stims_per_repeat):
                        if stim_idx == 0 and repeat == 0:
                            start_stim = int(stim_start_times[0][0])
                            if roi_array_idx == 0:
                                start_timepoints.append(start_stim)
                        elif repeat == 0:
                            start_stim = int(np.round(stim_start_times[0][0] + stim_idx * start_btw_stim_frames))
                            if roi_array_idx == 0:
                                start_timepoints.append(start_stim)
                        else:
                            offset = (stim_idx * start_btw_stim_frames) + (repeat * (((num_stims_per_repeat - 1) * start_btw_stim_frames) + trial_delay_frames))
                            start_stim = int(np.round(stim_start_times[0][0] + offset))
                            if roi_array_idx == 0:
                                start_timepoints.append(start_stim)

                        pre_start = max(0, start_stim - pre_frames)
                        post_end = min(F.shape[1], start_stim + post_frames)
                        trace_segment = F[F_index_for_all, pre_start:post_end]

                        if trace_segment.shape[0] == 0:
                            trace_segment = np.zeros(total_frames)
                        all_traces_grand_avg[roi_array_idx, repeat, stim_idx, :] = trace_segment
            print(all_traces_grand_avg.shape)
            print(f'shape2: {all_traces_grand_avg.shape}')
            time = np.linspace(-1, 3, total_frames)
            # Create output dir
            sum_avg_dir = os.path.join(expDir, dir, 'sum_avg_dir')
            os.makedirs(sum_avg_dir, exist_ok=True)

            num_amps = len(amplitude_values)
            fig_combined, axes = plt.subplots(1, num_amps, figsize=(4 * num_amps, 4), sharey=True)
            if num_amps == 1:
                axes = [axes]

            sum_avg_per_amplitude = {}
            all_sum_avgs = []
            activation_count = {}
            # Compute average traces per amplitude
            for stim_idx, amplitude in enumerate(amplitude_values):
                print(f'stimidx {stim_idx}, ampl {amplitude}')
                roi_traces = []
                active_count = 0
                active_rois = []

                for roi_id in cell_indices:
                    roi_array_idx = np.where(cell_indices == roi_id)[0][0]
                    if any(activation_results[roi_id][repeat][sorted_indices[stim_idx]] == 1 for repeat in range(num_repeats)):
                        active_count +=1
                        active_rois.append(roi_id)
                    is_activated_in_any_repeat = any(activation_results[roi_id][repeat][sorted_indices[stim_idx]] == 1 for repeat in range(num_repeats))

                    if not is_activated_in_any_repeat:
                        continue
                    roi_trials = [
                        all_traces_grand_avg[roi_array_idx, repeat, sorted_indices[stim_idx], :]
                        for repeat in range(num_repeats)
                        if activation_results[roi_id][repeat][sorted_indices[stim_idx]] == 1
                    ]

                    if roi_trials:
                        roi_avg_trace = np.mean(roi_trials, axis=0)
                        '''plt.figure()
                        plt.plot(time, roi_avg_trace)
                        plt.show()'''
                        roi_traces.append(roi_avg_trace)
                activation_count[amplitude] = len(active_rois)
                df_counts = pd.DataFrame(list(activation_count.items()), columns = ["amplitude (ua)", "num of activated rois"])
                df_counts.to_csv(os.path.join(sum_avg_dir, "activation_counts.csv"), index = False)
                #print(len(roi_traces))
                xpix = []
                ypix = []
                masks = []
                Ly, Lx = ops['Ly'], ops['Lx']
                for roi_id in active_rois:
                    roi_s = stat[roi_id]
                    mask = np.zeros((Ly, Lx), dtype=np.uint8)
                    mask[roi_s['ypix'], roi_s['xpix']] = 1
                    masks.append(mask)

                if masks:
                    mask_stack = np.stack(masks, axis=-0).astype(np.double)  # [nROIs, Ly, Lx]
                    output_folder = os.path.join(expDir, 'cellreg_files')
                    os.makedirs(output_folder, exist_ok=True)
                    mat_filename = f'pix_data_for_{amplitude}.mat'
                    out_path = os.path.join(output_folder, mat_filename)
                    savemat(out_path, {'cells_map': mask_stack})
                    print(f"saved {mat_filename} file to {out_path} ")

                activation_count[amplitude] = len(active_rois)
                print(f'for amplitude: {amplitude} active number of roi: {len(active_rois)}')

                if roi_traces:
                    roi_traces = np.array(roi_traces)
                    if roi_traces.ndim != 2:
                        print(f"[Warning] roi_traces shape is {roi_traces.shape} — skipping this amplitude.")
                        continue
                    sum_avg_per_amplitude[amplitude] = np.mean(roi_traces, axis=0)
                    all_sum_avgs.append(sum_avg_per_amplitude[amplitude])
                    npy_path = os.path.join(sum_avg_dir, f'sum_avg_{amplitude}uA.npy')
                    np.save(npy_path, sum_avg_per_amplitude[amplitude])
                    #np.save(npy_path, sum_avg)

            # Calculate y-limits
            if all_sum_avgs:
                global_min = min(np.min(trace) for trace in all_sum_avgs)
                global_max = max(np.max(trace) for trace in all_sum_avgs)

            # Plot
            for stim_idx, amplitude in enumerate(amplitude_values):
                sum_avg = sum_avg_per_amplitude[amplitude]
                ax = axes[stim_idx]
                print(f'shape3:{sum_avg.shape}, {time.shape}')
                ax.plot(time, sum_avg, linewidth=2)
                ax.set_title(f"{amplitude} μA")
                ax.set_xlabel("Time (s)")
                ax.grid(True)
                ax.set_ylim(global_min, global_max)
                if stim_idx == 0:
                    ax.set_ylabel("Mean ΔF/F₀")

            df_counts = pd.DataFrame(list(activation_count.items()), columns=["Amplitude (μA)", "Num Activated ROIs"])
            df_counts.to_csv(os.path.join(sum_avg_dir, "activation_counts.csv"), index=False)

            fig_combined.suptitle("Average Traces per Amplitude", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            subplot_plot_path = os.path.join(sum_avg_dir, 'sum_avg_traces_subplot.svg')
            plt.savefig(subplot_plot_path)
            plt.show()

def plot_across_experiments(root_directory, tiff_dir, list_of_file_nums, frame_rate ):
    '''
    Plot traces from multiple experiments across different stimulation amplitudes.
    Parameters
    ----------
    root_directory
    tiff_dir
    list_of_file_nums
    frame_rate

    Returns
    -------

    '''
    # Settings
    amplitude_keys = ['10', '20', '30','15', '25']  # uA values to expect
    pre_frames = int(np.round(frame_rate, 0))  # 1 second before
    post_frames = int(np.round((frame_rate * 3), 0))  # 3 seconds after
    total_frames = int(np.round((pre_frames + post_frames), 0))

    base_dir = Path(tiff_dir)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]
    matched_dirs = []

    for numbers_to_merge in list_of_file_nums:
        suffix = '_'.join(map(str, numbers_to_merge))
        for dir in filenames:
            if f'MUnit_{suffix}' in dir:
                matched_dirs.append(os.path.join(root_directory, 'merged_tiffs/', dir))
    print(matched_dirs)
    # collect traces by amplitude
    traces_by_amplitude = {amp: [] for amp in amplitude_keys}
    for dir_path in matched_dirs:
        sum_avg_subdir = os.path.join(dir_path, 'sum_avg_dir')
        for file in os.listdir(sum_avg_subdir):
            match = re.match(r'sum_avg_(\d+)uA\.npy', file)
            if match:
                amp = match.group(1)
                print(traces_by_amplitude)
                if amp in traces_by_amplitude:
                    trace_path = os.path.join(sum_avg_subdir, file)
                    trace = np.load(trace_path)
                    traces_by_amplitude[amp].append(trace)

    # subplots for amplitude figs
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    time = np.linspace(-1, 3, total_frames)

    for idx, amp in enumerate(amplitude_keys):
        ax = axes[idx]
        traces = traces_by_amplitude[amp]
        if traces:
            for trace in traces:
                ax.plot(time, trace, alpha=0.6, label=f'Merged trace')
            ax.set_title(f"{amp} μA")
            ax.set_xlabel("Time (s)")
            ax.grid(True)
            if idx == 0:
                ax.set_ylabel("Mean ΔF/F₀")
        else:
            ax.set_title(f"{amp} μA\n(no data)")
            ax.set_axis_off()

    plt.suptitle("Overlay of sum_avg traces by amplitude across experiments", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    #plt.show()


def analyze_merged_activation_and_save(exp_dir, mesc_file_name, tiff_dir, list_of_file_nums,  stim_segm, threshold_value, trialNo, trialDur, frameRate):
    '''

    Parameters
    ----------
    exp_dir
    mesc_file_name
    tiff_dir
    list_of_file_nums
    stim_segm: how many frames to look at for the stim_avg
    threshold_value
    trialNo
    trialDur: trial delay and stimulatio
    frameRate

    Returns
    -------
    mask_stack for cellreg
    activation_df
    med_val_df

    '''

    fileId_path = os.path.join(exp_dir, 'fileId.txt')
    trigger_path = os.path.join(exp_dir, 'trigger.txt')
    frameNo_path = os.path.join(exp_dir, 'frameNo.txt')

    file_ids = []
    triggers = []
    frame_lens = []

    with open(fileId_path, 'r') as f_ids, open(trigger_path, 'r') as f_triggers, open(frameNo_path, 'r') as f_frames:
        for id_line, trig_line, frame_line in zip(f_ids, f_triggers, f_frames):
            trig_line = trig_line.strip()
            frame_line = frame_line.strip()
            if trig_line.lower() == 'none' or trig_line == '' or frame_line == '':
                #print(f" Skipping invalid line: trigger={trig_line}, frame={frame_line}")
                continue
            unit_id = int(id_line.strip().replace('MUnit_', ''))
            trigger_val = int(trig_line)
            frame_len = int(frame_line)

            file_ids.append(unit_id)
            triggers.append(trigger_val)
            frame_lens.append(frame_len)

    # unit_number --> (trigger, block_len)
    fileid_to_info = {
        file_id: {'trigger': trig, 'block_len': frame_len}
        for file_id, trig, frame_len in zip(file_ids, triggers, frame_lens)
    }
    base_dir = Path(tiff_dir)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]
    cellreg_dir = Path(os.path.join(base_dir, '/cellreg_files'))
    cellreg_dir.mkdir(exist_ok=True)

    for group_idx, file_group in enumerate(list_of_file_nums):
        for block_idx in range(len(file_group)):
            file_num = file_group[block_idx]
            block_len = fileid_to_info[file_num]['block_len']

        suffix = '_'.join(map(str, file_group))
        matched_file = None
        for dir in filenames:
            if f'MUnit_{suffix}' in dir:
                matched_file = dir
                break
        if matched_file is None:
            print(f"No matched directory for MUnit_{suffix}")
            continue
        #print(matched_file)
        exp_dir = os.path.join(base_dir, matched_file)
        suite2p_dir = os.path.join(exp_dir, 'suite2p', 'plane0')
        # Load base data
        mesc_path = os.path.join(exp_dir, 'mesc_data.npy')
        F = np.load(os.path.join(suite2p_dir, 'F0.npy'), allow_pickle=True)
        iscell = np.load(os.path.join(suite2p_dir, 'iscell.npy'), allow_pickle=True)
        stat = np.load(os.path.join(suite2p_dir, 'stat.npy'), allow_pickle=True)
        ops = np.load(os.path.join(suite2p_dir, 'ops.npy'), allow_pickle=True).item()

        Ly, Lx = ops['Ly'], ops['Lx']
        valid_rois = np.where(iscell[:, 0] == 1)[0]

        activated_roi_count = 0
        all_activated_roi_indices = []
        all_traces = []
        all_y_coords = []
        all_x_coords = []
        all_block_indices = []
        all_masks = []

        for block_idx, file_num in enumerate(file_group):
            block_stim_time = fileid_to_info[file_num]['trigger']
            block_len = fileid_to_info[file_num]['block_len']

            start_frame = block_idx * block_len
            end_frame = start_frame + block_len

            for i, roi in enumerate(valid_rois):
                F_block = F[i, start_frame:end_frame]
                baseline = F_block[:block_stim_time]
                baseline_avg = np.mean(baseline)
                baseline_std = np.std(baseline)
                threshold = baseline_avg + threshold_value * baseline_std

                trialDurInFrames = int(round(trialDur * frameRate))
                stim_segments = []
                for j in range(trialNo):
                    seg_start = block_stim_time + (trialDurInFrames * j)
                    stim_segment = F_block[seg_start: seg_start + stim_segm]
                    stim_segments.append(stim_segment)

                stim_avg = np.mean(stim_segments)
                '''active = False
                activation = 1 if stim_avg > threshold else 0'''
                active = stim_avg > threshold
                if active:
                    #active = True
                    activated_roi_count +=1
                #if active:
                    all_activated_roi_indices.append(roi)
                    all_traces.append(F_block)
                    all_block_indices.append(block_idx)

                    #centroid coords:
                    roi_stat = stat[roi]
                    #print(roi_stat)
                    all_x_coords.append(roi_stat[roi]['med'][1])
                    all_y_coords.append(roi_stat[roi]['med'][0])

                    mask = np.zeros((Ly, Lx), dtype=np.uint8)
                    mask[roi_stat['ypix'], roi_stat['xpix']] = 1
                    all_masks.append(mask)

        out_path = os.path.join(tiff_dir, matched_file)
        if all_masks:
            mask_stack = np.stack(all_masks, axis=0).astype(np.double)
            mat_path = os.path.join(out_path, f'cellreg_input_{mesc_file_name}_{file_num}.mat')
            savemat(mat_path, {'cells_map': mask_stack})
        activation_df = pd.DataFrame({
            'Block_Index': all_block_indices,
            'ROI_Count': activated_roi_count,
            'ROI_Index': all_activated_roi_indices,
            'Trace': all_traces
        })

        med_val_df = pd.DataFrame({
            'Block_Index': all_block_indices,
            'ROI_Index': all_activated_roi_indices,
            'Y_coord': all_y_coords,
            'X_coord': all_x_coords
            })

            csv_path = os.path.join(out_path, f'activated_neurons_{mesc_file_name}_{file_num}.csv')
            activation_df.to_csv(csv_path, index=False)
            med_csv_path = os.path.join(out_path, f'med_of_act_ns_{mesc_file_name}_{file_num}.csv')
            med_val_df.to_csv(med_csv_path, index=False)

        print(f'Processed finished for {matched_file}')


def collect_file_paths_for_blocks(tiff_dir, list_of_file_nums):
    """
    For each set of TIFF numbers in list_of_file_nums, find the corresponding merged folder (MUnit_3_4_5 etc)
    and return paths to key files.

    :param tiff_dir: base directory containing merged TIFF folders
    :param list_of_file_nums: e.g. [[3,4,5], [6,7,8]]
    :return: list of dicts with file paths for each matched folder
    """
    base_dir = Path(tiff_dir)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]

    results = []

    for numbers_to_merge in list_of_file_nums:
        suffix = '_'.join(map(str, numbers_to_merge))
        matched_file = None
        for dir in filenames:
            num_to_search_split = dir.split('MUnit_')
            if len(num_to_search_split) > 1:
                file_suffix = num_to_search_split[1].rsplit('.', 1)[0]
                if file_suffix == suffix:
                    matched_file = dir
                    print(f"Matched: {matched_file}")
                    break
        else:
            continue  # skip to next block if no match found

        if matched_file:
            folder_path = base_dir / matched_file
            suite2p_path = folder_path / 'suite2p' / 'plane0'
            result = {
                'merged_folder': matched_file,
                'folder_path': folder_path,
                'iscell_path': suite2p_path / 'iscell.npy',
                'stat_path': suite2p_path / 'stat.npy',
                'ops_path': suite2p_path / 'ops.npy',
                'electrodeROI_path': folder_path / 'selected_elec_ROI.npy',
                'csv_file_path': folder_path / 'elec_roi_info.csv',
                'stimTimes_path': folder_path / 'stimTimes.npy',
                'frameNum_path': folder_path / 'frameNum.npy'
            }
            results.append(result)

    return results

def get_stim_frames_to_video(exp_dir, tiff_dir, list_of_file_nums, stim_segm=15, threshold_value=3.0, block_order=[5,6,8,2,10,9,3,1,7,4,0]):
    import cv2
    output_video_name = 'stim_activation_frames.avi'
    base_dir = Path(tiff_dir)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]
    all_frames = []

    for group_idx, file_group in enumerate(list_of_file_nums):
        suffix = '_'.join(map(str, file_group))
        matched_file = next((f for f in filenames if f'MUnit_{suffix}' in f), None)
        fileId_path = os.path.join(exp_dir, 'fileId.txt')
        trigger_path = os.path.join(exp_dir, 'trigger.txt')
        frameNo_path = os.path.join(exp_dir, 'frameNo.txt')
        if matched_file is None:
            print(f"No matched directory for MUnit_{suffix}")
            continue

        print(f"Processing group: {matched_file}")
        exp_dir_merged = os.path.join(base_dir, matched_file)
        suite2p_dir = os.path.join(exp_dir_merged, 'suite2p', 'plane0')

        F = np.load(os.path.join(suite2p_dir, 'F.npy'), allow_pickle=True)
        iscell = np.load(os.path.join(suite2p_dir, 'iscell.npy'), allow_pickle=True)
        stat = np.load(os.path.join(suite2p_dir, 'stat.npy'), allow_pickle=True)
        ops = np.load(os.path.join(suite2p_dir, 'ops.npy'), allow_pickle=True).item()
        Ly, Lx = ops['Ly'], ops['Lx']
        valid_rois = np.where(iscell[:, 0] == 1)[0]

        with open(fileId_path, 'r') as f:
            file_ids = [int(line.strip().replace('MUnit_', '')) for line in f]
            print(file_ids)

        with open(trigger_path, 'r') as f:
            triggers = [int(line.strip()) if line.strip().lower() != 'none' and line.strip() else None for line in f]

        with open(frameNo_path, 'r') as f:
            frame_lens = [int(line.strip()) for line in f if line.strip()]

        local_start_frames = np.cumsum([0] + frame_lens[:-1])
        local_block_info = list(zip(file_ids, triggers, frame_lens, local_start_frames))

        if block_order:
            try:
                ordered_file_ids = [file_group[i] for i in block_order]
            except IndexError:
                raise ValueError(f"Invalid block_order {block_order} for file_group {file_group}")
        else:
            ordered_file_ids = file_group
        ordered_block_info = []
        for file_id in ordered_file_ids:
            matched = False
            for block in local_block_info:
                if block[0] == file_id:
                    ordered_block_info.append(block)
                    matched = True
                    break
            if not matched:
                print(f"Warning: file_id {file_id} not found in merged block.")

        for file_id, trigger, frame_len, block_start in ordered_block_info:
            if trigger is None or trigger + stim_segm > frame_len:
                print(f"Skipping block {file_id} due to invalid trigger.")
                continue

            masks = []
            for roi in valid_rois:
                mask = np.zeros((Ly, Lx), dtype=np.uint8)
                stat_roi = stat[roi]
                mask[stat_roi['ypix'], stat_roi['xpix']] = 1
                masks.append(mask)

            # For 15 frames after trigger
            for frame_idx in range(stim_segm):
                composite = np.zeros((Ly, Lx), dtype=np.float32)

                for i, roi in enumerate(valid_rois):
                    trace = F[roi]
                    if trigger < 10: continue  # skip edge cases
                    baseline = np.mean(trace[trigger - 10:trigger])
                    deltaF = trace[trigger + frame_idx] - baseline
                    deltaF = max(deltaF, 0)
                    composite += masks[i] * deltaF

                # Normalize and convert
                if np.max(composite) > 0:
                    composite = 255 * (composite / np.max(composite))
                composite = composite.astype(np.uint8)
                bgr_frame = cv2.cvtColor(composite, cv2.COLOR_GRAY2BGR)
                all_frames.append(bgr_frame)

    height, width, _ = all_frames[0].shape
    out_path = os.path.join(tiff_dir, output_video_name)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_path, fourcc, 5, (width, height), isColor=True)

    if not out.isOpened():
        raise IOError(f"Failed to open video writer for: {out_path}")

    for frame in all_frames:
        out.write(frame)
    out.release()
    print(f"Saved video to: {out_path}")

def create_video_from_mesc_tiffs(mesc_dir, list_of_file_nums, output_video_name='first_stim_video.avi', stim_segm=15,block_order=None):
    import tifffile
    import cv2
    file_ids = []
    triggers = []
    frame_lens = []

    with open(os.path.join(mesc_dir, 'fileId.txt'), 'r') as f_ids, \
         open(os.path.join(mesc_dir, 'trigger.txt'), 'r') as f_trigs, \
         open(os.path.join(mesc_dir, 'frameNo.txt'), 'r') as f_lens:
        for id_line, trig_line, len_line in zip(f_ids, f_trigs, f_lens):
            trig_line = trig_line.strip().lower()
            if trig_line == 'none' or trig_line == '':
                continue
            file_ids.append(int(id_line.strip().replace('MUnit_', '')))
            triggers.append(int(trig_line.strip()))
            frame_lens.append(int(len_line.strip()))

    file_nums_flat = [item for sublist in list_of_file_nums for item in sublist]
    all_blocks = list(zip(file_ids, triggers, frame_lens))
    blocks = [b for b in all_blocks if b[0] in file_nums_flat]

    if block_order:
        try:
            blocks = [blocks[i] for i in block_order]
        except IndexError:
            raise ValueError(f"Invalid block_order: {block_order}")

    all_frames = []

    for file_id, trigger, frame_len in blocks:
        tiff_filename = f"{Path(mesc_dir).stem}_MUnit_{file_id}.tif"
        tiff_path = os.path.join(mesc_dir, tiff_filename)
        if not os.path.exists(tiff_path):
            print(f"tiff not found: {tiff_path}")
            continue

        if trigger + stim_segm > frame_len:
            print(f"Skipping file {file_id}, stim window exceeds recording length.")
            continue

        print(f"Loading {tiff_path} (frames {trigger}–{trigger+stim_segm})")
        tiff_stack = tifffile.imread(tiff_path)

        # Extract 15 frames from trigger
        stim_segment = tiff_stack[trigger:trigger + stim_segm]

        for frame in stim_segment:
            norm_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            norm_frame = norm_frame.astype(np.uint8)
            color_frame = cv2.cvtColor(norm_frame, cv2.COLOR_GRAY2BGR)
            all_frames.append(color_frame)

    if not all_frames:
        print("No frames collected.")
        return

    height, width, _ = all_frames[0].shape
    out_path = os.path.join(mesc_dir, output_video_name)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_path, fourcc, 5, (width, height), isColor=True)

    for frame in all_frames:
        out.write(frame)

    out.release()
    print(f"\n Saved video to: {out_path}")

def speed_up():
    import cv2

    cap = cv2.VideoCapture('C:/Users/NP/Documents/Hyperstim_Eszter/opencv_test/M20240311_0005.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    i = 0  # frame counter
    frameTime = 1  # time of each frame in ms, you can add logic to change this value.
    width = cap.get(3)
    heigth = cap.get(4)
    size = (width, heigth)
    frame_rate = cap.get(5)
    # frame_num_to_cut =
    # cut_frames = (frame_num_to_cut*(1/frame_rate)*1000)
    output = cv2.VideoWriter('C:/Users/NP/Documents/Hyperstim_Eszter/opencv_test/M20240311_0005_10x.avi', fourcc,
                             frame_rate, (int(width), int(heigth)))
    # cap.set(cv2.CAP_PROP_POS_MSEC, cut_frames)
    while (cap.isOpened()):
        ret = cap.grab()  # grab frame
        i = i + 1  # increment counter
        # if i > cut_frames:
        if i % 10 == 0:  # display only one third of the frames, you can change this parameter according to your needs
            ret, frame = cap.retrieve()  # decode frame
            cv2.imshow('frame', frame)
            output.write(frame)
            if cv2.waitKey(frameTime) & 0xFF == ord('q'):
                break
        # print(ret, frame)
    cap.release()
    output.release()
    cv2.destroyAllWindows()
