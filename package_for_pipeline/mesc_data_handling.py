
import numpy as np
import os
from pathlib import Path
import tifffile
import tifftools
import pandas as pd
import pickle

def tiff_merge(mesc_file_name, list_of_file_nums, output_root_directory, mesc_DATA_file, stimulation):
    '''
    Parameters
    ----------
    mesc_file_name: name of the MESc from which the tiffs have been extracted
    list_of_file_nums : which tiff files to merge together (list of lists of ints)
    output_root_directory: outermost directory of the experiment
    stimulation: whether the files contain stimulation data. if True, saves frequency and electrode ROI info

    Returns
    -------
    makes 'merged_tiffs' directory, saves merged files to separate directories in it
    if stimulation is True, also saves frequencies and electrode ROI numbers to each merged directory.
    '''

    suffix = '_MUnit_'
    outer_directory = os.path.join(output_root_directory, 'merged_tiffs')
    os.makedirs(outer_directory, exist_ok=True)
    print(f"MESc file name: {mesc_file_name}")

    mesc_data = np.load(os.path.join(outer_directory, mesc_DATA_file), allow_pickle=True)
    fileIds = mesc_data[:, 0]
    file_nums = [int(fid.split('_')[-1]) for fid in fileIds]
    file_index_map = {num: idx for idx, num in enumerate(file_nums)}
    #filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]

    if stimulation:
        freq_path = os.path.join(outer_directory, 'frequencies.npy')
        frequencies = np.load(freq_path)
        electrodeROI_path = os.path.join(outer_directory, 'electrode_rois.npy')
        elec_ROIs = np.load(electrodeROI_path)

    for numbers_to_merge in list_of_file_nums:
        base_filename = mesc_file_name + suffix
        tiff_files_li = [os.path.join(output_root_directory, f"{base_filename}{num}.tif") for num in numbers_to_merge]

        for file in tiff_files_li:
            print(file)
            if not os.path.isfile(file):
                print(f"Error: File {file} does not exist:(")
                exit(1)

        output_dirname = 'merged_' + base_filename + '_'.join(map(str, numbers_to_merge))
        output_filepath = os.path.join(outer_directory, output_dirname)
        os.makedirs(output_filepath, exist_ok=True)

        output_filename = output_dirname + '.tif'
        output_fpath = os.path.join(output_filepath, output_filename)

        all_pages = []
        for file in tiff_files_li:
            with tifffile.TiffFile(file) as tif:
                all_pages.append(tif.asarray())

        merged_stack = np.concatenate(all_pages, axis=0)
        tifffile.imwrite(output_fpath, merged_stack.astype(np.uint16))
        print(f"Files {tiff_files_li} merged into {output_fpath}")

        if stimulation:
            try:
                selected_indices = [file_index_map[num] for num in numbers_to_merge]
            except KeyError as e:
                print(f"File number {e} not found in MESc metadata.")
                exit(1)

            selected_freqs = frequencies[selected_indices]
            print(f"Used frequency: {selected_freqs}")
            np.save(os.path.join(output_filepath, 'selected_freqs.npy'), selected_freqs)

            selected_electrode_ROIs = elec_ROIs[selected_indices]
            np.save(os.path.join(output_filepath, 'selected_elec_ROI.npy'), selected_electrode_ROIs)

            print(f"Saved selected frequency and electrode ROI info for group {numbers_to_merge}")



def extract_stim_frame(root_directory, mesc_DATA_file, list_of_file_nums):
    '''

    Parameters
    ----------
    root_directory
    mesc_DATA_file
    list_of_file_nums

    Returns
    -------
    stimTimes.npy
    frameNum.npy
    '''
    s = 'merged_tiffs/'
    base_dir = Path(os.path.join(root_directory, s))
    print(base_dir)
    mesc_data = np.load(f'{base_dir}/{mesc_DATA_file}', allow_pickle=True)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]

    fileIds = mesc_data[:, 0]
    frame_nos = mesc_data[:, 1]
    print(mesc_data)
    triggers = mesc_data[:, 2]

    for sublist in list_of_file_nums:
        suffix = '_'.join(map(str, sublist))
        matched_dir = None
        for file in filenames:
            parts = file.split('MUnit_')
            if len(parts) > 1:
                file_suffix = parts[1].rsplit('.', 1)[0]
                if file_suffix == suffix:
                    matched_dir = file
                    break
        if matched_dir:
            save_dir = base_dir / matched_dir
            all_frames = []
            all_triggers = []

            for num_id in sublist:
                num_id_str = str(num_id)
                mask = [fid.split('_')[-1] == num_id_str for fid in fileIds]
                filtered_frame_nos = frame_nos[mask]
                filtered_triggers = triggers[mask]
                np.save(save_dir/ f'stimTime_{num_id}', filtered_triggers, allow_pickle= True)
                all_frames.append(filtered_frame_nos)
                all_triggers.append(filtered_triggers)

            np.save(save_dir / f'frameNum.npy', all_frames, allow_pickle=True)  # lists of arrays saved
            np.save(save_dir / f'stimTimes.npy', all_triggers, allow_pickle=True)
            print(f"Frame numbers and stimulation timepoints for merged tiffs {sublist} are saved to dir: {matched_dir}!")
        else:
            print(f"No matching directory found for suffix {suffix}")



