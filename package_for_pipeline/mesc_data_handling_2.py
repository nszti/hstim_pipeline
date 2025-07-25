import numpy as np
import os
from pathlib import Path
import tifftools
import pandas as pd
import pickle


def tiff_merge(mesc_file_name, list_of_file_nums, output_root_directory):
    '''

    Parameters
    ----------
    mesc_file_name: name of the MESc from which the tiffs have been extracted
    list_of_file_nums : which tiff files to merge together
    output_root_directory: outermost directory of the experiment

    Returns
    -------
    makes 'merged_tiffs' directory, saves merged files to separate directories in it
    '''
    suffix = '_MUnit_'
    outer_directory = os.path.join(output_root_directory, 'merged_tiffs')
    os.makedirs(outer_directory, exist_ok=True)
    print(f"MESc file name: {mesc_file_name}")
    #output_directory = output_root_directory

    freq_path = outer_directory + '/frequencies.npy'
    frequencies = np.load(freq_path)
    electrodeROI_path = outer_directory + '/electrode_rois.npy'
    elec_ROIs = np.load(electrodeROI_path)

    for numbers_to_merge in list_of_file_nums:
        #print(numbers_to_merge)
        base_filename = mesc_file_name + suffix
        #output_path = output_directory

        tiff_files_li = [os.path.join(output_root_directory, f"{base_filename}{num}.tif") for num in numbers_to_merge]
        for file in tiff_files_li:
            #print(file)
            if not os.path.isfile(file):
                print(f"Error: File {file} does not exist:(")
                exit(1)

        output_dirname = 'merged_' + base_filename + '_'.join(map(str, numbers_to_merge))
        output_filepath = os.path.join(outer_directory, output_dirname)
        os.makedirs(output_filepath, exist_ok=True)

        output_filename = 'merged_' + base_filename + '_'.join(map(str, numbers_to_merge)) + '.tif'
        output_fpath = os.path.join(output_filepath, output_filename)

        tifftools.tiff_concat(tiff_files_li, output_fpath, overwrite=False)
        print(f"Files {tiff_files_li} merged into {output_filepath}")

        selected_freqs = frequencies[numbers_to_merge]
        print(f"Used frequency: {selected_freqs}")
        output_dirname = 'merged_' + base_filename + '_'.join(map(str, numbers_to_merge))
        np.save(outer_directory + '/' + output_dirname + '/selected_freqs.npy', selected_freqs)

        selected_electrode_ROIs = elec_ROIs[numbers_to_merge]
        #print(f"Used electrode(s): {selected_electrode_ROIs}")
        np.save(outer_directory + '/' + output_dirname + '/selected_elec_ROI.npy', selected_electrode_ROIs)

def extract_stim_frame(directory, mesc_DATA_file, list_of_file_nums):
    s = 'merged_tiffs/'
    base_dir = Path(os.path.join(directory, s))
    mesc_data = np.load(f'{base_dir}/{mesc_DATA_file}', allow_pickle=True)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]

    fileIds = mesc_data[:, 0]
    frame_nos = mesc_data[:, 1]
    triggers = mesc_data[:, 2]

    for sublist in list_of_file_nums:
        suffix = '_'.join(map(str, sublist))
        '''
        if len(sublist) == 1:
            suffix = f'_{sublist[0]}'
        else:
        '''
        #print(f"for sublist {sublist} : {suffix} suffix")
        #matched_dir = next((file for file in filenames if file.endswith(suffix) and file.startswith('merged')), None)
        matched_dir = None
        for file in filenames:
            #print(file)
            parts = file.split('MUnit_')
            #print(parts)
            if len(parts) > 1:
                file_suffix = parts[1].rsplit('.', 1)[0]
                #print(file_suffix)
                if file_suffix == suffix:
                    matched_dir = file
                    #print(matched_dir)
                    break
            '''
            if all(str(num) in file for num in sublist) and file.endswith(suffix) and file.startswith('merged'):
                matched_dir = file
                #print(f" matched dir : {matched_dir}")
                break
            '''

        if matched_dir:
            save_dir = base_dir / matched_dir
            #print(f"saved directory : {save_dir}")
            all_frames = []
            all_triggers = []

            for num_id in sublist:
                #print(num_id)
                #search_frameNumber
                num_id_str = str(num_id)
                mask = [fid.split('_')[-1] == num_id_str for fid in fileIds]
                filtered_frame_nos = frame_nos[mask]
                filtered_triggers = triggers[mask]
                #print(f"num id for frame {num_id}, {filtered_frame_nos}")
                #print(f"num id for trigger {num_id}, {filtered_triggers} ")
                all_frames.append(filtered_frame_nos)
                all_triggers.append(filtered_triggers)

            np.save(save_dir / f'frameNum.npy', all_frames, allow_pickle=True)  # lists of arrays saved
            np.save(save_dir / f'stimTimes.npy', all_triggers, allow_pickle=True)
            print(f"Frame numbers and stimulation timepoints for merged tiffs {sublist} are saved to dir: {matched_dir}!")
        else:
            print(f"No matching directory found for suffix {suffix}")





'''
def search_frameNumber(directory, mesc_data_file, num_id):
    mesc_data = np.load(directory + mesc_data_file, allow_pickle=True)

    fileIds = mesc_data[:, 0]
    frame_nos = mesc_data[:, 1]

    num_id_str = str(num_id)
    mask = [fid.split('_')[-1] == num_id_str for fid in fileIds]
    print(mask)
    filtered_frame_nos = frame_nos[mask]
    return filtered_frame_nos
def search_stim_start(directory, mesc_data_file, num_id):
    mesc_data = np.load(directory + mesc_data_file, allow_pickle=True)

    fileIds = mesc_data[:, 0]
    triggers = mesc_data[:, 2]

    num_id_str = str(num_id)
    mask = [fid.split('_')[-1] == num_id_str for fid in fileIds]
    #print(mask)
    filtered_triggers = triggers[mask]
    #print(filtered_triggers)
    return filtered_triggers




# for i, dir in enumerate(filenames):
# if i < len(list_of_file_nums):
# sublist = list_of_file_nums[i]
'''



