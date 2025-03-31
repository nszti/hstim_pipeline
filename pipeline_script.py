from pathlib import Path
import numpy as np
from package_for_pipeline import functions

from package_for_pipeline import mesc_data_handling
from package_for_pipeline import suite2p_script
from mesc_loader import extract_useful_xml_params
from general import ascii_to_str, find_frame_index_from_timestamp
from package_for_pipeline import mesc_tiff_extract
from package_for_pipeline import functions_og
#from package_for_pipeline import suite2p_neuropil
from package_for_pipeline import overlap

import os

#------STEPS IN PIPELINE------
'''
 0.: modify values in the 'values to change' section. NB! gcamp & stim_type are to be changes also,params for them can be modifited in 'suite2p_script.py' & 'functions.data_analysis_values()' respectively
 1.: run mesc_tiff_extract.analyse_mesc_file() (--> now 'merged_tiffs' folder is created, 'mesc_data.npy' is in it)
//manual work #1
 2.: modify 'frequency_to_save.py', 'electrode_roi_to_save.py' to match current file data (use experiment reports for fequency, electrode_Roi should be 0 for all files) 
 3.: add tiff numbers to merge in the 'list_of_file_nums' list. NB! it's possible to add 1 and multiple nums in a list & to add multiple lists to merge
//manual work #1 end
 4.: run: 'mesc_data_handling.extract_stim_frame()'-> 'mesc_data_handling.tiff_merge()'-> 'suite2p_script.run_suite2p()' 
//manual work #2
 5.: load the processed file into suite2p & manually add electrode ROI where needed in suite2p (suite2p GUI>file>manual labelling>alt+click /might have to wait for it to load the traces, often makes the software to freez/)
 6.: manually modify ROIs if needed (add, remove). NB! check the electrode and the side of the FOV for false ROIs & check the not cell ROIs as well you can look back the og mesc file & check the registered binary file in suite2p (ctrl+b)
//manual work #2 end
 7.: run 'functions.py' for analysis. NB! for timecourse_vals() the 3rd param ('num_trials') should be changed manually + there's probably a bug with stim_durations (my bad, will fix it)  
'''
#------STEPS IN PIPELINE END------

#------VALUES TO CHANGE------
#root_directory = 'c:/Hyperstim/data_analysis/2025-03-25-Amouse-invivo-GCaMP6f/' #
root_directory = 'd:/2P/Experiments/AMouse-2025-03-05-invivo-GCaMP6f/'
#tiff_directory = 'c:/Hyperstim/data_analysis/2025-03-25-Amouse-invivo-GCaMP6f/merged_tiffs/'
tiff_directory= 'd:/2P/Experiments/AMouse-2025-03-05-invivo-GCaMP6f/merged_tiffs/'
mesc_file_name = '2025-03-25-Amouse-invivo-GCaMP6f'
mesc_DATA_file = 'mesc_data.npy' #from mesc_tiff_extract
list_of_file_nums = [
  [10],
  [13]
]

gcamp = 'f' #for GCaMP6s: 's'
stim_type = 'amp' # 'freq', 'pulse_dur',  'amp'

'''
root_directory = 'C:/Hyperstim/data_analysis/2024_09_18_GCamp6s_in_vivo/' #
tiff_directory = 'C:/Hyperstim/data_analysis/2024_09_18_GCamp6s_in_vivo/merged_tiffs/'
mesc_file_name = '2024_09_18_GCamp6s_in_vivo_2'
mesc_DATA_file = 'mesc_data.npy'
list_of_file_nums = [
  [4,5]

]
gcamp = 's' #for GCaMP6s: 's'
stim_type = 'amp' # 'freq', 'pulse_dur',  'amp'
'''
#------VALUES TO CHANGE END------
#mesc_tiff_extract.analyse_mesc_file(Path(root_directory)/mesc_file_name, root_directory, print_all_attributes=True, plot_curves = True)
#-----1.2.step: frequency_to_save, electrode_roi_to_save-->automatization pending-----
#mesc_data_handling.extract_stim_frame(root_directory, mesc_DATA_file, list_of_file_nums)
#mesc_data_handling.tiff_merge(mesc_file_name, list_of_file_nums, root_directory)
#suite2p_script.run_suite2p(os.path.join(root_directory,'merged_tiffs/'), gcamp)

#--------------Suite2p manual sorting------------------

'''functions.stim_dur_val(tiff_directory, list_of_file_nums)
functions.electROI_val(tiff_directory, list_of_file_nums)
functions.dist_vals(tiff_directory, list_of_file_nums)
functions.stim_dur_val(tiff_directory, list_of_file_nums)'''


#functions.baseline_val(root_directory, tiff_directory, list_of_file_nums)
#functions.activated_neurons_val(root_directory, tiff_directory, list_of_file_nums, 1)
#functions.timecourse_vals(tiff_directory, list_of_file_nums, 5)
#functions.data_analysis_values(stim_type, tiff_directory, list_of_file_nums)
#functions_current_steering.plot_stim_traces(tiff_directory, num_repeats=6, num_stims_per_repeat=5)

#functions.plot_stim_traces(tiff_directory, 31, 6, 5, list_of_file_nums, 8, 5.2, 8) #5.165

overlap.overlap_calc(tiff_directory, list_of_file_nums)
#overlap.create_roi_map(tiff_directory, list_of_file_nums)


'''
functions.stim_dur_val(tiff_directory, list_of_file_nums)
functions.electROI_val(tiff_directory, list_of_file_nums)
functions.dist_vals(tiff_directory, list_of_file_nums)
functions.stim_dur_val(tiff_directory, list_of_file_nums)
functions.baseline_val(root_directory, tiff_directory, list_of_file_nums)
functions.activated_neurons_val(root_directory, tiff_directory, list_of_file_nums)
functions.timecourse_vals(tiff_directory, list_of_file_nums, 5)
functions.data_analysis_values(stim_type, tiff_directory, list_of_file_nums)


#TBD

functions.scratch_val(tiff_directory)
'''
