from pathlib import Path
import numpy as np
from package_for_pipeline import functions
from package_for_pipeline import mesc_data_handling
from package_for_pipeline import suite2p_script
from mesc_loader import extract_useful_xml_params
from general import ascii_to_str, find_frame_index_from_timestamp
from  package_for_pipeline import mesc_tiff_extract
import os

#------VALUES TO CHANGE------

root_directory = 'C:/Hyperstim/data_analysis/2023_07_05_in_vivo_test_GCaMP6f/' #
tiff_directory = 'C:/Hyperstim/data_analysis/2023_07_05_in_vivo_test_GCaMP6f/merged_tiffs/'
mesc_file_name = '2023_07_05_igin_vivo_test_GCaMP6f'
mesc_DATA_file = 'mesc_data.npy'

list_of_file_nums = [
  [1],
  [1,2]


]
gcamp = 's' #for GCaMP6s: 's'
gcamp = 'f' #for GCaMP6s: 's'

#------VALUES TO CHANGE END------



#mesc_tiff_extract.analyse_mesc_file(Path(root_directory)/mesc_file_name, root_directory, print_all_attributes=True, plot_curves = True)
#mesc_data_handling.tiff_merge(mesc_file_name, list_of_file_nums, root_directory)
#mesc_data_handling.extract_stim_frame(root_directory, mesc_DATA_file, list_of_file_nums)
#suite2p_script.run_suite2p(os.path.join(root_directory,'merged_tiffs/'), gcamp)


#functions.stim_dur_val(tiff_directory)
#functions.electROI_val(tiff_directory)
#functions.baseline_val(tiff_directory)
#functions.dist_vals(tiff_directory)
#functions.activated_neurons_val(tiff_directory)
#functions.timecourse_vals(tiff_directory, 5)


'''
#TBD
functions.data_analysis_values('amp', directory)
functions.scratch_val(directory)
'''
