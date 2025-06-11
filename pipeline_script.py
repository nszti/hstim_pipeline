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
from package_for_pipeline import cellreg_preprocess
from package_for_pipeline import cellreg_analysis
import matplotlib.pyplot as plt
import os
from package_for_pipeline import frequency_to_save
#------STEPS IN PIPELINE------
'''
 0.: modify values in the 'values to change' section. NB! gcamp & stim_type are to be changes also,params for them can be modifited in 'suite2p_script.py' & 'functions.data_analysis_values()' respectively
 1.: run mesc_tiff_extract.analyse_mesc_file() (--> now 'merged_tiffs' folder is created, 'mesc_data.npy' is in it) NB! check & modify -if needed- 'stim_trig_channel' in analyse_mesc_file's parameter list
//manual work #1
 2.: modify 'frequency_to_save.py', 'electrode_roi_to_save.py' to match current file data (use experiment reports for fequency, electrode_Roi should be 0 for all files) 
 3.: add tiff numbers to merge in the 'list_of_file_nums' list. NB! it's possible to add 1 and multiple nums in a list & to add multiple lists to merge
//manual work #1 end
 4.: run: 'mesc_data_handling.extract_stim_frame()'-> 'mesc_data_handling.tiff_merge()'-> 'suite2p_script.run_suite2p()' --> functions.baseline_val()
//manual work #2
 5.: load the processed file into suite2p & manually add electrode ROI where needed in suite2p (suite2p GUI>file>manual labelling>alt+click /might have to wait for it to load the traces, often makes the software to freez/)
 6.: manually modify ROIs if needed (add, remove). NB! check the electrode and the side of the FOV for false ROIs & check the not cell ROIs as well you can look back the og mesc file & check the registered binary file in suite2p (ctrl+b)
//manual work #2 end
 7.: run 'functions.py' for analysis. NB! for timecourse_vals() the 3rd param ('num_trials') should be changed manually + there's probably a bug with stim_durations (my bad, will fix it)  
'''
#------STEPS IN PIPELINE END------

#------VALUES TO CHANGE------

#root_directory = 'c:/Hyperstim/data_analysis/2025-04-24-Amouse-invivo-GCaMP6f/'
#root_directory = 'c:/Hyperstim/data_analysis/2025-04-01-Amouse-invivo-GCaMP6f/' #
#root_directory = 'c:/Hyperstim/data_analysis/2023_09_25_GCAMP6F/'
root_directory = 'c:/Hyperstim/data_analysis/2025-03-25-Amouse-invivo-GCaMP6f/'
#root_directory = 'c:/Hyperstim/data_analysis/AMouse-2025-03-05-invivo-GCaMP6f/'
#tiff_directory = 'c:/Hyperstim/data_analysis/2025-04-24-Amouse-invivo-GCaMP6f/merged_tiffs/'
#tiff_directory = 'c:/Hyperstim/data_analysis/2025-04-01-Amouse-invivo-GCaMP6f/merged_tiffs/'
#tiff_directory = 'c:/Hyperstim/data_analysis/2023_09_25_GCAMP6F/merged_tiffs/'
tiff_directory= 'c:/Hyperstim/data_analysis/2025-03-25-Amouse-invivo-GCaMP6f/merged_tiffs/'
#tiff_directory = 'c:/Hyperstim/data_analysis/AMouse-2025-03-05-invivo-GCaMP6f/merged_tiffs/'
#mesc_file_name = 'AMouse-2025-03-05-invivo-GCaMP6f-2' '2025-04-01-Amouse-invivo-GCaMP6f-2'
#mesc_file_name = '2025-03-25-Amouse-invivo-GCaMP6f'
#mesc_file_name = 'AMouse-2025-03-05-invivo-GCaMP6f-2'
#mesc_file_name = '2025-04-01-Amouse-invivo-GCaMP6f-2'
mesc_file_name ='2025-04-24-Amouse-invivo-GCaMP6f'
mesc_DATA_file = 'mesc_data.npy' #from mesc_tiff_extract
mat_file = 'cellRegistered_20250505_175342.mat'
postfix = 'sum_avg_dir/activated_roi_masks/'
list_of_file_nums = [
  [15]

]

gcamp = 'f' #for GCaMP6s: 's'
stim_type = 'amp' # 'freq', 'pulse_dur',  'amp'
'''
#root_directory = 'c:/Hyperstim/data_analysis/2025-04-15-Amouse-invivo-GCaMP6f/' #
#root_directory = 'c:/Hyperstim/data_analysis/2025-04-24-Amouse-invivo-GCaMP6f/'
#root_directory = 'e:/2025-05-08-Amouse-invivo-GCaMP6f/'
#root_directory = 'c:/Hyperstim/data_analysis/2025-04-29-Amouse-invivo-GCaMP6f/'
#tiff_directory = 'c:/Hyperstim/data_analysis/2025-04-15-Amouse-invivo-GCaMP6f/merged_tiffs/'
#tiff_directory = 'e:/2025-05-08-Amouse-invivo-GCaMP6f/merged_tiffs/'
#tiff_directory = 'c:/Hyperstim/data_analysis/2025-04-29-Amouse-invivo-GCaMP6f/merged_tiffs/'
mesc_file_name = '2025-05-08-Amouse-invivo-GCaMP6f'
#mesc_file_name = '2025-04-15-Amouse-invivo-GCaMP6f'
mesc_DATA_file = 'mesc_data.npy'
mat_file = 'cellRegistered_20250504_121302.mat'
#postfix = '53_54_55_56_57_58_59_60_61_62_63_64_65_66_ordered'
postfix = ''

list_of_file_nums = [

[27]

]

gcamp = 'f' #for GCaMP6s: 's'
stim_type = 'amp' # 'freq', 'pulse_dur',  'amp'
'''
RUN_MESC_PREPROCESS = False  #tiff extraction
RUN_PREPROCESS = False # osszefuz listaban megadott tifeket
S2P = False #suite2p futtatás
#--------
RUN_ANALYSIS_PREP = False  #F0 savelodik, ha modositod a suite2p barmelyik propertyet, akkor ezt ujra kell futtatni h frissuljon az F0
PLOTS = False #Analysis plotok, utolso 3 a relevans
PLOT_BTW_EXP = False
RUN_CELLREG_PREP = False #cellreghez mat fileokat ment ki
#cellreg hasznalat: CellReg.m run > GUIban load new dataval berakod a mat fileokat & 1.07 micront megadod> Non-rigid alignment futtatas > 12 micronos probabilistc modeling futtatas > a tobbit csak megnyomkodod sorban
RUN_CELLREG_ANALYSIS = True

#------VALUES TO CHANGE END------
if RUN_MESC_PREPROCESS:
  mesc_tiff_extract.analyse_mesc_file(Path(root_directory)/mesc_file_name, root_directory, print_all_attributes=True, plot_curves = True)

#-----1.2.step: frequency_to_save, electrode_roi_to_save-->automatization pending-----
if RUN_PREPROCESS:
  #frequency_to_save.frequency_electrodeRoi_to_save(root_directory, tiff_directory)
  mesc_data_handling.tiff_merge(mesc_file_name, list_of_file_nums, root_directory)
  #mesc_data_handling.extract_stim_frame(root_directory, mesc_DATA_file, list_of_file_nums) #--> saves stimTimes.npy needed for baseline
if S2P:
  suite2p_script.run_suite2p(tiff_directory, list_of_file_nums, False, gcamp)

#--------------Suite2p manual sorting------------------

if RUN_ANALYSIS_PREP:
  '''functions.stim_dur_val(tiff_directory, list_of_file_nums)
  functions.electROI_val(tiff_directory, list_of_file_nums)
  functions.dist_vals(tiff_directory, list_of_file_nums)
  functions.stim_dur_val(tiff_directory, list_of_file_nums)'''
  #functions.baseline_val(root_directory, tiff_directory, list_of_file_nums) #--> saves F0.npy : if suite2p files changes make sure to rerun
  functions.analyze_merged_activation_and_save(root_directory, mesc_file_name, tiff_directory, list_of_file_nums)


#functions.activated_neurons_val(root_directory, tiff_directory, list_of_file_nums, 1)
#functions.timecourse_vals(tiff_directory, list_of_file_nums, 5)
#functions.data_analysis_values(stim_type, tiff_directory, list_of_file_nums)
#functions_current_steering.plot_stim_traces(tiff_directory, num_repeats=6, num_stims_per_repeat=5)
if PLOTS:
  functions.plot_stim_traces(tiff_directory, 30.97, 6, 5, list_of_file_nums, 8, 5.2, 2) #5.165
if PLOT_BTW_EXP:
  functions.plot_across_experiments(root_directory, tiff_directory, list_of_file_nums, 30.97)

#overlap.overlap_calc(tiff_directory, list_of_file_nums)
#overlap.create_roi_map(tiff_directory, list_of_file_nums)

if RUN_CELLREG_PREP:
  cellreg_preprocess.suite2p_to_cellreg_masks(tiff_directory, list_of_file_nums)
# manual: run cellreg pipeline
if RUN_CELLREG_ANALYSIS:
  #cellreg_preprocess.cellreg_analysis(tiff_directory, mat_file, list_of_file_nums, postfix, 5)
  #cellreg_preprocess.single_block_activation(tiff_directory,postfix, mat_file,  30.97, 10, list_of_file_nums, 2.4,200, 3 )
  cellreg_analysis.run_cellreg_matlab(tiff_directory)
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
'''
sum_avg_path = 'path/to/sum_avg_values.npy'
            sum_avg_per_amplitude = np.load(sum_avg_path, allow_pickle=True).item()

            # kell?
            sorted_items = sorted(sum_avg_per_amplitude.items())

            # time axis of 4s duration
            trace_length = len(next(iter(sum_avg_per_amplitude.values())))
            time = np.linspace(-1, 3, trace_length)

            plt.figure(figsize=(8, 5))
            for amplitude, trace in sorted_items:
                plt.plot(time, trace, label=f"{amplitude} μA", linewidth=2)

            plt.title("Sum Avg Traces per Amplitude: .npy mentes ell")
            plt.xlabel("Time (s)")
            plt.ylabel("Mean ΔF/F₀")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
'''