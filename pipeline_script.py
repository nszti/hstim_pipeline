from pathlib import Path
import numpy as np
from package_for_pipeline import functions

from package_for_pipeline import mesc_data_handling
from package_for_pipeline import suite2p_script
from mesc_loader import extract_useful_xml_params
from general import ascii_to_str, find_frame_index_from_timestamp
from package_for_pipeline import mesc_tiff_extract
from package_for_pipeline import overlap
from package_for_pipeline import cellreg_preprocess
#from package_for_pipeline import cellreg_analysis #matlap api cellreg
from package_for_pipeline import CoM
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

root_directory = 'c:/Users/Rendszergazda/Documents/ttk/data/'
tiff_directory = 'c:/Users/Rendszergazda/Documents/ttk/data/merged_tiffs/'
mesc_file_name ='2025-07-02-Amouse-invivo-GCaMP6f'
mesc_DATA_file = 'mesc_data.npy' #from mesc_tiff_extract
mat_file = ''
postfix = ''
list_of_file_nums = [
  [37,38,39]
]
gcamp = 'f' #for GCaMP6s: 's'
stim_type = 'amp' # 'freq', 'pulse_dur',  'amp'

#=========

RUN_MESC_PREPROCESS = False  #tiff extraction
RUN_PREPROCESS = False # osszefuz listaban megadott tifeket
S2P = False #suite2p futtatás
#--------

RUN_ANALYSIS_PREP = True  #F0 savelodik, ha modositod a suite2p barmelyik propertyet, akkor ezt ujra kell futtatni h frissuljon az F0
PLOTS = False #Analysis plotok, utolso 3 a relevans
PLOT_BTW_EXP = False

RUN_CELLREG_PREP = False #cellreghez mat fileokat ment ki
#cellreg hasznalat: CellReg.m run > GUIban load new dataval berakod a mat fileokat & 1.07 micront megadod> Non-rigid alignment futtatas > 12 micronos probabilistc modeling futtatas > a tobbit csak megnyomkodod sorban
RUN_CELLREG = False
RUN_CELLREG_ANALYSIS = False
VIDEO = False

#------VALUES TO CHANGE END------
if RUN_MESC_PREPROCESS:
  mesc_tiff_extract.analyse_mesc_file(Path(root_directory)/mesc_file_name, root_directory, print_all_attributes=True, plot_curves = True)

#-----1.2.step: frequency_to_save, electrode_roi_to_save-->automatization pending-----
if RUN_PREPROCESS:
  #frequency_to_save.frequency_electrodeRoi_to_save(root_directory, tiff_directory, mesc_DATA_file)
  #mesc_data_handling.tiff_merge(mesc_file_name, list_of_file_nums, root_directory, mesc_DATA_file, True)
  mesc_data_handling.extract_stim_frame(root_directory, mesc_DATA_file, list_of_file_nums) #--> saves stimTimes.npy needed for baseline
if S2P:
  suite2p_script.run_suite2p(tiff_directory, list_of_file_nums, False, gcamp)

#--------------Suite2p manual sorting------------------

if RUN_ANALYSIS_PREP:
  functions.stim_dur_val(root_directory,tiff_directory, list_of_file_nums)
  functions.save_roi_numbers_only(tiff_directory, list_of_file_nums)
  '''functions.electROI_val(tiff_directory, list_of_file_nums)'''
  #functions.dist_vals(tiff_directory, list_of_file_nums)
  #functions.spontaneous_baseline(tiff_directory,list_of_file_nums, [2,3,5,6,8,9,0,1,10,11,12,13,15], frame_rate= 30.97, plot_start_frame = 0, plot_end_frame=None)
  #functions.F_extract(tiff_directory, list_of_file_nums, [0]) #--> saves F.npy : if suite2p files changes make sure to rerun
  #functions.baseline_val(root_directory, tiff_directory, list_of_file_nums) #--> saves F0.npy : if suite2p files changes make sure to rerun
  # functions.activated_neurons_val(root_directory, tiff_directory, list_of_file_nums, 1)
  #functions.timecourse_vals(tiff_directory, list_of_file_nums, 5)
  functions.analyze_merged_activation_and_save(root_directory, mesc_file_name, tiff_directory, list_of_file_nums, 30.97, 100, 3, 10,3.5)

if PLOTS:
  #functions.plot_activation_summary(stim_type, tiff_directory, list_of_file_nums)
  functions.plot_activation_summary(
    activation_map_path='c:/Users/Rendszergazda/Documents/ttk/data/merged_tiffs/merged_2025-07-02-Amouse-invivo-GCaMP6f_MUnit_39/activation_map_valid_merged_2025-07-02-Amouse-invivo-GCaMP6f_MUnit_39.npy',
    save_dir='c:/Users/Rendszergazda/Documents/ttk/data/merged_tiffs/merged_2025-07-02-Amouse-invivo-GCaMP6f_MUnit_39/',
  )
  #functions.plot_full_traces_and_roi_overlay(tiff_directory, list_of_file_nums)
  #functions.plot_stim_traces(tiff_directory, 30.97, 6, 5, list_of_file_nums, 8, 5.2, 2) #5.165
  functions.data_analysis_values(stim_type, tiff_directory, list_of_file_nums)
  #CoM.plot_weighted_com([], [], tiff_directory)

if PLOT_BTW_EXP:
  functions.plot_across_experiments(root_directory, tiff_directory, list_of_file_nums, 30.97)

if RUN_CELLREG_PREP:
  cellreg_preprocess.suite2p_to_cellreg_masks(tiff_directory, list_of_file_nums)
# manual: run cellreg pipeline
if RUN_CELLREG:
  cellreg_analysis.run_cellreg_matlab(tiff_directory)
if RUN_CELLREG_ANALYSIS:
  cellreg_preprocess.cellreg_analysis_overlap(tiff_directory, mat_file, list_of_file_nums, postfix)
  #cellreg_preprocess.single_block_activation(tiff_directory,postfix, mat_file,  30.97, 10, list_of_file_nums, 2.4,200, 3 )

if VIDEO:
  #functions.get_stim_frames_to_video(root_directory, tiff_directory,list_of_file_nums)
  functions.create_video_from_mesc_tiffs(root_directory, list_of_file_nums)