import numpy as np
import os
import pickle
import pprint
from pathlib import Path
import suite2p
from suite2p import run_s2p
from suite2p.run_s2p import run_s2p
from suite2p import detection
from suite2p.detection import denoise
from suite2p.io import save


#data_path = 'C:/Hyperstim/pipeline_pending/mesc_preprocess_1/merged_tiffs/'

def search_merged_subfolders(data_path):
      merged_subfolders = []
      for path in data_path:
            if os.path.isdir(path):
                  subfolders = [os.path.join(path, f) for f in os.listdir(path) if f.startswith('merged') and os.path.isdir(os.path.join(path, f))]
                  merged_subfolders.append(subfolders)
      return merged_subfolders

def convert_to_plain_types(d):
    plain_dict = {}
    for k, v in d.items():
        if isinstance(v, Path):
            plain_dict[k] = str(v)
        elif hasattr(v, 'tolist'):  # handles numpy arrays
            plain_dict[k] = v.tolist()
        else:
            plain_dict[k] = v
    return plain_dict

plain_base_db = convert_to_plain_types(base_db)

txt_path = os.path.join(folder_path, 'suite2p_params.txt')
def run_suite2p(tiff_dir, list_of_file_nums, reused_params, gcamp):
      '''

      Parameters
      ----------
      data_path: outer folder containing 'merged' folders
      gcamp: for GCaMP6f: 'f', for GCaMP6s: 's'

      Returns
      -------
      runs suite2p with predefined parameters, output: suite2p folder in each subfolder with extracted data
      predefined set of ops parameters for GCaMP s & f
      '''
      ops = suite2p.default_ops()
      db_list = []
      base_dir = Path(tiff_dir)
      filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged') and file.is_dir()]

      for numbers_to_merge in list_of_file_nums:
            suffix = '_'.join(map(str, numbers_to_merge))
            matched_file = None

            for dir_name in filenames:
                  split_name = dir_name.split('MUnit_')
                  if len(split_name) > 1:
                        file_suffix = split_name[1].rsplit('.', 1)[0]
                        if file_suffix == suffix:
                              matched_file = dir_name
                              break

            if matched_file:
                  folder_path = os.path.join(tiff_dir, matched_file)
                  base_db = {
                        'h5py': [], # a single h5 file path
                        'h5py_key': 'data',
                        'look_one_level_down': False, # whether to look in ALL subfolders when searching for tiffs
                        'data_path': [folder_path], # a list of folders with tiffs
                                                               # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)
                        'subfolders': [], # choose subfolders of 'data_path' to look in (optional)
                        'reg_tif': False,
                        'neuropil_extract': True,
                        'denoise': True,
                        'batch_size': 500,
                        'fs': 30.97,
                        'neucoeff': 0.7,
                        'ratio_neuropil': 6,
                        'allow_overlap': False,
                        'inner_neuropil_radius': 2,
                        'high_pass': 300

                      }
                  #base params
                  # nedd to spec params path
                  if reused_params:
                        params_path = os.path.join(folder_path, 'suite2p_params.pkl')
                        if os.path.exists(params_path):
                              with open(params_path, 'rb') as f:
                                    saved_params = pickle.load(f)
                              base_db.update(saved_params)
                              print(f'Reusing saved parameters at {params_path}')
                        else:
                              raise FileNotFoundError(f"Parameter file not found at {params_path}")
                        db_list.append(base_db)
                  else:
                        if gcamp == 'f':
                              base_db.update({
                                    'tau': 0.5,
                                    'spatial_scale': 0,
                                    'threshold_scaling': 0.55,
                                    'max_overlap': 0.75
                              })
                              print(f'Using parameters given for GCaMP6f')
                              #04.29
                              '''if gcamp == 'f':
                                    base_db.update({
                                          'tau': 0.5,
                                          'spatial_scale': 2,
                                          'threshold_scaling': 0.46,
                                          'max_overlap': 0.75
                                    })'''
                              #04.14
                              '''if gcamp == 'f':
                                    base_db.update({
                                          'tau': 0.5,
                                          'spatial_scale': 0,
                                          'threshold_scaling': 0.45,
                                          'max_overlap': 0.75
                                    })'''
                              #04.15
                              '''if gcamp == 'f':
                                    base_db.update({
                                          'tau': 0.5,
                                          'spatial_scale': 0,
                                          'threshold_scaling': 0.32,
                                          'max_overlap': 0.75
                                    })'''
                        elif gcamp == 's':
                              base_db.update({
                                    'tau': 1.25,
                                    'spatial_scale': 2,
                                    'threshold_scaling': 0.26,
                                    'max_overlap': 0.7
                              })
                              print(f'Using parameters given for GCaMP6s')

                        save_path = os.path.join(folder_path, 'suite2p_params.pkl')
                        with open(save_path, 'wb') as f:
                              pickle.dump(base_db, f)

                        txt_path = os.path.join(folder_path, 'suite2p_params.txt')
                        with open(txt_path, 'w') as f:
                              pprint.pprint(base_db, stream = f)

                        db_list.append(base_db)
      for dbi in db_list:
            opsEnd = run_s2p(ops=ops, db=dbi)


