import numpy as np
import os
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

def run_suite2p(data_path, gcamp):
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

      db = {
            'h5py': [], # a single h5 file path
            'h5py_key': 'data',
            'look_one_level_down': False, # whether to look in ALL subfolders when searching for tiffs
            'data_path': [data_path], # a list of folders with tiffs
                                                   # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)
            'subfolders': [], # choose subfolders of 'data_path' to look in (optional)
            'reg_tif': False,
            'neuropil_extract': True,
            'neucoeff': 0.7,
            'ratio_neuropil': 6,
            'allow_overlap': False,
            'inner_neuropil_radius': 2

          }
      print(db['data_path'])
      db_list = []
      merged_subfolders_list = search_merged_subfolders(db['data_path'])
      print(merged_subfolders_list)
      for sublist in merged_subfolders_list:
            print(sublist)
            for subfolder in sublist:
                  print(subfolder)
                  if gcamp == 'f':
                        db_list.append({
                              'h5py': [],  # a single h5 file path
                              'h5py_key': 'data',
                              'look_one_level_down': False,  # whether to look in ALL subfolders when searching for tiffs
                              'data_path': [subfolder],
                              # a list of folders with tiffs
                              # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)
                              'subfolders': [],  # choose subfolders of 'data_path' to look in (optional)
                              'tau': 0.5,
                              'fs': 30.97,
                              'batch_size': 500,
                              'spatial_scale': 0,
                              'denoise': True,
                              'threshold_scaling': 0.25,
                              'max_overlap': 0.7,
                              'high_pass': 300
                        })
                  if gcamp == 's':
                        db_list.append({
                              'h5py': [],  # a single h5 file path
                              'h5py_key': 'data',
                              'look_one_level_down': False,  # whether to look in ALL subfolders when searching for tiffs
                              'data_path': [subfolder],
                              # a list of folders with tiffs
                              # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)
                              'subfolders': [],  # choose subfolders of 'data_path' to look in (optional)
                              'tau': 1.25,
                              'fs': 31.0,
                              'batch_size': 500,
                              'spatial_scale': 2,
                              'denoise': True,
                              'threshold_scaling': 0.1,
                              'max_overlap': 0.7,
                              'high_pass': 300
                        })
      for dbi in db_list:
            opsEnd = run_s2p(ops=ops, db=dbi)
