import numpy as np

#electrodeROIs = [0, 29, 45, 14, 14, 14, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#electrodeROI = np.array([0,0,0,0,0,0,0,0,0,0 ])
electrodeROI = np.zeros((25,), dtype=int)
output_path = 'c:/Hyperstim/data_analysis/AMouse-2025-03-05-invivo-GCaMP6f/merged_tiffs/'
np.save(output_path + 'electrode_rois.npy', electrodeROI)
print(electrodeROI)