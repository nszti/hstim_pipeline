import numpy as np

#electrodeROIs = [0, 29, 45, 14, 14, 14, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#electrodeROI = np.array([0,0,0,0,0,0,0,0,0,0 ])
electrodeROI = np.zeros((18,), dtype=int)
output_path = 'c:/Hyperstim/data_analysis/2025-04-29-Amouse-invivo-GCaMP6f-2/merged_tiffs/'
np.save(output_path + 'electrode_rois.npy', electrodeROI)
print(electrodeROI)