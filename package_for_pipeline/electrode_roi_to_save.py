import numpy as np

#electrodeROIs = [0, 29, 45, 47, 49, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
electrodeROI = np.array([0,0,0,0,0,0,0,0,0,0 ])
np.save('C:/Hyperstim/data_analysis/2023_07_05_in_vivo_test_GCaMP6f/merged_tiffs/electrode_rois.npy', electrodeROI)
print(electrodeROI)