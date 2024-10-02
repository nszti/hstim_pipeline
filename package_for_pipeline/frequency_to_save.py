import numpy as np

frequency = np.array([0, 200, 100, 300, 300, 0, 0, 300, 300, 0 ])
np.save('C:/Hyperstim/data_analysis/2023_07_05_in_vivo_test_GCaMP6f/merged_tiffs/frequencies.npy', frequency)
print(frequency)