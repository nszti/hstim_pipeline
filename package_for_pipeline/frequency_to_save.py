import numpy as np
'''
frequency = np.array([0, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 50, 100, 200, 300, 400, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 50, 100, 200, 300, 400, 200, 200, 200])
output_path = 'c:/Hyperstim/data_analysis/AMouse-2025-02-18-invivo-GCaMP6f/merged_tiffs/'
np.save(output_path +'frequencies.npy', frequency)
print(frequency)
'''
#frequency = np.array([0, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 50, 100, 200, 300, 400, 200])
#frequency = np.zeros((25,), dtype=int)
frequency = np.full(50,100)
output_path = 'c:/Hyperstim/data_analysis/2023_09_25_GCAMP6F/merged_tiffs/'
np.save(output_path +'frequencies.npy', frequency)
print(frequency)