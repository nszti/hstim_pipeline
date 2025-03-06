import numpy as np
'''
frequency = np.array([0, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 50, 100, 200, 300, 400, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 50, 100, 200, 300, 400, 200, 200, 200])
output_path = 'c:/Hyperstim/data_analysis/AMouse-2025-02-18-invivo-GCaMP6f/merged_tiffs/'
np.save(output_path +'frequencies.npy', frequency)
print(frequency)
'''
#frequency = np.array([0, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 50, 100, 200, 300, 400, 200])
#frequency = np.zeros((25,), dtype=int)
frequency = np.full(25,200)
#frequency[12] = 100
output_path = 'c:/Hyperstim/data_analysis/AMouse-2025-03-05-invivo-GCaMP6f/merged_tiffs/'
np.save(output_path +'frequencies.npy', frequency)
print(frequency)