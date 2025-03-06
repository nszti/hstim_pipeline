from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import suite2p

stat = np.load('c:/Hyperstim/pipeline_pending/test/2024_09_18_GCamp6s_in_vivo/merged_tiffs/merged_2024_09_18_GCamp6s_in_vivo_2_MUnit_24_25_26_27_28/suite2p/plane0/stat.npy', allow_pickle=True)
print(stat)

ops = np.load('c:/Hyperstim/pipeline_pending/test/2024_09_18_GCamp6s_in_vivo/merged_tiffs/merged_2024_09_18_GCamp6s_in_vivo_2_MUnit_24_25_26_27_28/suite2p/plane0/stat.npy', allow_pickle=True).item()
Ly, Lx = ops['Ly'], ops['Lx']

neuropil_mask_full = np.zeros((Ly, Lx), dtype=bool)

for roi in stat:
    neuropil_mask_full[roi['neuropil_mask']] = True

plt.figure(figsize=(10, 10))
plt.imshow(neuropil_mask_full, cmap='gray')
plt.title("Neuropil Mask")
plt.axis("off")
plt.show()
