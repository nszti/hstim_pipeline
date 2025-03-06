from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import suite2p
from suite2p.run_s2p import run_s2p
import matplotlib as mpl
from matplotlib import pyplot as pl
import matplotlib.cm as cm

mpl.rcParams.update({
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'figure.subplot.wspace': .01,
    'figure.subplot.hspace': .01,
    'figure.figsize': (18, 13),
    'ytick.major.left': False,
})
jet = pl.cm.get_cmap('jet')
jet.set_bad(color='k')

db = {
    'data_path': 'C:/Hyperstim/data_analysis/2024_09_18_GCamp6s_in_vivo/',
    'save_path0': 'C:/Hyperstim/data_analysis/2024_09_18_GCamp6s_in_vivo/merged_tiffs',
    'tiff_list': '2024_09_18_GCamp6s_in_vivo_2_MUnit_6.tif',

    'h5py': [],  # a single h5 file path
    'h5py_key': 'data',
    'look_one_level_down': False,  # whether to look in ALL subfolders when searching for tiffs
    #'data_path': 'C:/Hyperstim/data_analysis/2024_09_18_GCamp6s_in_vivo/',
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
}
ops = suite2p.default_ops()
#output_ops = suite2p.run_s2p(ops=ops, db=db)
opsEnd = run_s2p(ops=ops, db=db)

len(output_ops)
print(set(output_ops.keys()).difference(ops.keys()))
list(Path(output_ops['save_path']).iterdir())
output_op_file = np.load(Path(output_ops['save_path']).joinpath('ops.npy'), allow_pickle=True).item()
output_op_file.keys() == output_ops.keys()

f_cells = np.load(Path(output_ops['save_path']).joinpath('F.npy'))
f_neuropils = np.load(Path(output_ops['save_path']).joinpath('Fneu.npy'))
spks = np.load(Path(output_ops['save_path']).joinpath('spks.npy'))
f_cells.shape, f_neuropils.shape, spks.shape
plt.figure(figsize=[20,20])
plt.suptitle("Flourescence and Deconvolved Traces for Different ROIs", y=0.92);
rois = np.arange(len(f_cells))[::20]
for i, roi in enumerate(rois):
    plt.subplot(len(rois), 1, i+1, )
    f = f_cells[roi]
    f_neu = f_neuropils[roi]
    sp = spks[roi]
    # Adjust spks range to match range of fluroescence traces
    fmax = np.maximum(f.max(), f_neu.max())
    fmin = np.minimum(f.min(), f_neu.min())
    frange = fmax - fmin
    sp /= sp.max()
    sp *= frange
    plt.plot(f, label="Cell Fluorescence")
    plt.plot(f_neu, label="Neuropil Fluorescence")
    plt.plot(sp + fmin, label="Deconvolved")
    plt.xticks(np.arange(0, f_cells.shape[1], f_cells.shape[1]/10))
    plt.ylabel(f"ROI {roi}", rotation=0)
    plt.xlabel("frame")
    if i == 0:
        plt.legend(bbox_to_anchor=(0.93, 2))
    plt.show()