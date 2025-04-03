
from scipy.io import savemat

def suite2p_to_cellreg_masks(expDir, list_of_file_nums):
    base_dir = Path(expDir)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]
    for numbers_to_merge in list_of_file_nums:
        suffix = '_'.join(map(str, numbers_to_merge))
        num_to_search = []
        for dir in filenames:
            num_to_search_split = dir.split('MUnit_')
            # print(num_to_search_split)
            if len(num_to_search_split) > 1:
                file_suffix = num_to_search_split[1].rsplit('.', 1)[0]
                if file_suffix == suffix:
                    matched_file = dir
                    print(matched_file)
                    # print(matched_file)
                    break
        else:
            continue

        if matched_file:
            stat_path = expDir + dir + '/suite2p/plane0/stat.npy'
            ops_path = expDir + dir + '/suite2p/plane0/ops.npy'
            stat = np.load(stat_path, allow_pickle=True)
            ops = np.load(ops_path, allow_pickle=True).item()

    Ly, Lx = ops['Ly'], ops['Lx']
    masks = []
    for roi in stat:
        mask = np.zeros((Ly, Lx), dtype=np.uint8)
        ypix = roi['ypix']
        xpix = roi['xpix']
        mask[ypix, xpix] = 1
        masks.append(mask)
    return np.stack(masks, axis=-1)  # shape: (Ly, Lx, nROIs)

masks = suite2p_to_cellreg_masks(stat, ops)
savemat('session1_cells.mat', {'cells_map': masks})
