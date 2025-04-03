import numpy as np
from pathlib import Path
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
from pathlib import Path
import re
import ast
from scipy.spatial.distance import cdist
from scipy.spatial.distance import cdist, directed_hausdorff
from skimage.draw import polygon
from tqdm import tqdm


def find_dir():
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
            print(f"\nAnalyzing directory: {dir}")
            # Load required data
            F_path = expDir + dir + '/suite2p/plane0/F0.npy'
            iscelll_path = expDir + dir + '/suite2p/plane0/iscell.npy'
            stim_start_times_path = expDir + dir + '/stimTimes.npy'
            stat_path = expDir + dir + '/suite2p/plane0/stat.npy'
            ops_path = expDir + dir + '/suite2p/plane0/ops.npy'
            activation_results_path = expDir + dir + '/activation_results.csv'

            F = np.load(F_path, allow_pickle=True) # NB! baseline corrected fluorescence
            iscell = np.load(iscelll_path, allow_pickle=True)
            stat = np.load(stat_path, allow_pickle=True)
            ops = np.load(ops_path, allow_pickle=True).item()
            activation_results = np.load(activation_results_path, allow_pickle=True)
    return stat, F, iscell, stim_start_times, ops

def get_roi_centroids(stat):
    roi_centroids = []
    for roi in stat:
        roi_centroids.append(roi['med'])
    return np.array(roi_centroids)

def dice_coefficient(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    return 2 * intersection.sum() / (mask1.sum() + mask2.sum())

def create_roi_map(expDir, list_of_file_nums):
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
            print(f"\nAnalyzing directory: {dir}")
            # Load required data
            F_path = expDir + dir + '/suite2p/plane0/F0.npy'
            iscelll_path = expDir + dir + '/suite2p/plane0/iscell.npy'
            stim_start_times_path = expDir + dir + '/stimTimes.npy'
            stat_path = expDir + dir + '/suite2p/plane0/stat.npy'
            ops_path = expDir + dir + '/suite2p/plane0/ops.npy'
            activation_results_path = expDir + dir + '/activation_results.csv'

            F = np.load(F_path, allow_pickle=True)  # NB! baseline corrected fluorescence
            iscell = np.load(iscelll_path, allow_pickle=True)
            stat = np.load(stat_path, allow_pickle=True)
            ops = np.load(ops_path, allow_pickle=True).item()
            activation_results = pd.read_csv(activation_results_path)

            #create roi map
            im = np.zeros((ops['Ly'], ops['Lx']))
            cell_indices = np.where(iscell[:, 0] == 1)[0]
            cell_stat = [stat[i] for i in cell_indices]
            for n in range(len(cell_stat)):
                ypix = cell_stat[n]['ypix'][~cell_stat[n]['overlap']]
                xpix = cell_stat[n]['xpix'][~cell_stat[n]['overlap']]
                im[ypix, xpix] = n + 1
            med_val = []
            roi_names= []
            for roi_idx in cell_indices:
                roi_names.append(roi_idx)
                med_val.append((stat[roi_idx]['med'][0], stat[roi_idx]['med'][1]))
            for i, coord in enumerate(med_val):
                plt.text(coord[1], coord[0], f'{roi_names[i]}', fontsize=9, color='white')

            #plt.imshow(im, alpha=0.5)
            #plt.title(f'roi map for file - {file_suffix}')
            #plt.savefig(os.path.join(expDir, dir, 'roi_map.png'))
            #plt.show()


def overlap_calc(expDir, list_of_file_nums):
    base_dir = Path(expDir)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]

    matched_files = []
    for numbers_to_merge in list_of_file_nums:
        suffix = '_'.join(map(str, numbers_to_merge))
        for dir in filenames:
            num_to_search_split = dir.split('MUnit_')
            if len(num_to_search_split) > 1:
                file_suffix = num_to_search_split[1].rsplit('.', 1)[0]
                if file_suffix == suffix:
                    matched_files.append(dir)
                    print(f"Matched directory: {dir}")
                    break  # Go to next numbers_to_merge
        else:
            continue

    dir1 = matched_files[0]
    dir2 = matched_files[1]

    # file #1
    F1_path = os.path.join(expDir, dir1) + '/suite2p/plane0/F0.npy'
    iscell1_path = os.path.join(expDir, dir1) + '/suite2p/plane0/iscell.npy'
    stat1_path = os.path.join(expDir, dir1) + '/suite2p/plane0/stat.npy'
    ops1_path = os.path.join(expDir, dir1) + '/suite2p/plane0/ops.npy'
    # load data
    F1 = np.load(F1_path, allow_pickle=True)
    iscell1 = np.load(iscell1_path, allow_pickle=True)
    stat1 = np.load(stat1_path, allow_pickle=True)
    ops1 = np.load(ops1_path, allow_pickle=True).item()
    image_shapeA = (ops1['Ly'], ops1['Lx'])
    # file #2
    F2_path = os.path.join(expDir, dir2) + '/suite2p/plane0/F0.npy'
    iscell2_path = os.path.join(expDir, dir2) + '/suite2p/plane0/iscell.npy'
    stat2_path = os.path.join(expDir, dir2) + '/suite2p/plane0/stat.npy'
    ops2_path = os.path.join(expDir, dir2) + '/suite2p/plane0/ops.npy'
    # load data
    F2 = np.load(F2_path, allow_pickle=True)
    iscell2 = np.load(iscell2_path, allow_pickle=True)
    stat2 = np.load(stat2_path, allow_pickle=True)
    ops2 = np.load(ops2_path, allow_pickle=True).item()
    image_shapeB = (ops2['Ly'], ops2['Lx'])

    # --- Compute masks ---
    cell_indicesA = np.where(iscell1[:, 0] == 1)[0]
    cell_indicesB = np.where(iscell2[:, 0] == 1)[0]
    cell_statsA = [stat1[i] for i in cell_indicesA]
    cell_statsB = [stat2[i] for i in cell_indicesB]
    for idx in range(len(cell_statsA)):
        maskA = cell_statsA[idx]['lam']
        centroidsA = cell_statsA[idx]['med']
        centroidsA = np.array(centroidsA).reshape(-1,2)
    for idx in range(len(cell_statsB)):
        maskB = cell_statsB[idx]['lam']
        centroidsB = cell_statsB[idx]['med']
        centroidsB = np.array(centroidsB).reshape(-1, 2)

    #Building a KD tree from centroidB points
    tree = KDTree(centroidsB)
    # find the nearest neighbors in centroidsB for each point in centroidsA
    dist, idx = tree.query(centroidsA, k=1)
    dist = dist.flatten()
    idx = idx.flatten()

    matches = []
    for i, maskA_i in enumerate(maskA):
        if i < len(dist):  # Ensure index is within bounds
            for j, maskB_j in enumerate(maskB):
                d_score = dice_coefficient(maskA_i, maskB_j)
                matches.append({
                    'ROI_A': i,
                    'ROI_B': j,
                    'centroid_distance': dist[i],
                    'dice_coefficient': d_score
                })
    print(f"\nTotal matches: {len(matches)}")

    for i, m1 in enumerate(maskA):
        for j, m2 in enumerate(maskB):
            dice = dice_coefficient(m1, m2)
            print(f"ROI {i} in {dir1} matches ROI {j} in {dir2} with Dice {dice:.2f}")
            #if dice > 0.3:
                #print(f"ROI {i} in {dir1} matches ROI {j} in {dir2} with Dice {dice:.2f}")

    #overlap roi maps
    fig, ax = plt.subplots()
    imA = np.zeros(image_shapeA)
    imB = np.zeros(image_shapeB)

    med_valA = []
    roi_namesA = []
    for roi_idx in cell_indicesA:
        roi_namesA.append(roi_idx)
        med_valA.append((stat1[roi_idx]['med'][0], stat1[roi_idx]['med'][1]))
    med_valB = []
    roi_namesB = []
    for roi_idx in cell_indicesB:
        roi_namesB.append(roi_idx)
        med_valB.append((stat2[roi_idx]['med'][0], stat2[roi_idx]['med'][1]))

    for n, stat in enumerate(cell_statsA):
        ypix = stat['ypix'][~stat['overlap']]
        xpix = stat['xpix'][~stat['overlap']]
        imA[ypix, xpix] = n + 1

    for n, stat in enumerate(cell_statsB):
        ypix = stat['ypix'][~stat['overlap']]
        xpix = stat['xpix'][~stat['overlap']]
        imB[ypix, xpix] = n + 1
    for i, coord in enumerate(med_valA):
        plt.text(coord[1], coord[0], f'{roi_namesA[i]}', fontsize=9, color='white')
    for i, coord in enumerate(med_valB):
        plt.text(coord[1], coord[0], f'{roi_namesB[i]}', fontsize=9, color='black')

    ax.imshow(imB, alpha=0.5, cmap='Dark2')
    #plt.imshow(imA, alpha=0.5, cmap ='Blues')
    #plt.imshow(imB, alpha=0.5)
    ax.imshow(imA, alpha=0.5, cmap='Purples_r')


    plt.title('Overlapping ROI Maps')
    plt.savefig(os.path.join(expDir, 'overlapping_roi_maps.png'))
    plt.show()



#******

def load_suite2p_data(expDir, list_of_file_nums):
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
    return stat, ops

def create_binary_mask(stat, ops):
    masks = []
    Ly, Lx = ops['Ly'], ops['Lx']
    for roi in stat:
        mask = np.zeros((Ly, Lx), dtype=bool)
        ypix = roi['ypix']
        xpix = roi['xpix']
        mask[ypix, xpix] = True
        masks.append(mask)
    return masks


def hausdorff_distance(mask1, mask2):
    # [0,inf] 0-perfect shape match
    pts1 = np.column_stack(np.nonzero(mask1))
    pts2 = np.column_stack(np.nonzero(mask2))

    if len(pts1) == 0 or len(pts2) == 0:
        return np.inf  # one mask is empty

    d1 = directed_hausdorff(pts1, pts2)[0]
    d2 = directed_hausdorff(pts2, pts1)[0]
    return max(d1, d2)
#in similarity score
def scaled_hausdorff(mask1, mask2, max_dist=20):
    #max_dist = dist at which 2 masks are totally different
    hd = hausdorff_distance(mask1, mask2)
    return max(0, 1 - min(hd, max_dist) / max_dist)  #normalize

def compute_jaccard(mask1, mask2):
    # [0,1] 1-perfect overlap
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def match_rois(stat_a, masks_a, stat_b, masks_b, distance_thresh=15, jaccard_thresh=0.3, score_thresh=0.5,
               w1=0.5, w2=0.4, w3=0.1):
    matches = []
    meds_a = np.array([roi['med'] for roi in stat_a])
    meds_b = np.array([roi['med'] for roi in stat_b])

    # compute pairwise distances between medians, uses euclidean by default
    dist_matrix = cdist(meds_a, meds_b)
    # normalize distance by diag ( diag.
    fov_diag = np.linalg.norm([masks_a[0].shape[0], masks_a[0].shape[1]])

    for i, row in enumerate(dist_matrix):
        for j, dist in enumerate(row):
            if dist < distance_thresh:
                norm_dist = dist / fov_diag
                jaccard = compute_jaccard(masks_a[i], masks_b[j])
                shape_sim = scaled_hausdorff_similarity(masks_a[i], masks_b[j])

                score = w1 * (1 - norm_dist) + w2 * jaccard + w3 * shape_sim

                if jaccard >= jaccard_thresh and score >= score_thresh:
                    matches.append((i, j, dist, jaccard, shape_sim, score))

    return sorted(matches, key=lambda x: -x[-1])  # sort by similarity score descending


def match_suite2p_rois(stat_a_path, ops_a_path, stat_b_path, ops_b_path,
                       distance_thresh=15,
                       jaccard_thresh=0.3,
                       score_thresh=0.5,
                       w1=0.4, w2=0.3, w3=0.3):
    """
    Matches ROIs between two Suite2p recordings and returns list of matches:
    (index_A, index_B, distance, jaccard, shape_similarity, combined_score)
    """
    # Load Suite2p outputs
    stat_a, ops_a = load_suite2p_data(stat_a_path, ops_a_path)
    stat_b, ops_b = load_suite2p_data(stat_b_path, ops_b_path)

    # Generate full-size binary ROI masks
    masks_a = create_binary_mask(stat_a, ops_a)
    masks_b = create_binary_mask(stat_b, ops_b)

    # Match ROIs with combined similarity score
    matches = match_rois(stat_a, masks_a, stat_b, masks_b,
                         distance_thresh=distance_thresh,
                         jaccard_thresh=jaccard_thresh,
                         score_thresh=score_thresh,
                         w1=w1, w2=w2, w3=w3)

    return matches

