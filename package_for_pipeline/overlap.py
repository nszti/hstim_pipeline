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

from torch.ao.nn.quantized.functional import threshold


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
    for i, j in enumerate(idx):
        d_score = dice_coefficient(maskA[i], maskB[j])
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
            if dice > 0.3:
                print(f"ROI {i} in {dir1} matches ROI {j} in {dir2} with Dice {dice:.2f}")

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





