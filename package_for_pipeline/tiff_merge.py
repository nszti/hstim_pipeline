import os

import numpy as np
import tifffile
import tifftools
#output_path= 'C:/Hyperstim/pipeline_teszt/mesc_preprocess/'
#tiff_files_li=['C:/Hyperstim/Experiments/stimulation/2023_09_25_in_vivo_test_GCAMP6F/2023_09_25_in_vivo_test_GCAMP6f_MUnit_46.tif', 'C:/Hyperstim/Experiments/stimulation/2023_09_25_in_vivo_test_GCAMP6F/2023_09_25_in_vivo_test_GCAMP6f_MUnit_47.tif','C:/Hyperstim/Experiments/stimulation/2023_09_25_in_vivo_test_GCAMP6F/2023_09_25_in_vivo_test_GCAMP6f_MUnit_48.tif','C:/Hyperstim/Experiments/stimulation/2023_09_25_in_vivo_test_GCAMP6F/2023_09_25_in_vivo_test_GCAMP6f_MUnit_49.tif' ]

##
#'2024_02_09_in_vivo_GCAMP6F_2'
#[1, 2]
#'C:/Hyperstim/pipeline_pending/mesc_preprocess_1/'
##

def tiff_merge(mesc_file_name, numbers_to_merge, output_root_directory):
    suffix = '_MUnit_'
    # MESc file név csere!
    base_filename =  mesc_file_name + suffix
    output_directory = output_root_directory
    output_path = output_directory

    tiff_files_li = [os.path.join(output_path, f"{base_filename}{num}.tif") for num in numbers_to_merge]

    for file in tiff_files_li:
        if not os.path.isfile(file):
            print(f"Error: File {file} does not exist:(")
            exit(1)

    output_dirname = 'merged_' + base_filename + '_'.join(map(str, numbers_to_merge))
    output_filepath = os.path.join(output_directory, output_dirname)
    os.makedirs(output_filepath, exist_ok=True)

    output_filename = 'merged_' + base_filename + '_'.join(map(str, numbers_to_merge)) + '.tif'
    output_fpath = os.path.join(output_filepath, output_filename)

    '''tifftools.tiff_concat(tiff_files_li, output_fpath, overwrite=False)
    print(f"files {tiff_files_li} merged into {output_filepath}")'''

    all_pages = []
    for file in tiff_files_li:
        with tifffile.TiffFile(file) as tif:
            all_pages.append(tif.asarray())

    merged_stack = np.concatenate(all_pages, axis=0)
    tifffile.imwrite(output_fpath, merged_stack.astype(np.uint16))
    print(f"files {tiff_files_li} merged into {output_fpath}")


'''
suffix = '_MUnit_'
#MESc file név csere!
base_filename = '2024_02_09_in_vivo_GCAMP6F_2' + suffix
output_directory ='C:/Hyperstim/pipeline_pending/mesc_preprocess_1/'
output_path = output_directory


numbers_to_merge = [1,2]
tiff_files_li = [os.path.join(output_path, f"{base_filename}{num}.tif") for num in numbers_to_merge]

for file in tiff_files_li:
    if not os.path.isfile(file):
        print(f"Error: File {file} does not exist:(")
        exit(1)

output_dirname = 'merged_' +base_filename + '_'.join(map(str, numbers_to_merge))
output_filepath = os.path.join(output_directory, output_dirname)
os.makedirs(output_filepath, exist_ok=True)

output_filename = 'merged_' +base_filename + '_'.join(map(str, numbers_to_merge)) + '.tif'
output_fpath = os.path.join(output_filepath, output_filename)

tifftools.tiff_concat(tiff_files_li, output_fpath, overwrite=False)
print(f"files {tiff_files_li} merged into {output_filepath}")
'''




'''
prefix = output_path + '2023_09_25_in_vivo_test_GCAMP6f_MUnit_'
file1= prefix + '21.tif'
file2= prefix + '2.tif'
file3= prefix + '11.tif'
file4= prefix + '9.tif'
file5= prefix + '0.tif'
file6= prefix + '72.tif'
file7= prefix + '73.tif'
#file5= prefix + '9.tif'
#file4= prefix + '22.tif'
tiff_files_li=[file1, file2, file3, file4, file5]
#tiff_files_li=[file3, file4]



for ti in os.listdir():
    if '.tif' in ti:
        tiff_files_li.append(ti)


tifftools.tiff_concat(tiff_files_li, output_directory +'/' + output_filename + '.tif', overwrite=False)
#tifftools.tiff_concat(tiff_files_li, output_path + 'merged_2023_09_11_in_vivo_test_GCAMP6s_67-73.tif',overwrite=False)
'''
