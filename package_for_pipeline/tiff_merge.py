import os

import numpy as np
import tifffile

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

    all_pages = []
    for file in tiff_files_li:
        with tifffile.TiffFile(file) as tif:
            all_pages.append(tif.asarray())

    merged_stack = np.concatenate(all_pages, axis=0)
    tifffile.imwrite(output_fpath, merged_stack.astype(np.uint16))
    print(f"files {tiff_files_li} merged into {output_fpath}")

