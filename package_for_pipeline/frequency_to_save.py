import numpy as np
import os
def frequency_electrodeRoi_to_save(root_directory, tiff_directory, mesc_DATA_file):
    # Set the output directory
    output_path = tiff_directory
    mesc_data_path = os.path.join(tiff_directory, mesc_DATA_file)
    mesc_data = np.load(mesc_data_path,allow_pickle=True)
    file_ids = []
    for row in mesc_data:
        row_id = row[0]
        if isinstance(row_id, str) and row_id.startswith("MUnit_"):
            file_num = int(row_id.replace("MUnit_", ""))
            file_ids.append(file_num)
        else:
            raise ValueError(f"FileID format {row_id} is not right")
    frame_nos = mesc_data[:,1]
    trigger  = mesc_data[:,2]
    fileid_to_index = {}
    for idx, fid in enumerate(file_ids):
        fileid_to_index[fid] = idx

    file_id_path = root_directory + '/fileID.txt'
    # Check if the file exists
    if not os.path.isfile(file_id_path):
        raise FileNotFoundError(f"File {file_id_path} does not exist.")
    # Read the file to count the number of lines
    with open(file_id_path, 'r') as f:
        num_files = sum(1 for _ in f)
    print(f'Number of values to save: {num_files}')
    print(f'List of the FileIDs: {file_ids}')
    # Create an array to hold the electrode ROI values
    electrodeROI = np.zeros((num_files,), dtype=int)

    print(f"Detected {num_files} TIFF files.")
    print("Choose input method for frequencies:")
    print("1 - Single frequency for all")
    print("2 - Repeating pattern (50,100,200)")  #not sure if it's feasible
    print("3 - Enter values manually")
    print("4 - Single base frequency for all and overwrite values on specific index")

    choice = input("Enter 1, 2, 3 or 4: ")

    if choice == '1':
        freq = int(input("Enter frequency value: "))
        frequency = np.full(num_files, freq)

    elif choice == '2':
        pattern_input = input("Enter comma-separated values (e.g., 50,100,200): ")
        pattern = [int(x.strip()) for x in pattern_input.split(',')]
        frequency = np.tile(pattern, int(np.ceil(num_files / len(pattern))))[:num_files]

    elif choice == '3':
        frequency = []
        print(f"Enter {num_files} values:")
        for i in range(num_files):
            val = int(input(f"Frequency for file {i + 1} (FileID {num_files[i]}: "))
            frequency.append(val)
        frequency = np.array(frequency)
    elif choice == '4':
        base_freq = int(input("Enter base frequency value: "))
        frequency = np.full(num_files, base_freq)
        print("Base frequencies set.")
        while True:
            file_num = int(input(f"Enter FileID to overwrite: "))
            print(file_num)
            if 0 <= file_num < num_files:
                idx = fileid_to_index[file_num] + 1
                new_freq = int(input(f"Enter new frequency for FileID {file_num} which is index {idx}: "))
                frequency[idx] = new_freq
                print(f"Frequency for FileID {file_num} updated to {new_freq}.")
                while True:
                    more = input("Do you want to overwrite another value? (y/n): ").lower()
                    if more in ('y', 'n'):
                        break
                    print("Invalid input. Please enter 'y' or 'n'.")
                if more == 'n':
                    break
            else:
                print("FileID out of range. Try again.")
    else:
        raise ValueError("Invalid choice. Try again")

    # Save and print
    save_path_e = output_path + '/electrode_rois.npy'
    np.save(save_path_e, electrodeROI)
    print("Saved electrode rois:")
    print(electrodeROI)
    save_path_f = output_path + '/frequencies.npy'
    np.save(save_path_f, frequency)
    print("Saved frequencies:")
    print(frequency)
