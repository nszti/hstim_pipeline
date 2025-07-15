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

    file_id_path = root_directory + '/fileID.txt'
    # Check if the file exists
    if not os.path.isfile(file_id_path):
        raise FileNotFoundError(f"File {file_id_path} does not exist.")

    file_ids_fromtxt = []
    with open(file_id_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("MUnit_"):
                line = line.replace("MUnit_", "")
            if line:
                file_ids_fromtxt.append(int(line))
    num_files = len(file_ids_fromtxt)
    print(f'Number of values to save: {num_files}')
    print(f'List of the FileIDs: {file_ids_fromtxt}')


    fileid_to_index = {}
    for idx, fid in enumerate(file_ids_fromtxt):
        fileid_to_index[fid] = idx

    # Create an array to hold the electrode ROI values
    electrodeROI = np.zeros((num_files,), dtype=int)

    print(f"Detected {num_files} TIFF files.")
    print("Choose input method for frequencies:")
    print("1 - Single frequency for all")
    print("2 - Single base frequency for all and overwrite values on specific index")  #not sure if it's feasible
    print("3 - Enter values manually")
    print("4 - Repeating pattern (50,100,200)")

    choice = input("Enter 1, 2, 3 or 4: ")

    if choice == '1':
        freq = int(input("Enter frequency value: "))
        frequency = np.full(num_files, freq)

    elif choice == '2':
        while True:
            try:
                base_freq = int(input("Enter base frequency value: "))
                break
            except ValueError:
                print("Invalid input. Please enter an integer value for the base frequency.")
        frequency = np.full(num_files, base_freq)
        print("Base frequencies set.")
        while True:
            try:
                file_num = int(input(f"Enter FileID to overwrite: "))
            except ValueError:
                print("Invalid input. Please enter a valid FileID.")
                continue
            print(file_num)

            if file_num in file_ids_fromtxt:
                idx = fileid_to_index[file_num]
                while True:
                    try:
                        new_freq = int(input(f"Enter new frequency for FileID {file_num} which is index {idx}: "))
                        break
                    except ValueError:
                        print("Invalid input. Please enter an integer.")
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

    elif choice == '3':
        frequency = []
        print(f"Enter {num_files} values:")
        for i in range(num_files):
            val = int(input(f"Frequency for file {i + 1} (FileID {num_files[i]}: "))
            frequency.append(val)
        frequency = np.array(frequency)

    elif choice == '4':
        pattern_input = input("Enter comma-separated values (e.g., 50,100,200): ")
        pattern = [int(x.strip()) for x in pattern_input.split(',')]
        frequency = np.tile(pattern, int(np.ceil(num_files / len(pattern))))[:num_files]

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
