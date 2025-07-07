import numpy as np
import os
def frequency_electrodeRoi_to_save(root_directory, tiff_directory):
    # Set the output directory
    output_path = tiff_directory
    file_id_path = root_directory + '/fileID.txt'
    # Check if the file exists
    if not os.path.isfile(file_id_path):
        raise FileNotFoundError(f"File {file_id_path} does not exist.")
    # Read the file to count the number of lines
    with open(file_id_path, 'r') as f:
        num_files = sum(1 for _ in f)
    print(f'Number of values to save: {num_files}')
    # Create an array to hold the electrode ROI values
    electrodeROI = np.zeros((num_files,), dtype=int)

    print(f"Detected {num_files} TIFF files.")
    print("Choose input method for frequencies:")
    print("1 - Single frequency for all")
    print("2 - Repeating pattern (50,100,200)")  #not sure if it's feasible
    print("3 - Enter values manually")

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
            val = int(input(f"Frequency for file {i + 1}: "))
            frequency.append(val)
        frequency = np.array(frequency)
    elif choice == '4':
        base_freq = int(input("Enter base frequency value: "))
        frequency = np.full(num_files, base_freq)
        print("Base frequencies set.")
        while True:
            idx = int(input(f"Enter index (0 to {num_files - 1}) to overwrite: "))
            if 0 <= idx < num_files:
                new_freq = int(input(f"Enter new frequency for index {idx}: "))
                frequency[idx] = new_freq
                print(f"Frequency at index {idx} updated to {new_freq}.")
                more = input("Do you want to overwrite another value? (y/n): ").lower()
                if more != 'y':
                    break
            else:
                print("Index out of range. Try again.")
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
