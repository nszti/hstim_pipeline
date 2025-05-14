'''
import numpy as np
frequency = np.array([0, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 50, 100, 200, 300, 400, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 50, 100, 200, 300, 400, 200, 200, 200])
output_path = 'c:/Hyperstim/data_analysis/AMouse-2025-02-18-invivo-GCaMP6f/merged_tiffs/'
np.save(output_path +'frequencies.npy', frequency)
print(frequency)
#frequency = np.array([0, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 50, 100, 200, 300, 400, 200])
#frequency = np.zeros((25,), dtype=int)
frequency = np.full(18,200)
output_path = 'c:/Hyperstim/data_analysis/2025-04-29-Amouse-invivo-GCaMP6f-2/merged_tiffs/'
np.save(output_path +'frequencies.npy', frequency)
print(frequency)
'''

import numpy as np
import os
def frequency_to_save(root_directory, tiff_directory):
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

    print(f"Detected {num_files} TIFF files.")
    print("Choose input method for frequencies:")
    print("1 - Single frequency for all")
    print("2 - Repeating pattern (50,100,200)")  #not sure if it's feasible
    print("3 - Enter values manually")

    choice = input("Enter 1, 2, or 3: ")

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

    else:
        raise ValueError("Invalid choice.")

    # Save and print
    save_path = output_path + '/frequencies.npy'
    np.save(save_path, frequency)
    print("Saved frequencies:")
    print(frequency)
