import os
import global_variables
import re


def rename_files(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # Separate the base name and the extension
            base_name, extension = os.path.splitext(filename)

            # Detect parts like "game4"
            match = re.search(r'(game\d+)', base_name)
            if match:
                game_part = match.group(0)  # The 'game' followed by digits (e.g., 'game4')
                base_name = base_name.replace(game_part, '')  # Remove 'game4' from base
            else:
                game_part = ''  # No 'game' followed by digits part found

            # Remove 'TF' and append '_TransferLearning' or append '_NoTL'
            if 'TF' in base_name:
                new_base_name = base_name.replace('TF_', '') + 'TransferLearning'
            else:
                new_base_name = base_name + 'NoTL'

            # Construct the new filename, placing the 'game' part at the end
            new_filename = new_base_name
            if game_part:
                new_filename += '_' + game_part  # Append 'game4' at the end if it was found
            new_filename += extension  # Append the file extension

            # Full path with the new file name
            new_file_path = os.path.join(root, new_filename)

            # Rename the file
            os.rename(os.path.join(root, filename), new_file_path)
            print(f'Renamed {os.path.join(root, filename)} to {new_file_path}')


# Replace 'your_directory_path' with the path to the directory you want to process
rename_files(f'{global_variables.GLOBAL_PATH_REPO}/Results/Comparison4')
