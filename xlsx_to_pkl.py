import pandas as pd
import pickle
import os

# Define the folder containing the Excel files
folder_path = "results/OfflineCausalInference_tradeoff_batch_episodes_enemies"

# List all files in the folder
files = os.listdir(folder_path)

# Iterate through each file
for file in files:
    if file.endswith('.xlsx'):
        # Construct full path to the Excel file
        file_path = os.path.join(folder_path, file)

        # Read data from Excel file
        df = pd.read_excel(file_path)

        # Define the new file path for the pickle file
        new_file_path = os.path.splitext(file_path)[0] + '.pkl'

        # Save data into the pickle file
        with open(new_file_path, 'wb') as f:
            pickle.dump(df, f)

        # Delete the original Excel file
        os.remove(file_path)

        print(f"Converted {file} to {os.path.basename(new_file_path)} and deleted the original file.")
