# THIS IS A SCRIPT TO RUN THROUGH DATA ANALYSIS OF CYR WHEEL DATA.
import numpy as np
import scipy.io as sci
import matplotlib.pyplot as plt
import os
import zipfile

# Unzipping the data into a data folder
zip_file_path = "data.zip"
extract_to = "./data"
os.makedirs(extract_to, exist_ok=True)

with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
    zip_file.extractall(extract_to)

print(f"All files extracted to {extract_to}")

def plotting(key, data, tracker, folder):
    if data.ndim == 1:
        time = np.arange(len(data))
        plt.plot(time, data, label=f'{key} - 1D')
        
    elif data.ndim == 2:
        time = np.arange(data.shape[0])
        for i in range(data.shape[1]):
            plt.plot(time, data[:, i], label=f'{key} - Dim {i+1}')
            
    else:
        print(f"Skipping {key}: Data has unsupported dimensions {data.ndim}")
        return

    plt.xlabel("Time")
    plt.ylabel("Center of Mass" if data.ndim == 1 else f"{key} Values")
    plt.title(f"Data from {tracker}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/Analyzed_{tracker}_{key}.png")
    # plt.show()

# Running through analysis
files = os.listdir('./data')
for file in files:
    if not file.endswith('.mat'):
        continue  # Skip non-MAT files
    
    mat_data = sci.loadmat(f"./data/{file}")
    tracker = file
    keys = list(mat_data.keys())
    
    for key in keys:
        # Filter out default MATLAB keys
        if key.startswith("__"):
            continue
        
        data = mat_data[key]        
        try:
            # If data is nested, unpack it down to an array or matrix
            while isinstance(data, np.ndarray) and data.size == 1:
                data = data[0]
            
            # Determine folder based on the key name
            if "EMG" in key:
                folder = "Figures/EMG"
            elif "Kinematics" in key:
                folder = "Figures/Kinematics"
            elif "Force" in key:
                folder = "Figures/Force"
            else:
                folder = "Figures/Other"  # Default folder for unclassified data
            
            # Plotting             
            plotting(key, data, tracker, folder)
        
        except Exception as e:
            print(f"Skipping {key} in {file}: Error {e}")
