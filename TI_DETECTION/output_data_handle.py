import pandas as pd
import numpy as np
import os
from .input_data_handle import load_info, load_data, one_hot_encode;

# save the training data to csv
# combine all files to a one file
# last row contain the information need to extract one hot encoding
#  # file data save path
#  # file data read path
#  # file data label value

def save_combined_data(save_fl_path,
                       read_fl_path,
                       label_vals):
    
    # assert the existance of the read file
    assert os.path.exists(read_fl_path), "Read file path does not exist";
    # read the data from the read file path
    num_sets,len_input_neurons,data = load_data(read_fl_path);
    # create an array to hold the combined data
    combined_data = data;
    combined_data[" "] = label_vals;
    
    # save the combined data to csv
    combined_data.to_csv(save_fl_path,index=False,header=False);


# function to combine all the training data to a single file
# need to keep all files in a single folder
# all files in the folder will be combined
# files must be similar in structure

def combine_all_data_in_folder(folder_path):
    assert os.path.exists(folder_path), "Folder path does not exist"
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not file_list:
        raise ValueError("No CSV files found in folder")
    frames = []
    for file_name in sorted(file_list):
        file_path = os.path.join(folder_path, file_name)
        data = pd.read_csv(file_path,delimiter=',',header=None);
        frames.append(data)
    all_data = pd.concat(frames, ignore_index=True)
    all_data.to_csv(os.path.join(folder_path, 'combined_data.csv'), index=False, header=False)     
    
