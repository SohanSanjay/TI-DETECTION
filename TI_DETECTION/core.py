from .input_data_handle import *;
from .output_data_handle import *;
import pandas as pd;
import numpy as np;
import os;

########################################################################
########################### Workflow Core ##############################
########################################################################
#
# 1. create the input data .csv file
#         - need - input array of parameters
#         - need - constant parameters
#         - need - read/write file paths

########################################################################

def create_input_data(path_array,
                      parm_array,
                      save_path):
    
    # create the input data file

    for i in range(len(path_array)):
        save_combined_data(save_path,
                       path_array[i],
                       parm_array[i]);
    


