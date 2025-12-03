import pandas as pd
import numpy as np
import os

# load data from the ml data extracted file
# file format
#  #  # file name - complete_ossh_honeycomb_w0.1000_a0.0390_mu0.0000_L6_b10.0000-1-train-data-v7.csv
#  #  # file name - contain the information about omega, alpha, mu, L, beta 
#. #. # file data - no header, comma delimiter, each row contain 100 simulation averages, lasr column is the phase

def load_info(FL_NAME):
    # obtain omega
    OMEGA = float(FL_NAME.split('_w')[1].split('_')[0]);
    # obtain alpha
    ALPHA = float(FL_NAME.split('_a')[1].split('_')[0]);
    # obtain mu
    MU = float(FL_NAME.split('_mu')[1].split('_')[0]);
    # obtain L
    L = int(FL_NAME.split('_L')[1].split('_')[0]);
    # obtain beta
    BETA = float(FL_NAME.split('_b')[1].split('-')[0]);
    return OMEGA, ALPHA, MU, L, BETA;


# load data from a csv file
# return number of sets, length of input neurons, data array

def load_data(FILE_PATH):
    # assert if the file is a csv and exists
    assert FILE_PATH.endswith('.csv'), "File must be a CSV";
    assert os.path.exists(FILE_PATH), "File does not exist";
    # read the csv file without header, delimiter 
    data = pd.read_csv(FILE_PATH,delimiter=',',header=None);
    # remove the last column (phase column)
    data = data.drop(columns=data.columns[-1],axis=1);
    # size of data
    num_sets,len_input_neurons = np.shape(data);

    return num_sets,len_input_neurons,data;

# create one hot encoding for a given label
# # # choose the label from Temperature, coupling type
# # # choose range of labels 
# # # can only create for one lable type at a time

def one_hot_encode(LABEL_VAL,LABELS):
    # set labels in order
    labels_ordered = sorted(LABELS);
    # number of labels
    num_labels = len(labels_ordered);
    # create one hot encoding array
    one_hot = np.zeros(num_labels);
    # find the index of the label value
    label_index = labels_ordered.index(LABEL_VAL);
    # set the corresponding index to 1
    one_hot[label_index] = 1;
    return one_hot;


