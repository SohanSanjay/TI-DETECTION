import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
##########################################################################################
########################################### The TI METHOD ################################
##########################################################################################
# Parameters required to create the CNN handle for TI detection
# # Input nurones - number of input nurones for the CNN handle
# # Output nurones - number of output nurones for the CNN handle
# # Total number of hidden layers + 1 hidden layer similar to output layer with Relu
##########################################################################################
#                                 # 
#                               ###
#                              ####
#                             ## ##
#                                ##
##                               ##
#                                ##
#                                ##
#                                ##
#                              ######
#
##########################################################################################
# Thie is the initial CNN
# Input layer - LxL lattice sites
# Hidden layers = 3 layers
# Hidden layer 1 -> 2X input nurones Relu
# Hidden layer 2 -> 2X input nurones Softmax
# Hidden layer 3 -> 1x output nurones Relu
# Output layer -> depend on the number of temperature/coupling measurements
##########################################################################################

def ti_cnn_test_module(input_nurones,output_nurones):
    model = Sequential();

    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_nurones,1)));
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(input_nurones*2, activation='softmax')) 
    model.add(Dense(output_nurones, activation='relu'))
    model.add(Dense(output_nurones, activation='softmax'))
    
    return model

##########################################################################################

def train_cnn_model(X_train,Y_train,X_test,Y_test,model):
    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']);
    # train the model
    history = model.fit(X_train, Y_train, epochs=1000, batch_size=32, validation_data=(X_test, Y_test));
    return model,history;

##########################################################################################


