import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import sequence
import random

# set up user paths
data_dir = "../resource"


def read_data():
    # training data inputs: x and targets: y
    x_train_path = os.path.join(data_dir, "X_train.hdf")
    y_train_path = os.path.join(data_dir, "y_train.hdf")
    # validation data inputs: x and targets: y
    x_valid_path = os.path.join(data_dir, "X_test.hdf")
    y_valid_path = os.path.join(data_dir, "y_test.hdf")

    X_train = pd.read_hdf(x_train_path)
    y_train = pd.read_hdf(y_train_path)

    X_valid = pd.read_hdf(x_valid_path)
    y_valid = pd.read_hdf(y_valid_path)

    # first select a random patient counter (encounter identifier)
    eIdx = random.choice(list(X_train.index.levels[0]))
    # next specify a few variables to look at
    variables = [
        'Age',
        'Heart rate (bpm)',
        'PulseOximetry',
        'Weight',
        'SystolicBP',
        'DiastolicBP',
        'Respiratory rate (bpm)',
        'MotorResponse',
        'Capillary refill rate (sec)'
    ]

    X_train.loc[8, 'Heart rate (bpm)'].plot()
    plt.ylabel("Heart rate (bpm)")
    plt.xlabel("Hours since first encounter")
    plt.show()

    with tf.name_scope("normalization"):
        # create file path for csv file with metadata about variables
        metadata = os.path.join(data_dir, "ehr_features.csv")
        # read in variables from csv file (using pandas) since each variable there is tagged with a category
        variables = pd.read_csv(metadata, index_col=0)
        # next, select only variables of a particular category for normalization
        normvars = variables[variables['type'].isin(['Interventions', 'Labs', 'Vitals'])]
        # finally, iterate over each variable in both training and validation data
        for vId, dat in normvars.iterrows():
            X_train[vId] = X_train[vId] - dat['mean']
            X_valid[vId] = X_valid[vId] - dat['mean']
            X_train[vId] = X_train[vId] / (dat['std'] + 1e-12)
            X_valid[vId] = X_valid[vId] / (dat['std'] + 1e-12)

    with tf.name_scope("ffill"):
        # first select variables which will be filled in
        fillvars = variables[variables['type'].isin(['Vitals', 'Labs'])].index
        # next forward fill any missing values with more recently observed value
        X_train[fillvars] = X_train.groupby(level=0)[fillvars].ffill()
        X_valid[fillvars] = X_valid.groupby(level=0)[fillvars].ffill()
        # finally, fill in any still missing values with 0
        X_train.fillna(value=0, inplace=True)
        X_valid.fillna(value=0, inplace=True)

    X_train.loc[8, "Heart rate (bpm)"].plot()
    plt.title("Normalized and FFill")
    plt.ylabel("Heart rate (bpm)")
    plt.xlabel("Hours since first encounter")
    plt.show()

    with tf.name_scope("padding"):
        maxlen = 500
        # get a list of unique patient encounter IDs
        teId = X_train.index.levels[0]
        veId = X_valid.index.levels[0]
        # pad every patient sequence with 0s to be the same length
        X_train = [X_train.loc[patient].values for patient in teId]
        y_train = [y_train.loc[patient].values for patient in teId]

        X_train = sequence.pad_sequences(X_train, dtype='float32', maxlen=maxlen, padding='post', truncating='post')
        y_train = sequence.pad_sequences(y_train, dtype='float32', maxlen=maxlen, padding='post', truncating='post')
        # repeat for the validation data
        X_valid = [X_valid.loc[patient].values for patient in veId]
        y_valid = [y_valid.loc[patient].values for patient in veId]

        X_valid = sequence.pad_sequences(X_valid, dtype='float32', maxlen=maxlen, padding='post', truncating='post')
        y_valid = sequence.pad_sequences(y_valid, dtype='float32', maxlen=maxlen, padding='post', truncating='post')

    return X_train, y_train, X_valid, y_valid