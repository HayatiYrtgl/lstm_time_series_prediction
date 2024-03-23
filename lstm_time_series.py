import pandas as pd
import tensorflow as tf
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError, R2Score
from keras.optimizers import Adam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# download datasets
data = pd.read_csv("datasets_ml/jena_climate_2009_2016.csv")

# get every hour
data = data[5::6]

# set datetime
data.index = pd.to_datetime(data["Date Time"], format="%d.%m.%Y %H:%M:%S")

# dataframe to split
def data_to_split(data: pd.DataFrame, window_size: int):

    # df to numpy
    df_as_numpy = data.to_numpy()

    # X and y
    X = []
    y = []

    # for loop to create
    for i in range(len(df_as_numpy)-window_size):
        # format should be
        # [[[1], [2], [3], [4], [5]...[window size]]]
        row = [[a] for a in df_as_numpy[i:i+window_size]]

        # append row to x
        X.append(row)

        # y
        label = df_as_numpy[i+window_size]

        # append label to y
        y.append(label)

    # return numpy array
    return np.array(X), np.array(y)


X, y = data_to_split(data=data["T (degC)"], window_size=10)

# train test split
X_train, y_train = X[:25000], y[:25000]
X_val, y_val = X[25000:30000], y[25000:30000]
X_test, y_test = X[30000:], y[30000:]

tf.debugging.set_log_device_placement(False)


# create model
def model_create():

    model = Sequential()

    # Input Layer
    model.add(InputLayer((10, 1)))

    # LSTM
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(64))

    # dense
    model.add(Dense(8, "relu"))
    model.add(Dense(1, "linear"))
    # summary
    model.summary()

    return model


model = model_create()

# best model chekpoint
cp = ModelCheckpoint("models/", save_best_only=True)

# compile
model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()], run_eagerly=True)

# fit the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, callbacks=[cp], verbose=1)



