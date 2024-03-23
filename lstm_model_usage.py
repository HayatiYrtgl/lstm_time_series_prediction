from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


model = load_model("models/")
train_preddictions = model.predict(X_test).flatten()
df = pd.DataFrame({"predictions": train_preddictions, "real":y_test})
print(df.head(25))


plt.plot(train_preddictions[:100])
plt.plot(y_test[:100])
plt.legend(labels=["predicted", "real"])
plt.show()
