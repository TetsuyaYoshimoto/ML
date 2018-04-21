
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
np.random.seed(100)
n_of_neurons = 100

def main():
    model = Sequential()
    model.add(Dense(input_dim = 2, output_dim = n_of_neurons, activation = "relu"))
    model.add(Dense(n_of_neurons, activation = "relu"))
    model.add(Dense(n_of_neurons, activation = "relu"))
    model.add(Dense(n_of_neurons, activation = "relu"))
    model.add(Dense(1))

    adam = Adam(lr = 1e-6)

    data = pd.read_csv("ice.csv")
    x = data[["temp", "street"]]
    xnp = x.as_matrix()
    y = data["ice"]
    ynp = y.as_matrix()

    model.compile(optimizer="adam", loss="mse")
    model.fit(xnp, ynp, nb_epoch = 20000, verbose = 1, shuffle=True)
    print("score : ", r2_score(ynp, model.predict_proba(xnp)))

if __name__=="__main__":
    main()

