
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier, RadiusNeighborsClassifier

def main():
    data = pd.read_csv("ice.csv")
    x = data[["temp", "street"]]
    y = data["ice"]

    clf = KNeighborsRegressor(n_neighbors=1)
    clf.fit(x, y)
    print("KNeighborsRegressor : ", clf.score(x, y))

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(x, y)
    print("KNeighborsClassifier : ", clf.score(x, y))

    clf = RadiusNeighborsClassifier()
    clf.fit(x, y)
    print("RadiusNeighborsClassifier : ", clf.score(x, y))


if __name__=="__main__":
    main()
