
import numpy as np
import pandas as pd
from sklearn.svm import SVR

def main():
    data = pd.read_csv("ice.csv")
    x = data[["temp", "street"]]
    y = data["ice"]

    clf = SVR(kernel="rbf", C=1e7, epsilon=0.01, max_iter=-1, tol=1e-7, verbose=1, gamma=10.1).fit(x, y)
    print(clf.score(x, y))

if __name__=="__main__":
    main()
