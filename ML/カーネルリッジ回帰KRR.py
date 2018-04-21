
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge

def main():
    data = pd.read_csv("ice.csv")
    x = data[["temp", "street"]]
    y = data["ice"]

    clf = KernelRidge(kernel = "rbf", alpha = 1e-8).fit(x, y)
    print(clf.score(x, y))


if __name__=="__main__":
    main()
