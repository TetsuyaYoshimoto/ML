
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import linear_model

def main():
    data = pd.read_csv("./ice.csv")
    x = data[["temp", "street"]]
    x = sm.add_constant(x)
    y = data["ice"]
    clf = linear_model.Lasso()
    clf.fit(x, y)
    p = clf.predict(x)

    print("score : ", clf.score(x, y))
    print("intercept : ", clf.intercept_)
    print("coef : ", clf.coef_)


if __name__=="__main__":
    main()
