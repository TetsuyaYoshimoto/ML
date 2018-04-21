
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def main():
    data = pd.read_csv("ice.csv")
    x = data[["temp", "street"]]
    y = data["ice"]

    clf = DecisionTreeClassifier().fit(x, y)
    print(clf.score(x, y))
    print(clf.feature_importances_)



if __name__=="__main__":
    main()
