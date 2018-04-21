
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB

def main():
    data = pd.read_csv("ice.csv")
    x = data[["temp", "street"]]
    y = data["ice"]

    clf = GaussianNB()
    clf.fit(x, y)
    print(clf.score(x, y))

    clf = MultinomialNB(alpha = 1e-3)
    clf.fit(x, y)
    print(clf.score(x, y))


if __name__=="__main__":
    main()
