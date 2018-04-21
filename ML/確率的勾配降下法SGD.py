
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

def main():
    data = pd.read_csv("ice.csv")
    x = data[["temp", "street"]]
    y = data["ice"]
    rbf_feature = RBFSampler(gamma = 1, random_state = 0, n_components = 100)
    x_features = rbf_feature.fit_transform(x)
    
    clf = SGDClassifier()
    clf.fit(x_features, y)
    print(clf.score(x_features, y))



if __name__=="__main__":
    main()
