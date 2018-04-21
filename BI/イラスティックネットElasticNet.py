
import pandas as pd
from sklearn.linear_model import ElasticNet

def main():
    data = pd.read_csv("./ice.csv")
    x = data[["temp", "street"]]
    y = data["ice"]
    clf = ElasticNet(alpha = 0.001)
    clf.fit(x, y)
    print(clf.score(x, y))
 

if __name__=="__main__":
    main()
