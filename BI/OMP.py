
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import OrthogonalMatchingPursuit

def main():
    data = pd.read_csv("./ice.csv")
    x = data[["temp", "street"]]
    x = sm.add_constant(x)
    y = data["ice"]
    lm = OrthogonalMatchingPursuit(n_nonzero_coefs = 3)
    est = lm.fit(x, y)
    print("coef : ", est.coef_)
    print("intercept : ", est.intercept_)
    print("score : ", est.score(x, y))

if __name__=="__main__":
    main()
