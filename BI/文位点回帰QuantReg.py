
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

def main():
    data = pd.read_csv("./ice.csv")
    x = data[["temp", "street"]]
    x = sm.add_constant(x)
    y = data["ice"]
    est = QuantReg(y, x).fit(q = 0.99999)
    print(est.summary())


if __name__=="__main__":
    main()

