
import pandas as pd
import statsmodels.api as sm

def main():
    data = pd.read_csv("./ice.csv")
    x = data[["temp", "street"]]
    x = sm.add_constant(x)
    y = data["ice"]
    est = sm.OLS(y, x).fit()
    print(est.summary())



if __name__=="__main__":
    main()
