
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score

def main():
    data = pd.read_csv("./ice.csv")
    x = data[["temp", "street"]]
    x = sm.add_constant(x)
    y = data["ice"]

    est1 = sm.RLM(y, x, M=sm.robust.norms.HuberT()).fit()
    r2 = r2_score(y, est1.predict(x))
    print("R-squared : ", r2)
    print(est1.summary(), "\n")

    est1 = sm.RLM(y, x, M=sm.robust.norms.LeastSquares()).fit()
    r2 = r2_score(y, est1.predict(x))
    print("R-squared : ", r2)
    print(est1.summary(), "\n")

    est1 = sm.RLM(y, x, M=sm.robust.norms.AndrewWave()).fit()
    r2 = r2_score(y, est1.predict(x))
    print("R-squared : ", r2)
    print(est1.summary(), "\n")

    est1 = sm.RLM(y, x, M=sm.robust.norms.RamsayE()).fit()
    r2 = r2_score(y, est1.predict(x))
    print("R-squared : ", r2)
    print(est1.summary(), "\n")

    est1 = sm.RLM(y, x, M=sm.robust.norms.TrimmedMean()).fit()
    r2 = r2_score(y, est1.predict(x))
    print("R-squared : ", r2)
    print(est1.summary(), "\n")

    est1 = sm.RLM(y, x, M=sm.robust.norms.Hampel()).fit()
    r2 = r2_score(y, est1.predict(x))
    print("R-squared : ", r2)
    print(est1.summary())


if __name__=="__main__":
    main()
