import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("./train_set/G.csv")
    df.duplicated().any()
    df.loc[:, "kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)

    y = df.Yield.values
    skf = model_selection.KFold(n_splits=10,shuffle=True,random_state=2023)

    for f, (t_, v_)in enumerate(skf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = f

    df.to_csv("train_Yield_folds.csv", index=False)