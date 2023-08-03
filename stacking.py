import os
import sys
import glob
import joblib
import numpy as np
from time import *
import pandas as pd
from scipy.stats import pearsonr
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

begin_time = time()

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()
sys.stdout = Logger('./log/layer2_result.txt')

print(path)
print(os.path.dirname(__file__))
print('----------prediction results----------')

if __name__ == "__main__":
    files = glob.glob(".\Results\Yield_model_ture_pred\*.csv") #Base models prediction valves
    df = None
    for f in files:
        if df is None:
            df = pd.read_csv(f)
        else:
            temp_df = pd.read_csv(f)
            df = df.merge(temp_df, on=['Hybrid', 'Yield','kfold'], how='left')
            df = df.groupby(['Hybrid', 'Yield', 'kfold']).agg('first').reset_index()
    print(df)
    df.to_csv("./data/train_set/New_Yield_values.csv", index=False) #Generate a new dataset-Mate dataset

def run_training(fold):
    df = pd.read_csv("./data/train_set/New_Yield_values.csv")
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    print(np.shape(df))

    xtrain = df_train.iloc[:, 3:7]
    xvalid = df_valid.iloc[:, 3:7]
    # print(xtrain)
    #print(xvalid)

    ytrain = df_train.Yield.values
    yvalid = df_valid.Yield.values
    #print(ytrain)
    #print(yvalid)

    #clf = LinearRegression()
    clf = linear_model.Ridge(alpha=0.0001, normalize=False)
    clf.fit(xtrain, ytrain)
    pred = clf.predict(xvalid)

    test = pd.read_csv("./data/test_set/New_test_values.csv")
    test_x = test.iloc[:, 1:5]
    # print(test_x)
    # 保存
    model_save_path = './models/layer2model.pkl'
    # 保存模型
    joblib.dump(clf, model_save_path)
    # 加载模型并预测
    Ind_test = joblib.load(model_save_path)
    test_pred = Ind_test.predict(test_x)
    # 读取Pred_data.csv的前两列 "Hybrid","Env"
    Hybrid = []
    fr = open("./data/test_set/New_test_values.csv", 'r')
    for line in fr.readlines():
        # 删除引号
        line = line.replace('"', '"')
        d = line.split(',')
        Hybrid.append((d[1], d[2]))
    # 删除第一个
    # Env.pop(0)
    out_x = pd.DataFrame([Hybrid, test_pred])
    out_x.T.to_csv("./Results/Metamodel_LARS_pred/fold{}_layer2_test_pred.csv".format(fold), index=False)

    MSE = mean_squared_error(yvalid, pred)
    RMSE = np.sqrt(mean_squared_error(yvalid, pred))
    MAE = mean_absolute_error(yvalid, pred)
    PCCs = pearsonr(yvalid, pred)

    print(f"fold={fold}, MSE={MSE}")
    print(f"fold={fold}, RMSE={RMSE}")
    print(f"fold={fold}, MAE={MAE}")
    print(f"fold={fold}, PCCs={PCCs}")

    df_valid.loc[:, "Layer2_pred"] = pred

    return df_valid[["Hybrid", "Yield", "kfold", "Layer2_pred"]]

if __name__ == "__main__":
    dfs = []
    for j in range(10):
        temp_df = run_training(j)
        dfs.append(temp_df)
    fin_valid_df = pd.concat(dfs)
    print(fin_valid_df.shape)
    fin_valid_df.to_csv("./Results/layer2 ture result/LARS_pred.csv",index=False)

end_time = time()
run_time = end_time - begin_time
print("LARS program run time:" , run_time)
