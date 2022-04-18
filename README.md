# AI_assessment

タイトル：「AI Quest2021 アセスメント」  

データセットを用いた特定の課題に対してAIモデル構築・分析して、結果を提出する。  
受験結果は選考において非常に重視される。  

課題内容    
<img src="https://user-images.githubusercontent.com/93046615/163835098-95d1dfa4-e50a-4cf9-af2f-6720bd84e58d.png" width="800px">  

結果　　  
231位/1231　　  

# CODE  

ライブラリ  
```bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import time
import datetime
%matplotlib inline
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , accuracy_score, recall_score, precision_score, f1_score
```
前処理（Train)
```bash
train = pd.read_csv("train.csv")
train["re_rate"] = train["host_response_rate"].str.split("%",expand = True)[0].astype(float)
today = datetime.datetime.today()
train["first_review"] = pd.to_datetime(train["first_review"])
train["host_since"] = pd.to_datetime(train["host_since"])
train["last_review"] = pd.to_datetime(train["last_review"])
train["first_counts"] = today - train["first_review"]
train["first_counts"] = train["first_counts"].dt.days
train["host_counts"] = today - train["host_since"]
train["host_counts"] = train["host_counts"].dt.days
train["last_counts"] = today - train["last_review"]
train["last_counts"] = train["last_counts"].dt.days
train.drop(["amenities"],axis = 1,inplace=True)
train.drop(["description","first_review","host_response_rate","host_since","last_review","name","neighbourhood"],axis = 1,inplace=True)
train.drop(["thumbnail_url"],axis = 1,inplace=True)
train.drop(["zipcode"],axis = 1,inplace=True)
train["y"] = train["y"].astype("int")
train = train.interpolate()
train_d = pd.get_dummies(train)
train_d.drop(['property_type_Casa particular', 'property_type_Earth House', 'property_type_Island', 'property_type_Parking Space', 'property_type_Tipi', 'property_type_Train', 'property_type_Treehouse', 'property_type_Vacation home'],axis=1,inplace=True)
train_d = train_d.dropna()
train_d.drop(["id"],axis = 1,inplace=True)
```
トレインデータとテストデータに分割  
```bash
X = train_d.drop(["y"],axis = 1)
y = train_d["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=10)
```
パラメーターの設定  
```bash
params = {'objective': 'regression',
 'metric': 'rmse',
 'feature_pre_filter': False,
 'lambda_l1': 1.9,
 'lambda_l2': 0.9,
 'num_leaves': 19,
 'feature_fraction': 0.7,
 'feature_fraction': 0.73,
 'bagging_fraction': 0.999,
 'min_child_samples': 25,
 'num_iterations': 410,
 'early_stopping_round': None,
}
```
モデルの作成 
```bash
# 訓練データから回帰モデルを作る
gbm = lgb.train(params, lgb_train)
# テストデータを用いて予測精度を確認する
test_predicted = gbm.predict(X_test)
predicted_df = pd.concat([y_test.reset_index(drop=True), pd.Series(test_predicted)], axis = 1)
predicted_df.columns = ['true', 'predicted']
# 予測値を図で確認する関数の定義
def Prediction_accuracy(predicted_df):
    RMSE = np.sqrt(mean_squared_error(predicted_df['true'], predicted_df['predicted']))
    plt.figure(figsize = (7,7))
    ax = plt.subplot(111)
    ax.scatter('true', 'predicted', data = predicted_df)
    ax.set_xlabel('True Price', fontsize = 20)
    ax.set_ylabel('Predicted Price', fontsize = 20)
    plt.tick_params(labelsize = 15)
    x = np.linspace(5, 50)
    y = x
    ax.plot(x, y, 'r-')
    plt.text(0.1, 0.9, 'RMSE = {}'.format(str(round(RMSE,3))),transform = ax.transAxes, fontsize = 15)
# 予測値を図で確認する
Prediction_accuracy(predicted_df)
```
結果の図  
<img src="https://user-images.githubusercontent.com/93046615/163836704-60b5c578-ed7f-4335-ba95-d83da000e9e8.png" width="500px">  
トレインデータとテストデータに分割  
```bash
predictions = gbm.predict(test_d)
ids = test.loc[:,["id"]]
predictions_pd = pd.DataFrame(predictions)
result_last = pd.concat([ids,predictions_pd], axis=1)
result_last.to_csv("result.csv",index = False)
```
# Note  
LightGBMを使って回帰モデルを作成した。  
pandasの使い方が分かってなくて今見るとかなり汚い。  
パラメータのチューニングは手動で行った。  
 
# Author

* 作成者 KeiichiAdachi
* 所属 Japan/Aichi
* E-mail keiichimonoo@gmail.com
 
# License
なし  
