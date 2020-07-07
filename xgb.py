import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm

data_train = pd.read_csv('sensor_train.csv')
data_test = pd.read_csv('sensor_test.csv')
data_test['fragment_id'] += 10000
label = 'behavior_id'

data = pd.concat([data_train, data_test], sort=False)

df = data.drop_duplicates(subset=['fragment_id']).reset_index(drop=True)[['fragment_id', 'behavior_id']]

data['acc'] = (data['acc_x'] ** 2 + data['acc_y'] ** 2 + data['acc_z'] ** 2) ** 0.5
data['accg'] = (data['acc_xg'] ** 2 + data['acc_yg'] ** 2 + data['acc_zg'] ** 2) ** 0.5

for f in tqdm([f for f in data.columns if 'acc' in f]):
    for stat in ['min', 'max', 'mean', 'median', 'std', 'skew']:
        df[f+'_'+stat] = data.groupby('fragment_id')[f].agg(stat).values


train_df = df[df[label].isna()==False].reset_index(drop=True)
test_df = df[df[label].isna()==True].reset_index(drop=True)

drop_feat = []
used_feat = [f for f in train_df.columns if f not in (['fragment_id', label] + drop_feat)]

train_x = train_df[used_feat]
train_y = train_df[label]
test_x = test_df[used_feat]


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, SelectFromModel


X = train_x.values
y = train_y.values


params = {
    'learning_rate': 0.05,
    'n_estimators': 400,
    'max_depth': 5,
    'min_child_weight': 1,
    'gamma': 0.1,
    'colsample_bytree': 0.6,
    'subsample': 0.8,
    'objective': 'binary:logistic',
    'reg_alpha': 0.0125,
    'reg_lambda': 0.025

}

clf = XGBClassifier(**params, random_state=2019)
grid_params = {
    'learning_rate':np.linspace(0.01,0.2,20),
    'n_estimators': list(range(100, 601, 100)),
    # 'max_depth': list(range(3, 12, 1)),
    # 'min_child_weight': list(range(1, 6, 2.txt)),
    #  'gamma':[i/10.0 for i in range(0,5)],
    # 'subsample': [i / 10.0 for i in range(6, 10)],
    # 'colsample_bytree': [i / 10.0 for i in range(6, 10)],
    # 'reg_alpha':np.linspace(0,0.05,5),
    # 'reg_lambda':np.linspace(0,0.05,5)
}
# test_x=test_x.values
# from sklearn.preprocessing import MinMaxScaler
# minMax = MinMaxScaler()
# minMax.fit(X)
#
# X = minMax.transform(X)
# test_x=minMax.transform(test_x)
#
# clf.fit(X,y)
# pre=clf.predict(X)
# print(pre)
# print("ACC:{}".format(metrics.accuracy_score(y, pre)))
#
# res=clf.predict(test_x)
# result=pd.read_csv("res.csv")
# result["behavior_id"]=res
# print(result)
# result.to_csv("result.csv",index=None,)
#-------------------------------------------
grid = GridSearchCV(clf, grid_params, n_jobs=-1, cv=5)
grid.fit(X, y)
print(grid.best_params_)
print("Accuracy:{0:.1f}%".format(100 * grid.best_score_))




