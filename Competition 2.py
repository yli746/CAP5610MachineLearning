# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 21:34:53 2022

@author: SanSan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from catboost import CatBoostClassifier
from xgboost import XGBClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.info()
train.head()
#set passenger ID as default index
train.set_index('PassengerId', inplace=True)
test.set_index('PassengerId', inplace=True)

#split cabin into deck, num and side
train[['Deck','Num','Side']] = train.Cabin.str.split('/',expand=True) 
test[['Deck','Num','Side']] = test.Cabin.str.split('/',expand=True)

#drop name and cabin
train = train.drop(['Name','Cabin'], axis=1)
test = test.drop(['Name','Cabin'], axis=1)
#drop name and cabin, num, side
#train = train.drop(['Name','Cabin', 'Num', 'Side'], axis=1)
#test = test.drop(['Name','Cabin', 'Num', 'Side'], axis=1)

#make new colum of total spending
#train['TotalSpent'] =train['RoomService']+train['FoodCourt']+ train['ShoppingMall']+ train['Spa']+ train['VRDeck']
#test['TotalSpent'] = test['RoomService']+test['FoodCourt']+ test['ShoppingMall']+ test['Spa']+ test['VRDeck']

sns.heatmap(train.corr(), annot=True, cmap="PiYG")
#drop the small spending columns
#train = train.drop(['RoomService', 'FoodCourt', 'Spa', 'ShoppingMall', 'VRDeck'], axis=1)
#test = test.drop(['RoomService', 'FoodCourt', 'Spa', 'ShoppingMall', 'VRDeck'], axis=1)

#Make new column age group
#train['AgeGroup']  = 0
#for i in range(6):
    #train.loc[(train.Age >= 10*i) & (train.Age < 10*(i+1)),'AgeGroup'] = i
    
#test['AgeGroup']  = 0
#for i in range(6):
    #test.loc[(test.Age >= 10*i) & (test.Age < 10*(i+1)),'AgeGroup'] = i

#drop Age column
#train = train.drop('Age', axis=1)
#test = test.drop('Age', axis=1)

#change object to categorical
obj_cols = train.columns[train.dtypes == 'object'].tolist()
for i in obj_cols:
    train[i] = train[i].astype('category')
for i in obj_cols:
    test[i] = test[i].astype('category')

#####################Direct xgb without imputation###########################
#label encoder
train_label = train.copy(deep=True)
test_label = test.copy(deep=True)
le = LabelEncoder()
for i in obj_cols:
    series = train_label[i]
    train_label[i] = pd.Series(
        le.fit_transform(series[series.notnull()]),
        index = series[series.notnull()].index)
for i in obj_cols:
    series = test_label[i]
    test_label[i] = pd.Series(
        le.fit_transform(series[series.notnull()]),
        index = series[series.notnull()].index)
train_label.head()


#change transported to 1 and 0
train_label['Transported'] = train_label['Transported'].replace({True:1, False:0})
#Split X and y of train
X_xgb = train_label.drop('Transported', axis=1)
y_xgb = train_label['Transported']
#split training and validation set
X_xgb_train, X_xgb_val, y_xgb_train, y_xgb_val = train_test_split(X_xgb, y_xgb, test_size=0.3, random_state=888)
xgb = XGBClassifier()
xgb.fit(X_xgb_train, y_xgb_train)
y_predxgb = xgb.predict(X_xgb_val)
accuracy_score(y_xgb_val, [round(value) for value in y_predxgb])
#0.8041



##########XGB with multiple imputation##############################
train_label.columns
from fancyimpute import IterativeImputer as MICE
train_mice = MICE().fit_transform(train_label)
train_mice = pd.DataFrame(train_mice, columns=['HomePlanet', 'CryoSleep', 'Destination', 'Age','VIP', 'RoomService',
       'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported', 'Deck',
       'Num', 'Side'])
X_mice = train_mice.drop('Transported', axis=1)
y_mice = train_mice['Transported']
X_mice_train, X_mice_val, y_mice_train, y_mice_val = train_test_split(X_mice, y_mice, test_size=0.3, random_state=888)
xgb = XGBClassifier()
xgb.fit(X_mice_train, y_mice_train)
y_predmice_xgb = xgb.predict(X_mice_val)
accuracy_score(y_mice_val, [round(value) for value in y_predmice_xgb])
#0.8248
#tuning XGB
xgb_params = {'max_depth': [3, 6, 10, 15],
              'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.4],
              'subsample': [0.5, 1.0, 0.1],
              'colsample_bytree': [0.5, 1.0, 0.8],
              'colsample_bylevel': [0.5, 1.0, 0.8],
              'n_estimators': [100, 250, 500, 750]
              }
xgb_mice_tune = GridSearchCV(xgb, param_grid=xgb_params,cv=3,verbose=2)
xgb_mice_tune.fit(X_mice_train, y_mice_train)
xgb_mice_tune.best_params_
xgb_mice_tuned = XGBClassifier(colsample_bylevel=0.5, colsample_bytree=0.6, learning_rate=0.01, max_depth=10, n_estimators=250, subsample=0.8)
xgb_mice_tuned = XGBClassifier(colsample_bylevel=0.8, colsample_bytree=0.5, learning_rate=0.1, max_depth=6, n_estimators=100, subsample=1)
xgb_mice_tuned.fit(X_mice_train, y_mice_train)
y_predmice_tuned = xgb_mice_tuned.predict(X_mice_val)
accuracy_score(y_mice_val, [round(value) for value in y_predmice_tuned])
#0.8202

#process test data with MICE
test_mice = MICE().fit_transform(test_label)
test_mice = pd.DataFrame(test_mice, columns=['HomePlanet', 'CryoSleep', 'Destination', 'Age','VIP', 'RoomService',
       'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',  'Deck',
       'Num', 'Side'])

y_test_xgb_pred = xgb.predict(test_mice)
y_test_mice_pred = xgb_mice_tuned.predict(test_mice)

sub_1 = pd.DataFrame({'Transported': y_test_xgb_pred.astype(bool)}, index=test.index)
sub_2 = pd.DataFrame({'Transported': y_test_mice_pred.astype(bool)}, index=test.index)

sub_1.to_csv('submission_1.csv')
sub_2.to_csv('submission_2.csv')




