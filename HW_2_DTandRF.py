# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 22:40:42 2022

@author: SanSan
"""

import pandas as pd
import matplotlib.pyplot as plt


from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV




#Import data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
all_data = train.append(test)

all_data['Title'] = all_data['Name']
for name_string in all_data['Name']:
    all_data['Title'] = all_data['Name'].str.extract('([A-Za-z]+)\.', expand=True)

all_data['Title'].unique()

mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
all_data.replace({'Title': mapping}, inplace=True)


#impute age based on title median
titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']
for title in titles:
    age_to_impute = all_data.groupby('Title')['Age'].median()[titles.index(title)]
    all_data.loc[(all_data['Age'].isnull()) & (all_data['Title'] == title), 'Age'] = age_to_impute
    
#create new variable family size and drop sibsp, parch
all_data['Family_Size'] = all_data['Parch'] + all_data['SibSp']
all_data.drop(['Parch','SibSp'], axis=1, inplace=True)

#impute fare based on Pclass and age
all_data['Fare'] = train.groupby(['Pclass']).Fare.apply(lambda x: x.fillna(x.median()))
all_data.isnull().sum()

#impute Embarked based on mode
freq_port = train.Embarked.dropna().mode()[0]
all_data['Embarked'] = all_data['Embarked'].fillna(freq_port)

#drop name, passengerid, ticket, cabin
all_data.drop(['Name','PassengerId','Ticket','Cabin'],axis=1,inplace=True)

#Convert categorical titles to ordinal
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4,"Rev":5,"Dr":6}
all_data['Title'] = all_data['Title'].map(title_mapping)
all_data['Sex'] = all_data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
all_data['Embarked'] = all_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

#make fare and age into categorical 
all_data['Fare'].fillna(all_data['Fare'].median(), inplace = True)
all_data['FareBin'] = pd.qcut(all_data['Fare'], 5)

label = LabelEncoder()
all_data['FareBin_Code'] = label.fit_transform(all_data['FareBin'])

all_data['AgeBin'] = pd.qcut(all_data['Age'], 4)

label = LabelEncoder()
all_data['AgeBin_Code'] = label.fit_transform(all_data['AgeBin'])

all_data.drop(['Age','Fare','AgeBin','FareBin'],axis=1,inplace=True)

train_df = all_data[:891]
test_df = all_data[891:]
test_df.drop('Survived',axis=1,inplace=True)

X = train_df.drop("Survived", axis=1)
Y = train_df["Survived"]


###############Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
#fine tune the parameters
parameters = {'max_leaf_nodes':[10,50,80],
              'criterion':['gini','entropy'], 
              'max_depth':[2,10,18],'min_samples_leaf':[1,2,5],
              }

dec_tree = DecisionTreeClassifier()
DT_tune = GridSearchCV(dec_tree, param_grid=parameters,scoring='f1',cv=3,verbose=3)
DT_tune.fit(X,Y)
DT_tune.best_params_
#{'criterion': 'entropy','max_depth': 10,'max_leaf_nodes': 10,'min_samples_leaf': 2}
Tuned_DT = DecisionTreeClassifier(max_depth=10, min_samples_leaf=2,criterion='entropy',\
                                      max_leaf_nodes=10)
Tuned_DT.fit(X,Y)
#plot the decision Tree
from sklearn import tree
plt.figure(figsize=(25,10))
levels = ['Died','Survivded']
tree.plot_tree(Tuned_DT, feature_names=X_train.columns, class_names=levels, filled=True, rounded=True,fontsize=14)

#5-fold cross validation
from sklearn.model_selection import cross_val_score
DT_scores = cross_val_score(Tuned_DT, X,Y, cv=5)
DT_scores
#array([0.83798883, 0.81460674, 0.83146067, 0.79213483, 0.85955056])
DT_scores.mean()
#0.8271483271608814


######################Random Forest
from sklearn.ensemble import RandomForestClassifier
RF_para = {'max_depth':[10,20,50],'min_samples_split':[2,5,10],\
           'min_samples_leaf':[2,5,10],\
               'n_estimators':[5,10,50],'max_samples':[0.2,0.5,0.8]}
RF = RandomForestClassifier()
RF_tune = GridSearchCV(RF, param_grid=RF_para, scoring='f1', cv=3,verbose=3)
RF_tune.fit(X,Y)
RF_tune.best_params_
#{'max_depth': 20,'max_samples': 0.5,'min_samples_leaf': 2,'min_samples_split': 10,'n_estimators': 10}
Tuned_RF = RandomForestClassifier(max_depth=20, max_samples=0.5, min_samples_split=10,\
                                  min_samples_leaf=2, n_estimators=10)
Tuned_RF.fit(X, Y)
RF_scores = cross_val_score(Tuned_RF,X,Y,cv=5)
RF_scores
#array([0.83798883, 0.8258427 , 0.83146067, 0.81460674, 0.86516854])
RF_scores.mean()
#0.8350134957002071















