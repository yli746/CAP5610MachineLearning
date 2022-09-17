# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 11:33:21 2022

@author: SanSan
"""

import pandas as pd
import seaborn as sns


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

################Separate train and validation set
train_data, val_data = train_test_split(train_df, test_size=0.3, random_state=100)
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_val = val_data.drop("Survived", axis=1)
Y_val = val_data["Survived"]

########KNN
n_neighbors = [5,6,7,8,9,10,12,14,16,18,20]
algorithm = ['auto']
weights = ['uniform','distance']
leaf_size = list(range(1,50,5))
KNN_para = {'algorithm':algorithm, 'weights':weights,'leaf_size':leaf_size, 'n_neighbors':n_neighbors}
KNN_tune = GridSearchCV(KNeighborsClassifier(),param_grid=KNN_para,verbose=3,cv=10,scoring='roc_auc')
KNN_tune.fit(X_train,Y_train)
KNN_tune.best_params_

Tuned_KNN =KNeighborsClassifier(algorithm='auto',leaf_size=11,n_neighbors=18,weights='distance')
Tuned_KNN.fit(X_train,Y_train)
Tuned_KNN.score(X_val,Y_val)

Y_pred_KNN = Tuned_KNN.predict(test_df)

##############Random forest
RF_para = {'max_depth':[10,20,50,100,500],'min_samples_split':[2,5,10],\
           'min_samples_leaf':[2,5,10],\
               'n_estimators':[5,10,50,100],'max_samples':[0.2,0.5,0.8]}
RF = RandomForestClassifier()
RF_tune = GridSearchCV(RF, param_grid=RF_para, scoring='f1', cv=3,verbose=3)
RF_tune.fit(X_train,Y_train)
RF_tune.best_params_

Tuned_RF = RandomForestClassifier(max_depth=10, max_samples=0.8, min_samples_split=10,\
                                  min_samples_leaf=2, n_estimators=50)
Tuned_RF.fit(X_train,Y_train)
Tuned_RF.score(X_val,Y_val)*100
Tuned_RF.score(X_train,Y_train)

Y_pred_RF = Tuned_RF.predict(test_df)

test_pid = test[['PassengerId']]
test_pred = pd.DataFrame(Y_pred_RF,columns=['Survived'])
test_submit = test_pid.join(test_pred)
test_submit.to_csv('Test_Submission.csv',index=False)





























