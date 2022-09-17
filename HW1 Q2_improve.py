# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 22:14:09 2022

@author: SanSan
"""
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
from sklearn.preprocessing import StandardScaler
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#Import data
train_1 = pd.read_csv('train.csv')
test_1 = pd.read_csv('test.csv')
combine_1 = [train_1, test_1]
train_1['Survived'].value_counts().plot.bar(title='Survival')
#Drop ticket and cabin
train_1 = train_1.drop(['Ticket', 'Cabin'], axis=1)
test_1 = test_1.drop(['Ticket', 'Cabin'], axis=1)
combine_1 = [train_1, test_1]

#standardize age and fare
train_1[['Age','Fare']] = StandardScaler().fit_transform(train_1[['Age','Fare']])
test_1[['Age','Fare']] = StandardScaler().fit_transform(test_1[['Age','Fare']])
combine_1 = [train_1,test_1]

#Creating new feature extracting from existing
for dataset in combine_1:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_1['Title'], train_1['Sex'])
#Rewrite title
for dataset in combine_1:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_1[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#Convert categorical titles to ordinal
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine_1:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_1.head()
#Drop name feature and passengerid feature
train_1 = train_1.drop(['Name', 'PassengerId'], axis=1)
test_1 = test_1.drop(['Name'], axis=1)
combine_1 = [train_1, test_1]
train_1.shape, test_1.shape

#Convert categorical feature
for dataset in combine_1:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_1.head()

#Fill missing value with other correlated features
# grid = sns.FacetGrid(train_1, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_1, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
#create empty array to contain guessed age based on Pclass x Gender combinations
guess_ages = np.zeros((2,3))
guess_ages
#iterate over sex(0 or 1) and Pclass(1,2,3) to calculate guessed values of Age for the 6 combinations
for dataset in combine_1:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_1 = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_1.mean()
            # age_std = guess_1.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_1.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_1.head()




for dataset in combine_1:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_1[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

for dataset in combine_1:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_1[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

train_1 = train_1.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_1 = test_1.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine_1 = [train_1, test_1]

train_1.head()


freq_port = train_1.Embarked.dropna().mode()[0]
freq_port

for dataset in combine_1:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_1[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#convert categorical feature to numerical
for dataset in combine_1:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_1.head()
test_1['Fare'].fillna(test_1['Fare'].dropna().median(), inplace=True)
test_1.head()

    
train_1.head(10)

test_1.head(10)

######################Model, predict and solve#########################################
X_train = train_1.drop("Survived", axis=1)
Y_train = train_1["Survived"]
X_test  = test_1.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape

#Balance data in training set only
from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler(sampling_strategy=0.8)
X_over, Y_over = oversample.fit_resample(X_train, Y_train)
Y_over.value_counts()

# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_over, Y_over)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_over, Y_over) * 100, 2)
acc_log 

coeff_1 = pd.DataFrame(train_1.columns.delete(0))
coeff_1.columns = ['Feature']
coeff_1["Correlation"] = pd.Series(logreg.coef_[0])

coeff_1.sort_values(by='Correlation', ascending=False)

# Support Vector Machines

svc = SVC()
svc.fit(X_over, Y_over)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_over, Y_over) * 100, 2)
acc_svc 

#knn
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_over, Y_over)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_over, Y_over) * 100, 2)
acc_knn 

# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_over, Y_over)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_over, Y_over) * 100, 2)
acc_gaussian 

# Perceptron

perceptron = Perceptron()
perceptron.fit(X_over, Y_over)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_over, Y_over) * 100, 2)
acc_perceptron 

# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_over, Y_over)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_over, Y_over) * 100, 2)
acc_linear_svc 

# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_over, Y_over)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_over, Y_over) * 100, 2)
acc_sgd 

# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_over, Y_over)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_over, Y_over) * 100, 2)
acc_decision_tree 

# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_over, Y_over)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_over, Y_over)
acc_random_forest = round(random_forest.score(X_over, Y_over) * 100, 2)
acc_random_forest 

####Model evaluation
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
