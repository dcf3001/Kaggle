# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 21:25:36 2020

@author: David
"""
# %% Import libraries
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Sequential
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %% Read in training and test data, and combine them
os.chdir('C:/Users/David/OneDrive/Kaggle/Titanic')
train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)
both = pd.concat([train, test])

# %% Get dummies for where passengers embarked from
both = both.join(pd.get_dummies(both['Embarked'], 'from'))
both = both.drop('Embarked', axis=1)

# %% TITLES: Get dummies for titles in names (Mr, Mrs, Miss...)


def title(name):
    if 'Mr.' in name:
        return 'Mr'
    elif 'Mrs.' in name:
        return 'Mrs'
    elif 'Miss' in name:
        return 'Miss'
    elif 'Master' in name:
        return 'Master'
    else:
        return 'Other'


both['title'] = both['Name'].apply(title)
both = both.join(pd.get_dummies(both['title'], 'Title'))
both = both.drop('title', axis=1)

# %% CABIN: Split cabin number (e.g. C123) into parts and get dummies


def cabin_split(cabin):
    if type(cabin) is str:
        return [cabin[0], cabin[1:]]
    else:
        return [0, 0]


both['CabinPrefix'] = both['Cabin'].apply(cabin_split).apply(lambda x: x[0])
both['CabinNumber'] = both['Cabin'].apply(cabin_split).apply(lambda x: x[1])
both = both.drop('Cabin', axis=1)
both = both.join(pd.get_dummies(both['CabinPrefix'], 'Cabin'))
both = both.drop('Cabin_0', axis=1)
both = both.drop(['CabinPrefix', 'CabinNumber'], axis=1)


# %% SEX: Change to 1 for female, 0 for male
both['Female'] = (both['Sex'] == 'female')
both = both.drop('Sex', axis=1)

# %% TICKET: Count how many travelling on the same ticket
tickets = both['Ticket'].value_counts()
both['SameTicket'] = both['Ticket'].apply(lambda x: tickets[x])
both = both.drop('Ticket', axis=1)

# %% FAMILY: Count how many travelling in the same family
both['LastName'] = both['Name'].apply(lambda x: str.split(x, ",")[0])
both['LastNameFare'] = tuple(both[['LastName', 'Fare']].values)
families = both['LastNameFare'].value_counts()
both['SameFamily'] = both['LastNameFare'].apply(lambda x: families[x])
both = both.drop('LastNameFare', axis=1)

# %% IMPUTE FARE: for the guy without fare data, impute from Pclass
both[pd.isna(both['Fare'])]
both.groupby('Pclass').mean()['Fare']


def impute_fare(pclass, fare):
    if pd.isna(fare):
        return both.groupby('Pclass').mean()['Fare'][pclass]
    else:
        return fare


both['ImputedFare'] = both.apply(lambda x: impute_fare(x['Pclass'],
                                                       x['Fare']), axis=1)
both['Fare'] = both['ImputedFare']
both = both.drop('ImputedFare', axis=1)

# Say anything below GBP 24 is cheap
both['Cheap'] = both['Fare'].apply(lambda x: x < 24)

# %% IMPUTE AGE: we impute 263 missing ages by regression from the training set
train = both[pd.notna(both['Survived'])]
train.corr()['Age'].sort_values()  # examine correlations

# Use StatsModels to get p-values and pick regressors
age_regressors = ['Title_Master', 'Title_Mr', 'Title_Miss', 'Title_Mrs',
                  'Title_Other', 'Pclass', 'SibSp', 'Female', 'Age']
X = train[age_regressors].dropna().drop('Age', axis=1)
y = train['Age'].dropna()
X2 = sm.add_constant(X)  # required for regression
reg_age = sm.OLS(y, X2.astype(float))
# prints coefficients and p-values; R^2 is 0.41 #
print(reg_age.fit().summary())

# Use SciKitLearn to predict ages based on those regressors
sk_reg_age = LinearRegression()
sk_reg_age.fit(X, y)
both_X = both[age_regressors].drop('Age', axis=1)
age_predictions = sk_reg_age.predict(both_X)
both['ImputedAge'] = age_predictions

# Pick the imputed age if we don't have the age already


def impute_age(age, imputed_age):
    if pd.isna(age):
        return max(imputed_age, 0)
    else:
        return age


both['Age'] = both.apply(lambda x: impute_age(
    x['Age'], x['ImputedAge']), axis=1)
both = both.drop('ImputedAge', axis=1)

# Call anyone under 16 a child
both['Child'] = both['Age'].apply(lambda x: x < 16)

# %% DEFINE TRAIN AND TEST SETS
both = both.drop(['Name', 'LastName'], axis=1)
train = both[pd.notna(both['Survived'])]
test = both[pd.isna(both['Survived'])].drop('Survived', axis=1)

# %% SCALING
X_train = train.drop('Survived', axis=1)
y_train = train['Survived']
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index,
                       columns=X_train.columns)
X_test = pd.DataFrame(scaler.fit_transform(test), index=test.index,
                      columns=test.columns)

# %% TRAIN-TEST SPLIT
X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(
    X_train, y_train, test_size=0.3)

# %% NEURAL NETWORK (SCORE: 0.77751)

model = Sequential()

# input layer
model.add(Dense(26, activation='relu'))
model.add(Dropout(0.4))

# hidden layer
model.add(Dense(13, activation='relu'))
model.add(Dropout(0.4))

# hidden layer
model.add(Dense(6, activation='relu'))
model.add(Dropout(0.4))

# output layer
model.add(Dense(units=1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=np.array(X_train_train),
          y=np.array(y_train_train),
          epochs=200,
          batch_size=128,
          validation_data=(X_train_test, y_train_test),
          )

losses = pd.DataFrame(model.history.history)
losses[['loss', 'val_loss']].plot()

predictions = model.predict_classes(X_train_test)
print(classification_report(y_train_test, predictions))

predictions_out = model.predict_classes(X_test)
pd.DataFrame(predictions_out,
             columns=['Survived'],
             index=X_test.index).to_csv('neural network2.csv')
