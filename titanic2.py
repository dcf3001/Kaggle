# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 17:27:09 2020

@author: David
"""


# %% Import libraries
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
train.corr()['Age'].sort_values() # examine correlations

# Use StatsModels to get p-values and pick regressors
age_regressors = ['Title_Master', 'Title_Mr', 'Title_Miss', 'Title_Mrs', 
                  'Title_Other', 'Pclass', 'SibSp', 'Female', 'Age']
X = train[age_regressors].dropna().drop('Age', axis=1) 
y = train['Age'].dropna()
X2 = sm.add_constant(X) # required for regression
reg_age = sm.OLS(y, X2.astype(float))
print(reg_age.fit().summary()) # prints coefficients and p-values; R^2 is 0.41 #

# Use SciKitLearn to predict ages based on those regressors
from sklearn.linear_model import LinearRegression
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
both['Age'] = both.apply(lambda x: impute_age(x['Age'], x['ImputedAge']), axis=1)
both = both.drop('ImputedAge', axis=1)

# Call anyone under 16 a child
both['Child'] = both['Age'].apply(lambda x: x < 16)

# %% DEFINE TRAIN AND TEST SETS
both = both.drop(['Name', 'LastName'], axis=1)
train = both[pd.notna(both['Survived'])]
test = both[pd.isna(both['Survived'])].drop('Survived', axis=1)

# %% DEFINE SUBSETS OF INTEREST
train.groupby(['Female', 'Pclass', 'Child']).mean() # Shows the hard ones are first-class men, third-class women + children
both['RichMen'] = (both.Pclass==1) & (both.Female==0) & (both.Child==0)
both['PoorWomenKids'] = (both.Pclass==3) & ((both.Female==1) | (both.Child==1))
both['NotRichMen'] = (both.Pclass!=1) & (both.Female==0) & (both.Child==0)
both['NotPoorWomenKids'] = (both.Pclass!=3) & ((both.Female==1) | (both.Child==1))

rich_men = both[both.RichMen] # 35% of rich men survive
rich_men_train = rich_men[pd.notna(both['Survived'])]
rich_men_test = rich_men[pd.isna(both['Survived'])]

poor_women_kids = both[both.PoorWomenKids] # 31% of poor boys, 46% of poor girls, 51% of poor women survive
poor_women_kids_train = poor_women_kids[pd.notna(both['Survived'])]
poor_women_kids_test = poor_women_kids[pd.isna(both['Survived'])]

not_rich_men_test = test[test.NotRichMen]
not_poor_women_kids_test = test[test.NotPoorWomenKids]

pd.DataFrame(not_rich_men_test.index).to_csv('List of not rich men.csv')
pd.DataFrame(not_poor_women_kids_test.index).to_csv('List of not poor women and kids.csv')
# %% DEFINE TRAIN AND TEST SETS
train = both[pd.notna(both['Survived'])]
test = both[pd.isna(both['Survived'])].drop('Survived', axis=1)

# %% LOGISTIC REGRESSION ON RICH MEN

# Scaling
from sklearn.preprocessing import MinMaxScaler
X_train = rich_men_train.drop('Survived', axis=1)
y_train = rich_men_train['Survived']
X_test = rich_men_test
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index,
                       columns=X_train.columns)
X_test = pd.DataFrame(scaler.fit_transform(X_test), index=X_test.index,
                      columns=X_test.columns)

# Decide regressors using StatsModels to get p-values
rich_men.corr()['Survived'].sort_values()
regressors = ['Age', 'Cabin_E']

X = X_train[regressors]
Xt = X_test[regressors]
y = y_train
log_reg = sm.Logit(y, X.astype(float)).fit()
print(log_reg.summary())

# Use SciKitLearn to predict survival based on those regressors
from sklearn.linear_model import LogisticRegression
sk_log_reg = LogisticRegression()
sk_log_reg.fit(X, y)
sk_log_reg.coef_
in_sample_predictions = sk_log_reg.predict(X)
out_sample_predictions = sk_log_reg.predict(Xt)
out_sample_predictions = out_sample_predictions.astype(int)
passenger_list = Xt.index

# Print confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_train, in_sample_predictions))
print(classification_report(y_train, in_sample_predictions))

# Save to CSV file
pd.DataFrame(out_sample_predictions, 
             columns=['Survived'],
             index=passenger_list).to_csv('RichMen_Logit.csv')

# %% LOGISTIC REGRESSION ON POOR WOMEN AND CHILDREN

# Scaling
from sklearn.preprocessing import MinMaxScaler
X_train = poor_women_kids_train.drop('Survived', axis=1)
y_train = poor_women_kids_train['Survived']
X_test = poor_women_kids_test
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index,
                       columns=X_train.columns)
X_test = pd.DataFrame(scaler.fit_transform(X_test), index=X_test.index,
                      columns=X_test.columns)

# Decide regressors using StatsModels to get p-values
poor_women_kids.corr()['Survived'].sort_values()
regressors = ['Age', 'SameFamily', 'Cheap']

X = X_train[regressors]
Xt = X_test[regressors]
y = y_train
log_reg = sm.Logit(y, X.astype(float)).fit()
print(log_reg.summary())

# Use SciKitLearn to predict survival based on those regressors
from sklearn.linear_model import LogisticRegression
sk_log_reg = LogisticRegression()
sk_log_reg.fit(X, y)
sk_log_reg.coef_
in_sample_predictions = sk_log_reg.predict(X)
out_sample_predictions = sk_log_reg.predict(Xt)
out_sample_predictions = out_sample_predictions.astype(int)
passenger_list = Xt.index

# Print confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_train, in_sample_predictions))
print(classification_report(y_train, in_sample_predictions))

# Save to CSV file
pd.DataFrame(out_sample_predictions, 
             columns=['Survived'],
             index=passenger_list).to_csv('PoorWomenKids_Logit.csv')