# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 18:40:33 2020

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

# %% CABIN: Split cabin number (e.g. C123) into parts and get dummies
def cabin_split(cabin):
    if type(cabin) is str:
        return [cabin[0], cabin[1:]]
    else:
        return [0, 0] 

both['cabin_prefix'] = both['Cabin'].apply(cabin_split).apply(lambda x: x[0])
both['cabin_number'] = both['Cabin'].apply(cabin_split).apply(lambda x: x[1])
both = both.drop(['Cabin', 'Embarked'], axis=1)
both = both.join(pd.get_dummies(both['cabin_prefix'], 'Cabin'))
both = both.drop('Cabin_0', axis=1)
both = both.drop(['cabin_prefix', 'cabin_number'], axis=1)

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

# %% SEX: Change to 1 for female, 0 for male
both['Female'] = (both['Sex'] == 'female')
both = both.drop('Sex', axis=1)
both['FemaleSurvived'] = both.Female & both.Survived
both['FemaleDied'] = both.Female & (both.Survived==0)
both['MaleSurvived'] = (both.Female==0) & both.Survived
both['MaleDied'] = (both.Survived==0) & (both.Female==0)
both['GenderPredicted'] = both.FemaleSurvived | both.MaleDied

train = both[pd.notna(both['Survived'])].drop('Name', axis=1)
test = both[pd.isna(both['Survived'])].drop(['Name', 'Survived'], axis=1)

tst = train.groupby(['GenderPredicted', 'Female']).mean()

# %% TICKET: Count how many travelling on the same ticket
ticket_sales = both['Ticket'].value_counts()
def same_ticket(ticket):
    return ticket_sales[ticket]
both['same_ticket'] = both['Ticket'].apply(same_ticket)
# both = both.drop('Ticket', axis=1)

ticket_sales_train = pd.DataFrame(train['Ticket'].value_counts())
# ticket_sales_train['Number'] = ticket_sales_train.index

# Count how many survived on each ticket
ticket_survived = {}
ticket_female = {}
ticket_femalesurvived = {}
ticket_femaledied = {}
ticket_malesurvived = {}
ticket_maledied = {}
ticket_genderpredicted = {}
ticket_class = {}

for ticket in ticket_sales_train.index:
    ticket_survived[ticket] = train[train['Ticket']==ticket]['Survived'].sum().astype(int)
    ticket_female[ticket] = train[train['Ticket']==ticket]['Female'].sum().astype(int)
    ticket_femalesurvived[ticket] = train[train['Ticket']==ticket]['FemaleSurvived'].sum().astype(int)
    ticket_femaledied[ticket] = train[train['Ticket']==ticket]['FemaleDied'].sum().astype(int)
    ticket_malesurvived[ticket] = train[train['Ticket']==ticket]['MaleSurvived'].sum().astype(int)
    ticket_maledied[ticket] = train[train['Ticket']==ticket]['MaleDied'].sum().astype(int)
    ticket_genderpredicted[ticket] = train[train['Ticket']==ticket]['GenderPredicted'].sum().astype(int)
    ticket_class[ticket] = train[train['Ticket']==ticket]['Pclass'].mean()


def data_ticket(ticket):
    return [ticket_female[ticket], ticket_survived[ticket], ticket_femalesurvived[ticket], ticket_malesurvived[ticket],
            ticket_femaledied[ticket], ticket_maledied[ticket], ticket_gendersurvived[ticket], ticket_class[ticket]]

data_ticket('347082')

ticket_sales_train['Data'] = list(map(data_ticket, list(ticket_sales_train.index)))
ticket_sales_train['Female'] = ticket_sales_train.Data.apply(lambda x:x[0])
ticket_sales_train['Survived'] = ticket_sales_train.Data.apply(lambda x:x[1])
ticket_sales_train['FemaleSurvived'] = ticket_sales_train.Data.apply(lambda x:x[2])
ticket_sales_train['MaleSurvived'] = ticket_sales_train.Data.apply(lambda x:x[3])
ticket_sales_train['FemaleDied'] = ticket_sales_train.Data.apply(lambda x:x[4])
ticket_sales_train['MaleDied'] = ticket_sales_train.Data.apply(lambda x:x[5])
ticket_sales_train['GenderPredicted'] = ticket_sales_train.Data.apply(lambda x:x[6])
ticket_sales_train['PClass'] = ticket_sales_train.Data.apply(lambda x:x[7])
ticket_sales_train = ticket_sales_train.drop('Data', axis=1)
ticket_sales_train.columns = ['Number', 'Female', 'Survived', 'FSurvived', 'MSurvived', 'FDied', 'MDied', 'GPredicted', 'PClass']
ticket_sales_train['AllSurvived'] = ticket_sales_train['Number'] == ticket_sales_train['Survived'] 
ticket_sales_train['AllDied'] = ticket_sales_train['Survived'] == 0
ticket_sales_train['SharedFate'] = ticket_sales_train['AllSurvived'] | ticket_sales_train['AllDied']
ticket_sales_train['AllGenderPredicted'] = ticket_sales_train['Number'] == ticket_sales_train['GPredicted']


group_data = ticket_sales_train.groupby(['Number', 'PClass']).mean()
group_data['Number']

def group_op(PClass):
    return group_data[group_data['PClass']==Pclass]

group_data[group_data['PClass']==3.0]
ticket_sales_train.corr()['Survived']


# %% MISSING DATA: Find out what's missing
missing = both.isnull().sum()

# %% IMPUTE FARE: for the guy without fare data, impute from Pclass
both[pd.isna(both['Fare'])]
both.groupby('Pclass').mean()['Fare']
def impute_fare(pclass, fare):
    if pd.isna(fare):
        return both.groupby('Pclass').mean()['Fare'][pclass]
    else:
        return fare
    
both['imputed_fare'] = both.apply(lambda x: impute_fare(x['Pclass'],
                                                        x['Fare']), axis=1)
both['Fare'] = both['imputed_fare']
both = both.drop('imputed_fare', axis=1)

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
both['imputed_age'] = age_predictions

# Pick the imputed age if we don't have the age already
def impute_age(age, imputed_age):
    if pd.isna(age):
        return max(imputed_age, 0)
    else:
        return age   
both['Age'] = both.apply(lambda x: impute_age(x['Age'], x['imputed_age']), axis=1)
both = both.drop('imputed_age', axis=1)

# Call anyone under 16 a child
both['Child'] = both['Age'].apply(lambda x: x < 16)

# %% INVESTIGATE WHOLE-FAMILY SURVIVAL

train['Ticket'].value_counts()

# %% DEFINE TRAIN AND TEST SETS
train = both[pd.notna(both['Survived'])].drop('Name', axis=1)
test = both[pd.isna(both['Survived'])].drop(['Name', 'Survived'], axis=1)

# %% ONLY WOMEN SURVIVE (SCORE: 0.76555)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(train['Survived'], train['Female']))
print(classification_report(train['Survived'], train['Female']))

# %% RICH WOMEN + WOMEN & CHILDREN ON CHEAP TICKETS SURVIVE (SCORE: 0.77033)
both['RichCheap'] = (both['Female'] & (both['Pclass'] != 3)) | \
(both['Cheap'] & both['Child'])

train.groupby('Female').mean()
train.groupby('RichCheap').mean()

pd.DataFrame(test['RichCheap'].astype(int)).to_csv('rich_or_cheap.csv')

# %% ABOVE BUT ANYONE IN A GROUP OF 5+ BELOW FIRST CLASS DIES (SCORE: 0.77033)
both['RichCheapBig'] = both['RichCheap'] & ((both['same_ticket'] < 5) | (both['Pclass'] == 1))
train = both[pd.notna(both['Survived'])].drop('Name', axis=1)
test = both[pd.isna(both['Survived'])].drop(['Name', 'Survived'], axis=1)
pd.DataFrame(test['RichCheap'].astype(int)).to_csv('RichCheapBig.csv')

# %% WOMEN SURVIVE UNLESS IN A GROUP OF 5+ BELOW FIRST CLASS
both['FSmallGroup'] = both['Female'] & ((both['same_ticket']<5) | (both['Pclass']==1))
train = both[pd.notna(both['Survived'])].drop('Name', axis=1)
test = both[pd.isna(both['Survived'])].drop(['Name', 'Survived'], axis=1)
pd.DataFrame(test['FSmallGroup'].astype(int)).to_csv('FSmallGroup.csv')

# %% SCALING
from sklearn.preprocessing import MinMaxScaler
X_train = train.drop('Survived', axis=1)
y_train = train['Survived']
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index,
                       columns=X_train.columns)
X_test = pd.DataFrame(scaler.fit_transform(test), index=test.index,
                      columns=test.columns)

# %% LOGISTIC REGRESSION (SCORE: 0.77511)

# Decide regressors using StatsModels to get p-values
train.corr()['Survived'].sort_values()
regressors = ['Title_Mrs', 'Title_Miss', 'Title_Master',
              'Pclass', 'Age', 'Cabin_D', 'Cabin_E', 'cheap']
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
print(confusion_matrix(y_train, in_sample_predictions))
print(classification_report(y_train, in_sample_predictions))

# Save to CSV file
pd.DataFrame(out_sample_predictions, 
             columns=['Survived'],
             index=passenger_list).to_csv('logistic_regression_Day3.csv')

# %% TRAIN-TEST SPLIT
from sklearn.model_selection import train_test_split
X_train_train, X_train_test, y_train_train, y_train_test = \
train_test_split(X_train, y_train, test_size=0.3)

# %% DECISION TREE (SCORE: 0.77033)
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.tree import plot_tree

dtree = DecisionTreeClassifier(max_depth=10, min_samples_split=5, 
                               min_samples_leaf=2, max_features='sqrt')

# fit on the whole training sample
dtree.fit(X_train, y_train)
tree_preds_in = dtree.predict(X_train) 
print(confusion_matrix(y_train, tree_preds_in))
print(classification_report(y_train, tree_preds_in))

# fit on training subsample
dtree.fit(X_train_train, y_train_train)
tree_preds_sub = dtree.predict(X_train_test)
print(confusion_matrix(y_train_test, tree_preds_sub))
print(classification_report(y_train_test, tree_preds_sub))

# fit on test sample
dtree.fit(X_train, y_train)
tree_preds_out = dtree.predict(X_test).astype(int)
print(export_text(dtree))
fig = plt.figure(figsize=(50,50))
view_tree = plot_tree(dtree, feature_names=X_train.columns, 
                      class_names=['Died', 'Survived'], filled=True)

# Save to CSV file
pd.DataFrame(tree_preds_out, 
             columns=['Survived'],
             index=passenger_list).to_csv('decision_tree.csv')

# %% RANDOM FOREST (SCORE: 0.77990) - HYPERPARAMETER TUNING
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf_reg = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf_reg, param_distributions 
                               = random_grid, n_iter = 100, cv = 3, verbose=2, 
                               random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train_train, y_train_train)
rf_random.best_params_

# %% RANDOM FOREST (SCORE: 0.77990 ) - RUNNING THE MODEL

rf = RandomForestClassifier(n_estimators=1200, min_samples_split=5, 
                            min_samples_leaf=2, max_features='sqrt',
                            max_depth=10, bootstrap=False)
rf.get_params()
# fit on the whole training sample
rf.fit(X_train, y_train)
rf_preds_in = rf.predict(X_train) 
print(confusion_matrix(y_train, rf_preds_in))
print(classification_report(y_train, rf_preds_in))

# fit on training subsample
rf.fit(X_train_train, y_train_train)
rf_preds_sub = rf.predict(X_train_test)
print(confusion_matrix(y_train_test, rf_preds_sub))
print(classification_report(y_train_test, rf_preds_sub))

# fit on test sample
rf.fit(X_train, y_train)
rf_preds_out = rf.predict(X_test).astype(int)

# Save to CSV file
pd.DataFrame(rf_preds_out, 
             columns=['Survived'],
             index=passenger_list).to_csv('random_forest.csv')

# %% K NEAREST NEIGHBOURS (SCORE: 0.76076)
from sklearn.neighbors import KNeighborsClassifier

# CHOOSING K
error_rate = []

# Will take some time
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_train,y_train_train)
    pred_i = knn.predict(X_train_test)
    error_rate.append(np.mean(pred_i != y_train_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

# RUNNING THE MODEL
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train_train, y_train_train)
knn_pred = knn.predict(X_train_test)
print(confusion_matrix(y_train_test, knn_pred))
print(classification_report(y_train_test, knn_pred))

knn.fit(X_train, y_train)
knn_pred_out = knn.predict(X_test).astype(int)

# Save to CSV file
pd.DataFrame(knn_pred_out, 
             columns=['Survived'],
             index=passenger_list).to_csv('knn.csv')

# %% SUPPORT VECTOR MACHINE (SCORE: 0.76794)
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train_train, y_train_train)
svc_pred = svc.predict(X_train_test)
print(confusion_matrix(y_train_test, svc_pred))
print(classification_report(y_train_test, svc_pred))

svc.fit(X_train, y_train)
svc_pred_out = svc.predict(X_test).astype(int)

# Save to CSV file
pd.DataFrame(svc_pred_out, 
             columns=['Survived'],
             index=passenger_list).to_csv('svm.csv')

# %% MISCELLANEOUS WORKINGS

children = train[train['Child']==1]
train.groupby(['female', 'Pclass', 'cheap']).mean()
poor_women = train[(train['female']==1) & (train['Pclass']==3)]

children.groupby(['female', 'cheap']).mean()
train.groupby(['female', 'cheap']).mean()
