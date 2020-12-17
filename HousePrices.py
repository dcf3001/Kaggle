# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 13:44:53 2020

@author: dcf30
"""
# %% HOUSE PRICES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sklearn.metrics as metrics
import math

os.chdir('C:\\Users\\David\\Onedrive\\Python\\House Prices')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
df = pd.concat([train, test])
df['test'] = df['SalePrice'].apply(pd.isna)
df.index = range(2919)

# %% FEATURE ENGINEERING
df['HouseStyle'].value_counts()
df['MSSubClass'].value_counts()

# Map stories from HouseStyle, assume split levels have 2 stories
stories_map = {'1Story':1, '1.5Fin':1.5, '1.5Unf':1.5, '2Story':2, 
               'SLvl':2, 'SFoyer':2, '2.5Fin':2.5, '2.5Unf':2.5}

# Get features for house type from HouseStyle, MSSubClass, BldgType
df['Stories'] = df['HouseStyle'].map(stories_map)
df['Finished'] = df['HouseStyle'].apply(lambda x: x[-3:] != 'Unf')
df['Split'] = df['HouseStyle'].apply(lambda x: x[0] == 'S')
df['Duplex'] = df['BldgType'] == 'Duplex'
df['PUD'] = df['MSSubClass'].apply(lambda x: x in [120, 150, 160, 180])
df['TwoFamilyConv'] = df['BldgType'] == '2fmCon'
df['Detached'] = df['BldgType'] == '1Fam'
df['TownhouseEnd'] = df['BldgType'] == 'TwnhsE'
df['TownhouseInside'] = df['BldgType'] == 'Twnhs'
df = df.drop(['BldgType', 'HouseStyle'], axis=1)

# Get zoning dummies from MSZoning
density_map = {'A':np.nan, 'C (all)':np.nan, 'FV':np.nan, 'I': np.nan,
               'RH':3, 'RM':2, 'RL':1, 'RP':1}

df['Residential'] = df['MSZoning'].apply(lambda x: x in ['FV', 'RH', 'RL', 'RP', 'RM'])
df['ResDensity'] = df['MSZoning'].map(density_map)
df['Floating'] = df['MSZoning'] == 'FV'
df = df.drop('MSZoning', axis=1)

# Get access dummies from Street, Alley, PavedDrive
alley_map = {'Grvl':0, 'Pave':1, 'NA':np.nan}
df['PavedStreet'] = df['Street'] == 'Pave' # either paved or gravel
df['PavedAlley'] = df['Alley'].map(alley_map)
df['PavedDrive'] = df['PavedDrive'].map({'Y':1, 'P':0.5, 'N':0})
df = df.drop(['Street', 'Alley'], axis=1)
# Get lot configuration dummies from LotConfig
df = df.join(pd.get_dummies(df['LotConfig'], prefix='LC')).drop('LotConfig', axis=1)

# Get lot irregularity scale from LotShape
lot_shape_map = {'Reg':0, 'IR1':1, 'IR2':2, 'IR3':3}
df['Irregularity'] = df['LotShape'].map(lot_shape_map)
df = df.drop('LotShape', axis=1)

# Get flatness dummies from LandContour
df = df.join(pd.get_dummies(df['LandContour'], prefix='Flat')).drop('LandContour', axis=1)
df['LandSlope'] = df['LandSlope'].map({'Gtl':1, 'Mod':2, 'Sev':3})

# Note: Utilities are all 'AllPub' except for one NoSeWa
df = df.drop('Utilities', axis=1)

# Get neighborhood dummies from Neighborhood
df = df.join(pd.get_dummies(df['Neighborhood'], prefix='NB')).drop('Neighborhood', axis=1)

# Get condition dummies from Condition1/2
df = df.join(pd.get_dummies(df['Condition1'], prefix='C1'))
df = df.join(pd.get_dummies(df['Condition2'], prefix='C2'))
only_conditions = ['RRAe', 'RRNe'] # no houses have these as second conditions
second_conditions = ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA']
for c in second_conditions:
    df['Cond_' + c] = df['C1_' + c] | df['C2_' + c]
    df = df.drop('C1_' + c, axis=1)
    df = df.drop('C2_' + c, axis=1)
for c in only_conditions:
    df['Cond_' + c] = df['C1_' + c]
    df = df.drop('C1_' + c, axis=1)
df = df.drop(['Condition1', 'Condition2'], axis=1)

# Get roof dummies from RoofStyle and RoofMatl
df = df.join(pd.get_dummies(df['RoofStyle'], prefix='Roof')).drop('RoofStyle', axis=1)
df = df.join(pd.get_dummies(df['RoofMatl'], prefix='RoofMat')).drop('RoofMatl', axis=1)

# Get exterior material dummies from Exterior1st/2nd
df = df.join(pd.get_dummies(df['Exterior1st'], prefix='E1'))
df = df.join(pd.get_dummies(df['Exterior2nd'], prefix='E2'))
ext1_values = set(df['Exterior1st'].unique()) - {np.nan}
ext2_values = set(df['Exterior2nd'].unique()) - {np.nan}
ext = list(ext1_values & ext2_values)
ext1 = list(ext1_values - ext2_values)
ext2 = list(ext2_values - ext1_values)
for v in ext:
    df['Ext_' + v] = df['E1_' + v] | df['E2_' + v]
    df = df.drop('E1_' + v, axis=1)
    df = df.drop('E2_' + v, axis=1)
for v in ext1:
    df['Ext_' + v] = df['E1_' + v] 
    df = df.drop('E1_' + v, axis=1)
for v in ext2:
    df['Ext_' + v] = df['E2_' + v]
    df = df.drop('E2_' + v, axis=1)
df = df.drop(['Exterior1st', 'Exterior2nd'], axis=1)    
# Get mason veneer dummies from MasVnrType
df = df.join(pd.get_dummies(df['MasVnrType'], prefix='Mason')).drop('MasVnrType', axis=1)

# Get external quality and condition scale from ExterQual and ExterCond
df['ExterQual'] = df['ExterQual'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1})
df['ExterCond'] = df['ExterCond'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1})

# Get foundation dummies from Foundation
df = df.join(pd.get_dummies(df['Foundation'], prefix='F')).drop('Foundation', axis=1)

# Get basement scale and dummies from BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1/2
df['Basement'] = pd.notna(df['BsmtQual'])
df['BsmtQual'] = df['BsmtQual'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1})
df['BsmtCond'] = df['BsmtCond'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1})
df['BsmtExposure'] = df['BsmtExposure'].map({'Gd':3, 'Av':2, 'Mn':1, 'No':0})

df['BsmtLiving1'] = df['BsmtFinType1'].apply(lambda x: x in ['GLQ', 'ALQ', 'BLQ'])
df['BsmtLiving2'] = df['BsmtFinType2'].apply(lambda x: x in ['GLQ', 'ALQ', 'BLQ'])
df['BsmtLiving'] = df['BsmtLiving1'] | df['BsmtLiving2']
df['BsmtLowQ'] = (df['BsmtFinType1'] == 'LwQ') | (df['BsmtFinType2'] == 'LwQ')
df['BsmtUnf'] = df['BsmtFinType1'] == 'Unf'
df['BsmtFinRatio'] = 1 - df.BsmtUnfSF / df.TotalBsmtSF
df = df.drop(['BsmtLiving1', 'BsmtLiving2', 'BsmtFinType1', 'BsmtFinType2'], axis=1)

# Get heating dummies and scale from Heating, HeatingQC, CentralAir
df = df.join(pd.get_dummies(df['Heating'], prefix='H')).drop('Heating', axis=1)
df['HeatinqQC2'] = df['HeatingQC'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1})
df = df.drop('HeatingQC', axis=1)
df['CentralAir'] = df['CentralAir'].map({'Y':1, 'N':0})

# Get electrical dummies from Electrical
df['FuseBox'] = df['Electrical'].map({'SBrkr':0, 'Mix':0.5, 'FuseA':1, 'FuseF':1, 'FuseP':1})
df['FuseBoxQual'] = df['Electrical'].map({'SBrkr':np.nan, 'Mix':np.nan, 'FuseA':3, 'FuseF':2, 'FuseP':1})
df = df.drop('Electrical', axis=1)

# Get quality ratio
df['QualRatio'] = 1 - df.LowQualFinSF / df.GrLivArea

# Get kitchen quality scale from KitchenQual
df['KitchenQual'] = df['KitchenQual'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1})

# Get functionality scale from Functional
df['FullyFunctional'] = df['Functional'] == 'Typ'
df['ProblemScale'] = df['Functional'].map({'Min1':1, 'Min2':2, 'Mod':3, 'Maj1':4, 
                                           'Maj2':5, 'Sev':6, 'Sal':7, 'Typ':np.nan})
df = df.drop('Functional', axis=1)
# Get fireplace quality scale from FireplaceQu
df['FireplaceQu'] = df['FireplaceQu'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1})

# Get garage scales and dummies
df['Garage'] = pd.notna(df['GarageType'])
df = df.join(pd.get_dummies(df['GarageType'], prefix='G')).drop('GarageType', axis=1)
df['GarageFinish'] = df['GarageFinish'].map({'Fin':1, 'RFn':0.5, 'Unf':0})
df['GarageQual'] = df['GarageQual'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1})
df['GarageCond'] = df['GarageCond'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1})

# Get pool dummy and scale from PoolQC
df['Pool'] = pd.notna(df['PoolQC'])
df['PoolQC'] = df['PoolQC'].map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1})

# Get fence dummies from Fence
df['HasFence'] = pd.notna(df['Fence'])
df['FencePrivacy'] = df['Fence'].apply(lambda x: x in ['GdPrv', 'MnPrv'])
df['FenceGood'] = df['Fence'].apply(lambda x: x in ['GdPrv', 'GdWo']) 
df = df.drop('Fence', axis=1)

# Get misc dummies from MiscFeature
df = df.join(pd.get_dummies(df['MiscFeature'], prefix='M')).drop('MiscFeature', axis=1)

# Get sale dummies from SaleType, SaleCondition
df['SaleType'] = df['SaleType'].fillna('WD')
df['SaleWD'] = df['SaleType'].apply(lambda x: x[-2:]=='WD')
df['Contract'] = df['SaleType'].apply(lambda x: x[:3]=='Con')
df['ConInterest'] = df['SaleType'].map({'WD':np.nan, 'CWD':np.nan, 'VWD':np.nan,
                                        'New':np.nan, 'COD':np.nan, 'Con':1,
                                        'ConLw':0, 'ConLI':0, 'ConLD':1})
df['ConDownPmt'] = df['SaleType'].map({'WD':np.nan, 'CWD':np.nan, 'VWD':np.nan,
                                        'New':np.nan, 'COD':np.nan, 'Con':1,
                                        'ConLw':0, 'ConLI':1, 'ConLD':0})
df['SaleNew'] = df['SaleType'] == 'New'
df['SaleCOD'] = df['SaleType'] == 'COD'
df = df.drop('SaleType', axis=1)

df = df.join(pd.get_dummies(df['SaleCondition'], prefix='Sale')).drop('SaleCondition', axis=1)

# %% REPLACAE MISSING VALUES WITH 0
df = df.drop('SalePrice', axis=1).fillna(0).join(df['SalePrice'])

# %% REDEFINE TRAIN AND TEST SETS
train = df[~df.test]
test = df[df.test]
X_test = test.drop('SalePrice', axis=1)
# %% TRAIN TEST SPLIT FOR HOLD
from sklearn.model_selection import train_test_split
X = train.drop('SalePrice', axis=1)
y = train.SalePrice
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3) 

# %% SCALING
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index,
                       columns=X_train.columns)
X_val = pd.DataFrame(scaler.transform(X_val), index=X_val.index,
                      columns=X_val.columns)

X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index,
                       columns=X_test.columns)

# %% DEFINE RESULTS FUNCTION

def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))

# %% SMART REGRESSION
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, classification_report
import statsmodels.api as sm
import statsmodels.formula.api as smf

corr = pd.DataFrame(train.corr()['SalePrice'].sort_values())

corr_threshold = 0.25
# regressors = corr[abs(corr['SalePrice']) > corr_threshold].index.tolist()[:-1]
regressors = ['Mason_None', 'ResDensity', 'GarageYrBlt', 'GarageQual', 'HalfBath', 'WoodDeckSF', 'NB_NoRidge', 'SaleNew',
              'BsmtExposure', 'BsmtFinSF1', 'NB_NridgHt', 'MasVnrArea', 'YearRemodAdd', 'YearBuilt', 'TotRmsAbvGrd',
              'BsmtQual', '1stFlrSF', 'TotalBsmtSF', 'GarageCars', 'KitchenQual', 'ExterQual', 'OverallQual']

X = X_train[regressors]
Xt = X_val[regressors]
y = y_train
log_reg = sm.OLS(y, X.astype(float)).fit()
print(log_reg.summary())

linear = LinearRegression()
linear.fit(X, y)
linear.score(X, y)
predictions_val = linear.predict(Xt)
regression_results(y_val, predictions_val)
predictions_out = linear.predict(X_test[regressors])
pd.DataFrame(predictions_out, index=X_test.index + 1, columns=['SalePrice']).to_csv('linear regression.csv')

# %% LASSO REGRESSION
from sklearn.linear_model import Lasso
# Add variable interactions
# for var in X_train:
#    X_train['Sale_Normal_' + var] = X_train['Sale_Normal'] * X_train[var] 
#    X_val['Sale_Normal_' + var] = X_val['Sale_Normal'] * X_val[var]
#    X_test['Sale_Normal' + var] = X_test['Sale_Normal'] * X_test[var]
    
lasso = Lasso(alpha=40, max_iter=10000)
lasso.fit(X_train, y_train)
lasso_predictions = lasso.predict(X_val)
lasso_predictions = list(map(lambda x: max(x, 50000), lasso_predictions))
lasso.coef_
regression_results(y_val, lasso_predictions)
predictions_out = lasso.predict(X_test)
pd.DataFrame(predictions_out, index=X_test.index + 1, columns=['SalePrice']).to_csv('lasso regression with Sale_Normal2.csv')


# %% NEURAL NETWORK

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.constraints import max_norm
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
model = Sequential()

# input layer
model.add(Dense(198, activation='relu'))
model.add(Dropout(0.4))

# hidden layer
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.4))

# hidden layer
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.4))

# hidden layer
model.add(Dense(5, activation='relu'))
model.add(Dropout(0.4))

# output layer
model.add(Dense(units=1,activation='relu'))

# Compile model
model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')

log_directory = 'logs\\fit2'
board = TensorBoard(log_dir=log_directory,histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch',
    profile_batch=2,
    embeddings_freq=0)

model.fit(x=np.array(X_train), 
          y=np.array(y_train), 
          epochs=250,
          batch_size=64,
          validation_data=(X_val, y_val), 
          )

losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()

predictions_out = model.predict(X_test)
pd.DataFrame(predictions_out, 
             columns=['SalePrice'], 
             index=X_test.index).to_csv('neural network.csv')
# %% FEATURE ENGINEERING
""" 
MSSubClass: Number of stories, age, style dummies
MSZoning: type & density dummies
LotFrontage / Lot Area: ---
Street / Alley: get dummies
LotShape: 0-3 scale
LandContour: get dummies
Utilities: Electricity, Water, Gas, Sewer dummies
LotConfig: get dummies
LandSlope: 0-2 scale
Neighborhood: get dummies
Condition 1/2: get dummies
BldgType: get dummies
HouseStyle: Number of stories, finished/unfinished dummies
OverallQual: ---
OverallCond: ---
YearBuilt: ---
YearRemodAdd: ---
RoofStyle: get dummies
RoofMatl: get dummies
Exterior1st/2nd: get dummies
MasVnrType: get dummies
MasVnrArea: ---
ExterQual: 1-5 scale
ExterCond: 1-5 scale
Foundation: get dummies
BsmtQual: get dummies, separate one for basement (1), unfinished (ratio) or none
BsmtExposure: 1-4 scale
BsmtFinType1/2: 1-5 scale
BsmtUnfSF: use for finished ratio
TotalBsmtSF: ---
Heating: get dummies
HeatingQC: 1-5 scale
CentralAir: 0-1 dummy
Electrical: get dummies
1stFlrSF: ---
2ndFLrSF: ---
LowQualFinSF: use for ratio
GrLivArea: ---
BsmtFullBath: ---
BsmtHalfBath: ---
FullBath / HalfBath: ---
Bedroom: ---
Kitchen: ---
KitchenQual: 1-5 scale
TotRmsAbvGrd: ---
Functional: 0-1 dummy plus 1-7 scale
Fireplaces: ---
FireplaceQu: 1-5 scale
GarageType: get dummies
GarageYrBlt: ---
GarageFinish: 0-3 scale
GarageCars: ---
GarageArea: ---
GarageQual: 1-5 scale
GarageCond: 1-5 scale
PavedDrive: 0-2 scale
WoodDeckSF-PoolArea: ---
PoolQC: 1-4 scale
MiscFeature: get dummies
MoSold/YrSold: ---
SaleType: Type / down payment / interest dummies
SaleCondition: get dummies
"""


