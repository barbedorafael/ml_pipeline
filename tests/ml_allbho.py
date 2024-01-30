# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 13:30:33 2022

@author: Rafael
"""

import numpy as np
import pandas as pd

from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn import preprocessing

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn.svm import SVR

from sklearn.metrics import r2_score
from sklearn.metrics import matthews_corrcoef as cc
from scipy.stats import pearsonr
import time

df_ml = pd.read_pickle('data/bho_attributes_agg.pkl')
df_ml.iloc[:, :30] = df_ml.iloc[:, :30].fillna(0)
df_ml.reset_index(drop=True, inplace=True)


targets = ['qm', 'q95'] # q95 or qm

df_results = df_ml[['cotrecho', 'cobacia']]
for target in targets:
    df_train = df_ml.dropna(subset=target)
    df_train = df_train.drop(['lat', 'lon', 'cotrecho', 'cobacia', 'cocursodag',
                              'nutrjus', 'nustrahler', 'dedominial', 'sub', 'posto',
                              'qm_inc', 'q95_inc', 'flag', 'A'], axis=1)
    df_test = df_ml.drop(['lat', 'lon', 'cotrecho', 'cobacia', 'cocursodag',
                            'nutrjus', 'nustrahler', 'dedominial', 'sub', 'posto',
                            'qm_inc', 'q95_inc', 'flag', 'A'], axis=1)    
    
    X_train = df_train.iloc[:, :-2].values
    y_train = (df_train[target].div(df_train['nuareamont'])*1000).values
    
    X_test = df_test.iloc[:, :-2].values
    
    X_train = preprocessing.MinMaxScaler().fit_transform(X_train)
    X_test = preprocessing.MinMaxScaler().fit_transform(X_test)
    
    # Feature selection
    estimator = RandomForestRegressor(n_estimators = 100)
    selector = RFE(estimator, n_features_to_select=12, step=1)
    # selector = SelectFromModel(estimator, threshold='median')
    selector = selector.fit(X_train, y_train)
    # selector.support_
    selector.get_support()
    # selector.ranking_
    feature_list = df_train.iloc[:, :-2].columns.to_numpy()
    feature_list[np.where(selector.get_support())[0]]
    X_train = selector.transform(X_train)    
    X_test = selector.transform(X_test)   
    
    model = KNeighborsRegressor(weights='distance', p=1)
    model.fit(X_train, y_train)
    result = model.predict(X_test)
    
    # area = df_test['nuareamont'].values
    # result = result * area / 1000
    
    df_results[target + '_esp_knn'] = result
    
    
df_results[['qm_knn', 'q95_knn']] = (df_results[['qm_esp_knn', 'q95_esp_knn']].multiply(df_ml['nuareamont'], axis=0) / 1000).astype(float)


# =============================================================================
# Save results
# =============================================================================
import geopandas as gpd

bho_att = gpd.read_file(r'D:\OneDrive\PRH\Brasil_data\BHO_database\BHO_2017_v_01_05_5k\geoft_bho_2017_5k_trecho_drenagem.gpkg')
bho_att = bho_att.dropna(subset=['cocursodag'])
bho_att.reset_index(drop=True, inplace=True)

df_new = df_ml.iloc[:, :-2]
df_new = pd.merge(df_new, df_results, on=['cotrecho', 'cobacia'])
gdf = gpd.GeoDataFrame(df_new, geometry=bho_att.geometry)

gdf.to_file('GIS/bho_database_knn.gpkg', driver='GPKG')
