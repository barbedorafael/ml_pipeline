# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 21:01:33 2021

@author: Rafael
"""
# =============================================================================
# ML prediction
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import randint, loguniform

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
# from yellowbrick.features import Rank2D

# from eli5.sklearn import PermutationImportance
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.model_selection import train_test_split, LeaveOneOut, GridSearchCV, RandomizedSearchCV, KFold
from sklearn import preprocessing

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, IsolationForest, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn.svm import SVR

from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.metrics import matthews_corrcoef as cc
from scipy.stats import pearsonr
# from pyearth import Earth
# import graphviz
# from sklearn.tree import export_graphviz
import time

# =============================================================================
# Initial parameters
# =============================================================================
mlmodel = 'SVM' # MLR, DT, KNN, SVM, GBM, RF
target = 'q95' # q95 or qm

dft = pd.read_parquet('data/data4ml.parquet')
dft = dft.sample(frac = 1) # Shuffle values MAKES ALL THE DIFFERENCE IDKW

# X = dft.iloc[:, :-2].values
# X = df_ml.iloc[:, [2]].values # Just precipitation


# X = preprocessing.MinMaxScaler().fit_transform(X)

# enc = preprocessing.OneHotEncoder()
# enc.fit(X)
# enc.categories_
# x1=enc.transform(X).toarray()

# poly = PolynomialFeatures(degree=2)
# X = poly.fit_transform(X)

# X = np.log(X)
# y = np.log(y)

# =============================================================================
# ### Feature selection ###
# =============================================================================

# datasum = dft.describe()
# datasum.loc['nan_count', :] = dft[dft==0].count()

# # visualizer = Rank2D(algorithm="pearson")
# # visualizer.fit_transform(dft)
# # visualizer.show()

# df_corr = dft.corr(method='spearman')
# # df_corr = df_corr.mask(np.tril(np.ones(df_corr.shape)).astype(np.bool)) # for plotting use where instead of mask
# df_corr_select = df_corr[abs(df_corr)>0.7]


# # Analyse every feature, delete highly correlated (the one with worse target corr)
# th = 0.75 # define the correlation threshold
# df_corr_sorted = abs(df_corr).sort_values(by=target, ascending=True)
# df_corr_sorted.drop(['qm', 'q95'], axis=1, inplace=True)
# df_corr_sorted.drop(['qm', 'q95'], axis=0, inplace=True)
# df_corr_sorted[df_corr_sorted==1] = 0
# for i, feat in df_corr_sorted.iterrows():
#     if any(feat>th) or all(feat.isna()):
#         # print(df_corr_sorted[i].mean())
#         df_corr_sorted.drop(i, axis=1, inplace=True)
#         df_corr_sorted.drop(i, axis=0, inplace=True)
#         print('Removed variable: ' + i)

# df_new = dft[df_corr_sorted.columns]


# sns.heatmap(df_corr, cmap='Spectral')
# First, drop the features with abs corr > 0.75
# dft = dft.drop([
#                                 'slp_avg', 'slp_std', # correlate a lot with sts and hand
#                                 'hnd_avg', 'hnd_std', # correlate a with sts
#                                 'tc_extreme', # have to remove one class for each category
#                                 'min_T', 'max_T', # they all correlate with each other
#                                 'min_ET', 'max_ET', # they all correlate with each other
#                                 'min_PET', 'max_PET', # they all correlate with each other
#                                 ], axis=1).reset_index()



# Calculate vif of the remaining variables
# Xv = add_constant(df_new)
# vif=pd.Series([variance_inflation_factor(Xv.values, i)
#                 for i in range(Xv.shape[1])], 
#               index=Xv.columns)


# The VIF requires (idk why) to add constant, this function is not working properly
# !!! Above issue resolved by using StandardScaler on data

# def calculate_vif_(dfi, thresh=5.0):
#     variables = list(range(dfi.shape[1]))
#     dropped = True
#     while dropped:
#         dropped = False
#         vif = [variance_inflation_factor(dfi.iloc[:, variables].values, ix)
#                 for ix in range(dfi.iloc[:, variables].shape[1])]
#         maxloc = vif.index(max(vif))
#         if max(vif) > thresh:
#             print('dropping \'' + dfi.iloc[:, variables].columns[maxloc] +
#                   '\' at index: ' + str(maxloc) + '| with value: ' + str(max(vif))
#                   )
#             del variables[maxloc]
#             dropped = True

#     print('Remaining variables:')
#     print(dfi.columns[variables])
#     return dfi.iloc[:, variables]

# Xv = preprocessing.StandardScaler().fit_transform(df_new.iloc[:, :-2].values) # FOR THE VIF
# Xv = pd.DataFrame(Xv, columns = df_new.iloc[:, :-2].columns)
# # Xv = add_constant(Xv)
# df_new = calculate_vif_(Xv, 10)

# df_newcorr = df_new.corr()
# df_newcorr = df_newcorr[abs(df_newcorr)>0.7]

### Identify outliers in the dataset ###
# old = LocalOutlierFactor()
# old = IsolationForest()
# yhat = old.fit_predict(X)
# mask = yhat != -1
# X, y = X[mask,:], y[mask]

# =============================================================================
# By hierarchical clustering
# =============================================================================
# https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py

from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from collections import defaultdict

dfcorr = dft.corr(method='spearman') # Change to pearson or spearman
# corr = abs(corr).sort_values(by=target, ascending=False, axis=0)
# corr = abs(corr).sort_values(by=target, ascending=False, axis=1)
corr = dfcorr.copy()
corr = corr.drop(['qm', 'q95'], axis=1)
corr = corr.drop(['qm', 'q95'], axis=0)
corr = corr.values
corr = np.nan_to_num(corr)

# Ensure the correlation matrix is symmetric
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)

# We convert the correlation matrix to a distance matrix before performing
# hierarchical clustering using Ward's linkage.
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))

# Plot the correlation matrix and the dendogram
# fig, ax = plt.subplots(1, 1, figsize=(6,6), dpi=300)
# dendro = hierarchy.dendrogram(
#     dist_linkage, labels=dft.iloc[:,:-2].columns.tolist(), ax=ax, leaf_rotation=90
# )
# dendro_idx = np.arange(0, len(dendro["ivl"]))
# fig.tight_layout()
# plt.show()

# fig, ax = plt.subplots(1, 1, figsize=(9,9), dpi=300)
# ax.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
# ax.set_xticks(dendro_idx)
# ax.set_yticks(dendro_idx)
# ax.set_xticklabels(dendro["ivl"], rotation="vertical")
# ax.set_yticklabels(dendro["ivl"])
# fig.tight_layout()
# plt.show()

# fig, ax = plt.subplots(1, 1, figsize=(15,12), dpi=300)
# sns.heatmap(abs(dfcorr), ax=ax, cmap='flare')
# fig.tight_layout()
# plt.show()


# Criteria: increment 0.01 until no corr > 0.7
cluster_ids = hierarchy.fcluster(dist_linkage, 0.4, criterion="distance")
cluster_id_to_feature_ids = defaultdict(list)
cluster_feature = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)
    cluster_feature['Cluster '+str(cluster_id)].append(dft.columns[idx])

# Select the feature with the greatest correlation between clusters
selected_features = []
for v in cluster_id_to_feature_ids.values():
    targetcorr = dfcorr[target].iloc[v].max()
    bestfeat = dfcorr.reset_index().index[dfcorr[target]==targetcorr].values[0]
    selected_features.append(bestfeat)

df_new = dft.iloc[:, selected_features]
newcorr = df_new.corr(method='spearman')

# cluster_feature = pd.DataFrame(cluster_feature)
fdf = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in cluster_feature.items() ]))
fdf = fdf.T
rep_ft = dft.columns[selected_features]

# Test the correlations of the left features (stop when no features left with corr > 0.7)
abs(newcorr[newcorr<1]).max().sort_values()

# =============================================================================
# 
# =============================================================================
X = df_new.values
X = preprocessing.MinMaxScaler().fit_transform(X)
y = dft[target].values

# =============================================================================
# Hyperparameter tuning
# =============================================================================

# Change parameters to distributions
if mlmodel=='MLR': # Maybe using SGD instead
    model = LinearRegression()
    grid = dict()
elif mlmodel=='DT':
    model = DecisionTreeRegressor()
    grid = dict(max_depth = randint(2, 20),
                min_samples_leaf = randint(5, 100))
elif mlmodel=='KNN':
    model = KNeighborsRegressor()
    grid = dict(leaf_size = randint(1, 50), 
                n_neighbors = randint(1, 30), 
                p = randint(1, 5))
elif mlmodel=='SVM':
    model = SVR()
    grid = dict(gamma = loguniform(1e-4, 1), 
                C = loguniform(0.1, 100))
elif mlmodel=='GBM':
    model = GradientBoostingRegressor()
    grid = dict(n_estimators = randint(1, 500),
                max_leaf_nodes = randint(2, 100),
                learning_rate = loguniform(0.01, 1))
elif mlmodel=='RF':
    model = RandomForestRegressor()
    grid = dict(n_estimators = randint(1, 500),
                max_leaf_nodes = randint(2, 100))
else:
    raise ValueError('PROVIDE VALID ML MODEL!')

# =============================================================================
# Normal split because the 10-Fold IS NOT FUCKING WORKING
# =============================================================================

# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     shuffle=True,
#                                                     test_size=0.1,
#                                                     random_state=None)

# randomSearch = RandomizedSearchCV(estimator=model, n_jobs=-1,
#     n_iter=10,
#  	cv=10, param_distributions=grid,
#  	scoring="r2"
#     # scoring="neg_mean_squared_error"
#     )
# searchResults = randomSearch.fit(X_train, y_train)

# print('Best score: ')
# print(randomSearch.best_score_)
# print('Best params: ')
# print(randomSearch.best_params_)
# print('\n')

# model_opt = searchResults.best_estimator_

# =============================================================================
# Feature importance
# =============================================================================

# model_opt.fit(X, y)
# r = permutation_importance(model_opt, X, y,
#                             n_repeats=30,
#                             random_state=0,
#                             scoring='r2'
#                             #scoring="neg_mean_squared_error"
#                             )

# for i in r.importances_mean.argsort()[::-1]:
#     # if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
#     print(f"{df_new.columns[i]:<16}"
#           f"{r.importances_mean[i]:.3f}"
#           f" +/- {r.importances_std[i]:.3f}")

# # With CV 10 folds
# pi = PermutationImportance(model_opt, 
#                           cv=5,
#                           # scoring='r2'
#                           scoring="neg_mean_squared_error"
#                           )

# r = pi.fit(X,y)
# for i in r.feature_importances_.argsort()[::-1]:
#     if r.feature_importances_[i] - 1 * r.feature_importances_std_[i] > 0:
#         print(f"{df_new.columns[i]:<16}"
#               f"{r.feature_importances_[i]:.3f}"
#               f" +/- {r.feature_importances_std_[i]:.3f}")


# sel = SelectFromModel(
#     pi,
#     threshold=0.05, # 0.05 for RMSE | 0.01 for R2
#     ).fit(X, y)
# feature_list = df_new.columns.to_numpy()
# feature_list[np.where(sel.get_support())[0]]
# X = sel.transform(X)

# RFE with permutation importance: https://stackoverflow.com/questions/62537457/right-way-to-use-rfecv-and-permutation-importance-sklearn

# =============================================================================
# Performance evaluation on test data that have never seen the model
# =============================================================================

# I don't know if it's right to fit the optimal model with all the training data
# or if I should fit several models from the k-fold to test on the test data
# model_opt.fit(X, y)

# result = model_opt.predict(X_test)


# # Permutation importance
# r = permutation_importance(model_opt, X_test, y_test,
#                             n_repeats=10,
#                             random_state=0,
#                             scoring='r2'
#                             # scoring="neg_mean_squared_error"
#                             )

# # for i in r.importances_mean.argsort()[::-1]:
# #     # if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
# #     print(f"{df_new.columns[i]:<16}"
# #           f"{r.importances_mean[i]:.3f}"
# #           f" +/- {r.importances_std[i]:.3f}")

# perm_sorted_idx = r.importances_mean.argsort()

# fig, ax = plt.subplots(1, 1, figsize=(4, 5), dpi=300)
# ax.boxplot(
#     r.importances[perm_sorted_idx][-12:].T,
#     vert=False,
#     labels=df_new.columns[perm_sorted_idx][-12:],
# )
# ax.set_title('{m}, {t}'.format(m=mlmodel, t=target))
# fig.tight_layout()
# plt.show()

# =============================================================================
# ### 10-Fold Model Running with best params ###
# =============================================================================
# from mapie.regression import MapieRegressor
# from mapie.metrics import regression_coverage_score
# from mapie.conformity_scores import GammaConformityScore

print('\n Running '+mlmodel+' for '+target + '\n')
start_time = time.time()

n_splits = 10
splits=[]
cv = KFold(n_splits=n_splits) # , shuffle=True)
result = np.zeros(y.size)
y_pis = np.zeros((y.size, 2, 1))
imps = pd.DataFrame(columns=df_new.columns, index=range(n_splits*10))
i=0
for train_index, test_index in cv.split(X, y):
    splits.append([train_index, test_index])
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    randomSearch = RandomizedSearchCV(estimator=model, n_jobs=-1,
        n_iter=10,
    	cv=10, param_distributions=grid,
    	scoring="r2"
        # scoring="neg_mean_squared_error"
        )
    searchResults = randomSearch.fit(X_train, y_train)
    
    print('Best score: ')
    print(randomSearch.best_score_)
    print('Best params: ')
    print(randomSearch.best_params_)
    print('\n')

    model_opt = searchResults.best_estimator_
    
    model_opt.fit(X_train, y_train)
    yhat = model_opt.predict(X_test)
    
    # Mapie for uncertainty estimation (idk why i need another cv here)
    # mapie = MapieRegressor(model_opt, method='base', cv=5)#, conformity_score=GammaConformityScore())
    # mapie.fit(X_train, y_train)
    # yhat, yrange = mapie.predict(X_test, alpha=[0.05])
    
    # # Allocate results in overall matrix
    result[test_index] = yhat
    # y_pis[test_index] = yrange
    
    processing = i+100/n_splits
    print('Processing: {:.0f}%'.format(processing))
    
    # Permutation importance
    r = permutation_importance(model_opt, X_test, y_test,
                                n_repeats=10,
                                random_state=0,
                                scoring='r2'
                                # scoring="neg_mean_squared_error"
                                )
    
    imps.iloc[i:i+10,:] = r.importances.T
    i+=10


end_time = time.time()
print('\n'+mlmodel+' | Time to process: ' + str((end_time - start_time)/60) + ' min \n')

# Save results to csv
# dfe = df_ml.loc[:, ['cobacia', 'code', 'A']]

# error = y - result
# dfe[target+'_obs'] = y
# dfe[target+'_pred'] = result
# dfe[target+'_error'] = error
# dfe[target+'_low'] = y_pis[:, 0, 0]
# dfe[target+'_high'] = y_pis[:, 1, 0]

# dfe.to_csv('data/results_'+target+'_'+mlmodel+'.csv', index=False)
# imps.to_csv('data/permimp_'+target+'_'+mlmodel+'.csv', index=False)


# =============================================================================
# Stats
# =============================================================================
# y[y==0] = 0.0001
# result[result==0] = 0.0001

# def stats(s1, s2):
#     rq75 = np.percentile(np.maximum(abs(s1/s2), abs(s2/s1)), 75)
#     r2 = 1 - ((s2-s1)**2).sum()/((s2-s2.mean())**2).sum() # == nash
#     rmse = ((s1 - s2) ** 2).mean() ** .5
#     bias = ((s1-s2).sum())/s2.sum() * 100
#     return {'rq75': rq75,
#             'r2': r2,
#             'rmse': rmse,
#             'bias': bias}

# # stats_train = stats(result_train, y_train)
# # stats_test = stats(result_test, y_test)
# stats_loo = stats(result, y)
# bias = stats_loo['bias'].round(2)
# rmse = stats_loo['rmse'].round(2)
# r2 = stats_loo['r2'].round(2)
# rq75 = stats_loo['rq75'].round(2)

# rq = np.maximum(abs(y/result), abs(result/y))

# # # Plot P x q
# # fig, ax = plt.subplots(dpi=300, figsize=(5,5))
# # ax.scatter(X[:,1], y)
# # # ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
# # ax.set_xlabel('Precipitation')
# # ax.set_ylabel('Discharge')
# # # ax.set_xlim([0, np.percentile(X[:,1], 99)])
# # # ax.set_ylim([0,np.percentile(y, 99)])
# # plt.show()


# # Plot observed x predicted
# fig, (ax1) = plt.subplots(1, 1, dpi=300, figsize=(5,5))
# ax1.scatter(y, result, s=1)
# ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
# ax1.set_xlabel('Obs ' + target)
# ax1.set_ylabel('Pred ' + target)
# ax1.set_title(mlmodel + ' \n '
#              + 'BIAS: ' + str(bias)
#              + ', RMSE: ' + str(rmse)
#              + ', R²: ' + str(r2)
#              # + ', RQ75: ' + str(rq75)
#              )
# cutlim = np.percentile(y, 95)
# ax1.set_xlim([-1, cutlim])
# ax1.set_ylim([-1, cutlim])
# ax1.set_aspect('equal', 'box')
# fig.tight_layout()
# plt.show()


# fig, (ax2) = plt.subplots(1, 1, dpi=300, figsize=(2,5))
# ax2.boxplot(rq, showfliers=False)#, manage_ticks=True)
# # ax2.set_xticklabels('RQ')
# # ax.boxplot(vel_arr, vert=False, manage_ticks=True)
# # ax.set_xlabel('value')
# # ax.set_yticks([1])
# # ax.set_yticklabels(['RQ'])

# quantiles = np.percentile(rq, np.array([10, 25, 50, 75, 90]))
# # ax2.hlines(quantiles, [0] * quantiles.size, [1] * quantiles.size,
# #           color='b', ls=':', lw=0.5, zorder=0)
# # ax2.set_ylim(0.5, 1.5)
# # ax2.set_yticks(quantiles)

# fig.tight_layout()
# plt.show()


# error model
# validation_error = (result - y) ** 2 
# error_model = KNeighborsRegressor(weights='distance', p=1)
# error_model.fit(X, validation_error) 
# compute the mean and standard deviation of the distribution 
# mean = base_model.predict(X_test)
# st_dev = error_model.predict(X)**0.5

# =============================================================================
# Plot importances
# =============================================================================
imps_mean = imps.median()
imps_std = imps.std()
imps_sortindex = imps_mean.argsort()[::-1]
for i in imps_sortindex:
    if imps_mean[i] - 1 * imps_std[i] > 0:
        print(f"{df_new.columns[i]:<16}"
              f"{imps_mean[i]:.3f}"
              f" +/- {imps_std[i]:.3f}")

fig, ax = plt.subplots(1, 1, figsize=(6, 8))
ax.boxplot(
    imps.iloc[:, imps_sortindex[::-1][-12:]],
    vert=False,
    showfliers=False,
    labels=imps.columns[imps_sortindex[::-1][-12:]],
)
ax.set_title('{m}, {t}'.format(m=mlmodel, t=target))
# ax.set_xlim([-0.25,0.75])
fig.tight_layout()
plt.show()
# fig.savefig('figures/permimp_'+target+'_'+mlmodel+'.png', dpi=300)

# =============================================================================
# Stats FOR SEPARATE SPLIT
# =============================================================================
y[y==0] = 0.0001
result[result==0] = 0.0001

def stats(s1, s2):
    rq75 = np.percentile(np.maximum(abs(s1/s2), abs(s2/s1)), 75)
    r2 = 1 - ((s2-s1)**2).sum()/((s2-s2.mean())**2).sum() # == nash
    rmse = ((s1 - s2) ** 2).mean() ** .5
    bias = ((s1-s2).sum())/s2.sum() * 100
    return {'rq75': rq75,
            'r2': r2,
            'rmse': rmse,
            'bias': bias}

# stats_train = stats(result_train, y_train)
# stats_test = stats(result_test, y_test)
stats_loo = stats(result, y)
bias = stats_loo['bias'].round(2)
rmse = stats_loo['rmse'].round(2)
r2 = stats_loo['r2'].round(2)
rq75 = stats_loo['rq75'].round(2)

rq = np.maximum(abs(y/result), abs(result/y))

# # Plot P x q
# fig, ax = plt.subplots(dpi=300, figsize=(5,5))
# ax.scatter(X[:,1], y)
# # ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
# ax.set_xlabel('Precipitation')
# ax.set_ylabel('Discharge')
# # ax.set_xlim([0, np.percentile(X[:,1], 99)])
# # ax.set_ylim([0,np.percentile(y, 99)])
# plt.show()


# Plot observed x predicted
fig, (ax1) = plt.subplots(1, 1, dpi=300, figsize=(5,5))
ax1.scatter(y, result, s=5, alpha=0.5)
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
ax1.set_xlabel('Obs ' + target + ' [l/s]')
ax1.set_ylabel('Pred ' + target + ' [l/s]')
ax1.set_title(mlmodel + ' \n '
             + 'BIAS: ' + str(bias)
             + ', RMSE: ' + str(rmse)
             + ', R²: ' + str(r2)
             # + ', RQ75: ' + str(rq75)
             )
cutlim = np.percentile(y, 99)
ax1.set_xlim([-1, cutlim])
ax1.set_ylim([-1, cutlim])
ax1.set_aspect('equal', 'box')
fig.tight_layout()
plt.show()
# fig.savefig('figures/result_'+target+'_'+mlmodel+'.png', dpi=300)

# =============================================================================
# Uncertainty
# =============================================================================
'''
cov = regression_coverage_score(
    y, y_pis[:, 0, 0], y_pis[:, 1, 0]
)

def get_yerr(y_pred, y_pis):
    return np.concatenate(
        [
            np.expand_dims(y_pred, 0) - y_pis[:, 0, 0].T,
            y_pis[:, 1, 0].T - np.expand_dims(y_pred, 0),
        ],
        axis=0,
    )


y_err = get_yerr(result, y_pis)
int_width = (
    y_pis[:, 1, 0] - y_pis[:, 0, 0]
)
y_err[y_err<0]=0.01


fig, axs = plt.subplots(1, 2, figsize=(10, 5))

class_name = "Residual Score"

axs[0].errorbar(
    y,
    result,
    yerr=y_err,
    alpha=0.5,
    linestyle="None",
)
axs[0].scatter(y, result, s=1, color="black")
axs[0].plot(
    [0, max(max(y), max(result))],
    [0, max(max(y), max(result))],
    "-r",
)
axs[0].set_xlabel('Obs ' + target + ' [l/s]')
axs[0].set_ylabel('Pred ' + target + ' [l/s]')
axs[0].grid()
# axs[0].set_title(f"{class_name} - coverage={cov:.0%}")
cutlim = np.percentile(y, 99)
axs[0].set_xlim([-1, cutlim])
axs[0].set_ylim([-1, cutlim])

xmin, xmax = axs[0].get_xlim() + np.array([0, 0])
ymin, ymax = axs[0].get_ylim() + np.array([0, 0])
axs[1].scatter(y, int_width, marker="+")
axs[1].set_xlabel('Obs ' + target + ' [l/s]')
axs[1].set_ylabel('Prediction interval width' + ' [l/s]')
axs[1].grid()
axs[1].set_xlim([xmin, xmax])
axs[1].set_ylim([ymin, ymax])

fig.suptitle(
    # f"Predicted values with the prediction intervals of level [0.05]"
    f"{mlmodel} {class_name} - coverage={cov:.0%}"
)
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()
# fig.savefig('figures/uncertainty_'+target+'_'+mlmodel+'.png', dpi=300)


# =============================================================================
# Plot errors
# =============================================================================

fig, (ax1) = plt.subplots(1, 1, dpi=300, figsize=(5,5))
ax1.scatter(dft.avg_P, abs(error)/y, s=5, alpha=0.5)
# ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
ax1.set_xlabel('Predictor')
ax1.set_ylabel('error [%]')
# cutlim = np.percentile(y, 99)
ax1.set_xlim([0, np.percentile(dft.avg_P, 99)])
ax1.set_ylim([0, 1])
# ax1.set_aspect('equal', 'box')
fig.tight_layout()
plt.show()
'''














'''
# # =============================================================================
# # Variable importances
# # =============================================================================
# # Correlation between variables
# # X1 = X
# # X1 = np.log(X.astype(float))
# # X1 = X.astype(float)**3

# # pearsonr(y, X1[:, 0].astype(int))[0] # sub
# # pearsonr(y, X1[:, 1].astype(int))[0] # Area
# # pearsonr(y, X1[:, 2].astype(int))[0] # Precipitation
# # pearsonr(y, X1[:, 3].astype(float))[0] # Drainage density
# # pearsonr(y, X1[:, 4].astype(float))[0] # Plateau
# # pearsonr(y, X1[:, 5].astype(float))[0] # Hillslope
# # pearsonr(y, X1[:, 6].astype(float))[0] # Wetland


# Plot importances
estimator.fit(X, y)
if mlmodel=='MLR':
    importances = estimator.coef_
else:
    importances = estimator.feature_importances_
feature_list = dft.iloc[:, :-2].columns.to_numpy()
feature_list = feature_list[np.where(selector.get_support())[0]]

# Set the style
# plt.style.use('default')
# list of x locations for plotting
x_values = list(range(len(importances)))

fig, ax = plt.subplots(1, 1, dpi=300, figsize=(5,5))
# Make a bar chart
ax.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importância'); plt.xlabel('Variável'); plt.title('KNN, RF e SVM')


# labelencoder = LabelEncoder()
# X[:, 0] = labelencoder.fit_transform(X[:, 0])
# X[:, 1] = labelencoder.fit_transform(X[:, 1])
# X[:, 2] = labelencoder.fit_transform(X[:, 2])
# X[:, 3] = labelencoder.fit_transform(X[:, 3])
# X[:, 4] = labelencoder.fit_transform(X[:, 4])




# confusion = confusion_matrix(y_test, result)
# taxa_acerto = accuracy_score(y_test, result)
# taxa_erro = 1 - taxa_acerto

# =============================================================================
# Errors
# =============================================================================

# dfe = df_ml.loc[:, ['A', 'cobacia', 'posto']]

# error = y - result
# dfe[target+'_obs'] = y
# dfe[target+'_pred'] = result
# dfe[target+'_error'] = error

# dfe.to_csv('data/'+target+'_results_knn.csv', index=False)


# rq = np.maximum(abs(y/result), abs(result/y))

# fig, (ax1) = plt.subplots(1, 1, dpi=300, figsize=(5,5))
# ax1.scatter(dfe['A'], rq)
# ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
# ax1.set_xlabel('A ' + target)
# ax1.set_ylabel('ABSR ' + target)
# # ax1.set_aspect('equal')
# ax1.set_title('BIAS: ' + str(bias) + 
#              ', RMSE: ' + str(rmse) +
#              ', R²: ' + str(r2) + 
#              ', RQ75: ' + str(rq75)
#              )
# ax1.set_xscale('log')
# cutlim = np.percentile(rq, 95)
# ax1.set_xlim([10, 1e7])
# ax1.set_ylim([-cutlim/10, cutlim])
# fig.tight_layout()
# plt.show()



# =============================================================================
# Retrieve nearest neighboors
# =============================================================================

# import geopandas as gpd

# knn = KNeighborsRegressor(weights='distance', p=1)
# knn.fit(X, y)
# nbrs_dw, nbrs_idx = knn.kneighbors()

# df_nbrs = pd.DataFrame(columns=['posto_0', 'posto', 'dist_weight', 'lat', 'lon'])
# for i, s in df_ml.iterrows():
#     # CSV version
#     s_nbrs = df_ml.loc[nbrs_idx[i], ['posto', 'lat', 'lon']]
#     s_nbrs['dist_weight'] = nbrs_dw[i]
#     s_nbrs = s_nbrs.append(s[['posto', 'lat', 'lon']])
#     s_nbrs['posto_0'] = s['posto']
#     df_nbrs = df_nbrs.append(s_nbrs, ignore_index=True)
    
    
#     # Multipoint version
#     # gs = gpd.GeoDataFrame(s_nbrs, geometry=gpd.points_from_xy(s_nbrs.lon, s_nbrs.lat))
    
# df_nbrs.to_csv('data/knn_neighbors_reg.csv')

# gdf = gpd.GeoDataFrame(df_nbrs, geometry=gpd.points_from_xy(df_nbrs.lon, df_nbrs.lat)).set_crs('epsg:4326')
# gdf = gdf.dissolve(by='posto_0', as_index=False)[['posto_0', 'geometry']]
# gdf.to_file('GIS/knn_neighbors_multipoint.gpkg', driver='GPKG')
'''