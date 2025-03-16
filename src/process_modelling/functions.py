# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 06:55:46 2023

@author: RafBar
"""

import numpy as np
import pandas as pd
import time

from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from collections import defaultdict

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV, KFold, LeaveOneOut
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.utils import resample
from sklearn.metrics import r2_score

from scipy.stats import randint, loguniform

import matplotlib.pyplot as plt
import seaborn as sns


# def plot_correlation_matrix(data, savefig=False):
#     """
#     Plots a heatmap of the correlation matrix with sizes proportional to the correlation values.
    
#     Parameters:
#     data (pd.DataFrame): The input dataframe containing the data.
#     title (str): The title of the plot.
#     figsize (tuple): The size of the figure.
    
#     Returns:
#     None
#     """
    
#     # Compute the correlation matrix
#     corr = data.corr()
    
#     # Create a mask for the upper triangle
#     mask = np.triu(np.ones_like(corr, dtype=bool))

#     # Initialize the matplotlib figure
#     f, ax = plt.subplots(figsize=(15, 12), dpi=300)

#     # Generate a custom diverging colormap
#     cmap = sns.diverging_palette(20, 220, as_cmap=True)

#     # Draw the heatmap with the mask and correct aspect ratio
#     sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-0.75, vmax=0.75, center=0,
#                 square=True, linewidths=.3, annot=False, cbar_kws={"shrink": .5})

#     # Add the sizes proportional to the correlation values
#     # for y in range(corr.shape[0]):
#     #     for x in range(y + 1, corr.shape[1]):
#     #         size = abs(corr.iloc[y, x]) * 1000  # Adjust size scaling factor as needed
#     #         plt.scatter(x + 0.5, y + 0.5, s=size, alpha=0.5, c='black', edgecolors='w', lw=0.5)

#     # Add title and labels
#     # plt.title(title, size=15)
#     plt.xticks(ticks=np.arange(len(corr.columns)) + 0.5, labels=corr.columns, rotation=90)
#     plt.yticks(ticks=np.arange(len(corr.index)) + 0.5, labels=corr.index, rotation=0)

#     plt.tight_layout()
#     plt.show()
    
#     if savefig:
#         f.savefig('docs/figures/correlation_matrix_all.png', dpi=300)

def fs_hcluster(df, features, target, cluster_threshold, link_method, plot=False, savefig=False):
    """
    Perform feature selection based on hierarchical clustering and correlation analysis.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing features and target.
    target (str): The name of the target variable.
    cluster_threshold (float): Threshold for forming flat clusters.
    link_method (string): Linkage method  for clustering features.

    Returns:
    cluster_feature (dict): Dictionaire containing the cluster with their respective features.
    selected_features (list): The selected feature for model running.
    """

    # Compute the correlation matrix
    dfcorr = df.corr(method='spearman')
    corr = dfcorr.loc[features, features].values
    corr = np.nan_to_num(corr)

    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # Convert the correlation matrix to a distance matrix
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.linkage(squareform(distance_matrix), method=link_method)

    # Hierarchical clustering
    cluster_ids = hierarchy.fcluster(dist_linkage, cluster_threshold, criterion="distance")
    cluster_id_to_feature_ids = defaultdict(list)
    cluster_feature = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
        cluster_feature['Cluster '+str(cluster_id)].append(df.columns[idx])

    # Select the representative feature closest to the cluster mean
    selected_features = []
    for cluster_id, feature_ids in cluster_id_to_feature_ids.items():
        cluster_dist = distance_matrix[feature_ids][:, feature_ids]
        cluster_mean = cluster_dist.mean(axis=0)
        closest_feature_idx = feature_ids[np.argmin(np.linalg.norm(cluster_dist - cluster_mean, axis=1))]
        representative_feature = df.columns[closest_feature_idx]
        selected_features.append(representative_feature)  

    # Select feature with highest correlation within cluster
    selected_features = []
    for cluster_id, feature_ids in cluster_id_to_feature_ids.items():
        cluster_dist = distance_matrix[feature_ids][:, feature_ids]
        avg_corr = cluster_dist.mean(axis=1)
        representative_feature_idx = feature_ids[np.argmin(avg_corr)]
        representative_feature = df.columns[representative_feature_idx]
        selected_features.append(representative_feature)
    
    # abscorr = dfcorr.loc[features, features].abs()
    # selected_features = []
    # for feats in cluster_feature.values():
    #     corrout = abscorr[feats].drop(feats)
    #     print(corrout)
    
    # Select features with the greatest correlation to the target
    # abscorr = dfcorr.abs()
    # selected_features = []
    # for v in cluster_id_to_feature_ids.values():
    #     targetcorr = abscorr[target].iloc[v].max()
    #     bestfeat = abscorr.reset_index().index[abscorr[target] == targetcorr].values[0]
    #     selected_features.append(df.columns[bestfeat])
        
    # Plot the correlation matrix and the dendogram
    if plot:
        
        flabels = df[features].rename(columns=lambda x: f"{x.split('_')[0].upper()}$_{{{x.split('_')[1]}}}$").columns.tolist()
        flabels = [f"{label}*" if label in selected_features else label for label in flabels]

        fig, ax = plt.subplots(1, 1, figsize=(6,8), dpi=300)
        dendro = hierarchy.dendrogram(
            dist_linkage,
            labels=flabels,
            color_threshold=cluster_threshold,
            above_threshold_color='gray',
            ax=ax,
            # leaf_rotation=90,
            orientation='left',
            )
        # dendro_idx = np.arange(0, len(dendro["ivl"]))
        leaf_colors = {}
        for color, leaf in zip(dendro['leaves_color_list'], dendro['leaves']):
            leaf_colors[leaf] = color
                
        # Apply the custom colors to the tick labels
        ax = plt.gca()
        f_labels = ax.get_ymajorticklabels()
        
        for i, label in zip(dendro['leaves'], f_labels):
            label_text = label.get_text()
            label.set_color(leaf_colors[i])
            if label_text.endswith('*'):
                label.set_fontweight('bold')
                label.set_color('red')  # Highlight representative feature in red
        
        ax.axvline(x=cluster_threshold, color='black', linestyle='--', label=f'Cluster Threshold')
        fig.tight_layout()
        plt.show()
        
        fig, ax = plt.subplots(1, 1, figsize=(15,12), dpi=300)
        sns.heatmap(corr[dendro["leaves"], :][:, dendro["leaves"]][::-1, ::-1], 
                    mask=np.triu(np.ones_like(corr, dtype=bool)), 
                    cmap=sns.diverging_palette(20, 220, as_cmap=True), 
                    vmin=-0.75, vmax=0.75, center=0,
                    square=True, linewidths=.3, annot=False, cbar_kws={"shrink": .6},
                    ax=ax)
        ax.set_xticklabels(dendro["ivl"][::-1], rotation="vertical")
        ax.set_yticklabels(dendro["ivl"][::-1], rotation="horizontal")
        fig.tight_layout()
        plt.show()
    
    return cluster_feature, selected_features

# def pca_cluster_transform(df, cluster_feature):
    
#     X = MinMaxScaler().fit_transform(df.values)
    
#     pcs = {}
    
#     for cluster, features in cluster_feature.items():
#         pca = PCA(n_components=1)  # Only the first principal component
#         pca.fit(X[:,features])
#         pcs[cluster] = pca.transform(X[:,features])
        
#     feature_matrix = np.hstack([pcs[cluster] for cluster in sorted(pcs.keys())])
    
#     return feature_matrix

# def calculate_vif(df, thresh=10):
#     Xv = StandardScaler().fit_transform(df.values) # FOR THE VIF
#     dfi = pd.DataFrame(Xv, columns = df.columns)
    
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
    
#     return dfi.columns[variables]

# def loo_cv(X, y, model, grid):
#     """
#     Perform leave-one-out cross-validation on a given dataset using a specified model and parameter grid.

#     Parameters:
#     - X (pd.DataFrame or np.ndarray): Input features.
#     - y (pd.Series or np.ndarray): Target variable.
#     - model: Machine learning model to be trained.
#     - grid (dict): Hyperparameter grid for tuning the model.

#     Returns:
#     - result (pd.DataFrame): DataFrame with observed values, predicted values.
#     - imps (pd.DataFrame): DataFrame with feature importances.
#     """
#     n_repeats=10
#     cv = LeaveOneOut()
#     yhat = np.zeros(y.size)
#     imps = pd.DataFrame(columns=range(1, X.shape[1]+1), index=range(10))
#     i = 0
    
#     for train_index, test_index in cv.split(X, y):
#         y_pred, pfi = train_test(X, y, train_index, test_index, grid, model, n_repeats)
#         yhat[test_index] = y_pred
#         imps.iloc[i:i+n_repeats, :] = pfi
#         processing = (i // n_repeats + 1) * 100 / y.size
#         print(f'Processing: {processing:.0f}%')
#         i += n_repeats

#     return yhat, imps

def kfold_cv(X, y, model, grid):
    """
    Perform k-fold cross-validation on a given dataset using a specified model and parameter grid.

    Parameters:
    - X (pd.DataFrame or np.ndarray): Input features.
    - y (pd.Series or np.ndarray): Target variable.
    - model: Machine learning model to be trained.
    - grid (dict): Hyperparameter grid for tuning the model.

    Returns:
    - result (pd.DataFrame): DataFrame with observed values, predicted values.
    - imps (pd.DataFrame): DataFrame with feature importances.
    """
    n_splits = 10
    n_repeats = 10
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)  # Ensure reproducibility
    yhat = np.zeros(y.shape)
    imps = pd.DataFrame(columns=range(1, X.shape[1]+1), index=range(n_splits * n_repeats))
    i = 0
    
    for train_index, test_index in cv.split(X, y):
        y_pred, pfi = train_test(X, y, train_index, test_index, grid, model, n_repeats)
        yhat[test_index] = y_pred
        imps.iloc[i:i+n_repeats, :] = pfi
        processing = (i // n_repeats + 1) * 100 / n_splits
        print(f'Processing: {processing:.0f}%')
        i += n_repeats

    return yhat, imps

def train_test(X, y, train_index, test_index, grid, model, n_repeats=10):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    randomSearch = RandomizedSearchCV(estimator=model,
                                      n_jobs=-1,
                                      n_iter=30,
                                      cv=10,
                                      param_distributions=grid,
                                      scoring="r2")
    searchResults = randomSearch.fit(X_train, y_train)
    
    print('\n')
    print('Best score: ' + str(randomSearch.best_score_))
    print('Best params: ' + str(randomSearch.best_params_) + '\n')
    
    model_opt = searchResults.best_estimator_
    
    model_opt.fit(X_train, y_train)
    y_pred = model_opt.predict(X_test)

    try:
        # r = permutation_importance(model_opt, X_test, y_test,
        #                            n_repeats=n_repeats,
        #                            random_state=42, scoring='r2')
        # pfi = r.importances.T
        
        # Compute permutation feature importance
        base_score = r2_score(y_test, y_pred)
        pfi = np.zeros([n_repeats, X.shape[1]])
        for i in range(X.shape[1]):
            perm_importances = []
            for _ in range(n_repeats):  # Repeat N times
                # Shuffle the feature
                # X_train_new = X_train.copy()
                # X_test_new = X_test.copy()
                # np.random.shuffle(X_train_new[:, i])
                # np.random.shuffle(X_test_new[:, i])

                # Drop the feature
                X_train_new = X_train.copy()
                X_test_new = X_test.copy()
                X_train_new = np.delete(X_train_new, i, axis=1)
                X_test_new = np.delete(X_test_new, i, axis=1)
                
                # Retrain the model with the shuffled feature
                model_opt.fit(X_train_new, y_train)
                y_pred_new = model_opt.predict(X_test_new)
                new_score = r2_score(y_test, y_pred_new)
                perm_importances.append(base_score - new_score)
            
            # Average the importance over N repeats
            pfi[:, i] = perm_importances
    except:
        pfi = 0
    
    return y_pred, pfi

def model_run(X, y, mlmodel, method='kfold'):
    """
    Tune hyperparameters, train and test a machine learning model.

    Parameters:
    df (pd.DataFrame): DataFrame containing the features and target.
    selected_features (list): The selected feature for model running.
    target (str): Name of the target variable.
    mlmodel (str): Machine learning model type ('MLR', 'DT', 'KNN', 'SVM', 'GBM', 'RF').

    Returns:
    dfe: DataFrame with specified columns, observed and predicted values, and errors.
    """
    start_time = time.time()

    # Model selection
    if mlmodel == 'MLR':
        model = LinearRegression()
        grid = dict()
    elif mlmodel == 'DT':
        model = DecisionTreeRegressor()
        grid = dict(max_depth=randint(2, 20), min_samples_leaf=randint(5, 100))
    elif mlmodel == 'KNN':
        model = KNeighborsRegressor()
        grid = dict(leaf_size=randint(1, 50), n_neighbors=randint(1, 30), p=randint(1, 5))
    elif mlmodel == 'SVM':
        model = SVR()
        grid = dict(gamma=loguniform(1e-4, 1), C=loguniform(0.1, 100))
    elif mlmodel == 'GBM':
        model = GradientBoostingRegressor()
        grid = dict(n_estimators=randint(1, 500), max_leaf_nodes=randint(2, 100),
                    learning_rate=loguniform(0.01, 1))
    elif mlmodel == 'RF':
        model = RandomForestRegressor()
        grid = dict(n_estimators=randint(1, 500), max_leaf_nodes=randint(2, 100))
    else:
        raise ValueError('Please provide a valid ML model type!')
    
    # Prepare data
    X = MinMaxScaler().fit_transform(X)
    
    ## 10-Fold Cross Validation
    if method=='k-fold':
        yhat, imps = kfold_cv(X, y, model, grid)
    # elif method=='loo':
    #     yhat, imps = loo_cv(X, y, model, grid)
    elif method=='dataset':
        train_index = ~np.isnan(y)
        test_index = [True] * train_index.size
        yhat, imps = train_test(X, y, train_index, test_index, grid, model)
    else:
        return print('PROVIDE VALID METHOD')

    end_time = time.time()
    print('\n' + mlmodel + ' | Time to process: ' + str((end_time - start_time) / 60) + ' min \n')
    
    return yhat, imps

def stats(pred, obs):
    # rq75 = np.percentile(np.maximum(abs(pred/obs), abs(obs/pred)), 75)
    r2 = 1 - ((pred-obs)**2).sum()/((obs-obs.mean())**2).sum() # == nash
    rmse = ((pred - obs) ** 2).mean() ** .5
    bias = ((pred-obs).sum())/obs.sum() * 100
    return {
        # 'rq75': rq75,
        'r2': r2,
        'rmse': rmse,
        'bias': bias
        }

def plot_results(yobs, ypred, imps, target, mlmodel, savefigs=False, folder='docs/figures/'):
    """    
    plot (bool): If True, plot feature importance and observed vs predicted values.
    savefigs (bool): If True, save figures plotted (plot must be True).
    """
    
    # Compute statistic metrics of the cross validations
    stats_cv = stats(ypred, yobs)
    
    # Feature Importance Visualization
    imps_mean = imps.median()
    imps_sortindex = imps_mean.argsort()[::-1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    ax.boxplot(imps.iloc[:, imps_sortindex[::-1][-12:]], vert=False, showfliers=False,
               labels=imps.columns[imps_sortindex[::-1][-12:]])
    ax.set_title(f'{mlmodel}, {target}')
    fig.tight_layout()
    plt.show()

    bias = stats_cv['bias'].round(2)
    rmse = stats_cv['rmse'].round(2)
    r2 = stats_cv['r2'].round(2)
    
    if savefigs:
        fig.savefig(folder+'permimp_'+target+'_'+mlmodel+'.png', dpi=300)

    # Observed vs Predicted Visualization
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    ax1.scatter(yobs, ypred, s=5, alpha=0.5)
    ax1.plot([yobs.min(), yobs.max()], [yobs.min(), yobs.max()], 'k--')
    ax1.set_xlabel(f'Obs {target} [l/s]')
    ax1.set_ylabel(f'Pred {target} [l/s]')
    ax1.set_title(f'{mlmodel} \n BIAS: {bias:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}')
    cutlim = np.percentile(yobs, 99)
    ax1.set_xlim([0, cutlim])
    ax1.set_ylim([0, cutlim])
    ax1.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.show()
    
    if savefigs:
        fig.savefig(folder+'result_'+target+'_'+mlmodel+'.png', dpi=300)




