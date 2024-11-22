# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:10:51 2024

@author: rafbar
"""

import numpy as np
import pandas as pd
from itertools import combinations
from src import functions as mlp

# Function to compute ensemble prediction
def compute_ensemble_prediction(df, model_indices):
    preds = [df['pred_{}'.format(model_names[i])] for i in model_indices]
    ensemble_pred = np.mean(preds, axis=0)
    return ensemble_pred

# Function to evaluate ensemble
def evaluate_ensemble(df, model_indices, compute_stats_func):
    ensemble_pred = compute_ensemble_prediction(df, model_indices)
    return compute_stats_func(ensemble_pred, df['obs'])

# Choose targets
targets = ['qm', 'q95']
model_names =  ['MLR', 'DT', 'KNN', 'SVM', 'GBM', 'RF']


for target in targets:
    models = [pd.read_parquet('data/output/results_raw_{}_{}_{}.parquet'.format(target, ml, 'k-fold'))
              for ml in model_names]

    # Assuming each model's DataFrame has 'obs' and 'pred' columns
    # Merge all model predictions into a single DataFrame
    ensemble_df = models[0].rename(columns={'pred': 'pred_{}'.format(model_names[0])})
    for i, model in enumerate(models[1:], start=1):
        ensemble_df = ensemble_df.join(model.pred.rename('pred_{}'.format(model_names[i])))
    
    # Generate all possible model combinations (ensembles)
    model_indices = range(len(models))
    ensemble_combinations = [comb for r in range(1, len(model_indices)+1) for comb in combinations(model_indices, r)]
    
    # Evaluate each ensemble
    ensemble_results = []
    for comb in ensemble_combinations:
        stats = evaluate_ensemble(ensemble_df, comb, mlp.stats)
        ensemble_results.append((np.array(model_names)[np.array(comb)], stats))
    
    # Convert results to DataFrame and sort by R²
    results_df = pd.DataFrame(ensemble_results, columns=['combination', 'stats'])
    for metric in stats.keys():
        results_df[metric] = results_df['stats'].apply(lambda x: x[metric])
    results_df = results_df.drop('stats', axis=1)
    results_df = results_df.sort_values(by='r2', ascending=False)
    
    # Select top 5 ensembles
    top_5_ensembles = results_df.head(10)
    
    # Display the top 5 ensembles
    print(target)
    print(top_5_ensembles)
    print('\n')


