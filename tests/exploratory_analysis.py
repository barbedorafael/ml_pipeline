# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 14:22:19 2022

@author: Rafael
"""

import numpy as np
import pandas as pd

subs_new = pd.read_excel('data/base_baciasinc_qinc_calc_sem_ons_sem_climadj.xlsx')
df_ml = pd.read_pickle('data/subs_attributes.pkl')
df_ml = df_ml[df_ml['posto'].isin(subs_new['codigo'])]
df_ml.fillna(0, inplace=True)
df_ml.reset_index(drop=True, inplace=True)



df_ml.describe()


results = pd.read_csv('data/q95_results_knn.csv')
distances = pd.read_csv('data/knn_neighbors_reg.csv', index_col=0)
dist_weights = distances.groupby('posto_0').mean()

dist_results = dist_weights.merge(results, left_on='posto_0', right_on='posto')
dists = dist_results['dist_weight']
qobs = dist_results['q95_obs']
qsim = dist_results['q95_pred']
qerr = np.maximum(abs(qobs/qsim), abs(qsim/qobs))

import matplotlib.pyplot as plt


# Plot observed x predicted
fig, (ax1) = plt.subplots(1, 1, dpi=300, figsize=(5,5))
ax1.scatter(dists, qerr)
# ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
# ax1.set_xlabel('Obs ' + target)
# ax1.set_ylabel('Pred ' + target)
# ax1.set_title(mlmodel)
ax1.set_xlim([0, np.percentile(dists, 100)])
ax1.set_ylim([1,np.percentile(qerr, 85)])
