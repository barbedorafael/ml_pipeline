# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:08:29 2024

@author: RafBar
"""

import numpy as np
import pandas as pd
from src import functions as mlp
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

df = pd.read_parquet('data/processed/data4ml_gauges.parquet')
df = df.loc[:, ~(df==0).all(axis=0)]
# df = df.drop(['code', 'g_area', 'g_lat', 'g_lon'], axis=1)
# df = df.drop(['cocursodag', 'cobacia', 'nucomptrec'], axis=1)

# Choose targets
targets = ['qm', 'q95'] # ['Wavg', 'Havg'] # 
mlmodel = 'ensemble'
for target in targets:
    
    dfr = pd.read_parquet('data/post/results_raw_'+target+'_'+mlmodel+'_k-fold.parquet')
    
    dfr = dfr.sort_values(by='pred')
    # min_value = dfr.obs.quantile([0.01]).iloc[0]
    dfr['pred'] = np.maximum(dfr.pred, 0.0001)
    
    # dfr = dfr[dfr.pred < 12.5]
    
    y = dfr['obs'].values
    yhat = dfr['pred'].values
    
    dfr['error'] = y - yhat
    dfr['error_rel'] = dfr.error / dfr.pred
    dfr[['lat', 'lon']] = df[['g_lat', 'g_lon']]
    
    # dfcorrx = df.corrwith(abs(dfr.error), method='spearman')
    # dfcorry = dfr.corrwith(abs(dfr.error), method='spearman')
    # print(dfcorry)
    
    u_bands = ['90_low', '90_high', '75_low', '75_high']
    u_alphas = [0.05, 0.95, 0.125, 0.875]
    
    # This processes the empiric error bands
    step = 0.1
    steps = np.arange(step, 1.05, step).round(2)
    quantiles = dfr.pred.quantile(steps)
    lower = 0
    e_range = pd.DataFrame(columns=u_bands)
    for q, upper in quantiles.items():
        subset = (dfr.pred > lower) & (dfr.pred <= upper)
        dfq = dfr[subset]
        e_range.loc[q, '90_low'] = dfq.error.quantile([0.05]).iloc[0]
        e_range.loc[q, '90_high'] = dfq.error.quantile([0.95]).iloc[0]
        e_range.loc[q, '75_low'] = dfq.error.quantile([0.125]).iloc[0]
        e_range.loc[q, '75_high'] = dfq.error.quantile([0.875]).iloc[0]
        lower = upper
        
    e_range['upper_limit'] = quantiles
    e_range.index.name = 'quantile'
    xdata = e_range.upper_limit.rolling(2).mean()
    xdata.iloc[0] = e_range.upper_limit.iloc[0] / 2
    xdata = xdata.values
    
    # This processes the quantile regression models
    params = pd.DataFrame(columns=u_bands)
    
    for n, qd in enumerate(u_alphas):
        quantile_model = smf.quantreg('error ~ pred', dfr).fit(q=qd)
        params.iloc[:,n] = quantile_model.params
    params.index = ['intercept', 'gradient']
    params.index.name = 'parameters'
    
    
    #### This will plot everything together
    
    # Generate the x values for plotting
    xx = np.linspace(dfr.pred.min() / 10, dfr.pred.max() * 1.2, 1000)
    
    # Create the plot
    fig = plt.figure(figsize=(10, 6))
    
    # Scatter plot for the data points
    plt.scatter(dfr.pred, dfr.error, s=15, alpha=0.2, label='Observed errors', color='gray')
    
    # Iterate over columns in params to plot lines and corresponding scatter points
    colors = plt.cm.Dark2(np.linspace(0, 1, len(params.columns)))
    ymin = 0
    ymax = 0
    for n, (column, color) in enumerate(zip(params.columns, colors)):
        a = params.iloc[-1][column]
        b = params.iloc[0][column]
        
        # Calculate the y values for the quantile fit line
        yy = a * xx + b
        yy[(xx + yy) < 0] = -xx[(xx + yy) < 0]
        
        ymin = np.minimum(ymin, np.quantile(yy, 0.05))
        ymax = np.maximum(ymax, np.quantile(yy, 0.95))
        
        # Plot the quantile fit line
        plt.plot(xx, yy, color=color)
        
        # Scatter plot for the noisy data
        ydata_noisy = e_range.iloc[:, n].values
        plt.scatter(xdata, ydata_noisy, color=color, s=50, alpha=0.75)
    
    # Set the x and y limits
    plt.xlim([0, dfr.pred.quantile(0.999)])
    plt.ylim([ymin, ymax])
   
    # Labels
    plt.xlabel('Predicted values (l/s km$^{2}$)')
    plt.ylabel('Error (l/s km$^{2}$)')
    
    # Set the plot title
    # plt.title(f'{mlmodel} {target}')
    
    # Add legend
    # plt.legend()
    
    # Show the plot
    plt.show()
    
    fig.savefig('docs/figures/uncfit_'+target+'_'+mlmodel+'.png', dpi=300)
    
    
    
    # # post-processed results
    # dfr.to_parquet('data/post/results_post_'+target+'_'+mlmodel+'_k-fold.parquet')
    
    # # empiric uncertainty bands
    # e_range.to_csv('data/post/empiric_uncertainty_'+target+'_'+mlmodel+'.csv')
    
    # # parameters of quantile regression
    # params.to_csv('data/post/parameters_uncertainty_'+target+'_'+mlmodel+'.csv')
    
fig = plt.figure(figsize=(2, 3))

plt.scatter([], [], s=15, alpha=0.2, label='Observed errors', color='gray')

# Custom legend for formats
plt.scatter([], [], color='gray', s=50, label='Empiric Quantiles')
plt.plot([], [], color='gray', linestyle='-', label='Fitted Quantiles')

# Custom legend for colors
for column, color in zip(params.columns, colors):
    parts = column.split('_')
    plt.scatter([], [], color=color, marker='s', s=100, label=f'{parts[0]}% {parts[1]}')

# Create the legend
plt.legend(title='', loc='center', ncol=1, frameon=False)

# Remove axes
plt.axis('off')

# Show the legend plot
plt.show()

fig.savefig('docs/figures/uncfit_legend.png', dpi=300)



