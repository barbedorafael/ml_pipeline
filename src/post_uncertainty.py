# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:08:29 2024

@author: RafBar
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

# Choose targets
targets = ['qm', 'q95'] # ['Wavg', 'Havg'] # 
for target in targets:
    
    dfr = pd.read_parquet('data/post/results_post_'+target+'_ensemble_k-fold.parquet')
    dfr = dfr.sort_values(by='pred')
    
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
        e_range.loc[q, '90_low'] = dfq.error_abs.quantile([0.05]).iloc[0]
        e_range.loc[q, '90_high'] = dfq.error_abs.quantile([0.95]).iloc[0]
        e_range.loc[q, '75_low'] = dfq.error_abs.quantile([0.125]).iloc[0]
        e_range.loc[q, '75_high'] = dfq.error_abs.quantile([0.875]).iloc[0]
        lower = upper
        
    e_range['upper_limit'] = quantiles
    e_range.index.name = 'quantile'
    xdata = e_range.upper_limit.rolling(2).mean()
    xdata.iloc[0] = e_range.upper_limit.iloc[0] / 2
    xdata = xdata.values
    
    # This processes the quantile regression models
    params = pd.DataFrame(columns=u_bands)
    
    for n, qd in enumerate(u_alphas):
        quantile_model = smf.quantreg('error_abs ~ pred', dfr).fit(q=qd)
        params.iloc[:,n] = quantile_model.params
    params.index = ['intercept', 'gradient']
    params.index.name = 'parameters'
    
    # empiric uncertainty bands
    e_range.to_csv('data/post/empiric_uncertainty_'+target+'_ensemble.csv', index=False)
    
    # parameters of quantile regression
    params.to_csv('data/post/parameters_uncertainty_'+target+'_ensemble.csv', index=False)
    
    #### This will plot everything together
    
    # Generate the x values for plotting
    xx = np.linspace(dfr.pred.min() / 10, dfr.pred.max() * 1.2, 1000)
    
    # Create the plot
    invert=False
    
    fig = plt.figure(figsize=(8, 4))
    
    # Scatter plot for the data points
    if invert:
        e = dfr.error_rel
    else:
        e = dfr.error_abs
    
    plt.scatter(dfr.pred, e, s=15, alpha=0.2, label='Observed errors', color='gray')
    
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
        
        ydata_noisy = e_range.iloc[:, n].values
        
        if invert:
            yy = yy/xx
            ydata_noisy = ydata_noisy/xdata
        
        ymin = np.minimum(ymin, np.quantile(yy, 0.025))
        ymax = np.maximum(ymax, np.quantile(yy, 0.975))
        
        # Plot the quantile fit line
        plt.plot(xx, yy, color=color)
        
        # Scatter plot for the noisy data
        plt.scatter(xdata, ydata_noisy, color=color, s=50, alpha=0.75)
    
    # Set the x and y limits
    plt.xlim([0, dfr.pred.quantile(0.999)])
    plt.ylim([ymin, ymax])
   
    # Labels
    plt.xlabel('Predicted (l/s km$^{2}$)')
    
    if invert:
        plt.ylabel('Relative Error (-)')
    else:
        plt.ylabel('Absolute Error (l/s km$^{2}$)')

    # Custom legend for formats
    plt.scatter([], [], color='gray', s=50, label='Empiric Quantiles')
    plt.plot([], [], color='gray', linestyle='-', label='Fitted Quantiles')

    # Custom legend for colors
    for column, color in zip(params.columns, colors):
        parts = column.split('_')
        plt.scatter([], [], color=color, marker='s', s=100, label=f'{parts[0]}% {parts[1]}')

    # Create the legend
    plt.legend(title='', loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, frameon=False)
    
    # Set the plot title
    # plt.title(f'{mlmodel} {target}')
    
    # Add legend
    # plt.legend()
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    if invert:
        fig.savefig('docs/figures/uncfit_rel_'+target+'.png', dpi=300)
    else:
        fig.savefig('docs/figures/uncfit_abs_'+target+'.png', dpi=300)

    
# fig = plt.figure(figsize=(2, 3))

# plt.scatter([], [], s=15, alpha=0.2, label='Observed errors', color='gray')

# # Custom legend for formats
# plt.scatter([], [], color='gray', s=50, label='Empiric Quantiles')
# plt.plot([], [], color='gray', linestyle='-', label='Fitted Quantiles')

# # Custom legend for colors
# for column, color in zip(params.columns, colors):
#     parts = column.split('_')
#     plt.scatter([], [], color=color, marker='s', s=100, label=f'{parts[0]}% {parts[1]}')

# # Create the legend
# plt.legend(title='', loc='center', ncol=1, frameon=False)

# # Remove axes
# plt.axis('off')

# # Show the legend plot
# plt.tight_layout()
# plt.show()

# fig.savefig('docs/figures/uncfit_legend.png', dpi=300)



