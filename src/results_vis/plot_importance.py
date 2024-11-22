# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 15:43:54 2024

@author: rafbar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")


# Choose targets
targets = ['qm', 'q95'] # ['Wavg', 'Havg'] # 
mlmodel = 'ensemble'
for target in targets:
    imps = pd.read_parquet('data/post/imps_'+target+'_'+mlmodel+'_k-fold.parquet')
    imps.rename(columns=lambda x: f"{x.split('_')[0].upper()}$_{{{x.split('_')[1]}}}$", inplace=True)
    
    nth = 15 # show up until the nth most important
    
    # Feature Importance Visualization
    imps_mean = imps.median()
    imps_sortindex = imps_mean.argsort()[::-1]
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 8), dpi=300)
    ax.boxplot(imps.iloc[:, imps_sortindex[::-1][-nth:]], vert=False, showfliers=False,
               patch_artist=True, notch=True,
               labels=imps.columns[imps_sortindex[::-1][-nth:]])
    fig.tight_layout()
    plt.show()
    
    fig.savefig('docs/figures/permimp_'+target+'_'+mlmodel+'.png', dpi=300)