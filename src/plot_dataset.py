# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:19:57 2024

@author: rafbar
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm, colors
import datashader as ds
import datashader.transfer_functions as tf
from datashader.mpl_ext import dsshow

# Load files for environmental descriptors
predictors_env = gpd.read_parquet("data/post/bho_area_descriptors.parquet")

# Load files for qm target
predictions_qm = gpd.read_parquet("data/post/bho_area_qm_ensemble.parquet")

# Load files for q95 target
predictions_q95 = gpd.read_parquet("data/post/bho_area_q95_ensemble.parquet")

# Exploratory analyssis (Comment after using)
desc_env = predictors_env.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
desc_qm = predictions_qm.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
desc_q95 = predictions_q95.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])

# Load base layers
hydro_boundaries = gpd.read_file(r"C:\Users\rafbar\OneDrive\PRH\Brasil_data\BRA_RegioesHidro.shp")
rivers = gpd.read_parquet("data/post/bho_trecho_qm_ensemble.parquet", columns=["area_upst", "geometry"])
rivers = rivers[rivers.area_upst>10000]
hydro_boundaries = hydro_boundaries.to_crs(rivers.crs)

# Convert the colormap to a format Datashader can use
def colormap_to_palette(cmap_name, N=256):
    cmap = plt.get_cmap(cmap_name)
    return [colors.to_hex(cmap(i/N)) for i in range(N)]

# def plot_pols(ax, gdf, column, datamin, datamax, cmap, title):
#     # cmap_lst = cmap_to_color_list(cmap)
#     palette = colormap_to_palette(cmap)
    
#     cvs = ds.Canvas(plot_width=1000, plot_height=1000)
#     agg = cvs.polygons(gdf, geometry='geometry', agg=ds.mean(column))
#     img = tf.shade(agg, cmap=palette, how='linear')
    
#     ax.imshow(tf.dynspread(img, threshold=0.5), origin='upper')
    
#     ax.set_title(title)
#     hydro_boundaries.boundary.plot(ax=ax, edgecolor='black')
#     rivers.plot(ax=ax, color='blue', linewidth=0.5)

#     # Histogram
#     inset_ax = ax.inset_axes([0.05, 0.05, 0.3, 0.3])
#     sns.histplot(gdf[column], kde=False, ax=inset_ax, color='gray')  # Using a solid color for the histogram
#     inset_ax.set_title("Histogram")
#     inset_ax.set_yticks([])
    
#     ax.axis('off')

def plot_pols(ax, gdf, column, datamin, datamax, cmap, title):
    print('start')
    norm = colors.Normalize(vmin=datamin, vmax=datamax)
    
    rivers.plot(ax=ax, color='lightgray', linewidth=0.2)
    hydro_boundaries.boundary.plot(ax=ax, edgecolor='gray', linewidth=0.3)
    print('rivers done')
    
    gdf.plot(column=column, norm=norm, cmap=cmap, legend=False, ax=ax, edgecolor='face', linewidth=0.4)
    ax.text(0.95, 0.95, title, transform=ax.transAxes,
            fontsize=16, verticalalignment='top', horizontalalignment='right')
    
    # Create colorbar in the lower right corner
    # norm = colors.Normalize(vmin=gdf[column].min(), vmax=gdf[column].max())
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Create an inset for the colorbar that is 30% of the plot's y-axis and positioned at 70% of the x-axis
    # divider = make_axes_locatable(ax)
    cax = ax.inset_axes([0.8, 0.05, 0.04, 0.2])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=8)
    cax.yaxis.set_ticks_position('right')
    cax.yaxis.set_label_position('right')

    # Histogram
    hax = ax.inset_axes([0.1, 0.1, 0.25, 0.25])
    colormap = plt.colormaps.get_cmap(cmap)
    data = gdf[column]
    data = data[data>datamin]
    data = data[data<datamax]
    
    # Plot histogram
    fw = np.zeros_like(data) + 1. / data.size
    n, bins, patches = hax.hist(data, bins=20, edgecolor='none', weights=fw)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Normalize bin centers to [0, 1]
    col = bin_centers - min(bin_centers)
    col /= max(col)
    
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', colormap(c))
        
    hax.set_xlabel('')
    hax.set_ylabel('frequency')
    
    print('done\n')
    ax.axis('off')


# Define target variables and corresponding data
targets = [
    ("qm", predictions_qm),
    ("q95", predictions_q95)
]

# norm_values = 
norm_errors = colors.CenteredNorm()

### Plot everything
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 30))

plot_pols(axes[0, 0], predictions_qm, 'pred', 2, 60,'magma', '$q_{m}$ \n(l/s km$^2$)')
plot_pols(axes[0, 1], predictions_q95, 'pred', 0, 15, 'magma', '$q_{95}$ \n(l/s km$^2$)')
plot_pols(axes[1, 0], predictors_env, 'p_avg', 50, 250, 'magma', '$P_{avg}$ \n(mm/month)')
plot_pols(axes[1, 1], predictors_env, 'tp_dd', 0.1, 0.9, 'magma', '$TP_{dd}$ \n(km/km$^2$)')
plot_pols(axes[2, 0], predictors_env, 'sc_org', 1.5, 10, 'magma', '$SC_{org}$ \n(%)')
plot_pols(axes[2, 1], predictors_env, 'tc_plt', 2, 45, 'magma', '$TC_{plt}$ \n(%)')

plt.tight_layout()
plt.show()

fig.savefig("docs/figures/dataset_maps.png", dpi=300)




