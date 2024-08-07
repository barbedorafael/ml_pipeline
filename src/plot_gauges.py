# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:19:57 2024

@author: rafbar
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from matplotlib import cm, colors

# Replace these with actual data loading steps
# Load files for qm target
gauged_data_qm = pd.read_parquet("data/post/results_post_qm_ensemble_k-fold.parquet")
# predictions_qm = gpd.read_parquet("data/post/bho_area_qm_ensemble.parquet")

# Load files for q95 target
gauged_data_q95 = pd.read_parquet("data/post/results_post_q95_ensemble_k-fold.parquet")
# predictions_q95 = gpd.read_parquet("data/post/bho_area_qm_ensemble.parquet")

# Load base layers
hydro_boundaries = gpd.read_file(r"C:\Users\rafbar\OneDrive\PRH\Brasil_data\BRA_RegioesHidro.shp")
rivers = gpd.read_parquet("data/post/bho_trecho_qm_ensemble.parquet", columns=["area_upst", "geometry"])
rivers = rivers[rivers.area_upst>10000]
hydro_boundaries = hydro_boundaries.to_crs(rivers.crs)

# Convert latitude and longitude to geometry for gauged data
def convert_to_geodataframe(df, crs="EPSG:4674"):
    geometry = [Point(xy) for xy in zip(df.lon, df.lat)]
    return gpd.GeoDataFrame(df, crs=crs, geometry=geometry)

gauged_points_qm = convert_to_geodataframe(gauged_data_qm)
gauged_points_q95 = convert_to_geodataframe(gauged_data_q95)

del gauged_data_qm, gauged_data_q95

def plot_points(ax, gdf, column, cmap, title, target):
    
    if target=="$q_{m}$":
        if column in ['obs', 'pred']:
            datamin = 2
            datamax = 40
        elif column == 'error_abs':
            datamin = -7
            datamax = 7
        elif column == 'error_rel':
            datamin = -0.5
            datamax = 0.5
    elif target=="$q_{95}$":
        if column in ['obs', 'pred']:
            datamin = 0.5
            datamax = 12
        elif column == 'error_abs':
            datamin = -3.5
            datamax = 3.5
        elif column == 'error_rel':
            datamin = -0.9
            datamax = 0.9

    norm = colors.Normalize(vmin=datamin, vmax=datamax)
    
    rivers.plot(ax=ax, color='black', linewidth=0.2)
    
    gdf.plot(column=column, norm=norm, cmap=cmap, legend=False, ax=ax, zorder=5, markersize=12)
    
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
    n, bins, patches = hax.hist(data, bins=20, edgecolor='none')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Normalize bin centers to [0, 1]
    col = bin_centers - min(bin_centers)
    col /= max(col)
    
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', colormap(c))
        
    hax.set_xlabel('')
    hax.set_ylabel('number of gauges')
    
    hydro_boundaries.boundary.plot(ax=ax, edgecolor='gray', linewidth=0.3)
    
    ax.axis('off')


# Define target variables and corresponding data
targets = [
    ("$q_{m}$", gauged_points_qm),#, predictions_qm),
    ("$q_{95}$", gauged_points_q95)#, predictions_q95)
]

### Plot everything
for target_name, gauged_points in targets:
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
    
    # Observed values
    plot_points(axes[0, 0], gauged_points, 'obs', 'magma', f'{target_name} \nobserved \n(l/s km$^2$)', target_name)

    # Predicted values
    plot_points(axes[0, 1], gauged_points, 'pred', 'magma', f'{target_name} \npredicted \n(l/s km$^2$)', target_name)

    # Prediction errors
    plot_points(axes[1, 0], gauged_points, 'error_abs', 'RdBu', f'{target_name} \nabsolute errors \n(l/s km$^2$)', target_name)

    # Prediction errors
    plot_points(axes[1, 1], gauged_points, 'error_rel', 'RdBu', f'{target_name} \nrelative errors \n(-)', target_name)
    
    plt.tight_layout()
    plt.show()
    
    target_fn = ''.join(filter(str.isalnum, target_name))
    fig.savefig(f"docs/figures/gauges_{target_fn}_py.png", dpi=300)




