# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:54:12 2024

@author: RafBar
"""

from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from src import functions as mlp


df = pd.read_parquet('data/processed/data4ml_gauges.parquet')
df = df.sample(frac = 1) # Shuffle values MAKES ALL THE DIFFERENCE IDKW
df = df.loc[:, ~(df==0).all(axis=0)]

# Convert DataFrame to GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.g_lat, df.g_lon))

gdf.crs = "EPSG:4326"  # Set the coordinate reference system to WGS84

# Convert GeoDataFrame to Web Mercator for distance calculation in meters
gdf_mercator = gdf.to_crs(epsg=3857)

# Extract points as an array of coordinates
points = np.array(list(zip(gdf_mercator.geometry.x, gdf_mercator.geometry.y)))

# Build a spatial index using cKDTree
tree = cKDTree(points)

# Query the tree for the 4 nearest neighbors of each point (including itself)
distances, _ = tree.query(points, k=20)  # k=4 because it includes the point itself

# Calculate the mean distance of the 3 nearest neighbors (excluding itself)
mean_distances = distances[:, 1:].mean(axis=1)

# Add the mean distances to the GeoDataFrame
df['Mean_Distance_3_Nearest'] = mean_distances


# Convert 100km to meters (as our projection is in meters)
radius_meters = 500 * 1000

# Query the tree for points within 100km
indices = tree.query_ball_point(points, r=radius_meters)

# Count the number of points within 100km (excluding the point itself)
counts_within_100km = [len(idx) - 1 for idx in indices]  # Subtract 1 to exclude the point itself

# Add the counts to the GeoDataFrame
df['Points_Within_100km'] = counts_within_100km



















