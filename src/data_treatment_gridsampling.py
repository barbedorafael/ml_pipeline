# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:44:17 2024

@author: RafBar
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box
import numpy as np
import matplotlib.pyplot as plt

# Create a sample DataFrame with latitude and longitude
df = pd.read_parquet('data/processed/data4ml_gauges.parquet')

# Convert the DataFrame to a GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.g_lon, df.g_lat, crs='epsg:4326'))

hydreg = gpd.read_file('data/raw/BRA_RegioesHidrográficas.shp', encoding='iso-8859-1')

# Spatial join to assign points to grid cells
joined = gpd.sjoin_nearest(hydreg, gdf, how='right')#, predicate='nearest')
joined = joined.rename(columns={'index_left': 'grid'})

percentages = joined.iloc[:, :5].groupby('RHI_NM').size() / joined.shape[0] * 100

# Number of chunks
num_chunks = 10
chunk_size = len(gdf) // num_chunks

# Function to create spatially distributed chunks
def create_spatial_chunks(joined, num_chunks, chunk_size):
    chunks = []
    for i in range(num_chunks):
        sampled_points = joined.groupby('grid', group_keys=False).apply(lambda x: x.sample(min(len(x), chunk_size // num_chunks), random_state=i))
        chunks.append(sampled_points)
        joined = joined.drop(sampled_points.index)
    return chunks

# Create the spatially distributed chunks
chunks = create_spatial_chunks(joined, num_chunks, chunk_size)

# Display the first chunk
print(chunks[0])

# If you want to see the number of points in each chunk
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} size: {len(chunk)}")

# Optionally, concatenate all chunks into a single GeoDataFrame
all_chunks = pd.concat(chunks).reset_index(drop=True)

# Display the concatenated chunks GeoDataFrame
print(all_chunks)

# Plot the grid and the points
fig, ax = plt.subplots(figsize=(10, 6))
grid.boundary.plot(ax=ax, linewidth=1, color='black')
gdf.plot(ax=ax, markersize=5, color='red', alpha=0.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Grid and Points')
plt.show()