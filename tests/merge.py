# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 16:52:13 2021

@author: Rafael
"""

import json
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
import glob
import os

folder = 'GIS/'
# dtype = 'int32'
file = 'MapBiomas_LC_rm'
searchC = folder + file + "*.tif"
files = glob.glob(searchC)


files_to_mosaic = []
for fp in files:
    src = rasterio.open(fp)
    files_to_mosaic.append(src)

mosaic, out_trans = merge(files_to_mosaic)

out_fp = folder + str(region) + '_' + file + '.tif'

out_meta = src.meta.copy()
out_meta.update({"driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
#                "dtype": dtype,
                "nodata": nodata,
                "compress": "LZW",
                "crs": 'EPSG:4326'
                
                })
             
with rasterio.open(out_fp, "w", **out_meta) as dest:
     dest.write(mosaic)

