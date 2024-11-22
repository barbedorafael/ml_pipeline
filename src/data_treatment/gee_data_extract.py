# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 13:20:59 2021

@author: Rafael
"""

import ee
# ee.Authenticate()

ee.Initialize()

def imgExtract(img, img_name, scale, refpoly, folder):
    img = img.clip(refpoly)
    
    task_config = {
            'image': img,
            'description': img_name,
            'folder': folder,
            'scale': scale,
            'region': refpoly,
            'maxPixels': 1e12
            }
    task = ee.batch.Export.image.toDrive(**task_config)
    task.start()

    return

def fcExtract(fc, description, folder):

    task_config = {
      'collection': fc,
      'folder': folder,
      'description': description,
      'fileFormat': 'CSV',
      # 'selectors': selectors
          }
    task = ee.batch.Export.table.toDrive(**task_config)
    task.start()

    return

def col2monthly(col):
    
    def monthly(m):
        return (col.filter(ee.Filter.calendarRange(y_start, y_end, 'year'))
                .filter(ee.Filter.calendarRange(m, m, 'month')).mean()
                .set('month', m)
                .set('system:index', ee.Number(m).format('%02d'))
                )

    
    months = ee.List.sequence(1, 12)
    col_m = ee.ImageCollection.fromImages(months.map(monthly).flatten())
    
    return col_m

def col2yearly(col):
    
    def yearly(y):
        return (col
                .filter(ee.Filter.calendarRange(y, y, 'year')).mean()
                .set('year', y)
                .set('system:index', ee.Number(y).format('%02d'))
                )

    
    years = ee.List.sequence(y_start, y_end)
    col_y = ee.ImageCollection.fromImages(years.map(yearly).flatten())
    
    return col_y

def grid2scale(img):
    return img.reproject(crs='EPSG:4326', scale=img.projection().nominalScale())#.clip(refpoly)

# Region of Interest
Fc = ee.FeatureCollection('projects/et-brasil/assets/bho/geoft_bho_2017_5k_area_drenagem')
Fc = Fc.select(['cotrecho', 'cobacia', 'cocursodag', 'nuareacont'])
# refpoly = roi.geometry().bounds()

folder = 'Regionalisation'

# Define the variables to extract
# =============================================================================
# # Precipitation
# =============================================================================
# gives mm/h
y_start = 2001
y_end = 2020

# climate_data = {'Precipitation':
#                     ee.ImageCollection("NASA/GPM_L3/IMERG_MONTHLY_V06").select('precipitation'),
#                 'Temperature': 
#                     ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY").select('temperature_2m'),
#                 'Potential Evapotranspiration':
#                     ee.ImageCollection("MODIS/NTSG/MOD16A2/105").select('PET'),
#                 'Actual Evapotranspiration':
#                     ee.ImageCollection("MODIS/NTSG/MOD16A2/105").select('ET')
#                     }

#p_y = col2yearly(ee.ImageCollection("NASA/GPM_L3/IMERG_MONTHLY_V06").select('precipitation'))#.map(grid2scale)
p = col2monthly(ee.ImageCollection("NASA/GPM_L3/IMERG_MONTHLY_V06")
                .select(['precipitation'], ['P'])
                .filter(ee.Filter.calendarRange(y_start, y_end, 'year'))
                ) # monthly precipitation

p_avg = p.mean().multiply(24*30).rename('p_avg')
p_min = p.min().multiply(24*30).rename('p_min')
p_max = p.max().multiply(24*30).rename('p_max')

# scale = p_mean.projection().nominalScale().getInfo() / 10
# imgExtract(p_mean, 'P_annual_mean_m', scale, refpoly, folder)

# =============================================================================
# # Temperature
# =============================================================================
# gives Celsius
y_start = 2001
y_end = 2020

t = col2monthly(ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY")
                .select(['temperature_2m'], ['T'])
                .filter(ee.Filter.calendarRange(y_start, y_end, 'year'))
                )

# t = col2monthly(ee.ImageCollection("MODIS/006/MOD11A1") #!!! scale factor 0.02
#                 .select('LST_Day_1km')
#                 .filter(ee.Filter.calendarRange(y_start, y_end, 'year'))
#                 # .map(grid2scale)
#                 ) # monthly temperature

t_avg = t.mean().rename('t_avg').subtract(273.15)
t_min = t.min().rename('t_min').subtract(273.15)
t_max = t.max().rename('t_max').subtract(273.15)

# img_t = t_avg.addBands(t_min).addBands(t_max)
# scale = 50
# folder = 'Regionalisation'
# imgExtract(img_t, 'T_avg_min_max', scale, Fc.geometry(), folder)

# =============================================================================
# # ETP
# =============================================================================
# Gives m
y_start = 2001
y_end = 2020

etp = col2monthly(ee.ImageCollection("MODIS/NTSG/MOD16A2/105")
                .select('PET')
                .filter(ee.Filter.calendarRange(y_start, y_end, 'year'))
                )

etp_avg = etp.mean().rename('pet_avg').divide(80)
etp_min = etp.min().rename('pet_min').divide(80)
etp_max = etp.max().rename('pet_max').divide(80)

# =============================================================================
# # ET
# =============================================================================
# Gives m
y_start = 2001
y_end = 2020

et = col2monthly(ee.ImageCollection("MODIS/NTSG/MOD16A2/105")
                .select('ET')
                .filter(ee.Filter.calendarRange(y_start, y_end, 'year'))
                )

et_avg = et.mean().rename('et_avg').divide(80)
et_min = et.min().rename('et_min').divide(80)
et_max = et.max().rename('et_max').divide(80)

# =============================================================================
# Elevation, Slope, Dd, HAND and Terrain Classes
# =============================================================================
import math

dem = ee.Image("MERIT/DEM/v1_0_3").select('dem')
slp = ee.Terrain.slope(dem).divide(180).multiply(ee.Number(math.pi)).tan().multiply(100).rename('slp')

hand = ee.Image('projects/ee-rbfontana/assets/HAND_SA_TPS_0615').rename('hand')
drain = ee.Image(0).where(hand.eq(0), 120).rename('dd')

tc = ee.Image(6).where(slp.lte(45), 5).where(
                       slp.lte(20), 4).where(
                       slp.lte(8), 3).where(
                       slp.lte(3), 2).where(            
                       hand.lte(50), 1).rename('terrain') # Scale factor 0.1

# =============================================================================
# Soil from USDA
# =============================================================================

# Have to filter to depth
soil_texture = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select('b0').rename('soiltext')

soil_clay_content = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select('b0').rename('soilclay')
soil_sand_content = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select('b0').rename('soilsand')
soil_organic_content = ee.Image("OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02").select('b0').rename('soilorga')
soil_water_content = ee.Image("OpenLandMap/SOL/SOL_WATERCONTENT-33KPA_USDA-4B1C_M/v01").select('b0').rename('soilwate')

soil_density = ee.Image("OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02").select('b0').rename('soildens')

# =============================================================================
# Lithology
# =============================================================================
lithology = ee.Image('projects/et-brasil/assets/glim2012_0point5j').rename('lithology')


# =============================================================================
# # Land cover
# =============================================================================
year = 2010
# MODIS #####
# 1 Evergreen Needleleaf Forests
# 2 Evergreen Broadleaf Forests
# 3 Deciduous Needleleaf Forests
# 4 Deciduous Broadleaf Forests
# 5 Mixed Forests
# 6 Closed Shrublands
# 7 Open Shrublands
# 8 Woody Savannas
# 9 Savannas
# 10 Grasslands
# 11 Permanent Wetlands
# 12 Croplands
# 13 Urban and Built-up Lands
# 14 Cropland/Natural Vegetation Mosaics
# 15 Permanent Snow and Ice
# 16 Barren
# 17 Water Bodies
imgcol = ee.ImageCollection("MODIS/006/MCD12Q1").select('LC_Type1')
img1 = imgcol.filter(ee.Filter.calendarRange(year, year, 'year')).first()
img = img1.reproject('EPSG:4326', scale=img1.projection().nominalScale())

old_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
new_classes = [1, 1, 1, 1, 1, 2, 2, 2, 3,  3,  3,  3,  4,  3,  4,  4,  5]
imgModis = img.remap(old_classes, new_classes)

# scale = img.projection().nominalScale().getInfo()
# imgExtract(imgRm, 'MODIS_LC_rm', scale, refpoly, folder)

# MapBiomas #####
# 3 - Forest
# 4 - Savanna
# 5 - Mangrove
# 9 - Forest Plantation
# 10 - Non-Forest Natural
# 11 - Wetland
# 12 - Grassland
# 32 - Salt Flat
# 29 - Rocky Outcrop
# 13 - Other Non-Forest Natural
# 15 - Pasture
# 39 - Soy Bean
# 20 - Sugar Cane
# 41 - Other Temporary Crops
# 36 - Perennial Crop
# 21 - Mosaic Agriculture and Plasture
# 23 - Beach and Dune
# 24 - Urban
# 30 - Minning
# 25 - Other Non-Vegetated
# 33 - Water Body
# 31 - Aquaculture
img = ee.Image(
    'projects/mapbiomas-workspace/public/collection5/mapbiomas_collection50_integration_v1',
    ).select('classification_'+str(year))

old_classes = [1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 18, 19, 39, 20, 41, 36, 21, 22, 23, 24, 25, 26, 29, 30, 31, 32, 33]
new_classes = [1, 1, 1, 2, 1, 1,  2,  2,  2,  2,  3,  2,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  5,  4,  4,  5,  4,  5]

imgMapbiomas = img.remap(old_classes, new_classes)

# scale = img.projection().nominalScale().getInfo()
# imgExtract(imgRm, 'MapBiomas_LC_rm', scale, refpoly, folder)

lc = ee.ImageCollection.fromImages([imgModis, imgMapbiomas]).mosaic().rename('landcover')

# =============================================================================
# Water storage variability Contributions
# =============================================================================

twsc = ee.Image('projects/et-brasil/assets/ws_sa/tws_c').rename('TWS')
gwc = ee.Image('projects/et-brasil/assets/ws_sa/gw_c').rename('GW')
smc = ee.Image('projects/et-brasil/assets/ws_sa/sm_c').rename('SM')
swc = ee.Image('projects/et-brasil/assets/ws_sa/sw_c').rename('SW')
rsc = ee.Image('projects/et-brasil/assets/ws_sa/rs_c').rename('RS')


# =============================================================================
# Collect data
# =============================================================================

# def getCentroids(feature):
#   return feature.set({'polyCent': feature.centroid()})

# # Agg coordinates
# Fc = Fc.map(getCentroids)

def img2scale(img, scale):
    return img.reproject('EPSG:4326', scale=scale)

def remove_geom(feat):
    return feat.setGeometry(None)

scale = 30 # This has to be the same to get data at basins big enough to fit at least a 30m x 30m pixel

# Agg climate features (P and T and E)    
img_p = p.toBands().multiply(24*30)
Fc_climate = img2scale(img_p, scale).reduceRegions(Fc, ee.Reducer.mean())

img_t = t.toBands().subtract(273.15)
Fc_climate = img2scale(img_t, scale).reduceRegions(Fc_climate, ee.Reducer.mean())

img_etp = etp.toBands().divide(80)
Fc_climate = img2scale(img_etp, scale).reduceRegions(Fc_climate, ee.Reducer.mean())

img_et = et.toBands().divide(80)
Fc_climate = img2scale(img_et, scale).reduceRegions(Fc_climate, ee.Reducer.mean())


# Agg terrain features
img_topo = dem.rename('elv_avg').addBands(slp.rename('slp_avg')).addBands(hand.rename('hnd_avg'))
Fc_topo = img2scale(img_topo, scale).reduceRegions(Fc, ee.Reducer.mean())

# img_topo = dem.rename('elv_').addBands(slp.rename('slp_')).addBands(hand.rename('hnd_'))
# Fc_topo = img2scale(img_topo, scale).reduceRegions(Fc_topo, ee.Reducer.percentile([10, 50, 90]))

# Agg drainage density
Fc_topo = img2scale(drain, 90).reduceRegions(Fc_topo, ee.Reducer.sum()) # Scale here has to be 90 because of dem

# Agg terrain and land cover features
img_classes = tc.addBands(lc)
Fc_topo = img2scale(img_classes, scale).reduceRegions(Fc_topo, ee.Reducer.frequencyHistogram())

# Agg soil and lithology features
img_content = soil_clay_content.addBands(soil_sand_content).addBands(soil_organic_content).addBands(soil_water_content)
Fc_content = img2scale(img_content, 50).reduceRegions(Fc, ee.Reducer.mean())

img_classes = soil_texture.addBands(lithology)
Fc_texlit = img2scale(img_classes, scale).reduceRegions(Fc, ee.Reducer.frequencyHistogram())

# Agg Water Storage features
img_ws = twsc.addBands(gwc).addBands(smc).addBands(swc).addBands(rsc)
Fc_ws = img2scale(img_ws, scale).reduceRegions(Fc, ee.Reducer.mean())

# Code to extract variables to basins
# If memory limit exceeded, do it in chunks accordingly (trial and error)
fcExtract(Fc_climate.map(remove_geom), 'bho_attributes_climate', folder)
fcExtract(Fc_topo.map(remove_geom), 'bho_attributes_topolc', folder)
fcExtract(Fc_content.map(remove_geom), 'bho_attributes_soilcontent', folder)
fcExtract(Fc_texlit.map(remove_geom), 'bho_attributes_textlito', folder)
fcExtract(Fc_ws.map(remove_geom), 'bho_attributes_ws', folder)





