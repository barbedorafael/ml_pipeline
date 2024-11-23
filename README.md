# ML regression pipeline

This repository contains the pipeline to generate a dataset containing mean and low reference flows for all Brazilian river stretches using Machine Learning models.

The attribution of the resulting variables was done using the BHO 5k [link]

Another important input is gauged flows at specific stations with more than 20 years of data...

The pipeline is processed in the following form:

1. Feature data collection using Google Earth Engine Python API
2. Data treatment using BHO topological information
3. Processing and evaluation of six Machine Learning models
4. Uncertainty estimation of the results

