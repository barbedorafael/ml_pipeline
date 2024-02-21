Steps for the ML regionalisation
1.	Initial feature selection
This involves collecting the variables at each basin, first at every BHO, then filtering 
to at every station point (lots of overlapping).

Climate:
•	Precipitation
•	Temperature
•	PET
•	ET
•	Soil Moisture
For the climate variables, collect in terms of mean and variability (Std doesn’t 
work because it is too correlated with the average).

Land Cover (Mapbiomas and MODIS):
•	Closed Forest
•	Open Forest
•	Grassland
•	Agriculture
•	Urban/semi-impervious
•	Open water
Simplified for the purposes of the study. Just rate of occupation of them all.

Terrain:
•	Elevation
•	Slope
Also collect in terms of average and variability within basin. Std has high correlation 
with mean, so maybe it would be worth taking percentile difference or something like 
that.

Hydrological:
•	Drainage density
•	HAND
•	HLCs
Just named hydrological because from the same database (TPS SA – have to upload to 
zenodo). Dd just in terms of mean. HAND same as terrain variables. HLCs same as LC 
variables.

Other:
•	Lithology
•	Soil types: USDA on GEE!!!
These ones we do not have on GEE. Maybe just take lithology, but it would be nice 
to see if soil and HLCs have some correlation.

2.	Feature selection (remove multicolinearity)
This enters in the ML pipeline. There’s correlation matrix and/or VIF (variation 
inflation factor) to use for that.
Correlation matrix is simply plotting all variables’ correlations between each other 
and removing the ones with the highest correlation (e.g. 0.8).
VIF is based on R2 to calculate the predictive power of each variable, without using 
the target (every variable is set as target at each step). VIF has a lot of documentation 
and papers to cite on the internet, so it won’t be difficult to back up.
Probably the best approach is to analyse the correlation matrix, and select variables 
that are highly correlated to each other to drop from the features. There’s no single 
approach to address this problem. Look at more references on the internet.
-	From the correlation matrix
In the first step of feature selection, these variables were excluded from the analysis:

slope_avg and slope_std: because they correlate a lot with hand (avg and std) and 
with some terrain classes (steep and extreme). 

3.	ML model
Now that we have our variables, we can start the predictions. Our ML pipeline consists 
in 3 main steps:
•	Hyperparameter tuning
•	Best model running
•	Uncertainty
•	Permutation importance
We are going to perform these steps for every model used (LR, KNN, SVM, GBM, RF). 
First, we must choose how we are going to split our training and testing datasets. 
We are going to use the K-Fold method, which consists in K runs of the model, in 
each one of them the testing is 1/K targets selected randomly.
For the hyperparameter tunning, we used 10 folds and a randomized search grid, which 
consists of testing different combinations of the parameters with the goal of achieving 
the best metric, which was set to be RMSE.
Then, the best model was run with 100 folds (or LOO-CV?). At this step, we also computed 
feature importances and prediction uncertainty. The predicted value was used to be 
evaluated against the observed data, and analysed in terms of statistical measures. 
The importances of the features were averaged from all time steps of the folds, as 
well as the uncertainty measures.
