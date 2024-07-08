#!/usr/bin/env python
# coding: utf-8

# %% IMPORT PACKAGES

import pandas as pd
import os
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
import numpy as np
from FW_sampling import my_forward_sample
import methods as m

# %% SETUP
only50YearsAndOlder = False
useOriginalSampling = False
onlyCottbus = False

# %% IMPORT DATA AND MODELS
# Import data
region_data_int = pd.read_csv("./output/basecase/region_data_int_2022.csv", header = 0, index_col=0)
trips_int = pd.read_csv("./output/learningData_SRV_int.csv", header = 0, index_col=0)

# Import models
model = BayesianNetwork.load("./output/model_complete.bif", filetype="bif")
model_swiss = BayesianNetwork.load("./output/model_swiss.bif", filetype="bif")

# Import distance data to assign float values to distance groups
distances_values = pd.read_csv("./output/distances_valuesAndGroups.csv", header = 0, index_col = 0)


# create output folder
try:
    os.mkdir("./output/basecase")
except:
    print("Output Ordner existiert bereits, die Dateien werden überschrieben!")

# %% FILTER
# Only older than 50 years
if only50YearsAndOlder == True:
    region_data_int = region_data_int[region_data_int.age >= 10]

# Only Cottbus example
if onlyCottbus == True:
    region_data_int = region_data_int[(region_data_int["LK-code"]== 12052)]

# %% SAMPLING FUNCTIONS
# %%% ORIGINAL SAMPLING 
def originalSampling(bn, testData):
    
    results = BayesianModelSampling(bn).forward_sample(size = len(testData),
                                                       partial_samples = testData)
    
    # append missing columns to results, since original sampling method doesn't copy them
    missingColumns = [x for x in testData.columns if x not in results.columns]
    results = pd.concat([testData[missingColumns].reset_index(drop = True), results], axis=1)
    
    # convert data to int64 except municipal name
    cols=[i for i in results.columns if i not in ["name"]]
    for col in cols:
        results[col]= results[col].astype(np.int64)
    
    return results

# %%% MODIFIED SAMPLING
def modifiedSampling(bn, testData):
    
    # sample data
    results = my_forward_sample(bn=bn,size=len(testData),
                                partial_samples=testData, 
                                include_latents=False,
                                show_progress=True)
    
    # convert data to int64 except municipal name
    cols=[i for i in results.columns if i not in ["name"]]
    for col in cols:
        results[col]= results[col].astype(np.int64)
        
    return results
    
# %% BASECASE - SAMPLE DATA

if useOriginalSampling == True:
    print("Original sampling is applied")
    
    results = originalSampling(model, region_data_int)  
    results_swiss = originalSampling(model_swiss, region_data_int)

else: 
    print("Modified sampling is applied")
    
    results = modifiedSampling(model, region_data_int)
    results_swiss = modifiedSampling(model_swiss, region_data_int)

# add distance value columns
results = m.convertToString(results)
results = m.addDistanceValueColumns(results, distances_values)
results = m.convertToInteger(results)
  
# %% SAVE DATA
results.to_csv("./output/basecase/results_complete_2022.csv")
results.to_csv("./output/scenarioAnalysis/outputData/results_complete_2022.csv")
results_swiss.to_csv("./output/basecase/results_swiss.csv")

