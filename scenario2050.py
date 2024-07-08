# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import methods as m
from FW_sampling import my_forward_sample
from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import BayesianNetwork

# %% SETUP
only50YearsAndOlder = False
useOriginalSampling = False
onlyCottbus = False

# %% DATA IMPORT
# Import Data
mittlereVariante = pd.read_csv("./output/scenario2050/region_data_int_2050_mittlereVariante.csv", header = 0, index_col=0, sep = ",")
jungeVariante = pd.read_csv("./output/scenario2050/region_data_int_2050_jungeVariante.csv", header = 0, index_col=0, sep = ",")
alteVariante = pd.read_csv("./output/scenario2050/region_data_int_2050_alteVariante.csv", header = 0, index_col=0, sep = ",")

mittlereVariante = m.convertToInteger(mittlereVariante)
jungeVariante = m.convertToInteger(jungeVariante)
alteVariante = m.convertToInteger(alteVariante)

mittlereVariante.name = "mittlereVariante"
alteVariante.name = "alteVariante"
jungeVariante.name = "jungeVariante"

# Import distance data to assign float values to distance groups
distances_values = pd.read_csv("./output/distances_valuesAndGroups.csv", header = 0, index_col = 0)

# Import models
model = BayesianNetwork.load("./output/model_complete.bif", filetype="bif")
model_swiss = BayesianNetwork.load("./output/model_swiss.bif", filetype="bif")


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
 


# %% LOOP THROUGH LIST TO SAMPLE EACH 2050 VERSION

varianten = [mittlereVariante, alteVariante, jungeVariante]

for i in varianten:
    testData = i
    varianteName = i.name
    
    # FILTER
    # Only older than 50 years
    if only50YearsAndOlder == True:
        testData = testData[testData.age >= 10]

    # Only Cottbus example
    if onlyCottbus == True:
        testData = testData[(testData["LK-code"] == 12052)]    

    # SAMPLE DATA
    if useOriginalSampling == True:
        print("Original sampling is applied")
        
        results = originalSampling(model, testData)  
        results_swiss = originalSampling(model_swiss, testData)
    
    else: 
        print("Modified sampling is applied")
        
        results = modifiedSampling(model, testData)
        results_swiss = modifiedSampling(model_swiss, testData)
         
        
    # add distance value columns
    results = m.convertToString(results)
    results = m.addDistanceValueColumns(results, distances_values)
    results = m.convertToInteger(results)
        
    # SAVE DATA
    results.to_csv("./output/scenario2050/results_complete_2050_"+varianteName+".csv")
    results.to_csv("./output/scenarioAnalysis/outputData/results_complete_2050_"+varianteName+".csv")
    results_swiss.to_csv("./output/scenario2050/results_swiss_2050_"+varianteName+".csv")


