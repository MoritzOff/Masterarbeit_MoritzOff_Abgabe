#!/usr/bin/env python
# coding: utf-8

# %% IMPORT PACKAGES

import pandas as pd
import os
from pgmpy.sampling import BayesianModelSampling
import numpy as np
from FW_sampling import my_forward_sample
import methods as m
from createModels import modelStructure,modelSwissStructure, modelTraining

# %% SETUP
only50YearsAndOlder = False
useOriginalSampling = False
onlyCottbus = False

# %% IMPORT DATA AND MODELS
# Import data
trips_int = pd.read_csv("./output/learningData_SRV_int.csv", header = 0, index_col=0)

# Import distance data to assign float values to distance groups
distances_values = pd.read_csv("./output/distances_valuesAndGroups.csv", header = 0, index_col = 0)


# create output folder
try:
    os.mkdir("./output/crossValidation")
except:
    print("Output Ordner existiert bereits, die Dateien werden überschrieben!")

# %% CREATE 80/20 DATA SPLIT
trainData = trips_int.sample(frac = 0.8)
testData_full = trips_int.drop(trainData.index)
testData = testData_full[["age", "sex", "employment"]]

# %% BUILD MODEL
args = {"useAge" : True,
        "useFullAge" : True,
        "fullDrivingLicense" : False,
        "fullEmployment" : True,
        "useFullSex" : True,
        "onlyEmploymentActivity" : False,
        "onlyAllSex_Activity" : False,
        "noActivity_driving" : False,
        "durationsAndStartTimes" : True}
    
model = modelStructure(**args)
model = modelTraining(model, trainData)

model_swiss = modelSwissStructure()
model_swiss = modelTraining(model_swiss, trainData)

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
  
# %% SAVE DATA
results.to_csv("./output/crossValidation/crossValidationSplit20_generated.csv")
results_swiss.to_csv("./output/crossValidation/crossValidationSplit20_generated_swiss.csv")

testData_full.to_csv("./output/crossValidation/crossValidationSplit20_originalData.csv")
trainData.to_csv("./output/crossValidation/crossValidationSplit80_trainData.csv")

results.to_csv("./output/scenarioAnalysis/outputData/crossValidationSplit20_generated.csv")
results_swiss.to_csv("./output/scenarioAnalysis/outputData/crossValidationSplit20_generated_swiss.csv")

testData_full.to_csv("./output/scenarioAnalysis/outputData/crossValidationSplit20_originalData.csv")
trainData.to_csv("./output/scenarioAnalysis/outputData/crossValidationSplit80_trainData.csv")



