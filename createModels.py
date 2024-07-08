# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 12:56:02 2023

@author: Moritz Off
"""
# %% IMPORT PACKAGES
import pandas as pd
import matplotlib.pyplot as plt
import os
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.sampling import BayesianModelSampling
import numpy as np
import methods as m
import pickle
from matplotlib.ticker import StrMethodFormatter

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# %% DATA IMPORT
if __name__ == "__main__":
    trips_int = pd.read_csv("./output/learningData_SRV_int.csv", header = 0, index_col=0)
    
    layout1 = ["age", "sex", "region_type", "employment", "economic_status", "driving_license",
              "leavingHomeTime",
              "activity1", "activity2", "activity3", "activity4", "activity5", "activity6", "activity7",
              "distance1", "distance2", "distance3", "distance4", "distance5", "distance6", "distance7",
              "distance1_value", "distance2_value", "distance3_value", "distance4_value", "distance5_value", "distance6_value", "distance7_value",
              "legDuration1", "legDuration2", "legDuration3", "legDuration4", "legDuration5", "legDuration6", "legDuration7",
              "startTimeOfActivity1", "startTimeOfActivity2", "startTimeOfActivity3", "startTimeOfActivity4", "startTimeOfActivity5", "startTimeOfActivity6", "startTimeOfActivity7",
              "totalDistance", "totalDistance_value",
              "durationOfActivity1", "durationOfActivity2", "durationOfActivity3", "durationOfActivity4", "durationOfActivity5", "durationOfActivity6", "durationOfActivity7", "durationOfActivitiesTotal"]
    
    trainData = trips_int.sample(frac = 0.8)
    testData = trips_int.drop(trainData.index)
    
    # Output Ordner erstellen
    try:
        os.mkdir("./output/model_tests_fScores")
    except:
        print("Output Ordner existiert bereits, die Dateien werden ueberschrieben!")   

# %% FUNCTIONS TO CREATE MODELS
# %%% MODEL SWISS STRUCTURE
def modelSwissStructure():
    # Erstellung des Vergleichsmodells
    edges = [("age", "employment"),
             ("sex", "employment"),
             ("age", "driving_license"),
             ("sex", "driving_license"),
             ("employment", "activity1"),
             ("driving_license", "activity1"),
             ("employment", "activity2"),
             ("activity1", "activity2"),
             ("activity1", "activity3"),
             ("activity2", "activity3"),
             ("activity2", "activity4"),
             ("activity3", "activity4"),
             ("activity3", "activity5"),
             ("activity4", "activity5"),
             ("activity4", "activity6"),
             ("activity5", "activity6"),
             ("activity5", "activity7"),
             ("activity6", "activity7")]
    
    model_swiss = BayesianNetwork(edges)
    
    return model_swiss

# %%% MODEL STRUCTURE
def modelStructure(modelSwiss = True, useAge = False, useFullAge = False, useFullSex = False, onlyAllSex_Activity = False,
                   fullEmployment = False,onlyEmploymentActivity = False, fullDrivingLicense = False, noActivity_driving = False, durationsAndStartTimes = False, **kwargs):
    
    fullEdgesModel = True
    useTotalDistances = False
    useStartTimeOfActivities = True
    useLegDuration = False
    useLeavingHomeTime = False
    useSex = True
    useEconomicStatus = False
    
    edges = [("age", "driving_license"),
             ("sex", "driving_license"),
             ("age", "economic_status"),
             ("employment", "economic_status"),
             ("age", "activity1"),
             ("age", "activity2"),
             ("sex", "activity1"),
             ("sex", "activity2"),
             ("employment", "activity1"),
             ("employment", "activity2"),
             ("activity1", "distance1"),
             ("activity1", "activity2"),
             ("activity1", "activity3"),
             ("activity2", "distance2"),
             ("activity2", "activity3"),
             ("activity2", "activity4"),
             ("activity3", "distance3"),
             ("activity3", "activity4"),
             ("activity3", "activity5"),
             ("activity4", "distance4"),
             ("activity4", "activity5"),
             ("activity4", "activity6"),
             ("activity5", "distance5"),
             ("activity5", "activity6"),
             ("activity5", "activity7"),
             ("activity6", "distance6"),
             ("activity6", "activity7"),
             ("activity7", "distance7"),
             ("age", "distance1"),
             ("age", "distance2"),
             ("driving_license", "distance1"),
             ("driving_license", "distance2"),
             ("driving_license", "activity1"),
             ("driving_license", "activity2"),
             ("distance1", "distance2"),
             ("distance1", "distance3"),
             ("distance2", "distance3"),
             ("distance2", "distance4"),
             ("distance3", "distance4"),
             ("distance3", "distance5"),
             ("distance4", "distance5"),
             ("distance4", "distance6"),
             ("distance5", "distance6"),
             ("distance5", "distance7"),
             ("distance6", "distance7"),
             ("activity1", "startTimeOfActivity1"),
             ("activity2", "startTimeOfActivity2"),
             ("activity3", "startTimeOfActivity3"),
             ("activity4", "startTimeOfActivity4"),
             ("activity5", "startTimeOfActivity5"),
             ("activity6", "startTimeOfActivity6"),
             ("activity7", "startTimeOfActivity7"),
             ("startTimeOfActivity1", "startTimeOfActivity2"),
             ("startTimeOfActivity2", "startTimeOfActivity3"),
             ("startTimeOfActivity3", "startTimeOfActivity4"),
             ("startTimeOfActivity4", "startTimeOfActivity5"),
             ("startTimeOfActivity5", "startTimeOfActivity6"),
             ("startTimeOfActivity6", "startTimeOfActivity7"),
             ("distance1", "startTimeOfActivity1"),
             ("distance2", "startTimeOfActivity2"),
             ("distance3", "startTimeOfActivity3"),
             ("distance4", "startTimeOfActivity4"),
             ("distance5", "startTimeOfActivity5"),
             ("distance6", "startTimeOfActivity6"),
             ("distance7", "startTimeOfActivity7"),
             ("activity1", "durationOfActivity1"),
             ("activity2", "durationOfActivity2"),
             ("activity3", "durationOfActivity3"),
             ("activity4", "durationOfActivity4"),
             ("activity5", "durationOfActivity5"),
             ("activity6", "durationOfActivity6"),
             ("activity7", "durationOfActivity7"),
             ("startTimeOfActivity1", "durationOfActivity1"),
             ("startTimeOfActivity2", "durationOfActivity2"),
             ("startTimeOfActivity3", "durationOfActivity3"),
             ("startTimeOfActivity4", "durationOfActivity4"),
             ("startTimeOfActivity5", "durationOfActivity5"),
             ("startTimeOfActivity6", "durationOfActivity6"),
             ("startTimeOfActivity7", "durationOfActivity7"),
             ("durationOfActivity1", "startTimeOfActivity2"),
             ("durationOfActivity2", "startTimeOfActivity3"),
             ("durationOfActivity3", "startTimeOfActivity4"),
             ("durationOfActivity4", "startTimeOfActivity5"),
             ("durationOfActivity5", "startTimeOfActivity6"),
             ("durationOfActivity6", "startTimeOfActivity7"),
             ("distance1", "totalDistance1"),
             ("distance2", "totalDistance2"),
             ("distance3", "totalDistance3"),
             ("distance4", "totalDistance4"),
             ("distance5", "totalDistance5"),
             ("distance6", "totalDistance6"),
             ("distance7", "totalDistance7"),
             ("totalDistance1", "totalDistance2"),
             ("totalDistance2", "totalDistance3"),
             ("totalDistance3", "totalDistance4"),
             ("totalDistance4", "totalDistance5"),
             ("totalDistance5", "totalDistance6"),
             ("totalDistance6", "totalDistance7"),
             ("distance1", "legDuration1"),
             ("distance2", "legDuration2"),
             ("distance3", "legDuration3"),
             ("distance4", "legDuration4"),
             ("distance5", "legDuration5"),
             ("distance6", "legDuration6"),
             ("distance7", "legDuration7"),
             ("employment", "leavingHomeTime"),
             ("age", "leavingHomeTime"),
             ("activity1", "leavingHomeTime")]
    
    
    if fullEdgesModel == True:
        edges.extend([("age", "activity3"),
                      ("age", "activity4"),
                      ("age", "activity5"),
                      ("age", "activity6"),
                      ("age", "activity7"),
                      ("sex", "activity3"),
                      ("sex", "activity4"),
                      ("sex", "activity5"),
                      ("sex", "activity6"),
                      ("sex", "activity7"),
                      ("employment", "activity3"),
                      ("employment", "activity4"),
                      ("employment", "activity5"),
                      ("employment", "activity6"),
                      ("employment", "activity7"),
                      ("age", "distance3"),
                      ("age", "distance4"),
                      ("age", "distance5"),
                      ("age", "distance6"),
                      ("age", "distance7"),
                      ("driving_license", "distance3"),
                      ("driving_license", "distance4"),
                      ("driving_license", "distance5"),
                      ("driving_license", "distance6"),
                      ("driving_license", "distance7"),
                      ("driving_license", "activity3"),
                      ("driving_license", "activity4"),
                      ("driving_license", "activity5"),
                      ("driving_license", "activity6"),
                      ("driving_license", "activity7"),
                      ("economic_status", "distance3"),
                      ("economic_status", "distance4"),
                      ("economic_status", "distance5"),
                      ("economic_status", "distance6"),
                      ("economic_status", "distance7")
                      ])
    
    if useTotalDistances == False:
        edges = [x for x in edges if "totalDistance" not in (x[0] + x[1])]
    if useStartTimeOfActivities == False:
        edges = [x for x in edges if "startTimeOfActivity" not in (x[0] + x[1])]
    if useLegDuration == False:
        edges = [x for x in edges if "legDuration" not in (x[0] + x[1])]
    if useLeavingHomeTime == False:
        edges = [x for x in edges if "leavingHomeTime" not in (x[0] + x[1])]
    if useSex == False:
        edges = [x for x in edges if "sex" not in (x[0] + x[1])]
        edges.extend([("sex", "employment")])
        edges.extend([("sex", "driving_license")])     
    if useAge == False:
        edges = [x for x in edges if "age" not in (x[0] + x[1])]
        edges.extend([("age", "driving_license")])
    if useEconomicStatus == False:
        edges = [x for x in edges if "economic_status" not in (x[0] + x[1])]
        edges.extend([("age", "driving_license")])


    
    # Erstellung des Models (ohne modes aus Arbeitsspeicher Gruenden)
    
    if modelSwiss == True:
        model_complete = modelSwissStructure()
        edges = list(model_complete.edges())
        edges.extend([("employment", "distance1"),
                      ("employment", "distance2"),
                      ("driving_license", "distance1"),
                      ("distance1", "distance2"),
                      ("distance1", "distance3"),
                      ("distance2", "distance3"),
                      ("distance2", "distance4"),
                      ("distance3", "distance4"),
                      ("distance3", "distance5"),
                      ("distance4", "distance5"),
                      ("distance4", "distance6"),
                      ("distance5", "distance6"),
                      ("distance5", "distance7"),
                      ("distance6", "distance7"),
                      ("activity1", "distance1"),
                      ("activity2", "distance2"),
                      ("activity3", "distance3"),
                      ("activity4", "distance4"),
                      ("activity5", "distance5"),
                      ("activity6", "distance6"),
                      ("activity7", "distance7")])
        
        if fullEmployment == True:
            edges.extend([("employment", "distance3"),
                          ("employment", "distance4"),
                          ("employment", "distance5"),
                          ("employment", "distance6"),
                          ("employment", "distance7"),
                          ("employment", "activity3"),
                          ("employment", "activity4"),
                          ("employment", "activity5"),
                          ("employment", "activity6"),
                          ("employment", "activity7"),])
        if onlyEmploymentActivity == True:
            edges = [x for x in edges if "employment" not in (x[0])]        
            edges.extend([("employment", "activity1"),
                          ("employment", "activity2"),
                          ("employment", "activity3"),
                          ("employment", "activity4"),
                          ("employment", "activity5"),
                          ("employment", "activity6"),
                          ("employment", "activity7")])
            
        if fullDrivingLicense == True:
            edges.extend([("driving_license", "distance2"),
                          ("driving_license", "distance3"),
                          ("driving_license", "distance4"),
                          ("driving_license", "distance5"),
                          ("driving_license", "distance6"),
                          ("driving_license", "distance7"),
                          ("driving_license", "activity2"),
                          ("driving_license", "activity3"),
                          ("driving_license", "activity4"),
                          ("driving_license", "activity5"),
                          ("driving_license", "activity6"),
                          ("driving_license", "activity7")])
            
        if useAge == True:
            edges.extend([("age", "distance1"),
                          ("age", "distance2"),
                          ("age", "activity1"),
                          ("age", "activity2")])
        if useFullAge == True:
            edges.extend([("age", "distance3"),
                          ("age", "distance4"),
                          ("age", "distance5"),
                          ("age", "distance6"),
                          ("age", "distance7"),
                          ("age", "activity3"),
                          ("age", "activity4"),
                          ("age", "activity5"),
                          ("age", "activity6"),
                          ("age", "activity7")])
        if useFullSex == True:
            edges.extend([("sex", "distance1"),
                          ("sex", "distance2"),
                          ("sex", "distance3"),
                          ("sex", "distance4"),
                          ("sex", "distance5"),
                          ("sex", "distance6"),
                          ("sex", "distance7"),
                          ("sex", "activity1"),
                          ("sex", "activity2"),
                          ("sex", "activity3"),
                          ("sex", "activity4"),
                          ("sex", "activity5"),
                          ("sex", "activity6"),
                          ("sex", "activity7")])
            
        if onlyAllSex_Activity == True:
            edges = [x for x in edges if "sex" not in (x[0])]
            edges.extend([("sex", "employment"),
                          ("sex", "driving_license"),
                          ("sex", "activity1"),
                          ("sex", "activity2"),
                          ("sex", "activity3"),
                          ("sex", "activity4"),
                          ("sex", "activity5"),
                          ("sex", "activity6"),
                          ("sex", "activity7")])
            
        if noActivity_driving == True:
            edges = [x for x in edges if "driving_license" not in (x[0])]
            edges.extend([("driving_license", "distance1"),
                          ("driving_license", "distance2"),
                          ("driving_license", "distance3"),
                          ("driving_license", "distance4"),
                          ("driving_license", "distance5"),
                          ("driving_license", "distance6"),
                          ("driving_license", "distance7")])
        
        if durationsAndStartTimes == True:
            edges.extend([("activity1", "startTimeOfActivity1"),
                          ("activity2", "startTimeOfActivity2"),
                          ("activity3", "startTimeOfActivity3"),
                          ("activity4", "startTimeOfActivity4"),
                          ("activity5", "startTimeOfActivity5"),
                          ("activity6", "startTimeOfActivity6"),
                          ("activity7", "startTimeOfActivity7"),
                          ("startTimeOfActivity1", "startTimeOfActivity2"),
                          ("startTimeOfActivity2", "startTimeOfActivity3"),
                          ("startTimeOfActivity3", "startTimeOfActivity4"),
                          ("startTimeOfActivity4", "startTimeOfActivity5"),
                          ("startTimeOfActivity5", "startTimeOfActivity6"),
                          ("startTimeOfActivity6", "startTimeOfActivity7"),
                          ("distance1", "startTimeOfActivity1"),
                          ("distance2", "startTimeOfActivity2"),
                          ("distance3", "startTimeOfActivity3"),
                          ("distance4", "startTimeOfActivity4"),
                          ("distance5", "startTimeOfActivity5"),
                          ("distance6", "startTimeOfActivity6"),
                          ("distance7", "startTimeOfActivity7"),
                          ("activity1", "durationOfActivity1"),
                          ("activity2", "durationOfActivity2"),
                          ("activity3", "durationOfActivity3"),
                          ("activity4", "durationOfActivity4"),
                          ("activity5", "durationOfActivity5"),
                          ("activity6", "durationOfActivity6"),
                          ("activity7", "durationOfActivity7"),
                          ("startTimeOfActivity1", "durationOfActivity1"),
                          ("startTimeOfActivity2", "durationOfActivity2"),
                          ("startTimeOfActivity3", "durationOfActivity3"),
                          ("startTimeOfActivity4", "durationOfActivity4"),
                          ("startTimeOfActivity5", "durationOfActivity5"),
                          ("startTimeOfActivity6", "durationOfActivity6"),
                          ("startTimeOfActivity7", "durationOfActivity7"),
                          ("durationOfActivity1", "startTimeOfActivity2"),
                          ("durationOfActivity2", "startTimeOfActivity3"),
                          ("durationOfActivity3", "startTimeOfActivity4"),
                          ("durationOfActivity4", "startTimeOfActivity5"),
                          ("durationOfActivity5", "startTimeOfActivity6"),
                          ("durationOfActivity6", "startTimeOfActivity7")])
            
        model_complete = BayesianNetwork(edges)
        
        
    
    m.draw3DNetwork(model_complete, "model-complete")
    
    # safe the edges for dashGraph
    with open("./output/model_complete_edges", "wb") as fp:  # Pickling
        pickle.dump(edges, fp)
        
    return model_complete

# %%% TRAIN MODEL WITH GIVEN DATA
def modelTraining(model, df):
    model_complete = model
    trainData = df
    model_complete.fit(trainData[list(model_complete.nodes())].astype(str),
                       estimator=MaximumLikelihoodEstimator,
                       complete_samples_only=False)
    
    # correct model
    model_complete = m.correctWrongCPDs(model_complete)
    if "durationOfActivity1" in list(model_complete.nodes()):
        model_complete = m.correctWrongActivityDurationCPDs(model_complete, trainData)
    else:
        print("no durationOfActivity nodes in model")
    
    if "startTimeOfActivity1" in list(model_complete.nodes()):
        model_complete = m.correctWrongStartTimeCPDs(model_complete, trainData)
    else:
        print("no startTimeOfActivity nodes in model")
        
    #model_complete = m.correctWrongTotalDistanceCPDs(model_complete, trips_int)
    
    print(model_complete.check_model())
    
    return model_complete

# %% FUNCTIONS TO CHECK MODELS
# %%% MANUAL CALCULATION OF FSCORE

def getFScore(df1, df2, column):
    """
    Takes two dataframes: realData and predData.
    column: specifies the column on which to calculate the Score
    
    DataFrames must contain columns age, sex and employment
    
    1. Attributes column combine age, sex and employment in one column
    2. Count amount of predicted chains per attributes-combination
    3. Calculate the real amounts per attributes vs the predicted amounts
    4. Fillna(0), if one chain was not predicted
    5. Calculate weighted mean of the F-Scores with amount of occured chain as weight
    """
    
    realData = df1
    predData = df2
    col = column
    
    realData["attributes"] = realData.age.astype(str) + "-" + realData.sex.astype(str) + "-" + realData.employment.astype(str)
    predData["attributes"] = predData.age.astype(str) + "-" + predData.sex.astype(str) + "-" + predData.employment.astype(str)
    
    df_real = realData.groupby(["attributes"])[col].value_counts().reset_index()
    df_pred = predData.groupby(["attributes"])[col].value_counts().reset_index()
    
    df = df_real.groupby("attributes")["count"].sum().reset_index().rename(columns={"count":"sum"})
    df_real = pd.merge(df_real, df, on = "attributes")
    
    df_merged = pd.merge(df_real, df_pred, how = "left", on = ["attributes", col], suffixes=["_real", "_pred"]).fillna(0)
    
    df_merged["TP"] = np.where(df_merged.count_pred <= df_merged.count_real, df_merged.count_pred, df_merged.count_real)
    df_merged["FP"] = np.where(df_merged.count_pred <= df_merged.count_real, 0, df_merged.count_pred - df_merged.count_real)
    df_merged["TN"] = np.where(df_merged.count_pred <= df_merged.count_real, df_merged["sum"] - df_merged.count_real, df_merged["sum"] - df_merged.count_pred)
    df_merged["FN"] = np.where(df_merged.count_pred <= df_merged.count_real, df_merged.count_real - df_merged.count_pred, 0)
    
    df_merged["precision"] = df_merged.TP / (df_merged.TP + df_merged.FP)
    df_merged["recall"] = df_merged.TP / (df_merged.TP + df_merged.FN)
    df_merged["F-Score"] = (2 * df_merged.precision * df_merged.recall) / (df_merged.precision + df_merged.recall)
    
    df_merged = df_merged.fillna(0)
    weighted_average = np.average(a = df_merged["F-Score"], weights = df_merged.count_real)
    
    return weighted_average, df_merged[[col,"count_real", "precision", "recall", "F-Score"]]

# %%% CREATE BOXPLOTS FUNCTION
def createBoxplots(actFull, actReduced, distFull, distReduced, plot_details = None):
    # Beispiel-Daten
    Activity_full = actFull
    Activity_reduced = actReduced
    Dist_full = distFull
    Dist_reduced = distReduced
    
    # Erstellen von Subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))
    axLabels = plot_details["xLabels"]
    
    # Boxplot für den ersten Subplot
    axes[0].boxplot([Activity_full, Activity_reduced], vert=True)
    axes[0].set_xticklabels(axLabels)
    axes[0].set_title('Aktivitätenketten')
    axes[0].tick_params(axis='both', labelsize=12)
    
    # Boxplot für den zweiten Subplot
    axes[1].boxplot([Dist_full, Dist_reduced], vert=True)
    axes[1].set_xticklabels(axLabels)
    axes[1].set_title('Distanzketten')
    axes[1].tick_params(axis='both', labelsize=12)   
       
    fig.suptitle(plot_details["title"], fontsize = 13)
    plt.text(0.03, 0.95, "n = 20",ha = "left", transform=axes[0].transAxes)
    plt.text(0.03, 0.92, "m1 = " + str(round(np.median(Activity_full), 3)),ha = "left", transform=axes[0].transAxes)
    plt.text(0.03, 0.89, "m2 = " + str(round(np.median(Activity_reduced), 3)),ha = "left", transform=axes[0].transAxes)
    
    plt.text(0.03, 0.95, "n = 20",ha = "left", transform=axes[1].transAxes)
    plt.text(0.03, 0.92, "m1 = " + str(round(np.median(Dist_full), 3)),ha = "left", transform=axes[1].transAxes)
    plt.text(0.03, 0.89, "m2 = " + str(round(np.median(Dist_reduced), 3)),ha = "left", transform=axes[1].transAxes)
    
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}')) # 2 decimal places
    plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.25, hspace=None)
    


    plt.savefig(fname = plot_details["savePath"], bbox_inches = "tight", format = "svg")

    # Anzeigen der Plots
    plt.show()
    
# %%% FORWARD SAMPLE RESULTS WITH GIVEN TEST DATA AND COMPARE F1-SCORES OF ACTIVITY AND DISTANCE CHAINS
def calculateScores(model, df):
    model_complete = model
    testData = df
    
    # change Layout and sort the values for all columns
    testData = m.changeLayout(testData, layout1)
    testData =  testData.sort_values(list(testData.columns))
    
    # original sampling
    results = BayesianModelSampling(model_complete).forward_sample(size = len(testData),
                                                                   partial_samples = testData[["age", "sex", "employment"]])
    
    # get both dataframes in the same format and sort the values
    results = m.changeLayout(results, list(testData.columns))
    results = results.astype(np.int64)
    results = results.sort_values(list(results.columns)).reset_index(drop=True)
    testData = m.changeLayout(testData, list(results.columns)).reset_index(drop=True)
    
    
    results = m.convertToString(results)
    testData = m.convertToString(testData)
    
    # add the activity chains column
    y_pred_activities = m.addActivityChainColumn(results)
    y_true_activities = m.addActivityChainColumn(testData)
    
    # add the distance chains column
    y_pred_distances = m.addDistanceChainColumn(results)
    y_true_distances = m.addDistanceChainColumn(testData)
    
    f_distances, f_distances_table = getFScore(y_true_distances, y_pred_distances, "dist_chain")
    f_activities, f_activities_table = getFScore(y_true_activities, y_pred_activities, "act_chain")
    
    return f_activities, f_activities_table, f_distances,  f_distances_table

# %%% CALCULATE WEIGHTED MEAN WITH GROUPBY
def Groupby_weighted_avg(values, weighted_value, Group_Cols):

# calculate weighted average per group
    return (values * weighted_value).groupby(Group_Cols).sum() / weighted_value.groupby(
        Group_Cols
    ).sum()

# %% MODEL SWISS
if __name__ == "__main__":
    trips_int_swiss = m.convertToInteger(m.convertToString(trips_int), swissCategories=True)
    trainData_swiss = trips_int_swiss.sample(frac = 0.8)
    testData_swiss = trips_int_swiss.drop(trainData_swiss.index)

    model_swiss = modelSwissStructure()
    edges = list(model_swiss.edges())
    
    model_swiss.fit(trips_int_swiss,
                    complete_samples_only=False)
    
    # Model abspeichern
    model_swiss.save("./output/model_swiss.bif", filetype="bif")
    
    # safe the edges for dashGraph
    with open("./output/model_swiss_edges", "wb") as fp:  # Pickling
        pickle.dump(edges, fp)
 
# %% TEST MODEL COMPLETE
# %%% LOOP X TIMES THROUGH MODEL CREATION, SAMPLING AND SCORING TO GET AVG F1
if __name__ == "__main__":
    path = "./output/model_tests_fScores"
    tests_dict = {}
    tests_dict[1] = {"plot_details" : {"title" : "F-Scores - keine Änderung vs. erste age-Kanten",
                                       "xLabels" : ["unverändert", "inklusive age"],
                                       "savePath" : path + "/fScores-Boxplot_ageEdges.svg"},
                      "kwargs" : {"test1" : {"useAge" : False},
                                  "test2" : {"useAge" : True}}
                      }
    tests_dict[2] = {"plot_details" : {"title" : "F-Scores - nur erste age-Kanten vs. alle age-Kanten",
                                       "xLabels" : ["erste Kanten","alle Kanten"],
                                       "savePath" : path + "/fScores-Boxplot_allAgeEdges.svg"},
                      "kwargs" : {"test1" : {"useAge" : True,
                                             "useFullAge" : False},
                                  "test2" : {"useAge" : True,
                                             "useFullAge" : True}}
                      }
    tests_dict[3] = {"plot_details" : {"title" : "F-Scores - nur erste license-Kanten vs. alle license-Kanten",
                                        "xLabels" : ["erste Kanten", "alle Kanten"],
                                        "savePath" : path + "/fScores-Boxplot_allLicenseEdges.svg"},
                      "kwargs" : {"test1" : {"useAge" : True,
                                             "useFullAge" : True,
                                             "fullDrivingLicense" : False},
                                  "test2" : {"useAge" : True,
                                             "useFullAge" : True,
                                             "fullDrivingLicense" : True}}
                      }
    tests_dict[4] = {"plot_details" : {"title" : "F-Scores - nur erste employment-Kanten vs. alle employment-Kanten",
                                       "xLabels" : ["erste Kanten", "alle Kanten"],
                                       "savePath" : path + "/fScores-Boxplot_allEmploymentEdges.svg"},
                      "kwargs" : {"test1" : {"useAge" : True,
                                             "useFullAge" : True,
                                             "fullDrivingLicense" : False,
                                             "fullEmployment" : False},
                                  "test2" : {"useAge" : True,
                                             "useFullAge" : True,
                                             "fullDrivingLicense" : False,
                                             "fullEmployment" : True}}
                      }
    tests_dict[5] = {"plot_details" : {"title" : "F-Scores - keine sex-Kanten vs. alle sex-Kanten",
                                       "xLabels" : ["keine Kanten", "alle Kanten"],
                                       "savePath" : path + "/fScores-Boxplot_allSexEdges.svg"},
                      "kwargs" : {"test1" : {"useAge" : True,
                                             "useFullAge" : True,
                                             "fullDrivingLicense" : False,
                                             "fullEmployment" : True,
                                             "useFullSex" : False},
                                  "test2" : {"useAge" : True,
                                             "useFullAge" : True,
                                             "fullDrivingLicense" : False,
                                             "fullEmployment" : True,
                                             "useFullSex" : True,}}
                      }
    tests_dict[6] = {"plot_details" : {"title" : "F-Scores - alle employment-Kanten vs. nur employment-activity-Kanten",
                                       "xLabels" : ["alle empl.-Kanten", "nur empl.-activity"],
                                       "savePath" : path + "/fScores-Boxplot_onlyEmployment-ActivityEdges.svg"},
                      "kwargs" : {"test1" : {"useAge" : True,
                                             "useFullAge" : True,
                                             "fullDrivingLicense" : False,
                                             "fullEmployment" : True,
                                             "useFullSex" : True,
                                             "onlyEmploymentActivity" : False},
                                  "test2" : {"useAge" : True,
                                             "useFullAge" : True,
                                             "fullDrivingLicense" : False,
                                             "fullEmployment" : True,
                                             "useFullSex" : True,
                                             "onlyEmploymentActivity" : True}}
                      }
    tests_dict[7] = {"plot_details" : {"title" : "F-Scores - alle sex-Kanten vs. nur sex-activity-Kanten",
                                       "xLabels" : ["activity/distance", "activity"],
                                       "savePath" : path + "/fScores-Boxplot_OnlyAllSex-ActivityEdges.svg"},
                    "kwargs" : {"test1" : {"useAge" : True,
                                           "useFullAge" : True,
                                           "fullDrivingLicense" : False,
                                           "fullEmployment" : True,
                                           "useFullSex" : True,
                                           "onlyEmploymentActivity" : False,
                                           "onlyAllSex_Activity" : False},
                                "test2" : {"useAge" : True,
                                           "useFullAge" : True,
                                           "fullDrivingLicense" : False,
                                           "fullEmployment" : True,
                                           "useFullSex" : True,
                                           "onlyEmploymentActivity" : False,
                                           "onlyAllSex_Activity" : True}}
                  }
    tests_dict[8] = {"plot_details" : {"title" : "F-Scores - alle license Kanten vs. nur license-distance-Kanten",
                                       "xLabels" : ["alle Kanten", "nur dist.-Kanten"],
                                       "savePath" : path + "/fScores-Boxplot_onlyAllLicense-DistanceEdges_withoutActivityEdges.svg"},
                     "kwargs" : {"test1" : {"useAge" : True,
                                            "useFullAge" : True,
                                            "fullDrivingLicense" : False,
                                            "fullEmployment" : True,
                                            "useFullSex" : True,
                                            "onlyEmploymentActivity" : False,
                                            "onlyAllSex_Activity" : False,
                                            "noActivity_driving" : False},
                                  "test2" : {"useAge" : True,
                                             "useFullAge" : True,
                                             "fullDrivingLicense" : False,
                                             "fullEmployment" : True,
                                             "useFullSex" : True,
                                             "onlyEmploymentActivity" : False,
                                             "onlyAllSex_Activity" : False,
                                             "noActivity_driving" : True}}
                     }
    
    
    # calculate chain prevalences of real data
    actChain_prevalence = m.convertToString(trips_int)
    actChain_prevalence = m.addActivityChainColumn(actChain_prevalence)
    actChain_prevalence = (actChain_prevalence.act_chain.value_counts()/len(actChain_prevalence)).reset_index().rename(columns = {"count":"prevalence"}).sort_values("prevalence", ascending = False)
     
    distChain_prevalence = m.convertToString(trips_int)
    distChain_prevalence = m.addDistanceChainColumn(distChain_prevalence)
    distChain_prevalence = (distChain_prevalence.dist_chain.value_counts()/len(distChain_prevalence)).reset_index().rename(columns = {"count":"prevalence"}).sort_values("prevalence", ascending = False)
     
    
    
    for test in tests_dict.values():
        # final data
        ActivityScores = []
        DistScores = []
        ActivityScoresPerChain = actChain_prevalence
        DistScoresPerChain = distChain_prevalence
           
        for n in test["kwargs"].values():    
        
            fScoresActivities = []
            fScoresActivities_table = pd.DataFrame()
            fScoresDistances = []
            fScoresDistances_table = pd.DataFrame()
        
            # split dataset in train and test data, train model, predict and get F-Score
            for num in range(20):
                trips_int_data = trips_int.copy()
                
                # split dataset
                trainData = trips_int_data.sample(frac = 0.8)
                testData = trips_int_data.drop(trainData.index)
                
                model = modelStructure(**n)
                model = modelTraining(model, trainData)
                a, b, c, d = calculateScores(model, testData)
                
                fScoresActivities.append(a)
                fScoresDistances.append(c)
                fScoresActivities_table = pd.concat([fScoresActivities_table, b], axis = 0)
                fScoresDistances_table = pd.concat([fScoresDistances_table, d], axis = 0)
            
            # calculate F-Score per chains
            fScoresActivities_table = Groupby_weighted_avg(fScoresActivities_table["F-Score"], fScoresActivities_table["count_real"], fScoresActivities_table["act_chain"]).to_frame().rename(columns={0:"F-Score_"+str(n)})
            fScoresDistances_table = Groupby_weighted_avg(fScoresDistances_table["F-Score"], fScoresDistances_table["count_real"], fScoresDistances_table["dist_chain"]).to_frame().rename(columns={0:"F-Score_"+str(n)})
              
            # List to compare Scores of different models
            ActivityScores.append(fScoresActivities)
            DistScores.append(fScoresDistances)
        
            ActivityScoresPerChain = pd.merge(ActivityScoresPerChain, fScoresActivities_table, how = "left", on = "act_chain").fillna(0)
            DistScoresPerChain = pd.merge(DistScoresPerChain, fScoresDistances_table, how = "left", on = "dist_chain").fillna(0)
            
        
        createBoxplots(ActivityScores[0], ActivityScores[1], DistScores[0], DistScores[1], plot_details = test["plot_details"])


# %% BUILD AND SAVE MODEL WITH GIVEN CRITERIA
if __name__ == "__main__":
    args = {"useAge" : True,
            "useFullAge" : True,
            "fullDrivingLicense" : False,
            "fullEmployment" : True,
            "useFullSex" : True,
            "onlyEmploymentActivity" : False,
            "onlyAllSex_Activity" : False,
            "noActivity_driving" : False,
            "durationsAndStartTimes" : True}
        
    model_complete = modelStructure(**args)
    model_complete = modelTraining(model_complete, trips_int)
    
    # Model abspeichern
    model_complete.save("./output/model_complete.bif", filetype="bif")
    
    # Graph HTML abspeichern
    m.draw3DNetwork(model_complete, "model-complete")
    
    # cpds abspeichern
    for node in model_complete.nodes():
        cpd = model_complete.get_cpds(node)
        cpd.to_csv(filename="./output/cpds/"+node+'_cpd.csv')
        
# %% DIFFERENT STRUCUTRE LEARNING AND INDEPENDENCE TESTS
# %%% CHECK INDEPENDANCIES
# COLLINEARITY = according to the BN independant nodes are actually dependant

# =============================================================================
# 
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# 
# # Load data into a pandas DataFrame
# data = trips_int
# 
# # Select independent variables
# X = data[['age', 'employment', 'sex', 'activity1']]
# 
# # Calculate VIF for each independent variable
# vif_data = pd.DataFrame() 
# vif_data["feature"] = X.columns 
#   
# # calculating VIF for each feature 
# vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
#                           for i in range(len(X.columns))]
# 
# # Print the VIF results
# print(vif_data)
# =============================================================================
# %%% CORRELATION MATRIX
# =============================================================================
# data = m.convertToString(trips_int)
# dummies = pd.get_dummies(data, drop_first = False)
# corr_matrix = dummies.corr()
# =============================================================================

# %%% CALCULATE CRAMER CORRELATION SCORE FOR EVERY COMBINATION
#model = modelStructure(False)
#nodes = model.nodes()
# =============================================================================
# 
# def cramers_corrected_stat(confusion_matrix):
#     """ calculate Cramers V statistic for categorial-categorial association.
#         uses correction from Bergsma and Wicher, 
#         Journal of the Korean Statistical Society 42 (2013): 323-328
#     """
#     
#     chi2 = stats.chi2_contingency(confusion_matrix)[0]
#     n = confusion_matrix.sum().sum()
#     phi2 = chi2/n
#     r,k = confusion_matrix.shape
#     phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
#     rcorr = r - ((r-1)**2)/(n-1)
#     kcorr = k - ((k-1)**2)/(n-1)
#     return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
# 
# 
# df = m.convertToString(trips_int).astype(str)
# combis = combinations(df.columns,2)
# combis = [("activity1", "distance1"), 
#           ("activity2", "distance2"), 
#           ("activity3", "distance3"), 
#           ("activity4", "distance4"), 
#           ("activity5", "distance5"), 
#           ("activity6", "distance6"), 
#           ("activity7", "distance7")]
# 
# scores = pd.DataFrame()
# nodes = [1]
# for node in nodes:
#     #parents = model.get_parents(node)
#     #combis = combinations(parents, 2)
#     for combi in combis:    
#         data = pd.crosstab(df[combi[0]], df[combi[1]])
#         results = pd.DataFrame({"var1":combi[0],
#                                "var2":combi[1],
#                                "cramerScore":cramers_corrected_stat(data)}, index = [1])
#         scores = pd.concat([scores, results])    
# 
# #scores = scores.sort_values("cramerScore", ascending = False)
# 
# scores = (scores[~scores.filter(like='var').apply(frozenset, axis=1).duplicated()]
#            .reset_index(drop=True))
# 
# =============================================================================

# %%% STRUCTURE LEARNING TEST - AUSKOMMENTIERT
# =============================================================================
# nodesBN = list(model_complete.nodes())
# samples = m.convertToString(trips_int[nodesBN])
# 
# samples = samples.replace(["20 bis unter 25 Jahre","25 bis unter 30 Jahre","30 bis unter 35 Jahre","35 bis unter 40 Jahre","40 bis unter 45 Jahre"],["20 bis unter 45 Jahre","20 bis unter 45 Jahre","20 bis unter 45 Jahre","20 bis unter 45 Jahre","20 bis unter 45 Jahre",])
# samples = samples.replace(["45 bis unter 50 Jahre","50 bis unter 55 Jahre","55 bis unter 60 Jahre","60 bis unter 65 Jahre"],["45 bis unter 65 Jahre","45 bis unter 65 Jahre","45 bis unter 65 Jahre","45 bis unter 65 Jahre",])
# samples = samples.replace(["unter 3 Jahre","3 bis unter 6 Jahre","6 bis unter 10 Jahre","10 bis unter 15 Jahre"],["0 bis unter 15 Jahre","0 bis unter 15 Jahre","0 bis unter 15 Jahre","0 bis unter 15 Jahre"])
# 
# test = BayesianNetwork([('activity2', 'activity5'), ('activity1', 'activity5'), ('activity1', 'activity2'), ('activity1', 'activity3'), ('activity1', 'age'), 
#                         ('sex', 'activity5'), ('sex', 'activity2'), ('sex', 'activity3'), ('sex', 'age'), ('sex', 'activity1'), ('sex', 'activity6'), ('sex', 'activity4'), 
#                         ('age', 'activity5'), ('age', 'activity2'), ('age', 'activity3'), ('activity3', 'activity5'), ('activity3', 'activity2'), ('activity6', 'activity5'), 
#                         ('activity6', 'activity2'), ('activity6', 'activity3'), ('activity6', 'age'), ('activity6', 'activity1'), ('activity4', 'activity5'), 
#                         ('activity4', 'activity2'), ('activity4', 'activity3'), ('activity4', 'age'), ('activity4', 'activity1'), ('activity4', 'activity6'), 
#                         ('employment', 'activity5'), ('employment', 'activity2'), ('employment', 'activity3'), ('employment', 'age'), ('employment', 'activity1'), 
#                         ('employment', 'activity6'), ('employment', 'activity4'), ('employment', 'sex')])
# m.draw3DNetwork(test, "test")
# 
# import pandas as pd
# import numpy as np
# from itertools import combinations
# 
# import networkx as nx
# from sklearn.metrics import f1_score
# 
# from pgmpy.estimators import PC, HillClimbSearch, ExhaustiveSearch
# from pgmpy.estimators import K2Score
# from pgmpy.utils import get_example_model
# from pgmpy.sampling import BayesianModelSampling
# 
# # Funtion to evaluate the learned model structures.
# def get_f1_score(estimated_model, true_model):
#     nodes = estimated_model.nodes()
#     est_adj = nx.to_numpy_matrix(
#         estimated_model.to_undirected(), nodelist=nodes, weight=None
#     )
#     true_adj = nx.to_numpy_matrix(
#         true_model.to_undirected(), nodelist=nodes, weight=None
#     )
# 
#     f1 = f1_score(np.ravel(true_adj), np.ravel(est_adj))
#     print("F1-score for the model skeleton: ", f1)
#     
# est = PC(data=samples)#[["age", "sex", "employment", "activity1", "activity2", "activity3", "activity4", "activity5", "activity6", "activity7"]])
# estimated_model = est.estimate(variant="orig", max_cond_vars=4)
# get_f1_score(estimated_model, model_complete)
# 
# print(estimated_model.edges())
# =============================================================================

# %%% HILL CLIMB STRUCTURE LEARNING TEST
# =============================================================================
# from pgmpy.estimators import PC, HillClimbSearch, ExhaustiveSearch
# from pgmpy.estimators import K2Score, BicScore
# from pgmpy.utils import get_example_model
# from pgmpy.sampling import BayesianModelSampling
# 
# 
# nodes = list(model_complete.nodes())
# samples = m.changeLayout(trips_int, nodes)
# 
# demograficNodes = ["age", "employment", "sex", "driving_license", "economic_status"]
# travelNodes = ["activity1", "activity2","activity3","activity4","activity5","activity6","activity7",
#                "distance1", "distance2","distance3","distance4","distance5","distance6","distance7",
#                "startTimeOfActivity1", "startTimeOfActivity2","startTimeOfActivity3","startTimeOfActivity4","startTimeOfActivity5","startTimeOfActivity6","startTimeOfActivity7",
#                "durationOfActivity1", "durationOfActivity2","durationOfActivity3","durationOfActivity4","durationOfActivity5","durationOfActivity6","durationOfActivity7"]
# 
# 
# from itertools import product
# # initial nodes are independant
# bl = [("sex", "age"), ("age", "sex"), 
#       ("driving_license", "sex"), ("driving_license", "age"),
#       ("economic_status", "sex"), ("economic_status", "age"),
#       ("employment", "sex"), ("employment", "age"), 
#       ("driving_license", "economic_status"), ("driving_license", "employment")]
# 
# # travelNodes don't point to demografic nodes
# bl = bl + list(product(travelNodes, demograficNodes))
# 
# # distance2 doesn't point to distance1 etc.
# for i in travelNodes:
#     for j in travelNodes:
#         if (i[:6] == j[:6]) & (int(i[-1]) > int(j[-1])):
#             bl.append((i,j))
#         if (i.startswith("durationOfActivity")) & (j.startswith("activity")):
#             bl.append((i,j))
#         if (i.startswith("startTimeOfActivity")) & (j.startswith("activity")):
#             bl.append((i,j))
#         if (i.startswith("distance")) & (j.startswith("activity")):
#             bl.append((i,j))
# 
# scoring_method = K2Score(data=samples)
# est = HillClimbSearch(data=samples)
# estimated_model = est.estimate(
#     scoring_method=scoring_method, max_indegree=4, max_iter=int(1e4), black_list=bl, tabu_length = 1000
# )
# 
# estimated_model.edges()
# m.draw3DNetwork(estimated_model, "test_estimation")
# 
# model_estimated = BayesianNetwork(estimated_model.edges())
# model_estimated.fit(trainData.astype(str),
#                     estimator=MaximumLikelihoodEstimator,
#                     complete_samples_only=False)
# 
# model_estimated = m.correctWrongCPDs(model_estimated)
# #model_estimated = m.correctWrongActivityDurationCPDs(model_estimated, trips_int)
# #model_estimated = m.correctWrongStartTimeCPDs(model_estimated, trips_int)
# #model_estimated = m.correctWrongTotalDistanceCPDs(model_estimated, trips_int)
# 
# =============================================================================



