#!/usr/bin/env python
# coding: utf-8

#%% Import Packages
import pandas as pd
from matplotlib import pyplot as plt
import folium
import json
import os
import methods as m
from scipy.stats import wasserstein_distance
import math
import branca.colormap as cm
from folium.plugins import GroupedLayerControl
import altair as alt

plt.style.use("seaborn-v0_8-deep")
plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams.update({"axes.grid" : False})
plt.rcParams.update({'font.size': 16})

# Output Ordner erstellen Szenario Analyse
try:
    os.mkdir("./output/scenarioAnalysis/basecase_swiss")
    os.mkdir("./output/scenarioAnalysis/crossValidation")
    os.mkdir("./output/scenarioAnalysis/crossValidation_swiss")
    os.mkdir("./output/scenarioAnalysis/middleScenarios")
    os.mkdir("./output/scenarioAnalysis/scenario2030")
    os.mkdir("./output/scenarioAnalysis/scenario2050")
    os.mkdir("./output/scenarioAnalysis/SRV")
    os.mkdir("./output/scenarioAnalysis/SrvVsBasecase")

except:
    print("Output Ordner existiert bereits, die Dateien werden ueberschrieben!")

# %% SCENARIO SETTINGS, DATA IMPORT AND PREPARATION

def importData():
    
    pfad = "./output"
    srv_pfad = pfad + "/learningData_SRV_int.csv"
    ergebnisPfad = pfad + "/scenarioAnalysis"
    
    basecasePfad = ergebnisPfad + "/outputData/results_complete_2022.csv"
    
    crossValidationPfad_generated = ergebnisPfad + "/outputData/crossValidationSplit20_generated.csv"
    crossValidationPfad_originalData = ergebnisPfad + "/outputData/crossValidationSplit20_originalData.csv"
    crossValidationPfad_generatedSwiss = ergebnisPfad + "/outputData/crossValidationSplit20_generated_swiss.csv"
    crossValidationPfad_trainData = ergebnisPfad + "/outputData/crossValidationSplit80_trainData.csv"


    scenario2030_lowPfad = ergebnisPfad + "/outputData/results_complete_2030_untereVariante.csv"
    scenario2030_middlePfad = ergebnisPfad + "/outputData/results_complete_2030_mittlereVariante.csv"
    scenario2030_highPfad = ergebnisPfad + "/outputData/results_complete_2030_obereVariante.csv"
    
    scenario2050_youngPfad = ergebnisPfad + "/outputData/results_complete_2050_jungeVariante.csv"
    scenario2050_middlePfad = ergebnisPfad + "/outputData/results_complete_2050_mittlereVariante.csv"
    scenario2050_oldPfad = ergebnisPfad + "/outputData/results_complete_2050_alteVariante.csv"
    
    
    # define columns layouts.
    layout1 = ["age", "sex", "region_type", "LK-code", "code", "employment", "economic_status", "driving_license", 
              "leavingHomeTime",
              "activity1", "activity2", "activity3", "activity4", "activity5", "activity6", "activity7",
              "distance1", "distance2", "distance3", "distance4", "distance5", "distance6", "distance7",
              "legDuration1", "legDuration2", "legDuration3", "legDuration4", "legDuration5", "legDuration6", "legDuration7",
              "startTimeOfActivity1", "startTimeOfActivity2", "startTimeOfActivity3", "startTimeOfActivity4", "startTimeOfActivity5", "startTimeOfActivity6", "startTimeOfActivity7",
              "totalDistance1", "totalDistance2", "totalDistance3", "totalDistance4", "totalDistance5", "totalDistance6", "totalDistance7",
              "durationOfActivity1", "durationOfActivity2", "durationOfActivity3", "durationOfActivity4", "durationOfActivity5", "durationOfActivity6", "durationOfActivity7", 
              "distance1_value", "distance2_value", "distance3_value", "distance4_value", "distance5_value", "distance6_value", "distance7_value","totalDistance", "totalDistance_value"]
    
    
    # Import der Daten im Integer Format
    scenarios = {"lowScenario_2030" : pd.read_csv(scenario2030_lowPfad, header = 0, index_col=0),
                 "middleScenario_2030" : pd.read_csv(scenario2030_middlePfad, header = 0, index_col=0),
                 "highScenario_2030" : pd.read_csv(scenario2030_highPfad, header = 0, index_col=0),
                 "youngScenario_2050" : pd.read_csv(scenario2050_youngPfad, header = 0, index_col=0),
                 "middleScenario_2050" : pd.read_csv(scenario2050_middlePfad, header = 0, index_col=0),
                 "oldScenario_2050" : pd.read_csv(scenario2050_oldPfad, header = 0, index_col=0),
                 "basecase" : pd.read_csv(basecasePfad, header = 0, index_col=0),
                 "SRV_BB" : pd.read_csv(srv_pfad, header = 0, index_col=0),
                 "testset_generated" : pd.read_csv(crossValidationPfad_generated, header = 0, index_col=0),
                 "testset_original" : pd.read_csv(crossValidationPfad_originalData, header = 0, index_col=0),
                 "testset_generatedSwiss" : pd.read_csv(crossValidationPfad_generatedSwiss, header = 0, index_col=0),
                 "trainset" : pd.read_csv(crossValidationPfad_trainData, header = 0, index_col=0)}


    
    # convert to string, give names and change layouts if necessary
    for key in scenarios.keys():
        scenarios[key] = m.convertToString(scenarios[key])
        scenarios[key] = m.changeLayout(scenarios[key], layout1)
        scenarios[key].name = key
        
    return scenarios, ergebnisPfad
        
# %% PLOT METHODS
# %%% TOTAL DISTANCE DISTRIBUTION PER AGE GROUP
def plotTotalDistanceDistributionPerAge(scenarios, savePath = "./output/totalDistanceDistributionPerAge.pdf"):
    """
    Plots the totalDistance distribution per age group of different scenarios
    scenarios = List of DataFrames, which contain the Output Data of scenarios in string format.
    DataFrames need a name (e.g. "lowScenario"), a totalDistance columns and an age column
    """
    
    # create dataframe with distributions per scenario
    df = scenarios[0].copy()
    df.name = scenarios[0].name
    
    # dict for share of age group
    populationShare = {}
    populationShare[df.name] = (df.value_counts("age")*100/len(df)).to_dict()

    df.totalDistance = df.totalDistance.fillna("nan")
    df = (df.groupby("age")["totalDistance"].value_counts(dropna = False) / df.groupby("age")["totalDistance"].count()) * 100
    df = df.to_frame().reset_index().rename(columns = {0:scenarios[0].name})
    
    for scenario in scenarios[1:]:
        df_scenario = scenario.copy()
        colName = scenario.name
        
        populationShare[colName] = (df_scenario.value_counts("age")*100/len(df_scenario)).to_dict()
        df_scenario.totalDistance = df_scenario.totalDistance.fillna("nan")
        df_scenario = (df_scenario.groupby("age")["totalDistance"].value_counts(dropna = False) / df_scenario.groupby("age")["totalDistance"].count()) * 100
        df_scenario = df_scenario.to_frame().reset_index().rename(columns = {0:colName})
        
        df = pd.merge(df, df_scenario, on = ["age", "totalDistance"])

    # convert to dictionary
    age_order = ["unter 10 Jahre","10 bis unter 15 Jahre", 
                 "15 bis unter 18 Jahre","18 bis unter 20 Jahre","20 bis unter 25 Jahre","25 bis unter 30 Jahre",
                 "30 bis unter 35 Jahre","35 bis unter 40 Jahre","40 bis unter 45 Jahre","45 bis unter 50 Jahre",
                 "50 bis unter 55 Jahre","55 bis unter 60 Jahre","60 bis unter 65 Jahre","65 Jahre und mehr"]
    
    df_unsorted = dict(tuple(df.groupby("age")))
    age_order = [i for i in age_order if i in df_unsorted.keys()]

    df = {k: df_unsorted[k] for k in age_order}

    
    # plot
    subplotRows = math.ceil(len(df.keys())/2)

    idx = list(range(subplotRows))*2
    i = 0
    j = 0
    count = 0

    fig, axes = plt.subplots(subplotRows, 2, figsize=(16,subplotRows*5))

    for key in df.keys():
        
        # Anteile der Altersgruppe je Scenario
        shares = {}
        for keyPop in populationShare.keys():
            shares[keyPop] = round(populationShare[keyPop][key],2)
        
        df_age = df[key]
        df_age = df_age.drop("age", axis = 1)
        df_age.index = df_age.totalDistance
        df_age = df_age.reindex(index = ["0-1km", "1-2km", "2-5km", "5-10km", "10-20km", "20-50km", ">50km", "nan"])
        
        # Spalten umbenennen, so dass Share drin ist
        for col in df_age.columns:
            if col != "totalDistance":
                df_age = df_age.rename(columns = {col : col + " (" + str(shares[col]) + "%)"}) 
         
        # Plot
        df_age.plot(kind = "bar", title = key,stacked = False, ax = axes[idx[i]][j], alpha = 0.8)
        axes[idx[i]][j].set_ylabel("Anteil [%]")
        axes[idx[i]][j].set_xlabel("Gesamtdistanz")
        for tick in axes[idx[i]][j].get_xticklabels():
            tick.set_rotation(45)
        
        if count % 2 == 0:
            j = 1
        else:
            j = 0
            i = i + 1
        
        count = count + 1

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.7)

    plt.suptitle("Verteilung der Gesamtdistanz je Altersgruppe", y = 0.9)  
    
    plt.rcParams.update({'font.size': 16})
    plt.savefig(fname = savePath, bbox_inches = "tight", format = "pdf")

    
    return

# %%% TOTAL DISTANCE DISTRIBUTION
def plotTotalDistanceDistribution(scenarios, savePath = "./output/totalDistanceDistribution.pdf"):

    df = scenarios[0]
    name = scenarios[0].name
    
    # calculate shares of totalDistance
    totalDistanceDist = (df.value_counts("totalDistance") *100 / len(df)).to_frame().rename(columns = {"count" : name})
    
    # calculate for all scenarios and merge
    for scenario in scenarios[1:]:
        df = scenario
        name = scenario.name
        
        totalDistanceDist_scenario = (df.value_counts("totalDistance") *100 / len(df)).to_frame().rename(columns = {"count" : name})
    
        totalDistanceDist = pd.merge(totalDistanceDist, totalDistanceDist_scenario, left_index = True, right_index = True, how = "outer")
    
    totalDistanceDist = totalDistanceDist.reindex(["0-1km", "1-2km", "2-5km", "5-10km", "10-20km", "20-50km", ">50km"])
    
    # plot
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(1,1,1)
    
    totalDistanceDist.plot(kind = "bar", title = "Verteilung der täglichen Gesamtdistanzen", figsize = (8,8),stacked = False, ax = ax1)
    
    ax1.set_ylabel("Anteil [%]")
    ax1.set_xlabel("Gesamtdistanz")
    
    for tick in ax1.get_xticklabels():
        tick.set_rotation(45)
        
    plt.rcParams.update({'font.size': 16})
    plt.savefig(fname = savePath, bbox_inches = "tight", format = "pdf")
    
    return
# %%% LEG LENGTH DISTRIBUTION
def plotLegLengthDistribution(scenarios, savePath = "./output/legLengthDistribution.pdf"):

    df = scenarios[0]
    name = scenarios[0].name
    
    distCols = [x for x in df.columns if x.startswith("distance") and "value" not in x]
   
    legs = pd.DataFrame()
    for col in distCols: 
        legs = pd.concat([legs, df[col].to_frame().rename(columns = {col : "distance"})])

    legs = legs[legs.distance.astype(str) != "nan"]
    
    legDist = (legs.value_counts("distance") *100 / len(legs)).to_frame().rename(columns = {"count" : name})
    
    for scenario in scenarios[1:]:
        df = scenario
        name = scenario.name
        
        distCols = [x for x in df.columns if x.startswith("distance") and "value" not in x]
       
        legs = pd.DataFrame()
        for col in distCols: 
            legs = pd.concat([legs, df[col].to_frame().rename(columns = {col : "distance"})])

        legs = legs[legs.distance.astype(str) != "nan"]
        
        legDist_scenario = (legs.value_counts("distance") *100 / len(legs)).to_frame().rename(columns = {"count" : name})
        
        legDist = pd.merge(legDist, legDist_scenario, left_index=True, right_index=True, how = "outer")
        
    legDist = legDist.reindex(["0-1km", "1-2km", "2-5km", "5-10km", "10-20km", "20-50km", ">50km"])
    
    # plot
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(1,1,1)
    
    legDist.plot(kind = "bar", title = "Verteilung der Wegedistanzen", figsize = (8,8),stacked = False, ax = ax1)
    
    ax1.set_ylabel("Anteil [%]")
    ax1.set_xlabel("Wegedistanz")
    
    for tick in ax1.get_xticklabels():
        tick.set_rotation(45)
    
    plt.rcParams.update({'font.size': 16})
    plt.savefig(fname = savePath, bbox_inches = "tight", format = "pdf")
    
    return

# %%% AVERAGE DISTANCE PER AGE GROUP
def plotAvgDistancePerAge(scenarios, savePath = "./output/avgDistancePerAge.pdf"):
    
    # create dataframe with average total distances
    df = scenarios[0]
    df.name = scenarios[0].name
    
    df.totalDistance_value = df.totalDistance_value.fillna(0)
        
    df = df.groupby("age")["totalDistance_value"].mean().to_frame().reset_index().rename(columns = {"totalDistance_value" : df.name})
    
    
    for scenario in scenarios[1:]:
        df_age = scenario.copy()
        df_age.name = scenario.name
        df_age.totalDistance_value = df_age.totalDistance_value.fillna(0)
        
        df_age = df_age.groupby("age")["totalDistance_value"].mean().to_frame().reset_index().rename(columns = {"totalDistance_value" : df_age.name})
    
        df = pd.merge(df, df_age, on = "age")
    
    df.index = df.age
    df = df.reindex(index = ["unter 10 Jahre","10 bis unter 15 Jahre", 
                             "15 bis unter 18 Jahre","18 bis unter 20 Jahre","20 bis unter 25 Jahre","25 bis unter 30 Jahre",
                             "30 bis unter 35 Jahre","35 bis unter 40 Jahre","40 bis unter 45 Jahre","45 bis unter 50 Jahre",
                             "50 bis unter 55 Jahre","55 bis unter 60 Jahre","60 bis unter 65 Jahre","65 Jahre und mehr"])
    
    try:
        avgDistance = (df.iloc[:,2]/df.iloc[:,1])-1
        print("avg overestimation of distance per age group: " + str(round(sum(avgDistance)/len(avgDistance)*100,3)) +"%")
    except:
        pass
    
    # plot
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(1,1,1)
    
    df.plot(kind = "bar", title = "", figsize = (8,8),stacked = False, ax = ax1)
    ax1.set_ylabel("durchschnittliche Tagesdistanz in km")
    ax1.set_xlabel("Altersgruppe")
    
    for tick in ax1.get_xticklabels():
        tick.set_rotation(90)
    
    plt.rcParams.update({'font.size': 16})
    plt.savefig(fname = savePath, bbox_inches = "tight", format = "pdf")

        
    return
# %%% AVERAGE DISTANCE PER DISTANCE GROUP
def plotAvgDistancePerDistanceGroup(scenarios, savePath = "./output/avgDistancePerDistanceGroup.pdf"):
    
    # create dataframe with average total distances
    df = scenarios[0]
    name = scenarios[0].name
    
    cols = [x for x in df.columns if x.startswith("distance") and "value" not in x]
    
    distances = pd.DataFrame()
    
    for col in cols:
        distance_value_col = col + "_value"
        dist = pd.DataFrame({"distance":df[col], name:df[distance_value_col]})
        distances = pd.concat([distances, dist])
    
    distances = distances.groupby("distance")[name].mean().to_frame()
    distances = distances.reindex(["0-1km", "1-2km", "2-5km", "5-10km", "10-20km", "20-50km", ">50km"])
    
    
    for scenario in scenarios[1:]:
        
        df = scenario
        name = scenario.name
        
        cols = [x for x in df.columns if x.startswith("distance") and "value" not in x]
    
        distances_scenario = pd.DataFrame()
    
        for col in cols:
            distance_value_col = col + "_value"
            dist = pd.DataFrame({"distance":df[col], name:df[distance_value_col]})
            distances_scenario = pd.concat([distances_scenario, dist])
    
        distances_scenario = distances_scenario.groupby("distance")[name].mean().to_frame()
    
        distances = pd.merge(distances, distances_scenario, left_index = True, right_index=True, how = "outer")
        
    distances = distances.reindex(index = ["0-1km","1-2km","2-5km","5-10km","10-20km","20-50km",">50km"])
    
    # plot
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(1,1,1)
    
    distances.plot(kind = "bar", title = "", figsize = (8,8),stacked = False, ax = ax1)
    ax1.set_ylabel("durchschnittliche Wegedistanz in km")
    ax1.set_xlabel("Distanzgruppe")
    
    for tick in ax1.get_xticklabels():
        tick.set_rotation(90)
    
    plt.rcParams.update({'font.size': 16})
    plt.savefig(fname = savePath, bbox_inches = "tight", format = "pdf")
    
    return
        
# %%% BEVOELKERUNGSPYRAMIDE
def getBevoelkerungspyramidePlot(scenarios, savePath = "./output/bevoelkerungspyramide.pdf", municipal = None):
    mun = municipal
    
    # prepare data
    df = scenarios[0].copy()
    
    # filter for municipal
    if municipal != None:
        mun = int(mun)
        df = df[df.code == mun]
    
    df.name = scenarios[0].name
    
    df = df.groupby("age")["sex"].value_counts().to_frame().rename(columns = {"count" : df.name}).reset_index()
    
    
    for scenario in scenarios[1:]:
        df_age = scenario
        
        if municipal != None:
            mun = int(mun)
            df_age = df_age[df_age.code == mun]
            
        df_age.name = scenario.name
        
        df_age = df_age.groupby("age")["sex"].value_counts().to_frame().rename(columns = {"count" : df_age.name}).reset_index()
        
        df = pd.merge(df, df_age, on = ["age", "sex"])
    
    df = {"df_m" : df[df.sex == "m"],
          "df_f" : df[df.sex == "f"]}
    
    for key in df.keys():
        df[key] = df[key].set_index("age")
        df[key] = df[key].reindex(["unter 10 Jahre","10 bis unter 15 Jahre",
                         "15 bis unter 18 Jahre","18 bis unter 20 Jahre","20 bis unter 25 Jahre","25 bis unter 30 Jahre",
                         "30 bis unter 35 Jahre","35 bis unter 40 Jahre","40 bis unter 45 Jahre","45 bis unter 50 Jahre",
                         "50 bis unter 55 Jahre","55 bis unter 60 Jahre","60 bis unter 65 Jahre","65 Jahre und mehr"])  
       
    # plot
    #define plot parameters
    fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(9, 6))
    
    #specify background color and plot title
    #fig.patch.set_facecolor('xkcd:light grey')
    #plt.figtext(.5,.9,"Bevölkerungspyramide", fontsize=15, ha='center')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.03, hspace=None)
        
    #define male and female bars
    df["df_m"].plot(kind = "barh", stacked = False, ax = axes[0], orientation = "horizontal")
    df["df_f"].plot(kind = "barh", stacked = False, ax = axes[1], orientation = "horizontal")
    
    axes[0].invert_xaxis()
    axes[0].set(title='Männlich')
    axes[1].set(title='Weiblich')
    axes[0].grid()
    axes[1].grid()
    axes[0].set_xticks(axes[0].get_xticks()[::2])
    axes[1].set_xticks(axes[0].get_xticks())
    axes[0].set_ylabel("Alter")
    fig.supxlabel('Anzahl')
        
    
    return fig, axes
 
def plotBevoelkerungspyramide(scenarios, savePath = "./output/bevoelkerungspyramide.pdf", municipal = None):
    
    fig, axes = getBevoelkerungspyramidePlot(scenarios, savePath, municipal)
    
    plt.rcParams.update({'font.size': 12})
    plt.savefig(fname = savePath, bbox_inches = "tight", format = "pdf")

# %%% BEVOELKERUNGSPYRAMIDE PROZENTUAL    
def getProzentualBevoelkerungspyramidePlot(scenarios, savePath = "./output/bevoelkerungspyramide.pdf", municipal = None):
    mun = municipal
    
    # prepare data
    df = scenarios[0].copy()
    
    # filter for municipal
    if municipal != None:
        mun = int(mun)
        df = df[df.code == mun]
    
    df.name = scenarios[0].name
    
    df = (df.groupby("age")["sex"].value_counts()/len(df)).to_frame().rename(columns = {"count" : df.name}).reset_index()

    for scenario in scenarios[1:]:
        df_age = scenario
        
        if municipal != None:
            mun = int(mun)
            df_age = df_age[df_age.code == mun]
            
        df_age.name = scenario.name
        
        df_age = (df_age.groupby("age")["sex"].value_counts()/len(df_age)).to_frame().rename(columns = {"count" : df_age.name}).reset_index()
        
        df = pd.merge(df, df_age, on = ["age", "sex"], how = "outer")

    df = {"df_m" : df[df.sex == "m"],
          "df_f" : df[df.sex == "f"]}
    
    for key in df.keys():
        df[key] = df[key].set_index("age")
        df[key] = df[key].reindex(["unter 10 Jahre","10 bis unter 15 Jahre",
                         "15 bis unter 18 Jahre","18 bis unter 20 Jahre","20 bis unter 25 Jahre","25 bis unter 30 Jahre",
                         "30 bis unter 35 Jahre","35 bis unter 40 Jahre","40 bis unter 45 Jahre","45 bis unter 50 Jahre",
                         "50 bis unter 55 Jahre","55 bis unter 60 Jahre","60 bis unter 65 Jahre","65 Jahre und mehr"])  
        df[key] = df[key] * 100
       
    # plot
    #define plot parameters
    fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(9, 6))
    
    #specify background color and plot title
    #fig.patch.set_facecolor('xkcd:light grey')
    #plt.figtext(.5,.9,"Bevölkerungspyramide", fontsize=15, ha='center')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.03, hspace=None)
        
    #define male and female bars
    df["df_m"].plot(kind = "barh", stacked = False, ax = axes[0], orientation = "horizontal")
    df["df_f"].plot(kind = "barh", stacked = False, ax = axes[1], orientation = "horizontal")
    
    axes[0].invert_xaxis()
    axes[0].set(title='Männlich')
    axes[1].set(title='Weiblich')
    axes[0].grid()
    axes[1].grid()
    axes[1].set_xticks(axes[0].get_xticks())
    axes[0].set_ylabel("Alter")
    axes[0].legend(loc = "center left")
    axes[1].get_legend().remove()
    
    fig.supxlabel('Anteil [%]')
    
    #plt.savefig(fname = savePath, bbox_inches = "tight", format = "pdf")
    
    return fig, axes
 
def plotProzentualBevoelkerungspyramide(scenarios, savePath = "./output/bevoelkerungspyramide.pdf", municipal = None):
    
    fig, axes = getProzentualBevoelkerungspyramidePlot(scenarios, savePath, municipal)
    plt.rcParams.update({'font.size': 12})
    plt.savefig(fname = savePath, bbox_inches = "tight", format = "pdf")
   
    
# %%% AGE DISTRIBUTION 3 GROUPS
def getBevoelkerung3Groups(scenarios, savePath = "./output/bevoelkerungs_3Gruppen.pdf"):
    
    # data preparation    
    age_dict = {"unter 10 Jahre":"unter 18 Jahre", 
                "10 bis unter 15 Jahre":"unter 18 Jahre", 
                "15 bis unter 18 Jahre":"unter 18 Jahre",
                "18 bis unter 20 Jahre":"18 bis unter 65 Jahre",
                "20 bis unter 25 Jahre":"18 bis unter 65 Jahre",
                "25 bis unter 30 Jahre":"18 bis unter 65 Jahre",
                "30 bis unter 35 Jahre":"18 bis unter 65 Jahre",
                "35 bis unter 40 Jahre":"18 bis unter 65 Jahre",
                "40 bis unter 45 Jahre":"18 bis unter 65 Jahre",
                "45 bis unter 50 Jahre":"18 bis unter 65 Jahre",
                "50 bis unter 55 Jahre":"18 bis unter 65 Jahre",
                "55 bis unter 60 Jahre":"18 bis unter 65 Jahre",
                "60 bis unter 65 Jahre":"18 bis unter 65 Jahre",
                "65 Jahre und mehr":"65 Jahre und mehr"}
    
    df = scenarios[0].copy()
    name = scenarios[0].name
    
    df["age"] = df["age"].apply(lambda x: age_dict[x])
    
    df = df.value_counts("age").reset_index().rename(columns = {"count":name}).set_index("age").reindex(index = ["unter 18 Jahre","18 bis unter 65 Jahre", "65 Jahre und mehr"])
    
    for scenario in scenarios[1:]:
        df_age = scenario.copy()
        name = scenario.name
        
        df_age["age"] = df_age["age"].apply(lambda x: age_dict[x])
        df_age = df_age.value_counts("age").reset_index().rename(columns = {"count":name}).set_index("age")
        
        df = pd.merge(df, df_age, how = "left", left_index=True, right_index=True)
     
    # plot
    #define plot parameters
    # plot
    fig = plt.figure(figsize=(10, 20))
    ax1 = fig.add_subplot(1,1,1)
    
    df.plot(kind = "bar", title = "", figsize = (14,8),stacked = False, ax = ax1)
    ax1.set_ylabel("Bevölkerung")
    ax1.set_xlabel("Altersgruppe")
    
    for tick in ax1.get_xticklabels():
        tick.set_rotation(45)
    
    plt.rcParams.update({'font.size': 16})
    plt.savefig(fname = savePath, bbox_inches = "tight", format = "pdf")
 
# %%% TOTAL DISTANCE PER AGE
def plotTotalDistancePerAge(scenarios, savePath = "./output/totalDistancePerAge.pdf"):
    
    df = scenarios[0]
    df.name = scenarios[0].name
    df = df.groupby("age")["totalDistance_value"].sum().to_frame().reset_index().rename(columns = {"totalDistance_value" : df.name})
    
    for scenario in scenarios[1:]:
        df_age = scenario
        df_age.name = scenario.name
        
        df_age = df_age.groupby("age")["totalDistance_value"].sum().to_frame().reset_index().rename(columns = {"totalDistance_value" : df_age.name})
        
        df = pd.merge(df, df_age, on = ["age"])
    
    df = df.set_index("age")
    df = df.reindex(["unter 10 Jahre","10 bis unter 15 Jahre",
                     "15 bis unter 18 Jahre","18 bis unter 20 Jahre","20 bis unter 25 Jahre","25 bis unter 30 Jahre",
                     "30 bis unter 35 Jahre","35 bis unter 40 Jahre","40 bis unter 45 Jahre","45 bis unter 50 Jahre",
                     "50 bis unter 55 Jahre","55 bis unter 60 Jahre","60 bis unter 65 Jahre","65 Jahre und mehr"])  
    
    df = df/1000
    
    # plot
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(1,1,1)
    
    df.plot(kind = "bar", title = "Tägliche Gesamtdistanzen der Altersgruppen", figsize = (8,8),stacked = False, ax = ax1)
    
    ax1.set_ylabel("km in 1000")
    ax1.set_xlabel("Alter")
    
    for tick in ax1.get_xticklabels():
        tick.set_rotation(90)
        
    plt.rcParams.update({'font.size': 16})
    plt.savefig(fname = savePath, bbox_inches = "tight", format = "pdf")
    
    return

# %%% TOTAL DISTANCE PER EMPLOYMENT
def plotAverageDistancePerEmployment(scenarios, savePath = "./output/averageDistancePerEmployment.pdf"):
    
    df = scenarios[0]
    name = scenarios[0].name
    df.totalDistance_value = df.totalDistance_value.fillna(0)
    
    df = (df.groupby("employment")["totalDistance_value"].sum() / df.groupby("employment")["totalDistance_value"].count()).to_frame().reset_index().rename(columns = {"totalDistance_value" : name})
    
    for scenario in scenarios[1:]:
        df_emloyment = scenario
        name = scenario.name
        df_emloyment.totalDistance_value = df_emloyment.totalDistance_value.fillna(0)
        
        df_emloyment = (df_emloyment.groupby("employment")["totalDistance_value"].sum() / df_emloyment.groupby("employment")["totalDistance_value"].count()).to_frame().reset_index().rename(columns = {"totalDistance_value" : name})
        
        df = pd.merge(df, df_emloyment, on = ["employment"])
    
    df = df.set_index("employment")
    df = df.reindex(["erwerbstaetig", "erwerbslos", "nichtErwerbsP"])  
        
    # plot
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(1,1,1)
    
    df.plot(kind = "bar", title = "Tägliche durchschnittl. Gesamtdistanz nach Erwerbsstatus", figsize = (8,8),stacked = False, ax = ax1)
    
    ax1.set_ylabel("Kilometer")
    ax1.set_xlabel("Erwerbsstatus")
    
    for tick in ax1.get_xticklabels():
        tick.set_rotation(90)
        
    for rect in ax1.patches:
        height = rect.get_height()
        ax1.annotate(f'{height:.2f}', 
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 Punkte vertikal über dem Balken
                    textcoords="offset points",
                    ha='center')
        
    plt.rcParams.update({'font.size': 16})
    plt.savefig(fname = savePath, bbox_inches = "tight", format = "pdf")
    
    return


# %%% PEOPLE SHARE PER EMPLOYMENT
def plotPeopleSharePerEmployment(scenarios, savePath = "./output/peopleSharePerEmployment.pdf"):
    
    df = scenarios[0]
    name = scenarios[0].name
    
    df = (df.employment.value_counts() * 100 / len(df)).to_frame().rename(columns = {"count" : name})
    
    for scenario in scenarios[1:]:
        df_emloyment = scenario
        name = scenario.name
        
        df_emloyment = (df_emloyment.employment.value_counts() * 100 / len(df_emloyment)).to_frame().rename(columns = {"count" : name})
        
        df = pd.merge(df, df_emloyment, on = ["employment"])
        
    df = round(df, 2)
        
    # plot
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(1,1,1)
    
    df.plot(kind = "bar", title = "Verteilung der Erwerbsstatus", figsize = (8,8),stacked = False, ax = ax1)
    
    
    ax1.set_ylabel("Anteil [%]")
    ax1.set_xlabel("Erwerbsstatus")
    
    for tick in ax1.get_xticklabels():
        tick.set_rotation(0)
    
    for rect in ax1.patches:
        height = rect.get_height()
        ax1.annotate(f'{height:.2f}', 
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 Punkte vertikal über dem Balken
                    textcoords="offset points",
                    ha='center')
         
    plt.rcParams.update({'font.size': 16})
    plt.savefig(fname = savePath, bbox_inches = "tight", format = "pdf")
    
    return
        
# %%% AVG AMOUNT OF LEGS
def plotLegsDistribution(scenarios, savePath = "./output/legDistribution.pdf"):
    df = scenarios[0]
    cols = [x for x in df.columns if "distance" in x and "value" not in x]
    df["legs"] = (df[cols].isnull() == False).sum(axis = 1)
    name = scenarios[0].name
    
    averages = {name : df.legs.sum()/len(df)}
    
    df_mobile = df[df.legs > 0]
    averages_mobile = {name : df_mobile.legs.sum()/len(df_mobile)}
                       
    df = (df["legs"].value_counts() / len(df)).to_frame().rename(columns = {"count" : name}) * 100
       
    for scenario in scenarios[1:]:
        df_scenario = scenario
        cols = [x for x in df_scenario.columns if "distance" in x and "value" not in x]    
        df_scenario["legs"] = (df_scenario[cols].isnull() == False).sum(axis = 1)
        name = scenario.name    
        
        averages[name] = df_scenario.legs.sum()/len(df_scenario)
        
        df_mobile = df_scenario[df_scenario.legs > 0]
        averages_mobile[name] = df_mobile.legs.sum()/len(df_mobile)
        
        df_scenario = (df_scenario["legs"].value_counts() / len(df_scenario)).to_frame().rename(columns = {"count" : name}) * 100
    
        df = pd.merge(df, df_scenario, on = "legs")
    
    df = df.reindex([0,1,2,3,4,5,6,7])
    
    # add average to column name
    for col in df.columns:
        df = df.rename(columns = {col : col + " (avg.: " + str(round(averages[col], 3)) + " (" + str(round(averages_mobile[col],3)) +  "))"})
                                                                                                    
    # plot
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(1,1,1)
    
    df.plot(kind = "bar", title = "Verteilung der Wege", figsize = (9,8),stacked = False, ax = ax1)
    
    
    ax1.set_ylabel("Anteil [%]")
    ax1.set_xlabel("Anzahl der Wege pro Tag")
    
    for tick in ax1.get_xticklabels():
        tick.set_rotation(0)
        
    plt.rcParams.update({'font.size': 16})
    plt.savefig(fname = savePath, bbox_inches = "tight", format = "pdf")
    
    return

# %%% AMOUNTS OF ACTIVITIES PER START TIME
def activityStartTimesAbsolut(scenarios, savePath = "./output/startTimesAmount.pdf"):
    
    df = scenarios[0]
    name = scenarios[0].name
    cols = [x for x in df.columns if "startTimeOfActivity" in x]
    
    startTimes = pd.DataFrame()
    
    for col in cols:   
        startTimes = pd.concat([startTimes, df[col].to_frame().rename(columns = {col : "startTime"})])
    
    startTimes = startTimes[startTimes["startTime"].astype(str) != "nan"]
    startTimesDistribution = (startTimes["startTime"].value_counts()/1000000).to_frame().rename(columns = {"count" : name}).reindex(index = ["0-6Uhr", "6-10Uhr", "10-14Uhr", "14-18Uhr", "18-22Uhr", ">22Uhr"])
    
    for df in scenarios[1:]:
        name = df.name
        cols = [x for x in df.columns if "startTimeOfActivity" in x]
    
        startTimes = pd.DataFrame()
    
        for col in cols:   
            startTimes = pd.concat([startTimes, df[col].to_frame().rename(columns = {col : "startTime"})])
    
        startTimes = startTimes[startTimes["startTime"].astype(str) != "nan"]
        startTimesDistribution_scenario = (startTimes["startTime"].value_counts()/1000000).to_frame().rename(columns = {"count" : name}).reindex(index = ["0-6Uhr", "6-10Uhr", "10-14Uhr", "14-18Uhr", "18-22Uhr", ">22Uhr"])
        startTimesDistribution = pd.merge(startTimesDistribution, startTimesDistribution_scenario, left_index=True, right_index=True)    
    

    
    # plot
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(1,1,1)
    
    startTimesDistribution.plot(kind = "bar", title = "Anzahl gestarteter Aktivitäten je Uhrzeit", figsize = (8,8),stacked = False, ax = ax1)
    
    ax1.set_ylabel("Anzahl gestarteter Aktivitäten in Millionen")
    ax1.set_xlabel("Startzeit")
    
    for tick in ax1.get_xticklabels():
        tick.set_rotation(45)
        
    plt.rcParams.update({'font.size': 16})
    plt.savefig(fname = savePath, bbox_inches = "tight", format = "pdf")
    
    return

# %%% AMOUNTS OF WORK/EDUCATION ACTIVITIES PER START TIME
def workEducationStartTimesAbsolut(scenarios, savePath = "./output/startTimesAmountWorkEducation.pdf"):
    
    df = scenarios[0]
    name = scenarios[0].name
    cols = [x for x in df.columns if "startTimeOfActivity" in x]
    
    startTimesWork = pd.DataFrame()
    startTimesEducation = pd.DataFrame()

    for col in cols:
        # only start times for work and education_school activities
        activity = "activity" + col[-1]

        df_work = df[df[activity] == "work"]
        df_education = df[df[activity] == "education_school"]
        
        # concat all start times in one column
        startTimesWork = pd.concat([startTimesWork, df_work[col].to_frame().rename(columns = {col : "startTime"})])
        startTimesEducation = pd.concat([startTimesEducation, df_education[col].to_frame().rename(columns = {col : "startTime"})])
    
    # calculate the distribuion of the start times
    startTimesWorkDistribution = (startTimesWork["startTime"].value_counts()/1000).to_frame().rename(columns = {"count" : name}).reindex(index = ["0-6Uhr", "6-10Uhr", "10-14Uhr", "14-18Uhr", "18-22Uhr", ">22Uhr"])
    startTimesEducationDistribution = (startTimesEducation["startTime"].value_counts()/1000).to_frame().rename(columns = {"count" : name}).reindex(index = ["0-6Uhr", "6-10Uhr", "10-14Uhr", "14-18Uhr", "18-22Uhr", ">22Uhr"])

    # do the same for the other scenarios and merge them together
    for df in scenarios[1:]:
        name = df.name
        cols = [x for x in df.columns if "startTimeOfActivity" in x]
    
        startTimesWork = pd.DataFrame()
        startTimesEducation = pd.DataFrame()

        for col in cols:
            activity = "activity" + col[-1]

            df_work = df[df[activity] == "work"]
            df_education = df[df[activity] == "education_school"]
            
            startTimesWork = pd.concat([startTimesWork, df_work[col].to_frame().rename(columns = {col : "startTime"})])
            startTimesEducation = pd.concat([startTimesEducation, df_education[col].to_frame().rename(columns = {col : "startTime"})])
             
        startTimesWorkDistribution_scenario = (startTimesWork["startTime"].value_counts()/1000).to_frame().rename(columns = {"count" : name}).reindex(index = ["0-6Uhr", "6-10Uhr", "10-14Uhr", "14-18Uhr", "18-22Uhr", ">22Uhr"])
        startTimesEducationDistribution_scenario = (startTimesEducation["startTime"].value_counts()/1000).to_frame().rename(columns = {"count" : name}).reindex(index = ["0-6Uhr", "6-10Uhr", "10-14Uhr", "14-18Uhr", "18-22Uhr", ">22Uhr"])

        
        startTimesWorkDistribution = pd.merge(startTimesWorkDistribution, startTimesWorkDistribution_scenario, left_index=True, right_index=True)    
        startTimesEducationDistribution = pd.merge(startTimesEducationDistribution, startTimesEducationDistribution_scenario, left_index=True, right_index=True)    


    
    # plot1
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1,1,1)
    
    startTimesWorkDistribution.plot(kind = "bar", title = "gestartete work Aktivitäten", figsize = (10,5),stacked = False, ax = ax1)

    ax1.set_ylabel("gestartete Aktivitäten in Tausend")
    ax1.set_xlabel("Startzeit")
           
    
    for tick in ax1.get_xticklabels():
        tick.set_rotation(45)
        
    plt.rcParams.update({'font.size': 16})
    plt.savefig(fname = savePath + "_work.pdf", bbox_inches = "tight", format = "pdf")
    
    # plot2
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1,1,1)
    
    startTimesEducationDistribution.plot(kind = "bar", title = "gestartete education-school Aktivitäten", figsize = (10,5),stacked = False, ax = ax1)

    ax1.set_ylabel("gestartete Aktivitäten in Tausend")
    ax1.set_xlabel("Startzeit")    
    
    for tick in ax1.get_xticklabels():
        tick.set_rotation(45)
        
    plt.rcParams.update({'font.size': 16})
    plt.savefig(fname = savePath + "_eduSchool.pdf", bbox_inches = "tight", format = "pdf")
    
    return
# %%% DISTRIBUTION OF ACTIVITIES PER START TIME

def activityStartTimeDistribution(scenarios, savePath = "./output/startTimeDistribution.pdf"):
    
    df = scenarios[0]
    name = scenarios[0].name
    cols = [x for x in df.columns if "startTimeOfActivity" in x]
    
    startTimes = pd.DataFrame()
    
    for col in cols:   
        startTimes = pd.concat([startTimes, df[col].to_frame().rename(columns = {col : "startTime"})])
    
    startTimes = startTimes[startTimes["startTime"].astype(str) != "nan"]
    startTimesDistribution = (startTimes["startTime"].value_counts()*100/len(startTimes)).to_frame().rename(columns = {"count" : name}).reindex(index = ["0-6Uhr", "6-10Uhr", "10-14Uhr", "14-18Uhr", "18-22Uhr", ">22Uhr"])
    
    for df in scenarios[1:]:
        name = df.name
        cols = [x for x in df.columns if "startTimeOfActivity" in x]
    
        startTimes = pd.DataFrame()
    
        for col in cols:   
            startTimes = pd.concat([startTimes, df[col].to_frame().rename(columns = {col : "startTime"})])
    
        startTimes = startTimes[startTimes["startTime"].astype(str) != "nan"]
        startTimesDistribution_scenario = (startTimes["startTime"].value_counts()*100/len(startTimes)).to_frame().rename(columns = {"count" : name}).reindex(index = ["0-6Uhr", "6-10Uhr", "10-14Uhr", "14-18Uhr", "18-22Uhr", ">22Uhr"])
        startTimesDistribution = pd.merge(startTimesDistribution, startTimesDistribution_scenario, left_index=True, right_index=True)    
    

    
    # plot
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(1,1,1)
    
    startTimesDistribution.plot(kind = "bar", title = "Verteilung gestarteter Aktivitäten je Uhrzeit", figsize = (8,8),stacked = False, ax = ax1)
    
    ax1.set_ylabel("Anteil gestarteter Aktivitäten")
    ax1.set_xlabel("Startzeit")
    
    for tick in ax1.get_xticklabels():
        tick.set_rotation(0)
        
    plt.rcParams.update({'font.size': 16})
    plt.savefig(fname = savePath, bbox_inches = "tight", format = "pdf")
    
    return
# %%% DISTRIBUTION OF ACTIVITY START TIME PER AGE 
def activityStartTimeDistributionPerAge(scenarios, savePath = "./output/startTimesDistributionPerAge.pdf"):
    """
    Plots the startTime distribution per age group of different scenarios
    scenarios = List of DataFrames, which contain the Output Data of scenarios in string format.
    DataFrames need a name (e.g. "lowScenario"), a totalDistance columns and an age column
    """
    
    #scenarios = [scenarios["basecase"], scenarios["middleScenario_2030"], scenarios["middleScenario_2050"]]
    # create dataframe with distributions per scenario
    df = scenarios[0].copy()
    name = scenarios[0].name
    
    # dict for share of age group
    populationShare = {}
    populationShare[name] = (df.value_counts("age")*100/len(df)).to_dict()
    
    name = scenarios[0].name
    cols = [x for x in df.columns if "startTimeOfActivity" in x]
    
    startTimes = pd.DataFrame()
    
    for col in cols:   
        startTimes = pd.concat([startTimes, df[["age", col]].rename(columns = {col : "startTime"})])
    
    startTimes = startTimes[startTimes["startTime"].astype(str) != "nan"]
    startTimesDistribution = (startTimes.groupby("age")["startTime"].value_counts()*100/startTimes.groupby("age")["startTime"].count()).to_frame().reset_index().rename(columns = {0 : name})
    
    for scenario in scenarios[1:]:
        df = scenario.copy()
        name = scenario.name
    
        populationShare[name] = (df.value_counts("age")*100/len(df)).to_dict()
    
        cols = [x for x in df.columns if "startTimeOfActivity" in x]
    
        startTimes = pd.DataFrame()
    
        for col in cols:   
            startTimes = pd.concat([startTimes, df[["age", col]].rename(columns = {col : "startTime"})])
    
        startTimes = startTimes[startTimes["startTime"].astype(str) != "nan"]
        startTimesDistribution_scenario = (startTimes.groupby("age")["startTime"].value_counts()*100/startTimes.groupby("age")["startTime"].count()).to_frame().reset_index().rename(columns = {0 : name})
    
        startTimesDistribution = pd.merge(startTimesDistribution, startTimesDistribution_scenario, on = ["age", "startTime"])
    
    # convert to dictionary
    age_order = ["unter 10 Jahre","10 bis unter 15 Jahre", 
                 "15 bis unter 18 Jahre","18 bis unter 20 Jahre","20 bis unter 25 Jahre","25 bis unter 30 Jahre",
                 "30 bis unter 35 Jahre","35 bis unter 40 Jahre","40 bis unter 45 Jahre","45 bis unter 50 Jahre",
                 "50 bis unter 55 Jahre","55 bis unter 60 Jahre","60 bis unter 65 Jahre","65 Jahre und mehr"]
    
    df_unsorted = dict(tuple(startTimesDistribution.groupby("age")))
    age_order = [i for i in age_order if i in df_unsorted.keys()]
    
    df = {k: df_unsorted[k] for k in age_order}
    
    
    # plot
    subplotRows = math.ceil(len(df.keys())/2)
    
    idx = list(range(subplotRows))*2
    i = 0
    j = 0
    count = 0
    
    fig, axes = plt.subplots(subplotRows, 2, figsize=(16,subplotRows*5))
    
    for key in df.keys():
        
        # Anteile der Altersgruppe je Scenario
        shares = {}
        for keyPop in populationShare.keys():
            shares[keyPop] = round(populationShare[keyPop][key],2)
        
        df_age = df[key]
        df_age = df_age.drop("age", axis = 1)
        df_age.index = df_age.startTime
        df_age = df_age.reindex(index = ["0-6Uhr", "6-10Uhr", "10-14Uhr", "14-18Uhr", "18-22Uhr", ">22Uhr"])
        
        # Spalten umbenennen, so dass Share drin ist
        for col in df_age.columns:
            if col != "startTime":
                df_age = df_age.rename(columns = {col : col + " (" + str(shares[col]) + "%)"}) 
         
        # Plot
        df_age.plot(kind = "bar", title = key,stacked = False, ax = axes[idx[i]][j], alpha = 0.8)
        axes[idx[i]][j].set_ylabel("Anteil [%]")
        axes[idx[i]][j].set_xlabel("Startzeit")
        for tick in axes[idx[i]][j].get_xticklabels():
            tick.set_rotation(45)
        
        if count % 2 == 0:
            j = 1
        else:
            j = 0
            i = i + 1
        
        count = count + 1
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.7)
    
    plt.suptitle("Verteilung der Startzeiten je Altersgruppe", y = 0.9)  
    
    plt.rcParams.update({'font.size': 16})
    plt.savefig(fname = savePath, bbox_inches = "tight", format = "pdf")

    return

# %%% ACTIVITY CHAIN DISTRIBUTION
def activityChainDistribution(scenarios, savePath = "./output/activityChainDistribution.pdf"):

    # calculation of the activity chain shares
    df = scenarios[0]
    name = df.name
    cols = [x for x in df.columns if x.startswith("activity")]
    df = df[cols].astype(str)
    
    df["act_chain"] = "home / " + df["activity1"].str.cat(df.loc[:,"activity2":"activity7"].values,sep=" / ")
    df.act_chain = df.act_chain.str.replace(" / nan", "")
    
    actChainShares = (df.act_chain.value_counts()*100/len(df)).reset_index().rename(columns = {"count" : name})
    for df in scenarios[1:]: 
        name = df.name
        cols = [x for x in df.columns if x.startswith("activity")]
        df = df[cols].astype(str)
        
        df["act_chain"] = "home / " + df["activity1"].str.cat(df.loc[:,"activity2":"activity7"].values,sep=" / ")
        df.act_chain = df.act_chain.str.replace(" / nan", "")
        
        shares = (df.act_chain.value_counts()*100/len(df)).reset_index().rename(columns = {"count" : name})
        
        actChainShares = pd.merge(actChainShares, shares, how = "outer", on = "act_chain")
    
    actChainShares = actChainShares.set_index("act_chain")
    
    
    if len(actChainShares.columns)<=2:
        activity_WassersteinDistance = wasserstein_distance(actChainShares.iloc[:50,0], actChainShares.iloc[:50,1])
    else:
        activity_WassersteinDistance = 0
        
    # plot
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(1,1,1)
    
    actChainShares.iloc[:10,:].plot(kind = "bar", title = "Aktivitätenketten", figsize = (8,8),stacked = False, ax = ax1)
    
    plt.text(0.97, 0.81, f"Wasserstein-Distanz: {activity_WassersteinDistance:.3f}",ha = "right", transform=ax1.transAxes)
    
    ax1.set_ylabel("Anteil [%]")
    ax1.set_xlabel("Aktivitätenkette")
    
    for tick in ax1.get_xticklabels():
        tick.set_rotation(90)
        
    plt.rcParams.update({'font.size': 16})
    plt.savefig(fname = savePath, bbox_inches = "tight", format = "pdf")
    
    return

# %%% DISTANCE CHAIN DISTRIBUTION
def distanceChainDistribution(scenarios, savePath = "./output/distanceChainDistribution.pdf"):

    # calculation of the activity chain shares
    df = scenarios[0]
    name = df.name
    cols = [x for x in df.columns if x.startswith("distance")]
    df = df[cols].astype(str)
    
    df["dist_chain"] = df["distance1"].str.cat(df.loc[:,"distance2":"distance7"].values,sep=" / ")
    df.dist_chain = df.dist_chain.str.replace(" / nan", "")
    
    distChainShares = (df.dist_chain.value_counts()*100/len(df)).reset_index().rename(columns = {"count" : name})
    
    for df in scenarios[1:]: 
        name = df.name
        cols = [x for x in df.columns if x.startswith("distance")]
        df = df[cols].astype(str)
        
        df["dist_chain"] = df["distance1"].str.cat(df.loc[:,"distance2":"distance7"].values,sep=" / ")
        df.dist_chain = df.dist_chain.str.replace(" / nan", "")
        
        shares = (df.dist_chain.value_counts()*100/len(df)).reset_index().rename(columns = {"count" : name})
        
        distChainShares = pd.merge(distChainShares, shares, how = "outer", on = "dist_chain")
    
    distChainShares = distChainShares.set_index("dist_chain")
    
    
    if len(distChainShares.columns)<=2:
        distance_WassersteinDistance = wasserstein_distance(distChainShares.iloc[:50,0], distChainShares.iloc[:50,1])
    else:
        distance_WassersteinDistance = 0
        
    # plot
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(1,1,1)
    
    distChainShares.iloc[:10,:].plot(kind = "bar", title = "Distanzketten", figsize = (8,8),stacked = False, ax = ax1)
    
    plt.text(0.97, 0.81, f"Wasserstein-Distanz: {distance_WassersteinDistance:.3f}",ha = "right", transform=ax1.transAxes)
    
    ax1.set_ylabel("Anteil [%]")
    ax1.set_xlabel("Distanzkette")
    
    for tick in ax1.get_xticklabels():
        tick.set_rotation(90)
        
    plt.rcParams.update({'font.size': 16})
    plt.savefig(fname = savePath, bbox_inches = "tight", format = "pdf")
    
    return


# %%% MISSING/NEW CHAINS
def missingChains(scenarios, savePath = "./output/missingChains.pdf"):
    # calculation of the activity chain shares
    df = scenarios[0]
    name = df.name
    cols = [x for x in df.columns if x.startswith("activity")]
    df = df[cols].astype(str)
    
    df["act_chain"] = "home / " + df["activity1"].str.cat(df.loc[:,"activity2":"activity7"].values,sep=" / ")
    df.act_chain = df.act_chain.str.replace(" / nan", "")
    
    actChainShares = (df.act_chain.value_counts()*100/len(df)).reset_index().rename(columns = {"count" : name})
    for df in scenarios[1:]: 
        name = df.name
        cols = [x for x in df.columns if x.startswith("activity")]
        df = df[cols].astype(str)
        
        df["act_chain"] = "home / " + df["activity1"].str.cat(df.loc[:,"activity2":"activity7"].values,sep=" / ")
        df.act_chain = df.act_chain.str.replace(" / nan", "")
        
        shares = (df.act_chain.value_counts()*100/len(df)).reset_index().rename(columns = {"count" : name})
        
        actChainShares = pd.merge(actChainShares, shares, how = "outer", on = "act_chain")
    
    actChainShares = actChainShares.set_index("act_chain").fillna(0)

    newChainsShare = sum(actChainShares.iloc[:,1][actChainShares.iloc[:,0] == 0])
    missingChainsShare = sum(actChainShares.iloc[:,0][actChainShares.iloc[:,1] == 0])
    chains = pd.DataFrame(data = [[newChainsShare, 100-newChainsShare], [missingChainsShare, 100-missingChainsShare]], columns = ["neu/fehlend","bestehend"],index = ["neue Ketten", "fehlende Ketten"])

    # plot
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(1,1,1)
    
    chains.plot(kind = "bar", title = "Neue/fehlende Aktivitätenketten", figsize = (8,8),stacked = True, ax = ax1)
    
    for rect in ax1.patches:
        height = rect.get_height()
        ax1.annotate(f'{height:.2f}%', 
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 Punkte vertikal über dem Balken
                    textcoords="offset points",
                    ha='center')
        
    plt.rcParams.update({'font.size': 16})
    plt.savefig(fname = savePath, bbox_inches = "tight", format = "pdf")
        
    return
# %%% ACTIVITY DURATIONS PER ACTIVITY
def plotDurationsPerActivity(scenarios, savePath = "./output/durationsPerActivity.pdf"):
    """
   
    """

    # create dataframe with distributions per scenario
    df = scenarios[0]
    name = scenarios[0].name
    
    
    actCols = [x for x in df if x.startswith("activity")]
    
    activityDurations = pd.DataFrame()
    
    # all activities and regarding durations below each other
    for act in actCols:
            durCol = "durationOfActivity" + act[-1]   
            activityDurations = pd.concat([activityDurations, df[[act, durCol]].rename(columns = {act : "activity", durCol : "duration"})])
    
    # leave out unused activities
    activityDurations = activityDurations[activityDurations.activity.astype(str) != "nan"]
    
    # dict for share of activity
    populationShare = {}
    populationShare[name] = round((activityDurations.value_counts("activity")*100/len(activityDurations)), 2).to_dict()
    
    # get the distribution of durations per activity
    durationsDist = (activityDurations.groupby("activity")["duration"].value_counts() *100 / activityDurations.groupby("activity")["duration"].count()).to_frame().reset_index().rename(columns={0:name})
    
    # do the same for the other scenarios and merge together      
    for scenario in scenarios[1:]:
            df = scenario
            name = scenario.name
            
            actCols = [x for x in df if x.startswith("activity")]
    
            activityDurations = pd.DataFrame()
            for act in actCols:
                    durCol = "durationOfActivity" + act[-1]   
                    activityDurations = pd.concat([activityDurations, df[[act, durCol]].rename(columns = {act : "activity", durCol : "duration"})])
                    
            activityDurations = activityDurations[activityDurations.activity.astype(str) != "nan"]
            
            populationShare[name] = round((activityDurations.value_counts("activity")*100/len(activityDurations)), 2).to_dict()
    
            durationsDist_scenario = (activityDurations.groupby("activity")["duration"].value_counts() *100 / activityDurations.groupby("activity")["duration"].count()).to_frame().reset_index().rename(columns={0:name})
            
            durationsDist = pd.merge(durationsDist, durationsDist_scenario, on = ["activity", "duration"], how = "outer")
    
    
    # create dictionary
    dict_unsorted = dict(tuple(durationsDist.groupby("activity")))      
    
    # set order of activities
    dict_order = ["work", "education_school", "education_higher", "shopping", "leisure", "dining", "personal_business", "transport", "home"]
    dict_order = [i for i in dict_order if i in dict_unsorted.keys()]
    
    dict_sorted = {k: dict_unsorted[k] for k in dict_order}
    
    # set order of durations
    duration_order = ["0-5min", "5-15min", "15-30min", "30-60min", "1-2h", "2-3h", "3-5h", "5-7h", "7-8h", "8-9h", "9-12h", ">12h"]
    for key in dict_sorted.keys():
            dict_sorted[key] = dict_sorted[key].drop("activity", axis = 1)
            dict_sorted[key] = dict_sorted[key].set_index("duration")
            dict_sorted[key] = dict_sorted[key].reindex(duration_order)
            dict_sorted[key] = dict_sorted[key].fillna(0)
    
    # Plot
    subplotRows = math.ceil(len(dict_sorted.keys())/2)
    
    idx = list(range(subplotRows))*2
    i = 0
    j = 0
    count = 0
    
    fig, axes = plt.subplots(subplotRows, 2, figsize=(16,subplotRows*5))
    
    for key in dict_sorted.keys():
            
            # Spalten umbenennen, so dass Share drin ist
            for col in dict_sorted[key].columns:
                dict_sorted[key] = dict_sorted[key].rename(columns = {col : col + " (" + str(populationShare[col][key]) + "%)"}) 
         
            # Plot
            dict_sorted[key].plot(kind = "bar", title = key, stacked = False, ax = axes[idx[i]][j], alpha = 0.8)
            axes[idx[i]][j].set_ylabel("Anteil [%]")
            axes[idx[i]][j].set_xlabel("Aktivitätendauer")
            
            for tick in axes[idx[i]][j].get_xticklabels():
                tick.set_rotation(45)
            
            if count % 2 == 0:
                j = 1
            else:
                j = 0
                i = i + 1
            
            count = count + 1
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.7)
    
    plt.suptitle("Verteilung der Aktivitätendauer je Aktivität", y = 0.9)  
    
    plt.rcParams.update({'font.size': 16})
    plt.savefig(fname = savePath, bbox_inches = "tight", format = "pdf")
    
    return
 
# %%% CREATE HEATMAP
# %%%% create population pyramid per municipalty
def getPyramid(scenarios, name1, municipal = None):
    scenarios = scenarios
    df = scenarios[0]
    name = scenarios[0].name
    mun = municipal
    names = []
    
    for scen in scenarios:
        nameScen = scen.name
        names.append(nameScen)
        if mun != None:         
            # code Anpassung fuer kreisfreie Staedt
            scen.code[scen.code < 100000] = scen.code * 1000
            scen = scen[scen.code == int(mun)]    
        
    df = df[df.code == int(mun)]   
    df = df.groupby("age")["sex"].value_counts().to_frame().rename(columns = {"count" : name}).reset_index()
    
    for scenario in scenarios[1:]:
        df_age = scenario
        name = scenario.name
        
        df_age = df_age[df_age.code == int(mun)] 
        df_age = df_age.groupby("age")["sex"].value_counts().to_frame().rename(columns = {"count" : name}).reset_index()
        
        df = pd.merge(df, df_age, on = ["age", "sex"])
    
    df = {"df_m" : df[df.sex == "m"],
          "df_f" : df[df.sex == "f"]}
    
    for key in df.keys():
        df[key] = df[key].set_index("age")
        df[key] = df[key].reindex(["unter 10 Jahre","10 bis unter 15 Jahre",
                         "15 bis unter 18 Jahre","18 bis unter 20 Jahre","20 bis unter 25 Jahre","25 bis unter 30 Jahre",
                         "30 bis unter 35 Jahre","35 bis unter 40 Jahre","40 bis unter 45 Jahre","45 bis unter 50 Jahre",
                         "50 bis unter 55 Jahre","55 bis unter 60 Jahre","60 bis unter 65 Jahre","65 Jahre und mehr"])  
        df[key] = df[key].reindex(df[key].index[::-1])
        df[key] = df[key].reset_index()
        df[key] = pd.melt(df[key], id_vars=["age", "sex", ], value_vars=names, var_name = "Scenario", value_name = "amount")
   
    df_m = df["df_m"].reset_index()
    df_f = df["df_f"].reset_index()
    
    # create altair population chart
    chart_m = alt.Chart(df_m).mark_bar(height=alt.RelativeBandSize(0.5)).encode(
        x=alt.X("amount:Q",
                title='Male',
                sort='descending'),
        
        y=alt.Y('age:N',
                axis=alt.Axis(title='Age Groups'),
                sort=alt.EncodingSortField(field="index:N",  order='ascending')),
        yOffset='Scenario:N',
        color = alt.Color("Scenario:N")
        ).properties(height = 400)
    
    chart_f = alt.Chart(df_f).mark_bar(height=alt.RelativeBandSize(0.5)).encode(
        x=alt.X("amount:Q",
                title='Female',
                sort='ascending'),     
        y=alt.Y('age:N', 
                axis=None,
                sort=alt.EncodingSortField(field="index", order='ascending')),
        yOffset='Scenario:N',
        color = alt.Color("Scenario:N")
        ).properties(height = 400)
    
    chart = alt.concat(chart_m, chart_f, spacing=5)
    #chart.save('./output/chart.html')

    return json.loads(chart.to_json())


# %%%% add layer to map
def toMap(JsonData, scen, scenarios, valueToDisplay = "kmPerPerson"):

    geo_objects = JsonData
    name = scen
    scenarios = scenarios
    
    cms = {}
    # colormap scale totalDistance
    cms["kmPerPerson"] = cm.LinearColormap(["white", "yellow", "green", "blue", "purple", "red"], 
                                            index= [24, 25, 26, 27, 28, 29], vmin=23, vmax=30, caption = "gemittelte zurückgelegte Kilometer pro Person")
        
    cms["totalDistance_diffToBasecase"] = cm.LinearColormap(["#cc7a00", "#ffcc33", "#5cd65c", "#00b300"], 
                                            index= [-15,-5, 5, 15], vmin=-20, vmax=20, caption = "Entwicklung Gesamtdistanz vom Basecase in Prozent")
    
    cms["kmPerPerson_diffToBasecase"] = cm.LinearColormap(["#cc7a00", "#ffcc33", "#5cd65c", "#00b300"], 
                                            index= [-15,-5, 5, 15], vmin=-20, vmax=20, caption = "Entwicklung der Personenkilometer vom Basecase in Prozent")
    
    cms["legAmount_diffToBasecase"] = cm.LinearColormap(["#cc7a00", "#ffcc33", "#5cd65c", "#00b300"], 
                                            index= [-15,-5, 5, 15], vmin=-20, vmax=20, caption = "Entwicklung der Gesamtanzahl der Wege vom Basecase in Prozent")
    
    
    cmToUse = cms[valueToDisplay]
    
    layer = folium.FeatureGroup(name=name,control=False)

    # for every feature (=municipal): create layer and add to feature group
    for feature in geo_objects["features"]:
        temp_geojson = {"features":[feature],"type":"FeatureCollection"}
        
        # temporary layer of municipal
        temp_geojson_layer = folium.GeoJson(temp_geojson,
                                            highlight_function=lambda x: {'weight':3, 'color':'black'},
                                            control=False,
                                            name = name,
                                            style_function = lambda feature: {'fillColor':cmToUse(feature["properties"][valueToDisplay + "_" + name]),
                                                                              'color': 'black',       #border color for the color fills
                                                                              'weight': 1,            #how thick the border has to be
                                                                              'fillOpacity':0.6,
                                                                              'dashArray': '5, 3'},
                                            tooltip = folium.GeoJsonTooltip(['Name', "KilometersStr_"+name, "PopulationStr_"+name, "kmPerPersonStr_"+name, "Erwerbslosenquote"],
                                                                            aliases = ['Name', "Kilometers", "Population", "kmPerPerson", "Erwerbslosenquote (%)"])
                        )
        # add popup with graph to layer
        graph = getPyramid(scenarios, name, feature["properties"]["AGS"])
        popup = folium.Popup(max_width = 1000)
        folium.features.VegaLite(graph, height=400, width=950).add_to(popup)
        popup.add_to(temp_geojson_layer)
        
        # add municipal to scenario layer group
        temp_geojson_layer.add_to(layer)
        
    return layer

# %%%% Create Heatmap
def createHeatmap(compareScenarios, savePath, valueToDisplay = "kmPerPerson"):
    
    # list of the scenarios
    names = []
    scenarios_dict = {}
    for scen in compareScenarios:
        names.append(scen.name)
        scenarios_dict[scen.name] = scen
    
    # read in json Gemeinden 
    #geo_objects = json.load(open("./BB_PLZ_Shapefile/landkreise_simplify200.geojson", encoding="UTF-8"))
    geo_objects = json.load(open("./input_data/BB_PLZ_Shapefile/gemeinden_simplify200.geojson", encoding="UTF-8"))
    
    # Heatmap
    karte = folium.Map(location=[52.5244, 13.4105], zoom_start = 8)
    
    cms = {}
    # colormap scale totalDistance
    cms["kmPerPerson"] = cm.LinearColormap(["white", "yellow", "green", "blue", "purple", "red"], 
                                            index= [24, 25, 26, 27, 28, 29], vmin=23, vmax=30, caption = "gemittelte zurückgelegte Kilometer pro Person")
        
    cms["totalDistance_diffToBasecase"] = cm.LinearColormap(["#cc7a00", "#ffcc33", "#5cd65c", "#00b300"], 
                                            index= [-15,-5, 5, 15], vmin=-20, vmax=20, caption = "Entwicklung Gesamtdistanz vom Basecase in Prozent")
    
    cms["kmPerPerson_diffToBasecase"] = cm.LinearColormap(["#cc7a00", "#ffcc33", "#5cd65c", "#00b300"], 
                                            index= [-15,-5, 5, 15], vmin=-20, vmax=20, caption = "Entwicklung der Personenkilometer vom Basecase in Prozent")
    
    cms["legAmount_diffToBasecase"] = cm.LinearColormap(["#cc7a00", "#ffcc33", "#5cd65c", "#00b300"], 
                                            index= [-15,-5, 5, 15], vmin=-20, vmax=20, caption = "Entwicklung der Gesamtanzahl der Wege vom Basecase in Prozent")
    
    cmToUse = cms[valueToDisplay]

    cmToUse.width = 1200
    cmToUse.add_to(karte)
    
    # basecase scenario for difference 
    if "basecase" in scenarios_dict.keys():
        # total Distance
        baseScen = scenarios_dict["basecase"][["code", "totalDistance_value"]].fillna(0)
        baseScen = baseScen.groupby("code")['totalDistance_value'].agg(['sum','count']).reset_index().rename(columns={"sum":"totalDistance_base", "count":"population_base"})
        baseScen["kmPerPerson_base"] = baseScen.totalDistance_base / baseScen.population_base
        
        # leg amount
        baseLegs = scenarios["basecase"]
        cols = [x for x in baseLegs.columns if x.startswith("distance") and "value" not in x]
        baseLegs["legAmount_base"] = (baseLegs[cols].isnull() == False).sum(axis = 1)
        baseLegs = baseLegs.groupby("code")["legAmount_base"].sum().to_frame()
        baseScen = pd.merge(baseScen, baseLegs, on = "code", how = "left")
        
    # manipulate data and add layer for every scenario
    for scen in compareScenarios:   
        scenario = scen
        name = scen.name
        
        # calculate the employment rate
        emplRate = (scenario.groupby("code")["employment"].value_counts() * 100 / scenario.groupby("code")["employment"].count()).to_frame().reset_index().rename(columns = {0:"Erwerbslosenquote"})
        emplRate = emplRate[emplRate.employment == "erwerbslos"]
        emplRate = emplRate.drop("employment", axis = 1)
        emplRate.Erwerbslosenquote = round(emplRate.Erwerbslosenquote, 2)
        
        # leg amount 
        scenLegs = scenario
        cols = [x for x in scenLegs.columns if x.startswith("distance") and "value" not in x]
        scenLegs["legAmount"] = (scenLegs[cols].isnull() == False).sum(axis = 1)
        scenLegs = scenLegs.groupby("code")["legAmount"].sum().to_frame()
              
        # nur total Distance und Code sind relevant
        scenario = scenario[["code", "totalDistance_value"]].fillna(0)
        
        # je Gemeinde die Gesamtdistanz und Population
        scenario = scenario.groupby("code")['totalDistance_value'].agg(['sum','count']).reset_index().rename(columns={"sum":"totalDistance_value", "count":"population"})
        
        # merge emlpoyment rate
        scenario = pd.merge(scenario, emplRate, on = "code", how = "left")
        
        if "basecase" in scenarios_dict.keys(): 
            scenario = pd.merge(scenario, baseScen, on = "code", how = "left")
        
        scenario = pd.merge(scenario, scenLegs, on = "code", how = "left")
            
        
        # fuer die kreisfreien Staedte die den Code anpassen
        scenario.code[scenario.code < 1000000] = scenario.code * 1000
        
        # avg km per person berechnen
        scenario["kmPerPerson"] = scenario.totalDistance_value / scenario.population
        if "basecase" in scenarios_dict.keys():
            scenario["totalDistance_diffToBasecase"] = round((scenario.totalDistance_value *100 / scenario.totalDistance_base) - 100, 2)
            scenario["population_diffToBasecase"] = round((scenario.population *100 / scenario.population_base) - 100, 2)
            scenario["kmPerPerson_diffToBasecase"] = round((scenario.kmPerPerson *100 / scenario.kmPerPerson_base) - 100, 2)
            scenario["legAmount_diffToBasecase"] = round((scenario.legAmount *100 / scenario.legAmount_base) - 100, 2)
            
            scenario = scenario.drop(["totalDistance_base", "population_base", "kmPerPerson_base", "legAmount_base"], axis = 1)

        else: 
            scenario["totalDistance_diffToBasecase"] = 0.00
            scenario["population_diffToBasecase"] = 0.00
            scenario["kmPerPerson_diffToBasecase"] = 0.00
            scenario["legAmount_diffToBasecase"] = 0.00
        
        # add Berlin und fehlende Gemeinden without kilometers
        addData = pd.DataFrame([[11000000, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [12073032, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [12073386, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [12073505, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [12073603, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
                               columns=scenario.columns)
        
        scenario = pd.concat([scenario, addData]).set_index("code")
        scenario.index = scenario.index.astype(str)
        scenario.totalDistance_value = round(scenario.totalDistance_value,2)
        scenario.kmPerPerson = round(scenario.kmPerPerson,2)
        scenario.population = scenario.population.astype(int)
        
        # Alle Einträge des Features ändern
        for item in geo_objects['features']:
            item['properties']['Kilometers_'+name] = scenario.loc[item["properties"]["AGS"], 'totalDistance_value']
            item['properties']['KilometersStr_'+name] = "{:,.2f}".format(scenario.loc[item["properties"]["AGS"], 'totalDistance_value']).replace(","," ") + " (" + "{:+.2f}".format(scenario.loc[item["properties"]["AGS"], 'totalDistance_diffToBasecase']) + "%)"
            item['properties']['Name'] = item['properties']['GEN']
            item['properties']['Population_'+name] = int(scenario.loc[item["properties"]["AGS"], 'population'])
            item['properties']['PopulationStr_'+name] = "{:,d}".format(int(scenario.loc[item["properties"]["AGS"], 'population'])).replace(","," ") + " (" + "{:+.2f}".format(scenario.loc[item["properties"]["AGS"], 'population_diffToBasecase']) + "%)"
            item['properties']['kmPerPerson_'+name] = scenario.loc[item["properties"]["AGS"], 'kmPerPerson']
            item['properties']['kmPerPersonStr_'+name] = str(scenario.loc[item["properties"]["AGS"], 'kmPerPerson']) + " (" + "{:+.2f}".format(scenario.loc[item["properties"]["AGS"], 'kmPerPerson_diffToBasecase']) + "%)"
            item['properties']['kmPerPerson_diffToBasecase_'+name] = scenario.loc[item["properties"]["AGS"], 'kmPerPerson_diffToBasecase']
            item['properties']['population_diffToBasecase_'+name] = scenario.loc[item["properties"]["AGS"], 'population_diffToBasecase']
            item['properties']['totalDistance_diffToBasecase_'+name] = scenario.loc[item["properties"]["AGS"], 'totalDistance_diffToBasecase']
            item['properties']['legAmount_diffToBasecase_'+name] = scenario.loc[item["properties"]["AGS"], 'legAmount_diffToBasecase']
            item['properties']['Erwerbslosenquote'] = scenario.loc[item["properties"]["AGS"], 'Erwerbslosenquote']

    # Layer dictionary
    layers = {}
    
    # add for every scenario a layer
    for n in names:
        layers[n] = toMap(geo_objects, n, compareScenarios, valueToDisplay)
        karte.add_child(layers[n])   
    
    # =========== add employment layer
    cm_employment = cm.LinearColormap(["white", "yellow", "green", "blue", "purple", "red"], 
                                         index= [0, 1, 2, 3, 4, 5], vmin=0, vmax=6, caption = "Erwerbslosenquote (%)")
        
 
    layer = folium.FeatureGroup(name="Erwerbslosenquote",control=False)

    # for every feature (=municipal): create layer and add to feature group
    for feature in geo_objects["features"]:
        temp_geojson = {"features":[feature],"type":"FeatureCollection"}
        
        # temporary layer of municipal
        temp_geojson_layer = folium.GeoJson(temp_geojson,
                                            highlight_function=lambda x: {'weight':3, 'color':'black'},
                                            control=False,
                                            name = "Erwerbslosenquote",
                                            style_function = lambda feature: {'fillColor':cm_employment(feature["properties"]["Erwerbslosenquote"]),
                                                                              'color': 'black',       #border color for the color fills
                                                                              'weight': 1,            #how thick the border has to be
                                                                              'fillOpacity':0.6,
                                                                              'dashArray': '5, 3'},
                                            tooltip = folium.GeoJsonTooltip(['Name', "Kilometers_"+name, "Population_"+name, "kmPerPerson_"+name, "Erwerbslosenquote"],
                                                                            aliases = ['Name', "Kilometers", "Population", "kmPerPerson", "Erwerbslosenquote (%)"])
                        )
        # add municipal to scenario layer group
        temp_geojson_layer.add_to(layer)
    
    layers["Erwerbslosenquote"] = layer
    
    karte.add_child(layers["Erwerbslosenquote"])
    cm_employment.add_to(karte)
     # ========
    
    

    # Layer control for scenarios
    GroupedLayerControl(
        groups={'Scenarios': list(layers.values())},
        collapsed=False,
        exclusive_groups = True).add_to(karte)
    
    karte.save(savePath)
 
    #karte.show_in_browser()
    
    return

# %% RUN FUNCTIOS - PLOT SCENARIO COMPARISONS
# %%% SRV ANALYSIS

def runSRV(compareScenarios, ergebnisPfad):
    compareScenarios = compareScenarios #[scenarios["SRV_BB"]]
    folder = "/SRV"
    
    plotBevoelkerungspyramide(compareScenarios, savePath = ergebnisPfad + folder + "/bevoelkerungspyramiden_SRV.pdf")   
    plotTotalDistanceDistributionPerAge(compareScenarios, savePath = ergebnisPfad + folder + "/totalDistanceDistributionPerAge_SRV.pdf")
    plotTotalDistanceDistribution(compareScenarios, savePath = ergebnisPfad + folder + "/totalDistanceDistribution_SRV.pdf")
    plotAvgDistancePerAge(compareScenarios, savePath = ergebnisPfad + folder + "/avgDistancePerAge_SRV.pdf")
    plotLegsDistribution(compareScenarios, savePath = ergebnisPfad + folder + "/legAmountDistribution_SRV.pdf")
    plotLegLengthDistribution(compareScenarios, savePath = ergebnisPfad + folder + "/legLengthDistribution_SRV.pdf")
    activityStartTimeDistributionPerAge(compareScenarios, savePath = ergebnisPfad + folder + "/startTimesPerAge_SRV.pdf")
    plotPeopleSharePerEmployment(compareScenarios, savePath = ergebnisPfad + folder + "/peopleSharePerEmployment_SRV.pdf")
    plotAverageDistancePerEmployment(compareScenarios, savePath = ergebnisPfad + folder + "/distancePerEmployment_SRV.pdf")
    plotTotalDistancePerAge(compareScenarios, savePath = ergebnisPfad + folder + "/totalDistancePerAge_SRV.pdf")
    
    plt.close()

    return

# %%% CROSS VALIDATION 80/20 SPLIT - COMPARISON OF GENERATED DATA WITH ORIGINAL DATA

def runCrossValidation(compareScenarios, ergebnisPfad):
    compareScenarios = compareScenarios #[scenarios["SRV_BB"]]
    folder = "/crossValidation"
    
    plotTotalDistanceDistributionPerAge(compareScenarios, savePath = ergebnisPfad + folder + "/totalDistanceDistributionPerAge_crossValidation.pdf")
    plotTotalDistanceDistribution(compareScenarios, savePath = ergebnisPfad + folder + "/totalDistanceDistribution_crossValidation.pdf")
    plotAvgDistancePerAge(compareScenarios, savePath = ergebnisPfad + folder + "/avgDistancePerAge_crossValidation.pdf")
    plotAvgDistancePerDistanceGroup(compareScenarios, savePath = ergebnisPfad + folder + "/avgDistancePerDistanceGroup_crossValidation.pdf")
    plotLegsDistribution(compareScenarios, savePath = ergebnisPfad + folder + "/legAmountDistribution_crossValidation.pdf")
    plotLegLengthDistribution(compareScenarios, savePath = ergebnisPfad + folder + "/legLengthDistribution_crossValidation.pdf")
    activityStartTimeDistributionPerAge(compareScenarios, savePath = ergebnisPfad + folder + "/startTimesPerAge_crossValidation.pdf")
    activityChainDistribution(compareScenarios, savePath = ergebnisPfad + folder + "/activityChainDistribution_crossValidation.pdf")
    distanceChainDistribution(compareScenarios, savePath = ergebnisPfad + folder + "/distanceChainDistribution_crossValidation.pdf")
    missingChains(compareScenarios, savePath = ergebnisPfad + folder + "/missingChains_crossValidation.pdf")
    plotDurationsPerActivity(compareScenarios, savePath = ergebnisPfad + folder + "/durationsPerActivity_crossValidation.pdf")
    plotAverageDistancePerEmployment(compareScenarios, savePath = ergebnisPfad + folder + "/distancePerEmployment_crossValidation.pdf")
    
    plt.close()

    return

# %%% CROSS VALIDATION 80/20 SPLIT - COMPARISON OF GENERATED DATA WITH ORIGINAL DATA

def runCrossValidationSwiss(compareScenarios, ergebnisPfad):
    compareScenarios = compareScenarios #[scenarios["SRV_BB"]]
    folder = "/crossValidation_swiss"
    
    activityChainDistribution(compareScenarios, savePath = ergebnisPfad + folder + "/activityChainDistribution_crossValidationSwiss.pdf")
    missingChains(compareScenarios, savePath = ergebnisPfad + folder + "/missingChains_crossValidationSwiss.pdf")
    
    plt.close()

    return
# %%% SRV VS BASECASE

def runSrvVsBasecase(compareScenarios, ergebnisPfad):
    compareScenarios = compareScenarios #[scenarios["SRV_BB"], scenarios["basecase"]]
    folder = "/SrvVsBasecase"
    
    plotProzentualBevoelkerungspyramide(compareScenarios, savePath = ergebnisPfad + folder + "/bevoelkerungspyramiden_SrvVsBasecase.pdf")   
    plotTotalDistanceDistributionPerAge(compareScenarios, savePath = ergebnisPfad + folder + "/totalDistanceDistributionPerAge_SrvVsBasecase.pdf")
    plotTotalDistanceDistribution(compareScenarios, savePath = ergebnisPfad + folder + "/totalDistanceDistribution_SrvVsBasecase.pdf")
    plotAvgDistancePerAge(compareScenarios, savePath = ergebnisPfad + folder + "/avgDistancePerAge_SrvVsBasecase.pdf")
    plotAvgDistancePerDistanceGroup(compareScenarios, savePath = ergebnisPfad + folder + "/avgDistancePerDistanceGroup_SrvVsBasecase.pdf")
    plotLegsDistribution(compareScenarios, savePath = ergebnisPfad + folder + "/legAmountDistribution_SrvVsBasecase.pdf")
    plotLegLengthDistribution(compareScenarios, savePath = ergebnisPfad + folder + "/legLengthDistribution_SrvVsBasecase.pdf")
    activityStartTimeDistributionPerAge(compareScenarios, savePath = ergebnisPfad + folder + "/startTimesPerAge_SrvVsBasecase.pdf")
    activityChainDistribution(compareScenarios, savePath = ergebnisPfad + folder + "/activityChainDistribution_SrvVsBasecase.pdf")
    distanceChainDistribution(compareScenarios, savePath = ergebnisPfad + folder + "/distanceChainDistribution_SrvVsBasecase.pdf")
    missingChains(compareScenarios, savePath = ergebnisPfad + folder + "/missingChains_SrvVsBasecase.pdf")
    plotDurationsPerActivity(compareScenarios, savePath = ergebnisPfad + folder + "/durationsPerActivity_SrvVsBasecase.pdf")
    plotPeopleSharePerEmployment(compareScenarios, savePath = ergebnisPfad + folder + "/peopleSharePerEmployment_SrvVsBasecase.pdf")
    plotAverageDistancePerEmployment(compareScenarios, savePath = ergebnisPfad + folder + "/distancePerEmployment_SrvVsBasecase.pdf")
    
    plt.close()

    return

# %%% SCENARIO 2030
def runScenario2030(compareScenarios, ergebnisPfad):
    compareScenarios = compareScenarios #[scenarios["basecase"], scenarios["lowScenario_2030"], scenarios["middleScenario_2030"], scenarios["highScenario_2030"]]
    folder = "/scenario2030"   
    
    plotTotalDistanceDistributionPerAge(compareScenarios, savePath = ergebnisPfad + folder + "/totalDistanceDistributionPerAge_Scenario2030Cases.pdf")
    plotBevoelkerungspyramide(compareScenarios, savePath = ergebnisPfad + folder + "/bevoelkerungspyramiden_scenario2030.pdf")
    plotTotalDistancePerAge(compareScenarios, savePath = ergebnisPfad + folder + "/totalDistancePerAge_scenario2030.pdf")
    plotTotalDistanceDistribution(compareScenarios, savePath = ergebnisPfad + folder + "/totalDistanceDistribution_scenario2030.pdf")
    plotLegsDistribution(compareScenarios, savePath = ergebnisPfad + folder + "/legAmountDistribution_scenario2030.pdf")
    plotLegLengthDistribution(compareScenarios, savePath = ergebnisPfad + folder + "/legLengthDistribution_scenario2030.pdf")
    activityStartTimesAbsolut(compareScenarios, savePath = ergebnisPfad + folder + "/startTimesAbsolut_scenario2030.pdf")
    activityStartTimeDistribution(compareScenarios, savePath = ergebnisPfad + folder + "/startTimeDistribution_scenario2030.pdf")
    workEducationStartTimesAbsolut(compareScenarios, savePath = ergebnisPfad + folder + "/startTimesAmountWorkEducation_scenario2030")

    createHeatmap(compareScenarios, savePath = ergebnisPfad + folder + "/heatmap_scenario2030.html", valueToDisplay="kmPerPerson")
      
    plt.close()
    
    return

# %%% SCENARIO 2050
def runScenario2050(compareScenarios, ergebnisPfad):
    compareScenarios = compareScenarios #[scenarios["basecase"], scenarios["youngScenario_2050"], scenarios["middleScenario_2050"], scenarios["oldScenario_2050"]]
    folder = "/scenario2050" 
    
    plotTotalDistanceDistributionPerAge(compareScenarios, savePath = ergebnisPfad + folder + "/totalDistanceDistributionPerAge_Scenario2050Cases.pdf")
    plotBevoelkerungspyramide(compareScenarios, savePath = ergebnisPfad + folder + "/bevoelkerungspyramiden_scenario2050.pdf")
    plotTotalDistancePerAge(compareScenarios, savePath = ergebnisPfad + folder + "/totalDistancePerAge_scenario2050.pdf")
    plotTotalDistanceDistribution(compareScenarios, savePath = ergebnisPfad + folder + "/totalDistanceDistribution_scenario2050.pdf")
    plotLegsDistribution(compareScenarios, savePath = ergebnisPfad + folder + "/legAmountDistribution_scenario2050.pdf")
    plotLegLengthDistribution(compareScenarios, savePath = ergebnisPfad + folder + "/legLengthDistribution_scenario2050.pdf")
    activityStartTimesAbsolut(compareScenarios, savePath = ergebnisPfad + folder + "/startTimesAbsolut_scenario2050.pdf")
    activityStartTimeDistribution(compareScenarios, savePath = ergebnisPfad + folder + "/startTimeDistribution_scenario2050.pdf")
    workEducationStartTimesAbsolut(compareScenarios, savePath = ergebnisPfad + folder + "/startTimesAmountWorkEducation_scenario2050")
    
    createHeatmap(compareScenarios, savePath = ergebnisPfad + folder + "/heatmap_scenario2050.html", valueToDisplay="kmPerPerson")

    plt.close()
    
    return

# %%% BASECASE - MIDDLE SCENARIO 2030 - MIDDLE SCENARIO 2050
def runMiddleScenarios(compareScenarios, ergebnisPfad):

    compareScenarios = compareScenarios #[scenarios["basecase"], scenarios["middleScenario_2030"], scenarios["middleScenario_2050"]]
    folder = "/middleScenarios" 
    
    plotTotalDistanceDistributionPerAge(compareScenarios, savePath = ergebnisPfad + folder + "/totalDistanceDistributionPerAge_BasecaseVsScenario2030VsScenario2050.pdf")
    plotBevoelkerungspyramide(compareScenarios, savePath = ergebnisPfad + folder + "/bevoelkerungspyramiden_middleScenarios.pdf")
    plotTotalDistancePerAge(compareScenarios, savePath = ergebnisPfad + folder + "/totalDistancePerAge_middleScenarios.pdf")
    plotTotalDistanceDistribution(compareScenarios, savePath = ergebnisPfad + folder + "/totalDistanceDistribution_middleScenarios.pdf")
    plotLegsDistribution(compareScenarios, savePath = ergebnisPfad + folder + "/legAmountDistribution_middleScenarios.pdf")
    plotLegLengthDistribution(compareScenarios, savePath = ergebnisPfad + folder + "/legAmountLengthDistribution_middleScenarios.pdf")
    activityStartTimesAbsolut(compareScenarios, savePath = ergebnisPfad + folder + "/startTimesAbsolut_middleScenarios.pdf")
    activityStartTimeDistribution(compareScenarios, savePath = ergebnisPfad + folder + "/startTimeDistribution_middleScenarios.pdf")
    workEducationStartTimesAbsolut(compareScenarios, savePath = ergebnisPfad + folder + "/startTimesAmountWorkEducation_middleScenarios")
    
    createHeatmap(compareScenarios, savePath = ergebnisPfad + folder + "/heatmap_middleScenarios.html", valueToDisplay="kmPerPerson")
    createHeatmap(compareScenarios, savePath = ergebnisPfad + folder + "/heatmap_distanceDiff_middleScenarios.html", valueToDisplay="totalDistance_diffToBasecase")
    createHeatmap(compareScenarios, savePath = ergebnisPfad + folder + "/heatmap_kmPerPersonDiff_middleScenarios.html", valueToDisplay="kmPerPerson_diffToBasecase")
    createHeatmap(compareScenarios, savePath = ergebnisPfad + folder + "/heatmap_legAmountDiff_middleScenarios.html", valueToDisplay="legAmount_diffToBasecase")

    plt.close()
    
    return

# %% RUN ANALYSIS

scenarios, ergebnisPfad = importData()
  
runSRV([scenarios["SRV_BB"]], ergebnisPfad)
runCrossValidation([scenarios["testset_original"], scenarios["testset_generated"]], ergebnisPfad)
runCrossValidationSwiss([scenarios["testset_original"], scenarios["testset_generatedSwiss"]], ergebnisPfad)

runSrvVsBasecase([scenarios["SRV_BB"], scenarios["basecase"]], ergebnisPfad)
runScenario2030([scenarios["lowScenario_2030"], scenarios["middleScenario_2030"], scenarios["highScenario_2030"]], ergebnisPfad)
runScenario2050([scenarios["youngScenario_2050"], scenarios["middleScenario_2050"], scenarios["oldScenario_2050"]], ergebnisPfad) 
runMiddleScenarios([scenarios["basecase"], scenarios["middleScenario_2030"], scenarios["middleScenario_2050"]], ergebnisPfad)

# further heatmaps
createHeatmap([scenarios["basecase"], scenarios["lowScenario_2030"], scenarios["middleScenario_2030"], scenarios["highScenario_2030"]], savePath = ergebnisPfad + "/scenario2030/heatmap_distanceDiff_scenario2030.html", valueToDisplay="totalDistance_diffToBasecase")
createHeatmap([scenarios["basecase"], scenarios["lowScenario_2030"], scenarios["middleScenario_2030"], scenarios["highScenario_2030"]], savePath = ergebnisPfad + "/scenario2030/heatmap_kmPerPersonDiff_scenario2030.html", valueToDisplay="kmPerPerson_diffToBasecase")
createHeatmap([scenarios["basecase"], scenarios["youngScenario_2050"], scenarios["middleScenario_2050"], scenarios["oldScenario_2050"]], savePath = ergebnisPfad + "/scenario2050/heatmap_distanceDiff_scenario2050.html", valueToDisplay="totalDistance_diffToBasecase")
createHeatmap([scenarios["basecase"], scenarios["youngScenario_2050"], scenarios["middleScenario_2050"], scenarios["oldScenario_2050"]], savePath = ergebnisPfad + "/scenario2050/heatmap_kmPerPersonDiff_scenario2050.html", valueToDisplay="kmPerPerson_diffToBasecase")

# save the data sets in string format

# Output Ordner erstellen Szenario Analyse
try:
    os.mkdir("./output/outputDataframes_string")

except:
    print("Output Ordner existiert bereits, die Dateien werden ueberschrieben!")
    

layout2 = ["LK-code", "code", "region_type", "age", "sex", "employment", "economic_status", "driving_license", "leavingHomeTime",
           "activity1", "startTimeOfActivity1", "durationOfActivity1", "distance1", "distance1_value", "legDuration1", "totalDistance1",
           "activity2", "startTimeOfActivity2", "durationOfActivity2", "distance2", "distance2_value", "legDuration1", "totalDistance2",
           "activity3", "startTimeOfActivity3", "durationOfActivity3", "distance3", "distance3_value", "legDuration1", "totalDistance3",
           "activity4", "startTimeOfActivity4", "durationOfActivity4", "distance4", "distance4_value", "legDuration1", "totalDistance4", 
           "activity5", "startTimeOfActivity5", "durationOfActivity5", "distance5", "distance5_value", "legDuration1", "totalDistance5", 
           "activity6", "startTimeOfActivity6", "durationOfActivity6", "distance6", "distance6_value", "legDuration1", "totalDistance6", 
           "activity7", "startTimeOfActivity7", "durationOfActivity7", "distance7", "distance7_value", "legDuration1", "totalDistance7", "totalDistance", "totalDistance_value"]
    
    
for df in scenarios.values():
    name = df.name
    df_string = m.convertToString(df)
    df_string = m.changeLayout(df_string, layout2)
      
    df_string.to_csv("./output/outputDataframes_string/" + name + ".csv")

# %% SCENARIO OVERVIEW
  
# =============================================================================
# overview = pd.DataFrame(columns = ["Szenario", "Population", "Gesamtdistanz (km)", "Pro-Kopf-Distanz (km)"])
# 
# for scenario in scenarios.values():
#     name = scenario.name
#     pop = len(scenario)
#     totDist = round(scenario.totalDistance_value.sum())
#     totDist_perPerson = round(totDist/pop,3)
#     
#     df = pd.DataFrame([[name, pop, totDist, totDist_perPerson]], columns = ["Szenario", "Population", "Gesamtdistanz (km)", "Pro-Kopf-Distanz (km)"])
#     
#     overview = pd.concat([overview, df])
# 
# overview = overview.set_index("Szenario")    
# 
# basePop = overview.loc["basecase", "Population"]
# baseTotDist = overview.loc["basecase", "Gesamtdistanz (km)"]
# baseTotDistPP = overview.loc["basecase", "Pro-Kopf-Distanz (km)"]
# 
# for i in overview.index:
#     
#     overview.loc[i, "Population"] = str(overview.loc[i, "Population"]) +" (" + str(round(overview.loc[i, "Population"] *100 / basePop - 100,2)) + "%)"
#     overview.loc[i, "Gesamtdistanz (km)"] = str(overview.loc[i, "Gesamtdistanz (km)"]) +" (" + str(round(overview.loc[i, "Gesamtdistanz (km)"] * 100 / baseTotDist - 100,2)) + "%)"
#     overview.loc[i, "Pro-Kopf-Distanz (km)"] = str(overview.loc[i, "Pro-Kopf-Distanz (km)"]) +" (" + str(round(overview.loc[i, "Pro-Kopf-Distanz (km)"] * 100 / baseTotDistPP - 100,2)) + "%)"
#     
# overview = overview.apply(lambda col: col.str.replace(".", ","))
# 
# =============================================================================

# %% OUTPUT GRAPHS FOR ZWISCHENPRAESENTATION
# =============================================================================
# path = "./Zwischenpraesentation"
# 
# scenarios_cottbus = scenarios.copy()
# 
# for key in scenarios_cottbus.keys():
#     if key != "SRV_BB":
#         df = scenarios_cottbus[key]
#         scenarios_cottbus[key] = df[df.code == 12052]
#         scenarios_cottbus[key].name = key
#  
# 
# scenarios_2030_cottbus = [scenarios_cottbus[x] for x in scenarios_cottbus.keys() if "2030" in x]
# scenarios_2050_cottbus = [scenarios_cottbus[x] for x in scenarios_cottbus.keys() if "2050" in x]
# scenarios_2030_brandenburg = [scenarios[x] for x in scenarios.keys() if "2030" in x]
# scenarios_2050_brandenburg = [scenarios[x] for x in scenarios.keys() if "2050" in x]
# 
# plotBevoelkerungspyramide([scenarios_cottbus["basecase"]], savePath = path + "/basecase_bevoelkerungspyramide_cottbus.pdf")
# plotBevoelkerungspyramide([scenarios["basecase"]], savePath = path + "/basecase_bevoelkerungspyramide_brandenburg.pdf")
# 
# plotBevoelkerungspyramide(scenarios_2030_cottbus, savePath = path + "/scenario2030_bevoelkerungspyramide_cottbus.pdf")
# plotBevoelkerungspyramide(scenarios_2030_brandenburg, savePath = path + "/scenario2030_bevoelkerungspyramide_brandenburg.pdf")
# 
# plotBevoelkerungspyramide(scenarios_2050_cottbus, savePath = path + "/scenario2050_bevoelkerungspyramide_cottbus.pdf")
# plotBevoelkerungspyramide(scenarios_2050_brandenburg, savePath = path + "/scenario2050_bevoelkerungspyramide_brandenburg.pdf")
# 
# plotBevoelkerungspyramide([scenarios["SRV_BB"]], savePath = path + "/SrV_bevoelkerungspyramide.pdf")
# 
# =============================================================================
