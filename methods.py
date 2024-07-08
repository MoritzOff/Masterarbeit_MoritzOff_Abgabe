# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 17:06:10 2023

@author: Moritz Off
"""
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import graphviz
import os
import math
import numpy as np
from pgmpy.factors.discrete.CPD import TabularCPD
from itertools import product


# %% DRAW GRAPH FUNCTIONS
# Speichert 3D Netz als HTML
def draw3DNetwork(bn, name):

    graph_model = bn
    
    G = nx.DiGraph()
    i = 0
    j=0
    k=0
    l=0
    m = 0
    o = 0
    posi = {}
    for n in graph_model.nodes():
        G.add_node(n)
        if n.startswith("activity"):
            i = i+1
            if i%2!=0:
                posi[n]=[i, 4, 3]
            else:
                posi[n]=[i, 4.5, 3]
                
        if n.startswith("distance"):
            j = j+1
            if j%2!=0:
                posi[n]=(j, 2, 2)
            else:
                posi[n]=(j, 2.5, 2)
        if n.startswith("totalDistance"):
            k = k+1
            posi[n]=(k, 1.5, 1.5)

        if n.startswith("startTimeOfActivity"):
            l = l+1
            posi[n]=(l, 4, 1.5)
        if n.startswith("durationOfActivity"):
            m = m+1
            posi[n]=(m, 5.5, 1.5)

        if n.startswith("legDuration"):
            o = o+1
            posi[n]=(o, 2, 1)
        

    
            
    posi["age"]=[4,3,7]
    posi["employment"]=[4,4,7]
    posi["region_type"]=[4,1,7]
    posi["sex"]=[4,5,7]
    posi["driving_license"]=[4,1.5,5]
    posi["leavingHomeTime"]=[0,4,2]
    try:posi["pt_abo_avail"]=[4,1,5]
    except: pass
    try:posi["economic_status"]=[4,3.2,4]
    except: pass
    
    for n in graph_model.edges():
        G.add_edge(n[0],n[1])
    
    edges = G.edges()
       
    spring_3D = nx.spring_layout(G, dim = 3, pos = posi, fixed=posi.keys())
    
    
    x_nodes= [spring_3D[key][0] for key in spring_3D.keys()] # x-coordinates of nodes
    y_nodes = [spring_3D[key][1] for key in spring_3D.keys()] # y-coordinates
    z_nodes = [spring_3D[key][2] for key in spring_3D.keys()] # z-coordinates
    
    #we need to create lists that contain the starting and ending coordinates of each edge.
    x_edges=[]
    y_edges=[]
    z_edges=[]
    
    coords = pd.DataFrame()
    
    #need to fill these with all of the coordinates
    for edge in edges:
        #format: [beginning,ending,None]
        x_coords = [spring_3D[edge[0]][0],spring_3D[edge[1]][0],None]
        x_edges += x_coords
    
        y_coords = [spring_3D[edge[0]][1],spring_3D[edge[1]][1],None]
        y_edges += y_coords
    
        z_coords = [spring_3D[edge[0]][2],spring_3D[edge[1]][2],None]
        z_edges += z_coords
        
        df = pd.DataFrame({"x1":[x_coords[0]],"x2":[x_coords[1]], 
                           "y1":[y_coords[0]],"y2":[y_coords[1]], 
                           "z1":[z_coords[0]],"z2":[z_coords[1]]})
        coords = pd.concat([coords,df])
    
    
    nodeNames = [f'{w}' for w in G.nodes()]
    for idx in range(len(nodeNames)):
        if nodeNames[idx].startswith("activity"): nodeNames[idx] = "act"+nodeNames[idx][-1]
        if nodeNames[idx].startswith("distance"): nodeNames[idx] = "dist"+nodeNames[idx][-1]
        if nodeNames[idx].startswith("totalDistance"): nodeNames[idx] = "totDist"+nodeNames[idx][-1]
        if nodeNames[idx].startswith("startTimeOfActivity"): nodeNames[idx] = "startTime"+nodeNames[idx][-1]
        if nodeNames[idx].startswith("durationOfActivity"): nodeNames[idx] = "duration"+nodeNames[idx][-1]

    #create a trace for the edges
    trace_edges = go.Scatter3d(
        x=x_edges,
        y=y_edges,
        z=z_edges,
        mode='lines',
        line=dict(color='black', width=2),
        hoverinfo='none')
    
    #create a trace for the nodes
    trace_nodes = go.Scatter3d(x=x_nodes,
                               y=y_nodes,
                               z=z_nodes,
                               mode='markers+text',
                               text = nodeNames,
                               hoverinfo="text",
                               opacity=1,
                               marker=dict(symbol='circle',
                                           size=20,
                                           color='skyblue')
                            )
   
    #Include the traces we want to plot and create a figure
    data = [trace_edges, trace_nodes]#, trace_cones]
    fig = go.Figure(data=data)
      
    
    for index, row in coords.iterrows():
        arrow_start_ratio = 0.99
        arrow_tip_ratio = 0.01 
        
        length = math.sqrt((row.x2-row.x1)**2 + (row.y2-row.y1)**2 + (row.z2-row.z1)**2)
        
        if length < 10:
            arrow_start_ratio = 0.98-(0.1/length)
            arrow_tip_ratio = (1/length)*(0.4-(length)*0.0349)
            
        fig.add_trace(go.Cone(x=[row.x1+arrow_start_ratio*(row.x2-row.x1)], 
                              y=[row.y1+arrow_start_ratio*(row.y2-row.y1)],
                              z=[row.z1+arrow_start_ratio*(row.z2-row.z1)],
                              u=[arrow_tip_ratio*(row.x2-row.x1)],
                              v=[arrow_tip_ratio*(row.y2-row.y1)], 
                              w=[arrow_tip_ratio*(row.z2-row.z1)],
                              #anchor="tip",
                              showlegend=False,
                              showscale=False,
                              hoverinfo="none",
                              colorscale=[[0, 'black'], [1, 'black']]))
    
    
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)
    
    fig.write_html("./output/Graph_"+name+".html")
    fig.show()
    
    

# Speichert Netz als PNG unter vergebene Namen
def draw2DNetwork(bn, name):
    
    graph_model = bn
# Graph-Objekt erstellen
    graph = graphviz.Digraph(graph_attr={'rankdir': 'TB', 'center':'true'})

    i=12
    # Knoten hinzufuegen
    for n in graph_model.nodes():
        i=i-1
        if i%2!=0:
            graph.node(n, pos = "0,"+str(i+1)+"!")
        else:
            graph.node(n, pos = "-1.5,"+str(i+1)+"!")
        
    graph.node("sex", pos = "-1.5,10!")
    graph.node("age", pos = "1.5,10!")
    graph.node("employment", pos = "-1.5,9!")
    graph.node("driving_license", pos = "1.5,9!")
    
    
    # Kanten hinzufuegen
    for n in graph_model.edges():
        graph.edge(n[0],  n[1])
    
    # Layout-Algorithmus festlegen
    graph.attr(layout='neato', dir = "forward")
    
    # Graph als .png abspeichern
    graph.render("./"+name, format="png", view=False)
    os.remove("./"+name)
    
    graph.view()
 
# %% CONVERT TO STRING/INTEGER FUNCTIONS
# %%% CONVERT DATA FROM STRING FORMAT TO INTEGER
# Konvertiert die Integer Dataframes zu Strings zurück
# Für Schweizer Szenario werden entsprechen zusammengefasste Kategorien verwendet
def convertToInteger(dataFrame, swissCategories = False):
    df = dataFrame.copy()
    swissCategories = swissCategories
    
    # Dictionaries zur Zuordnung der Integers zu den Strings
    dictionaries = {"age" : {"unter 10 Jahre":0, 
                             "10 bis unter 15 Jahre":1, 
                             "15 bis unter 18 Jahre":2,
                             "18 bis unter 20 Jahre":3,
                             "20 bis unter 25 Jahre":4,
                             "25 bis unter 30 Jahre":5,
                             "30 bis unter 35 Jahre":6,
                             "35 bis unter 40 Jahre":7,
                             "40 bis unter 45 Jahre":8,
                             "45 bis unter 50 Jahre":9,
                             "50 bis unter 55 Jahre":10,
                             "55 bis unter 60 Jahre":11,
                             "60 bis unter 65 Jahre":12,
                             "65 Jahre und mehr":13},
                    "employment" : {"erwerbstaetig":0,
                                    "erwerbslos":1,
                                    "nichtErwerbsP":2},
                    "economic_status" : {"very_high":0,
                                         "high":1,
                                         "medium":2, 
                                         "low":3, 
                                         "very_low":4},
                    "sex" : {"m":0,
                             "f":1},
                    "driving_license" : {"no":0,
                                         "yes":1},
                    "pt_abo_avail" : {"no":0,
                                      "yes":1},
                    "activity" : {"home":1,
                                  "work":2,
                                  "education_school":3,
                                  "education_higher":4,
                                  "transport":5, 
                                  "shopping":6,
                                  "personal_business" : 7,
                                  "leisure":8,
                                  "dining":9,
                                  "other":10,
                                  np.NaN:-1},
                    "distance" : {"0-1km":1,
                                  "1-2km":2,
                                  "2-5km":3,
                                  "5-10km":4,
                                  "10-20km":5,
                                  "20-50km":6,
                                  ">50km":7,
                                  "nan":-1,
                                  np.NaN:-1},
                    "totalDistance" : {"0-1km":1,
                                      "1-2km":2,
                                      "2-5km":3,
                                      "5-10km":4,
                                      "10-20km":5,
                                      "20-50km":6,
                                      ">50km":7,
                                      "nan":-1,
                                      np.NaN:-1},
                    "durationOfActivit" : {"0-5min":1,
                                          "5-15min":2,
                                          "15-30min":3,
                                          "30-60min":4,
                                          "1-2h":5,
                                          "2-3h":6,
                                          "3-5h":7,
                                          "5-7h":8,
                                          "7-8h":9,
                                          "8-9h":10,
                                          "9-12h":11,
                                          ">12h":12,
                                          "nan":-1,
                                          np.NaN:-1},
                    "legDuration" : {"0-5min":1,
                                    "5-15min":2,
                                    "15-30min":3,
                                    "30-60min":4,
                                    "60-90min":5,
                                    "90-120min":6,
                                    "2-3h":7,
                                    "3-5h":8,
                                    "5-7h":9,
                                    ">7h":10,
                                    "nan":-1,
                                    np.NaN:-1},
                    "startTimeOfActivity" : {"0-6Uhr":1,
                                           "6-10Uhr":2,
                                           "10-14Uhr":3,
                                           "14-18Uhr":4,
                                           "18-22Uhr":5,
                                           ">22Uhr":6,
                                           "nan":-1,
                                           np.NaN:-1},
                    "leavingHomeTime" : {"0-6Uhr":1,
                                        "6-10Uhr":2,
                                        "10-14Uhr":3,
                                        "14-18Uhr":4,
                                        "18-22Uhr":5,
                                        ">22Uhr":6,
                                        "nan":-1,
                                        np.NaN:-1}
                    }

    for col in df.columns:
        if df.dtypes[col]=="float64":
            continue
        
        for key in dictionaries.keys():
            if col.startswith(key):
                try:
                    df[col] = df[col].apply(lambda x: dictionaries[key][x])
                except: 
                    pass

    cols=[i for i in df.columns if i not in ["name"] and df.dtypes[i] != "float64"]
    for col in cols:
        try:
            df[col]= df[col].astype(np.int64)
        except:
            pass
    
    return df                 

# %%% CONVERT DATA FROM INTEGER FORMAT TO STRING
# Konvertiert die Integer Dataframes zu Strings zurück
# Für Schweizer Szenario werden entsprechen zusammengefasste Kategorien verwendet
def convertToString(dataFrame, swissCategories = False):
    
    """
    Convert Integers to String and return the Dataframe.
    Let's user chose to use Swiss Categories or not
    """
    
    df = dataFrame.copy()
    swissCategories = swissCategories
    
    # Dictionaries zur Zuordnung der Integers zu den Strings
    dictionaries = {"age" : {0:"unter 10 Jahre", 
                             1:"10 bis unter 15 Jahre", 
                             2:"15 bis unter 18 Jahre",
                             3:"18 bis unter 20 Jahre",
                             4:"20 bis unter 25 Jahre",
                             5:"25 bis unter 30 Jahre",
                             6:"30 bis unter 35 Jahre",
                             7:"35 bis unter 40 Jahre",
                             8:"40 bis unter 45 Jahre",
                             9:"45 bis unter 50 Jahre",
                             10:"50 bis unter 55 Jahre",
                             11:"55 bis unter 60 Jahre",
                             12:"60 bis unter 65 Jahre",
                             13:"65 Jahre und mehr"},
                    "employment" : {0:"erwerbstaetig",
                                    1:"erwerbslos",
                                    2:"nichtErwerbsP"},
                    "economic_status" : {0:"very_high",
                                         1:"high",
                                         2:"medium", 
                                         3:"low", 
                                         4:"very_low"},
                    "sex" : {0:"m",
                             1:"f"},
                    "driving_license" : {0:"no",
                                         1:"yes"},
                    "pt_abo_avail" : {0:"no",
                                      1:"yes"},
                    "activity" : {1:"home",
                                  2:"work",
                                  3:"education_school",
                                  4:"education_higher",
                                  5:"transport", 
                                  6:"shopping",
                                  7:"personal_business",
                                  8:"leisure",
                                  9:"dining",
                                  10:"other",
                                 -1: np.NaN},
                    "distance" : {1:"0-1km",
                                  2:"1-2km",
                                  3:"2-5km",
                                  4:"5-10km",
                                  5:"10-20km",
                                  6:"20-50km",
                                  7:">50km",
                                  -1:np.NaN},
                    "totalDistance" : {1:"0-1km",
                                      2:"1-2km",
                                      3:"2-5km",
                                      4:"5-10km",
                                      5:"10-20km",
                                      6:"20-50km",
                                      7:">50km",
                                      -1:np.NaN},
                    "durationOfActivit" : {1:"0-5min",
                                          2:"5-15min",
                                          3:"15-30min",
                                          4:"30-60min",
                                          5:"1-2h",
                                          6:"2-3h",
                                          7:"3-5h",
                                          8:"5-7h",
                                          9:"7-8h",
                                          10:"8-9h",
                                          11:"9-12h",
                                          12:">12h",
                                          -1:np.NaN},
                    "legDuration" : {1:"0-5min",
                                    2:"5-15min",
                                    3:"15-30min",
                                    4:"30-60min",
                                    5:"60-90min",
                                    6:"90-120min",
                                    7:"2-3h",
                                    8:"3-5h",
                                    9:"5-7h",
                                    10:">7h",
                                    -1:np.NaN},
                    "startTimeOfActivity" : {1:"0-6Uhr",
                                           2:"6-10Uhr",
                                           3:"10-14Uhr",
                                           4:"14-18Uhr",
                                           5:"18-22Uhr",
                                           6:">22Uhr",
                                           -1:np.NaN},
                    "leavingHomeTime" : {1:"0-6Uhr",
                                        2:"6-10Uhr",
                                        3:"10-14Uhr",
                                        4:"14-18Uhr",
                                        5:"18-22Uhr",
                                        6:">22Uhr",
                                        -1:np.NaN}
                    }
    
    for col in df.columns:
        for key in dictionaries.keys():
            if col.startswith(key):
                try:
                    df[col] = df[col].apply(lambda x: dictionaries[key][x])
                except: 
                    pass
    return df

# %% DATA MANIPULATION AND ANALYSIS  
# %%% PRINT FULL CPD
# Methode, um die CPD vollständig anzuzeigen
def print_full(cpd):
    backup = TabularCPD._truncate_strtable
    TabularCPD._truncate_strtable = lambda self, x: x
    print(cpd)
    TabularCPD._truncate_strtable = backup
    return cpd

# %%% COMBINE CERTAIN AGE GROUPS
def replaceOver65(data):
    df = data.copy()
    replacement_dict = {"unter 3 Jahre" : "unter 10 Jahre", 
                        "3 bis unter 6 Jahre" : "unter 10 Jahre", 
                        "6 bis unter 10 Jahre" : "unter 10 Jahre",
                        "65 bis unter 75 Jahre" : "65 Jahre und mehr",
                        "75 Jahre und mehr" : "65 Jahre und mehr"}
    
    df.age = df["age"].replace(replacement_dict)
    
    return df

# %%% CHANGE COLUMN ORDER OF THE DATA
def changeLayout(df, column_order):
    """
    Change the column order of the table to a given layout and returns the updated DataFrame
    If some columns are missing in the layout: ignore them and change the remaining columns

    Parameters
    ----------
    df : pd.DataFrame
        
    column_order : List
        List of column names

    """
    
    layout = column_order.copy()
    layout = [i for i in layout if i in df.columns]
    return df[layout]       

# %%% ADD A COLUMN SHOWING THE ACTIVITY CHAIN
def addActivityChainColumn(df):
    """
    Adds an act_chain columns with the activity chains as strings
    Need columns activity1 - activity7
    """
    
    df = df.astype(str)
    
    layout = ["age", "sex", "region_type", "LK-code", "code", "employment", "economic_status", "driving_license", 
              "leavingHomeTime",
              "activity1", "activity2", "activity3", "activity4", "activity5", "activity6", "activity7",
              "distance1", "distance2", "distance3", "distance4", "distance5", "distance6", "distance7",
              "legDuration1", "legDuration2", "legDuration3", "legDuration4", "legDuration5", "legDuration6", "legDuration7",
              "startTimeOfActivity1", "startTimeOfActivity2", "startTimeOfActivity3", "startTimeOfActivity4", "startTimeOfActivity5", "startTimeOfActivity6", "startTimeOfActivity7",
              "totalDistance1", "totalDistance2", "totalDistance3", "totalDistance4", "totalDistance5", "totalDistance6", "totalDistance7",
              "durationOfActivity1", "durationOfActivity2", "durationOfActivity3", "durationOfActivity4", "durationOfActivity5", "durationOfActivity6", "durationOfActivity7", "totalDistance", "totalDistance_value"]

    df = changeLayout(df, layout)
    
    # Erstellung der Spalte act_chain fuer die verketteten Aktivitaeten
    indexAct2 = df.columns.get_loc("activity2")
    indexLastAct = df.columns.get_loc("activity7")
  
    df["act_chain"] = "home / " + df["activity1"].str.cat(df.iloc[:,indexAct2:indexLastAct+1].values,sep=" / ")
        
    # herausloeschen der stop-Werte
    df.act_chain = df["act_chain"].str.replace(" / nan","")
    df.act_chain = df["act_chain"].str.replace(" / stop","")
    df.act_chain = df["act_chain"].str.replace(" / -1","")
    #df = df.sort_values(list(df.columns))
    return df
    

# %%% GET THE DISTRIBUTION OF THE ACTIVITY CHAIN
def getActivityChainDistribution(dataframe):
    """
    Returns the activity chain distribution of a Dataframe with activity1 to activity7
    Activity values need to be in String format

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input DataFrame with agent's trips
    Returns
    -------
    act_distribution : pd.DataFrame
        Activity Chains with the percentage of their distribution.

    """
    df = dataframe.astype(str)
    
    # Erstellung der Spalte act_chain fuer die verketteten Aktivitaeten
    indexAct2 = df.columns.get_loc("activity2")
    indexLastAct = df.columns.get_loc("activity7")
  
    df["act_chain"] = "home / " + df["activity1"].str.cat(df.iloc[:,indexAct2:indexLastAct+1].values,sep=" / ")
        
    # herausloeschen der stop-Werte
    df.act_chain = df["act_chain"].str.replace(" / nan","")
    df.act_chain = df["act_chain"].str.replace(" / stop","")
    df.act_chain = df["act_chain"].str.replace(" / -1","")
    
    # nach chain gruppieren, Anzahl der Trips pro chain ausgeben, sortieren vom häufigstem Vorkommen der Aktivitaetenkette absteigend
    df_act = df.groupby("act_chain").count().sort_values("age",ascending=False)
    
    # Prozent Spalte fuer Verteilung berechnen
    df_act["distribution"]=df_act["age"]/sum(df_act["age"])
    
    act_distribution = pd.DataFrame(df_act["distribution"])
    
    return act_distribution

# %%% ADD A COLUMN DISPLAYING THE DISTANCE CHAIN
def addDistanceChainColumn(df):
   """
   Adds an dist_chain columns with the activity chains as strings
   Need columns distance1 - distance7
   """
   
   df = df.astype(str)
   
   layout = ["age", "sex", "region_type", "LK-code", "code", "employment", "economic_status", "driving_license", 
             "leavingHomeTime",
             "activity1", "activity2", "activity3", "activity4", "activity5", "activity6", "activity7",
             "distance1", "distance2", "distance3", "distance4", "distance5", "distance6", "distance7",
             "legDuration1", "legDuration2", "legDuration3", "legDuration4", "legDuration5", "legDuration6", "legDuration7",
             "startTimeOfActivity1", "startTimeOfActivity2", "startTimeOfActivity3", "startTimeOfActivity4", "startTimeOfActivity5", "startTimeOfActivity6", "startTimeOfActivity7",
             "totalDistance1", "totalDistance2", "totalDistance3", "totalDistance4", "totalDistance5", "totalDistance6", "totalDistance7",
             "durationOfActivity1", "durationOfActivity2", "durationOfActivity3", "durationOfActivity4", "durationOfActivity5", "durationOfActivity6", "durationOfActivity7", "totalDistance", "totalDistance_value"]

   df = changeLayout(df, layout)
   
   # Erstellung der Spalte act_chain fuer die verketteten Aktivitaeten
   indexAct2 = df.columns.get_loc("distance2")
   indexLastAct = df.columns.get_loc("distance7")
 
   df["dist_chain"] = df["distance1"].str.cat(df.iloc[:,indexAct2:indexLastAct+1].values,sep=" / ")
       
   # herausloeschen der stop-Werte
   df.dist_chain = df["dist_chain"].str.replace(" / nan","")
   df.dist_chain = df["dist_chain"].str.replace(" / stop","")
   df.dist_chain = df["dist_chain"].str.replace(" / -1","")
   #df = df.sort_values(list(df.columns))
   
   return df 
    
# %%% GET THE DISTRIBUTION OF THE DISTANCE CHAINS 
def getDistanceChainDistribution(dataframe):
    """
   Returns the distance chain distribution of a Dataframe with distance1 to distance7

   Parameters
   ----------
   dataframe : pd.DataFrame
       Input DataFrame with agent's trips and distances
   Returns
   -------
   dist_distribution : pd.DataFrame
       Distance Chains with the percentage of their distribution.

    """
    
    df = dataframe.astype(str)
    # Erstellung der Spalte dist_chain fuer die verketteten Distanzen
    try:
        indexDist2 = df.columns.get_loc("distance2")
        indexLastDist = df.columns.get_loc("distance7")
    except:
        return 0
    
    df["dist_chain"] = df["distance1"].str.cat(df.iloc[:,indexDist2:indexLastDist+1].values,sep=" / ")
    
    
    # herausloeschen der stop-Werte
    df.dist_chain = df["dist_chain"].str.replace(" / nan","")
    df.dist_chain = df["dist_chain"].str.replace(" / stop","")
    df.dist_chain = df["dist_chain"].str.replace(" / -1","")   
    
    # nach dist gruppieren, Anzahl der Trips pro distance ausgeben, sortieren vom häufigstem Vorkommen der Distanzenkette absteigend
    df_dist = df.groupby("dist_chain").count().sort_values("age",ascending=False)
    
    # Prozent Spalte fuer Verteilung berechnen
    df_dist["distribution"]=df_dist["age"]/sum(df_dist["age"])
    
    dist_distribution = pd.DataFrame(df_dist["distribution"])
    
    return dist_distribution

# %%% RETURNS COMPARISONS OF THE ACTIVITY AND DISTANCE CHAIN DISTRIBUTIONS
def compareDistribution(learningData, testData):
    """
    

    Parameters
    ----------
    learningData : pd.DataFrame
        SRV dataset
    testData : pd.DataFrame
        sampled prediction

    Returns
    -------
    Comparison of activity distribution and of distance distribution

    """
    
    
    srv = learningData
    results = testData


    act_srv = getActivityChainDistribution(srv)
    act_results = getActivityChainDistribution(results)
    
    activity_distribution = pd.merge(act_results, act_srv, how = "outer", left_index=True, right_index=True)
    activity_distribution = activity_distribution.rename(columns = {"distribution_x":"Prediction", "distribution_y":"SRV_BB"})
    activity_distribution = activity_distribution.sort_values("Prediction", ascending = False).fillna(0)
    activity_distribution = activity_distribution[["SRV_BB", "Prediction"]]
    
    try:
        dist_srv = getDistanceChainDistribution(srv)
        dist_results = getDistanceChainDistribution(results)
        
        distance_distribution = pd.merge(dist_results, dist_srv, how = "outer", left_index=True, right_index=True)
        distance_distribution = distance_distribution.rename(columns = {"distribution_x":"Prediction", "distribution_y":"SRV_BB"})
        distance_distribution = distance_distribution.sort_values("Prediction", ascending = False).fillna(0)
        distance_distribution = distance_distribution[["SRV_BB", "Prediction"]]
    except:
        distance_distribution = 0
    
    return activity_distribution, distance_distribution

# %%% DISTANCE DISTRIBUTIONS OF ALL LEGS
def getDistanceDistribution(data):
    """
    takes Data in String Format and calculate the distance distribution of all legs

    Parameters
    ----------
    data : pd.DataFrame
        Model data in  String Format with distance columns 1-7

    Returns
    -------
    df : pd.DataFrame
        Distribution of lengths

    """
     
        
    df = data
    dist1 = df.columns.get_loc("distance1")
    lastDist = df.columns.get_loc("distance7")

    # Distanzen Spalten untereinander anordnen
    df = pd.melt(df, id_vars= ["region_type"], value_vars=df.columns[dist1:lastDist+1].values, var_name="distance_type", value_name="distance")

    # Gruppieren nach Distanz, Anteile in distance_distribution speichern
    df = df.groupby("distance").count()
    df["distance_distribution"] = df.distance_type/sum(df.distance_type)
    df = df.drop(["region_type", "distance_type"], axis=1)
    df = df.reindex(index = ["0-1km", "1-2km", "2-5km", "5-10km", "10-20km", "20-50km", ">50km"])
    
    return df

# %%% TOTAL DISTANCE DISTRIBUTION
def getTotalDistanceDistribution(data):
    """
    takes Date in String Format and calculate the totalDistance distribution of all agents days

    Parameters
    ----------
    data : pd.DataFrame
        Model data in  String Format with totalDistance columns 1-7
    

    Returns
    -------
    df : pd.DataFrame
        Distribution of lengths

    """
    df = data.copy()
    
    
    # groupby values and calulate distribution
    df = df[["totalDistance"]].astype(str)
    dist = df.groupby("totalDistance")["totalDistance"].value_counts()/len(df)
    dist.index = dist.index.rename("totalDistance")
    dist = dist.reindex(index = ["0-1km", "1-2km", "2-5km", "5-10km", "10-20km", "20-50km", ">50km", "nan"])
    dist = dist.fillna(0)
    
    dist = pd.DataFrame(data = dist.values, index=dist.index, columns=["distribution"])
    
    return dist

# %%% AMOUNT OF LEGS TRAVELLED PER DAY
def getLegsPerDayDistribution(data):
    """
    Returns the distribution of the amount of legs driven during the day. 
    

    Parameters
    ----------
    data : pd.DataFrame
        data with 7 activities in string format.

    Returns
    -------
    distribution
    
    average amount

    """
    
    df = data

    cols = [x for x in df.columns if x.startswith("activity")]
    df = df[cols]
    
    df["legsPerDay"]=7-df.isnull().sum(axis = 1)
    
    legsPerDay = df.groupby("legsPerDay")["legsPerDay"].value_counts() / len(df)
    legsPerDay = legsPerDay.to_frame().rename(columns = {"count":"distribution"})
    legsPerDay = legsPerDay.reindex(index = [0, 1, 2, 3, 4, 5, 6, 7])
    legsPerDay = legsPerDay.fillna(0)
    
    avgLegsPerDay = np.mean(df.legsPerDay)
 
    return legsPerDay, avgLegsPerDay

# %%% TOTAL TIME SPENT IN ACTIVITIES
def getTotalActivitesTimeDistribution(data):
    """
    Returns the distribution of the total time spent in activities during the day. 
    

    Parameters
    ----------
    data : pd.DataFrame
        data with durationOfActivitiesTotal column.

    Returns
    -------
    distribution
    
    """
    
    df = data
    
    totalTime = df.groupby("durationOfActivitiesTotal",).count().sort_values(by = "age", ascending = False)
    totalTime["distribution"] = totalTime.age/totalTime.age.sum()
    
    return totalTime["distribution"].to_frame()

# %%% ACTIVITY DURATION PER ACTIVITY TYPE    
def getActivityTypesDurationDistribution(data):
    """
    Returns the distribution of the time spent for each activity. 
    

    Parameters
    ----------
    data : pd.DataFrame
        data with activity and durationOfActivity columns 1-7.

    Returns
    -------
    distribution dictionary 
    with the different types as keys and the distribution dataframes as values
    
    """
    
    df = data

    # write data for every activity (e.g. 1,2,3,...) in dictionary
    act = {}
    for i in range(1,8):
        key = "activity"+str(i)
        activity = df.loc[:,["activity"+str(i), "durationOfActivity"+str(i)]]
        activity = activity.rename(columns={"activity"+str(i) : "activity", "durationOfActivity"+str(i) : "duration"})
        act[key] = activity
 
    # create DataFrame with all data appended
    allActivities = pd.DataFrame()
    for key in act.keys():
        allActivities = pd.concat([allActivities, act[key]])
 
    # per type of activity: group by duration and get distribution
    activitiesGrouped = {}
    for activityType in allActivities.activity.unique():
        types = allActivities[allActivities.activity == activityType]
        activitiesGrouped[activityType] = types
        activitiesGrouped[activityType] = activitiesGrouped[activityType].groupby("duration").count()
        activitiesGrouped[activityType] = activitiesGrouped[activityType]/activitiesGrouped[activityType].sum()
        activitiesGrouped[activityType] = activitiesGrouped[activityType].rename(columns={"activity":"distribution"})
        activitiesGrouped[activityType] = activitiesGrouped[activityType].reindex(index = ["0-5min", "5-15min", "15-30min", "30-60min", "1-2h", "2-3h", "3-5h", "5-7h", "7-8h", "8-9h", "9-12h", ">12h"])
    
    try:
        del activitiesGrouped[np.nan]
    except:
        pass
    
    return activitiesGrouped 

# %%% ACTIVITY TYPE DISTRIBUTION
def getActivityTypeShares(data):
    df = data
    
    actCols = [x for x in df.columns if x.startswith("activity")]
    
    activities = pd.DataFrame(columns = ["activity"])
    for col in actCols:
        df["activity"] = df[col]
        activities = pd.concat([activities, df[["activity"]]], axis = 0)
    
    activities = activities[activities.activity.astype(str) != "nan"]
    # calculate the distribution
    activities = activities.groupby("activity")["activity"].value_counts()/len(activities)
    # convert to DataFrame    
    activities = pd.DataFrame(data = activities.values, index = activities.index, columns = ["activity"])
    return activities

# %% CORRECT CPDS
# %%% CHECK FOR WRONG DATA 
def filterWrongData(data):
    """
    This function takes the output data of the model (in integers) and returns the cleaned and the false data in two DataFrames.
    
    The false data contains those rows, where data for e.g. distance exists, even though the regarding or a previous
    activity was already nan.
    
    Also filters if following startTimes are earlier than previous.
    
    Clean data is the remaining data.

    Parameters
    ----------
    data : pd.DataFrame
        Output DataFrame with "activity", "distance", "startTimeOfActivity", "durationOfActivity", "totalDistance"
        columns 1-7.
        

    Returns
    -------
    dfCleaned : pd.DataFrame
        the clean Data
    dfFalse : pd.DataFrame
        the false data

    """
    
    dfCleaned = data.copy()
    dfAll = dfCleaned 

    # alle nachfolgenden nicht-nans flaggen, wenn schon eine activity nan war
    columns = ["activity", "distance", "startTimeOfActivity", "durationOfActivity", "totalDistance"]
    for col in columns:
        for i in range(1,8):
            rest = range(i, 8)
            for r in rest:
                # existiert eine distance/duration/activity etc, obwohl schon vorher die letzte Activity war?
                dfCleaned = dfCleaned.drop(dfCleaned[(dfCleaned["activity"+str(i)] == -1) & (dfCleaned[col+str(r)] != -1)].index)
    
    # alle nachfolgenden startTimes, welche früher sind als vorangegangene, flaggen
    for i in range(1,7):   
        col1 = "startTimeOfActivity"+str(i)
        col2 = "startTimeOfActivity"+str(i+1)
        
        dfCleaned = dfCleaned.drop(dfCleaned[(dfCleaned[col2]<dfCleaned[col1])&(dfCleaned[col2]!=-1)].index)
    
    # alle nachfolgenden totalDistances, welche kleiner sind als vorangegangene, flaggen
    for i in range(1,7):   
        col1 = "totalDistance"+str(i)
        col2 = "totalDistance"+str(i+1)
        
        dfCleaned = dfCleaned.drop(dfCleaned[(dfCleaned[col2]<dfCleaned[col1])&(dfCleaned[col2]!=-1)].index)
    
    
    dfFalse = dfAll.drop(dfCleaned.index)
        
    return dfCleaned, dfFalse

# %%% CORRECT NAN CPDS
def correctWrongCPDs(bayesianModel):
    """
    Takes the Bayesian Model and checks for every CPD in every node, if a parents value is nan.
    For all parent combinations containing nan, the CPDs is set to 100% for nan and 0% for all other possibilities.

    Parameters
    ----------
    bayesianModel : Learned Bayesian Model with calculated CPDs

    Returns
    -------
    bn : learned Bayesian Model with corrected CPDs

    """
    
    bn = bayesianModel
    nodes = ["activity", "distance", "startTimeOfActivity", "durationOfActivity", "legDuration", "leavingHomeTime"]
    indices = list(range(1,8))
    f = list([nodes, indices])
    nodesDF = pd.DataFrame(list(product(*f)), columns=["node", "indices"])

    nodesDF["combined"] = nodesDF.node + nodesDF.indices.astype(str)
    
    finalNodes = [x for x in nodesDF.combined if x in bn]
    counter = 0
    for n in finalNodes:       
        node = n
        cpd = bn.get_cpds(node) 
        parents_state_amount = {}
        
        for parent in cpd.variables[1:]:
            parents_state_amount[parent] = bn.get_cpds(parent).variable_card
        
        for parent in parents_state_amount.keys():
           
            # wenn ein parent einen nan enthält, update alle Kombinationen der anderen parents und dem nan parent
            if any("-1" in x for x in cpd.state_names[parent]): 
                kwargs = {}
                kwargs[parent] = "-1"
                
                rest_parents = [x for x in list(parents_state_amount.keys()) if x != parent]
                
                parents_state_ranges = {}
                for p in rest_parents: # save the range of states of rest_parents
                    parents_state_ranges[p] = cpd.state_names[p]#list(range(parents_state_amount[p]))
                   
                # df mit allen möglichen Kombination der parent states erstellen
                s = list(parents_state_ranges.values())
                combinations = pd.DataFrame(list(product(*s)), columns=list(parents_state_ranges.keys()))     
        
                # für jede Kombination: setze -1 Wert des nodes auf 1, alle anderen auf 0     
                for row in range(len(combinations)):
                    for col in combinations.columns:
                        kwargs[col]=combinations[col][row]
                        
                    # jede Ausprägung updaten    
                    for n in cpd.state_names[node]:
                        if n == "-1":
                            kwargs["value"]=1 # dem nan Wert 100% geben, wenn ein parent -1 ist
                        else:
                             kwargs["value"]=0 # allen anderen states 0% geben, wenn ein parent -1 ist    
                        
                        kwargs[node] = n               
                        cpd.set_value(**kwargs)
                    counter = counter + 1
                    if counter%10000 == 0:
                        print("changed "+str(counter)+"th cpd: "+node+" with parents: "+str(kwargs))
    print(str(counter)+" CPDs with nan parents changed in total")
    return bn

# %%% CORRECT ACTIVITY DURATION CPDS
def correctWrongActivityDurationCPDs(model, learningData):
    """
    Takes the Bayesian Model and checks for every CPD in every duration node, if all the values have the same probability.
    If so, the avg distribution of the learningData's time duration of the regarding activity is used.

    Parameters
    ----------
    bayesianModel : Learned Bayesian Model with calculated CPDs
    learningData: pd DataFrame, which was used to train the model

    Returns
    -------
    bn : learned Bayesian Model with corrected CPDs

    """
    
    srv = convertToString(learningData)
    activityTimeDistribution = getActivityTypesDurationDistribution(srv)
    bn = model

    nodes = ["durationOfActivity"]
    indices = list(range(1,8))
    f = list([nodes, indices])
    nodesDF = pd.DataFrame(list(product(*f)), columns=["node", "indices"])

    nodesDF["combined"] = nodesDF.node + nodesDF.indices.astype(str)
    
    finalNodes = [x for x in nodesDF.combined if x in bn]

    counter = 0
    amountOfCPDs = 0
    for n in finalNodes: 
        node = n
        
        cpd = bn.get_cpds(node)
        activityParent = ""
        parents_state_amount = {}
        parents_state_names = {}
        
        for parent in cpd.variables[1:]:
            parents_state_amount[parent] = bn.get_cpds(parent).variable_card # get the amount of states for all parents
            parents_state_names[parent] = bn.get_cpds(parent).state_names[parent] # get the state names for all parents
            if parent.startswith("activity"): # save the name of the activity parent
              activityParent = parent
              
        
        # create dictionary with combination of all parents possibilities
        s = list(parents_state_names.values())
        combinations = pd.DataFrame(list(product(*s)), columns=list(parents_state_names.keys())) # df mit allen möglichen Kombination der parent states erstellen    
        #combinations_string = convertToString(combinations.astype(int))
        
        amountOfCPDs = amountOfCPDs + len(combinations)
        
        # loop through each combination of parents states
        for row in range(len(combinations)):
            kwargs = {}
            states_prob = []
            for col in combinations.columns:
                kwargs[col] = combinations[col][row]
            
            # save the probabilities for every combination in list
            for i in cpd.state_names[node]:
                kwargs[node] = i      
                states_prob.append(cpd.get_value(**kwargs))
            
            # if all probabilities of a combination are the same --> happens if no data was available
            # update data to average distribution
            if all(x == states_prob[0] for x in states_prob):
                
                # get the time distribution for the activity type
                activityOfCombination = pd.DataFrame(data = [combinations[activityParent][row]], columns=[activityParent])
                activityOfCombination = convertToString(activityOfCombination.astype(int))
                activityOfCombination = activityOfCombination[activityParent][0]
                
                if str(activityOfCombination)=="nan":
                    continue
                
                dist = activityTimeDistribution[activityOfCombination].fillna(0)
                
                # only keep dist values, which exist in state names for that node
                # create dataframe of nodes's possible state_names and convert to string
                state_names = cpd.state_names[node]
                state_names = pd.DataFrame(data=state_names, columns = ["durationOfActivity"]).astype(int)
                state_names = convertToString(state_names)


                #test = test["dining"].fillna(0).rename(columns={"duration":"durationOfActivity"})
                # assign distribution values to state names and recalculate the distribution (values change, since one node may not have all states)
                dist = pd.merge(state_names, dist, how="left", left_on="durationOfActivity", right_index=True)
                dist.distribution = dist.distribution / dist.distribution.sum()

                # set the duration as index 
                dist.index = dist.durationOfActivity
                dist = dist.drop("durationOfActivity", axis = 1)
                                          
                # update all values to average distribution
                for n in cpd.state_names[node]:
                    # get the relevant time group in string format
                    timeString = pd.DataFrame(data = [n], columns = ["durationOfActivity"])
                    timeString = convertToString(timeString.astype(int))
                    timeString = timeString["durationOfActivity"][0]

                    # set nan value to probability 0, since those cases where covered before in correctWrongData function
                    if n == "-1":
                        kwargs[node] = "-1" 
                        kwargs["value"] = 0
                        cpd.set_value(**kwargs)
                    # set other values to average distribution
                    else:                
                        kwargs[node] = n
                        kwargs["value"] = dist.loc[timeString,"distribution"]
                        cpd.set_value(**kwargs)
                counter = counter+1
                if counter%10 == 0:
                    print("changed "+str(counter)+"th cpd: "+node+" with parents: "+str(kwargs))
    
    print("ActivityDurations: " + str(amountOfCPDs) +" CPDs existing")
    print(str(counter)+" activityDuration CPDs changed in total (" + str(round(counter/amountOfCPDs*100,2)) + "%)") 
    
    return bn

# %%% CORRECT STARTTIME CPDS
def correctWrongStartTimeCPDs(model, learningData):
    """
    Takes the Bayesian Model and checks for every CPD in every StartTime node, if all the values have the same probability.
    If so, the avg distribution of the learningData's StartTime regarding the last StartTime is used.

    Parameters
    ----------
    bayesianModel : Learned Bayesian Model with calculated CPDs
    learningData: pd DataFrame, which was used to train the model

    Returns
    -------
    bn : learned Bayesian Model with corrected CPDs

    """
    
    srv = convertToString(learningData)
    followingStartTimeDistribution = getFollowingStartTimeDistribution(srv)
    bn = model

    nodes = ["startTimeOfActivity"]
    indices = list(range(2,8))
    f = list([nodes, indices])
    nodesDF = pd.DataFrame(list(product(*f)), columns=["node", "indices"])

    nodesDF["combined"] = nodesDF.node + nodesDF.indices.astype(str)
    finalNodes = [x for x in nodesDF.combined if x in bn]

    counter = 0
    amountOfCPDs = 0
    for n in finalNodes: 
        node = n
        
        cpd = bn.get_cpds(node)
        startTimeParent = ""
        parents_state_amount = {}
        parents_state_names = {}
        
        for parent in cpd.variables[1:]:
            parents_state_amount[parent] = bn.get_cpds(parent).variable_card # get the amount of states for all parents
            parents_state_names[parent] = bn.get_cpds(parent).state_names[parent] # get the state names for all parents
            if parent.startswith("startTimeOfActivity"): # save the name of the activity parent
              startTimeParent = parent
        
        # create dictionary with combination of all parents possibilities
        s = list(parents_state_names.values())
        combinations = pd.DataFrame(list(product(*s)), columns=list(parents_state_names.keys())) # df mit allen möglichen Kombination der parent states erstellen    
        combinations_string = convertToString(combinations.astype(int))
        
        amountOfCPDs = amountOfCPDs + len(combinations)
        
        # loop through each combination of parents states
        for row in range(len(combinations)):
            kwargs = {}
            states_prob = []
            for col in combinations.columns:
                kwargs[col] = combinations[col][row]
            
            # save the probabilities for every combination in list
            for i in cpd.state_names[node]:
                kwargs[node] = i      
                states_prob.append(cpd.get_value(**kwargs))
            
            # if all probabilities of a combination are the same --> happens if no data was available
            # update data to average distribution for previous startTime
            if all(x == states_prob[0] for x in states_prob):
                
                # get state of the startTimeParent
                previousStartTime = combinations_string[startTimeParent][row]
                distribution = followingStartTimeDistribution[previousStartTime]
                
                # cases with previousStartTime as nan are covered in correctWrongCPDS()
                if str(previousStartTime)=="nan":
                    continue
                        
                # get cpd states as strings for distribution dictionary
                states = pd.DataFrame(data = cpd.state_names[node], columns = ["startTimeOfActivity"]).astype(int)
                states_string = convertToString(states)
                
                # only have distribution of possible states               
                distribution = {key : distribution[key] for key in states_string.startTimeOfActivity}
                
                # recalculate distribution, since irrelevant values are gone and values have to add up to 1
                distribution = {key : distribution[key] / sum(distribution.values()) for key in distribution.keys()}

                
                # update all values to average distribution
                for n, s in zip(cpd.state_names[node], states_string.startTimeOfActivity):              
                    kwargs[node] = n
                    kwargs["value"] = distribution[s]
                    cpd.set_value(**kwargs)
                counter = counter+1
                if counter%1000 == 0:
                    print("changed "+str(counter)+"th cpd: "+node+" with parents: "+str(kwargs))

    
    print("Startimes: " + str(amountOfCPDs) +" CPDs existing")
    print(str(counter)+" startTime CPDs changed in total (" + str(round(counter/amountOfCPDs*100,2)) + "%)") 
    return bn

# %%% CORRECT TOTAL DISTANCE CPDS
def correctWrongTotalDistanceCPDs(model, learningData):
    """
    Takes the Bayesian Model and checks for every CPD in every totalDistance node, if all the values have the same probability.
    If so, the avg distribution of the learningData's totalDistances regarding the last totalDistance and distance is used.

    Parameters
    ----------
    bayesianModel : Learned Bayesian Model with calculated CPDs
    learningData: pd DataFrame, which was used to train the model

    Returns
    -------
    bn : learned Bayesian Model with corrected CPDs

    """
    srv = convertToString(learningData)
    followingTotalDistanceDistribution = getFollowingTotalDistanceDistribution(srv)
    bn = model

    nodes = ["totalDistance"]
    indices = list(range(2,8))
    f = list([nodes, indices])
    nodesDF = pd.DataFrame(list(product(*f)), columns=["node", "indices"])

    nodesDF["combined"] = nodesDF.node + nodesDF.indices.astype(str)
    
    finalNodes = [x for x in nodesDF.combined if x in bn]

    counter = 0
    for n in finalNodes: 
        node = n
        
        cpd = bn.get_cpds(node)
        parents_state_amount = {}
        parents_state_names = {}
        
        for parent in cpd.variables[1:]:
            parents_state_amount[parent] = bn.get_cpds(parent).variable_card # get the amount of states for all parents
            parents_state_names[parent] = bn.get_cpds(parent).state_names[parent] # get the state names for all parents
            if parent.startswith("totalDistance"): # save the name of the activity parent
                totalDistanceParent = parent
            if parent.startswith("distance"): # save the name of the activity parent
                distanceParent = parent
        
        # create dictionary with combination of all parents possibilities
        s = list(parents_state_names.values())
        combinations = pd.DataFrame(list(product(*s)), columns=list(parents_state_names.keys())) # df mit allen möglichen Kombination der parent states erstellen    
        combinations_string = convertToString(combinations.astype(int))

        # loop through each combination of parents states
        for row in range(len(combinations)):
            kwargs = {}
            states_prob = []
            for col in combinations.columns:
                kwargs[col] = combinations[col][row]
            
            # save the probabilities for every combination in list
            for i in cpd.state_names[node]:
                kwargs[node] = i      
                states_prob.append(cpd.get_value(**kwargs))
            
            # if all probabilities of a combination are the same --> happens if no data was available
            # update data to average distribution for previous startTime
            if all(x == states_prob[0] for x in states_prob):
                
                # get state of the startTimeParent
                previousTotalDistance = combinations_string[totalDistanceParent][row]
                previousDistance = combinations_string[distanceParent][row]
                distribution = followingTotalDistanceDistribution[previousTotalDistance][previousDistance]
                
                # cases with previousDistance | previousTotalDistance as nan are covered in correctWrongCPDS()
                if str(previousTotalDistance)=="nan":
                    continue
                if str(previousDistance)=="nan":
                    continue
                        
                # get cpd states as strings for distribution dictionary
                states = pd.DataFrame(data = cpd.state_names[node], columns = ["totalDistance"]).astype(int)
                states_string = convertToString(states)
                
                # update all values to average distribution
                for n, s in zip(cpd.state_names[node], states_string.totalDistance):              
                    kwargs[node] = n
                    kwargs["value"] = distribution[s]
                    cpd.set_value(**kwargs)
                counter = counter+1
                if counter%10 == 0:
                    print("changed "+str(counter)+"th cpd: "+node+" with parents: "+str(kwargs))

    print(str(counter)+" totalDistance CPDs changed in total") 
    
    return bn

# %% FURTHER DATA MANIPULATON AND ANALYSIS
# %%% REPLACE GROUPED DATA WITH MEANS
def groupedDataToMeans(data, meansOfGroups):
    """
    Replaces the grouped data (strings) with means from the srv data

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with the result data of the model in string format
    meansOfGroups : dictionary with dataframes
        Dictionary, containing means to replace grouped data. 

    Returns
    -------
    df : pd.DataFrame
        DataFrame with replaced Data

    """
    
    df = data
    means_dict = meansOfGroups

    for i in range(1,8):
        for key in [x for x in means_dict.keys() if x not in "leavingHomeTime"]:            
            col = key+str(i)
            conditions_dict = means_dict[key].columns[0:-1]
            conditions = conditions_dict+str(i)
            
            df = pd.merge(df, means_dict[key], how="left", left_on=list(conditions), right_on=list(conditions_dict))
            df[col] = df.means
            df = df.drop(conditions_dict, axis = 1)
            df = df.drop("means", axis = 1)
    
    if "leavingHomeTime" in means_dict.keys():  
        key = "leavingHomeTime"
        col = key
        conditions_dict = means_dict[key].columns[0:-1]
        conditions = conditions_dict
        
        df = pd.merge(df, means_dict[key], how="left", left_on=list(conditions), right_on=list(conditions_dict))
        df[col] = df.means
        df = df.drop("means", axis = 1)
            
    return df

# %%% GET STARTTIME DISTRIBUTION OF FOLLOWING ACTIVITIES
def getFollowingStartTimeDistribution(data):
    """
    takes the learning data in string format and returns the distribution the following startTime

    Parameters
    ----------
    data : pd.DataFrame
        LearningData of the model

    Returns
    -------
    startTimes : nested dictionary
        dictionary

    """
    
    df = data

    possibleStartTimes = ["0-6Uhr", "6-10Uhr", "10-14Uhr", "14-18Uhr", "18-22Uhr", ">22Uhr", np.nan]
    
    startTimes = pd.DataFrame()    
    startTimes[["startTimeOfActivity0", "startTimeOfActivity1"]] = df[["startTimeOfActivity1", "startTimeOfActivity2"]]

    # DataFrame mit allen startTimes und nachfolgenden startTimes untereinander
    for i in range(2,7):
        newCols = {"startTimeOfActivity"+str(i):"startTimeOfActivity0", "startTimeOfActivity"+str(i+1):"startTimeOfActivity1"}
        startTimes = pd.concat([startTimes, df[["startTimeOfActivity"+str(i), "startTimeOfActivity"+str(i+1)]].rename(columns=newCols)], axis = 0)

    # nan rausfiltern, die werden in correctWrongCPDs gehandelt
    startTimes = startTimes[startTimes["startTimeOfActivity1"].isnull()==False]

    # Je startTime die Verteilung der nachfolgenden startTimes
    startTimes = pd.DataFrame(startTimes.groupby("startTimeOfActivity0").startTimeOfActivity1.value_counts()/startTimes.groupby("startTimeOfActivity0").startTimeOfActivity1.count(), columns=["distribution"]).reset_index()
    
    # convert to nested dictionary
    startTimes = startTimes.groupby('startTimeOfActivity0').apply(lambda x: pd.DataFrame(zip(x['startTimeOfActivity1'], x['distribution'])).groupby(0)[1].apply(list).to_dict()).to_dict()
    
    # write values as values instead of lists
    for key in startTimes.keys():
        for secKey in startTimes[key].keys():
            startTimes[key][secKey] = startTimes[key][secKey][0]
    
    # Verteilungen durch 0 ergänzen wenn startTime nicht vorhanden
    for key in startTimes.keys():
        for time in possibleStartTimes:
            if time not in startTimes[key].keys():
                startTimes[key][time]=0.0

    return startTimes

# %%% GET THE FOLLOWING TOTAL DISTANCE DISTRIBUTION
def getFollowingTotalDistanceDistribution(learningData):
    """
    takes the learning data in string format and returns the distribution 
    the following totalDistances depending on previous totalDistance and distance
    
    Parameters
    ----------
    data : pd.DataFrame
        LearningData of the model
    
    Returns
    -------
    startTimes : nested dictionary
        dictionary
    
    """
    df = learningData
    
    # possible distances
    possibleTotalDistances = ["0-1km", "1-2km", "2-5km", "5-10km", "10-20km", "20-50km", ">50km", np.nan]

    totalDistances = pd.DataFrame()    
    totalDistances[["totalDistance0", "distance1","totalDistance1"]] = df[["totalDistance1", "distance2", "totalDistance2"]]

    # DataFrame mit allen totalDistances, nachfolgenden distances und nachfolgenden totalDistances untereinander
    for i in range(2,7):
        newCols = {"totalDistance"+str(i):"totalDistance0","distance"+str(i+1):"distance1", "totalDistance"+str(i+1):"totalDistance1"}
        totalDistances = pd.concat([totalDistances, df[["totalDistance"+str(i), "distance"+str(i+1), "totalDistance"+str(i+1)]].rename(columns=newCols)], axis = 0)

    # nan rausfiltern, die werden in correctWrongCPDs behandelt
    totalDistances = totalDistances[totalDistances["distance1"].isnull()==False]

    # Je totalDistance und distance die Verteilung der nachfolgenden totalDistances
    totalDistances = pd.DataFrame(totalDistances.groupby(["totalDistance0", "distance1"]).totalDistance1.value_counts()/totalDistances.groupby(["totalDistance0", "distance1"]).totalDistance1.count(), columns=["distribution"]).reset_index()

    # convert to nested dictionary
    totalDistances = totalDistances.groupby(["totalDistance0", "distance1"]).apply(lambda x: pd.DataFrame(zip(x['totalDistance1'], x['distribution'])).groupby(0)[1].apply(list).to_dict()).to_dict()

    # erste keys sind noch Tupel --> in tiefer nested dict converten
    out = {}
    for key, value in totalDistances.items():
        k1, k2 = key
        out.setdefault(k1, {})[k2] = value

    totalDistances = out

    # write values as values instead of lists
    for key in totalDistances.keys():
        for secKey in totalDistances[key].keys():
            for thrdKey in totalDistances[key][secKey].keys():
                totalDistances[key][secKey][thrdKey] = totalDistances[key][secKey][thrdKey][0]

    # Verteilungen durch 0 ergänzen wenn startTime nicht vorhanden
    for key in totalDistances.keys():
        for secKey in totalDistances[key].keys():
            for dist in possibleTotalDistances:
                if dist not in totalDistances[key][secKey].keys():
                    totalDistances[key][secKey][dist]=0.0
    
    return totalDistances

# %%% ADD DISTANCE VALUE COLUMNS
def addDistanceValueColumns(data, distances):
    """
    add for every distance column a distance_value column with a random value out of the distance distribution of the group

    Parameters
    ----------
    data : pd.DataFrame
        DESCRIPTION.
    distances : pd.DataFrame
        DataFrame with one "distance" column (floats) and one "distance_group" column (string)

    Returns
    -------
    DataFrame with value columns added

    """
    
    df = data.copy()
    distances_values = distances

    # create dictionary with df per distance group
    distance_dict = dict(tuple(distances_values.groupby("distance_group")))
    
    # list of every distance column
    dist_cols = [x for x in df.columns if x.startswith("distance")]
    
    for dist_col in dist_cols:
        
        # list of every distance group in column
        dist_groups = [x for x in df[dist_col].unique() if str(x) != "nan"]
        
        # set every value to 0, then update each distance group with random choices from distance_dict
        df[dist_col + "_value"] = 0
        for dist_group in dist_groups:  
            rows = df[dist_col].eq(dist_group)
            df.loc[rows, dist_col+"_value"] = np.random.choice(list(distance_dict[dist_group].distance), size = rows.sum())
    
    # create totalDistance_value columns and group them
    distValue_cols = [x for x in df.columns if x.startswith("distance") and "value" in x]
    df["totalDistance_value"] = df[distValue_cols].sum(axis = 1)
    df["totalDistance_value"] = df["totalDistance_value"].replace(0,np.nan)
        
    df["totalDistance"] = pd.cut(df["totalDistance_value"],
                                    bins=[-1, 1, 2, 5, 10,
                                          20, 50, float("inf")],
                                    labels=["0-1km", "1-2km", "2-5km", "5-10km", "10-20km", "20-50km", ">50km"],
                                    right=False)  # von ... bis unter [)    
    df["totalDistance"] = df["totalDistance"].astype(str)
    df["totalDistance"] = df["totalDistance"].replace(np.NaN, "nan")
    
    return df
