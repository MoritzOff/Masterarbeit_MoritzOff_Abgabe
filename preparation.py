# -*- coding: utf-8 -*-

#%% IMPORT PACKAGES
import pandas as pd
import os
import numpy as np
import methods as m

# %% DATA INPUT
# Data Input
persons = pd.read_csv("./input_data/SRV_Data/table-persons.csv", header=0)
activities = pd.read_csv("./input_data/SRV_Data/table-activities.csv", header=0)
data_regionalstatistik = pd.read_csv("./input_data/Bevoelkerung_Alter_Geschlecht_Erwerbsstatus.csv",
                                     skiprows=9,
                                     skipfooter=7,
                                     engine="python",
                                     names=["code", "name", "altersgruppe", "bevoelkerung", "erwerbspersonen", "erwerbstaetig_gesamt", "erwerbstaetig_m", "erwerbstaetig_w",
                                            "erwerbslos_gesamt", "erwerbslos_m", "erwerbslos_w", "nichtErwerbsP_gesamt", "nichtErwerbsP_m", "nichtErwerbsP_w"],
                                     encoding="latin1",
                                     delimiter=";")

data2022 = pd.read_csv("./input_data/Bevoelkerung_31.12.2022.csv",
                       skiprows=8,
                       skipfooter=4,
                       engine="python",
                       names=["stichtag", "code", "name",
                              "unter 3 Jahre_insgesamt", "3 bis unter 6 Jahre_insgesamt", "6 bis unter 10 Jahre_insgesamt", "10 bis unter 15 Jahre_insgesamt", "15 bis unter 18 Jahre_insgesamt", "18 bis unter 20 Jahre_insgesamt", "20 bis unter 25 Jahre_insgesamt", "25 bis unter 30 Jahre_insgesamt", "30 bis unter 35 Jahre_insgesamt", "35 bis unter 40 Jahre_insgesamt", "40 bis unter 45 Jahre_insgesamt", "45 bis unter 50 Jahre_insgesamt", "50 bis unter 55 Jahre_insgesamt", "55 bis unter 60 Jahre_insgesamt", "60 bis unter 65 Jahre_insgesamt", "65 bis unter 75 Jahre_insgesamt", "75 Jahre und mehr_insgesamt", "insgesamt_insgesamt",
                              "unter 3 Jahre_m", "3 bis unter 6 Jahre_m", "6 bis unter 10 Jahre_m", "10 bis unter 15 Jahre_m", "15 bis unter 18 Jahre_m", "18 bis unter 20 Jahre_m", "20 bis unter 25 Jahre_m", "25 bis unter 30 Jahre_m", "30 bis unter 35 Jahre_m", "35 bis unter 40 Jahre_m", "40 bis unter 45 Jahre_m", "45 bis unter 50 Jahre_m", "50 bis unter 55 Jahre_m", "55 bis unter 60 Jahre_m", "60 bis unter 65 Jahre_m", "65 bis unter 75 Jahre_m", "75 Jahre und mehr_m", "insgesamt_m",
                              "unter 3 Jahre_f", "3 bis unter 6 Jahre_f", "6 bis unter 10 Jahre_f", "10 bis unter 15 Jahre_f", "15 bis unter 18 Jahre_f", "18 bis unter 20 Jahre_f", "20 bis unter 25 Jahre_f", "25 bis unter 30 Jahre_f", "30 bis unter 35 Jahre_f", "35 bis unter 40 Jahre_f", "40 bis unter 45 Jahre_f", "45 bis unter 50 Jahre_f", "50 bis unter 55 Jahre_f", "55 bis unter 60 Jahre_f", "60 bis unter 65 Jahre_f", "65 bis unter 75 Jahre_f", "75 Jahre und mehr_f", "insgesamt_f"],
                       encoding="latin1",
                       delimiter=";")

data2050 = pd.read_csv("./input_data/Bevoelkerung2050.csv", 
                       header=0, 
                       delimiter=";",
                       decimal=",")

mittlereVariante2030 = pd.read_csv("./input_data/Bevoelkerung2030_mittlereVariante.csv", header = 0, index_col=None, sep = ";", decimal=",")
untereVariante2030 = pd.read_csv("./input_data/Bevoelkerung2030_untereVariante.csv", header = 0, index_col=None, sep = ";", decimal=",")
obereVariante2030 = pd.read_csv("./input_data/Bevoelkerung2030_obereVariante.csv", header = 0, index_col=None, sep = ";", decimal=",")

mittlereVariante2030.name = "mittlereVariante"
obereVariante2030.name = "obereVariante"
untereVariante2030.name = "untereVariante"

# Output Ordner erstellen
try:
    os.mkdir("./output")
except:
    print("Output Ordner existiert bereits, die Dateien werden ueberschrieben!")
    
# Output Ordner erstellen
try:
    os.mkdir("./output/cpds")
except:
    print("Output Ordner existiert bereits, die Dateien werden ueberschrieben!")


# Output Ordner erstellen basecase
try:
    os.mkdir("./output/basecase")
except:
    print("Output Ordner existiert bereits, die Dateien werden ueberschrieben!")

# Output Ordner erstellen 2030
try:
    os.mkdir("./output/scenario2030")
except:
    print("Output Ordner existiert bereits, die Dateien werden ueberschrieben!")

# Output Ordner erstellen 2050
try:
    os.mkdir("./output/scenario2050")
except:
    print("Output Ordner existiert bereits, die Dateien werden ueberschrieben!")

# Output Ordner erstellen Szenario Analyse
try:
    os.mkdir("./output/scenarioAnalysis")
    os.mkdir("./output/scenarioAnalysis/outputData")
except:
    print("Output Ordner existiert bereits, die Dateien werden ueberschrieben!")

# %% BASECASE DATASET
# %%% ZENSUS 2011 EMPLOYMENT VERTEILUNG
# Vorbereitung des 2011 Regionalstatistik Datensatzes, so dass das gelernte Modell angewandt werden kann

# Bevoelkerung = Erwerbspersonen_gesamt + Nichterwerbspersonen_gesamt
# Erwerbspersonen = erwerbstaetig_gesamt + erwerbslos_gesamt

# auffüllen leerer Daten, welche sitz aus dem Kontext berechnen lassen 
data_regionalstatistik = data_regionalstatistik.replace(to_replace=["x", ".", "-", "/", "", np.NaN], value=0)

data_regionalstatistik["erwerbstaetig_m"] = np.where((data_regionalstatistik["erwerbstaetig_gesamt"] != 0) & (data_regionalstatistik["erwerbstaetig_m"] == 0),
                                                     round(data_regionalstatistik["erwerbstaetig_gesamt"].astype(int)/2), 
                                                     data_regionalstatistik["erwerbstaetig_m"])
data_regionalstatistik["erwerbstaetig_w"] = np.where((data_regionalstatistik["erwerbstaetig_gesamt"] != 0) & (data_regionalstatistik["erwerbstaetig_w"] == 0),
                                                     data_regionalstatistik["erwerbstaetig_gesamt"].astype(int) - data_regionalstatistik["erwerbstaetig_m"].astype(int), 
                                                     data_regionalstatistik["erwerbstaetig_w"])

data_regionalstatistik["erwerbslos_gesamt"] = np.where((data_regionalstatistik["erwerbstaetig_gesamt"] != 0) & (data_regionalstatistik["erwerbslos_gesamt"] == 0),
                                                       data_regionalstatistik["erwerbspersonen"].astype(int) - data_regionalstatistik["erwerbstaetig_gesamt"].astype(int), 
                                                       data_regionalstatistik["erwerbslos_gesamt"])

data_regionalstatistik["erwerbslos_m"] = np.where((data_regionalstatistik["erwerbslos_gesamt"] != 0) & (data_regionalstatistik["erwerbslos_m"] == 0) & (data_regionalstatistik["erwerbslos_w"] == 0),
                                                  round(data_regionalstatistik["erwerbslos_gesamt"].astype(int)/2), 
                                                  data_regionalstatistik["erwerbslos_m"])

data_regionalstatistik["erwerbslos_m"] = np.where((data_regionalstatistik["erwerbslos_gesamt"] != 0) & (data_regionalstatistik["erwerbslos_m"] == 0) & (data_regionalstatistik["erwerbslos_w"] != 0),
                                                  data_regionalstatistik["erwerbslos_gesamt"].astype(int) - data_regionalstatistik["erwerbslos_w"].astype(int), 
                                                  data_regionalstatistik["erwerbslos_m"])

data_regionalstatistik["erwerbslos_w"] = np.where((data_regionalstatistik["erwerbstaetig_gesamt"] != 0) & (data_regionalstatistik["erwerbslos_w"] == 0),
                                                  data_regionalstatistik["erwerbslos_gesamt"].astype(int) - data_regionalstatistik["erwerbslos_m"].astype(int), 
                                                  data_regionalstatistik["erwerbslos_w"])

data_regionalstatistik["nichtErwerbsP_gesamt"] = np.where((data_regionalstatistik["bevoelkerung"] != 0) & (data_regionalstatistik["nichtErwerbsP_gesamt"] == 0),
                                                          data_regionalstatistik["bevoelkerung"].astype(int) - data_regionalstatistik["erwerbspersonen"].astype(int), 
                                                          data_regionalstatistik["nichtErwerbsP_gesamt"])

data_regionalstatistik["nichtErwerbsP_m"] = np.where((data_regionalstatistik["nichtErwerbsP_gesamt"] != 0) & (data_regionalstatistik["nichtErwerbsP_m"] == 0) & (data_regionalstatistik["nichtErwerbsP_w"] == 0),
                                                     round(data_regionalstatistik["nichtErwerbsP_gesamt"].astype(int)/2), 
                                                     data_regionalstatistik["nichtErwerbsP_m"])

data_regionalstatistik["nichtErwerbsP_m"] = np.where((data_regionalstatistik["nichtErwerbsP_gesamt"] != 0) & (data_regionalstatistik["nichtErwerbsP_m"] == 0) & (data_regionalstatistik["nichtErwerbsP_w"] != 0),
                                                     data_regionalstatistik["nichtErwerbsP_gesamt"].astype(int) - data_regionalstatistik["nichtErwerbsP_w"].astype(int), 
                                                     data_regionalstatistik["nichtErwerbsP_m"])

data_regionalstatistik["nichtErwerbsP_w"] = np.where((data_regionalstatistik["nichtErwerbsP_gesamt"] != 0) & (data_regionalstatistik["nichtErwerbsP_w"] == 0),
                                                     data_regionalstatistik["nichtErwerbsP_gesamt"].astype(int) - data_regionalstatistik["nichtErwerbsP_m"].astype(int), 
                                                     data_regionalstatistik["nichtErwerbsP_w"])


# Data cleaning
data_regionalstatistik = data_regionalstatistik[data_regionalstatistik["altersgruppe"] != "Insgesamt"]


# Loeschen aller gesamt-Spalten, der Erwerbspersonen Spalte, der Bevoelkerung Spalte, da die Daten bereits in anderen Spalten vorhanden sind
data_regionalstatistik = data_regionalstatistik[["code", "name", "altersgruppe", "erwerbstaetig_m",
                                                 "erwerbstaetig_w", "erwerbslos_m", "erwerbslos_w", "nichtErwerbsP_m", "nichtErwerbsP_w"]]

# Anzahl der Personen in eine Zeile/Person umwandeln
# Spalten in Zeilenwerte umwandeln
melted = pd.melt(data_regionalstatistik,
                 id_vars=["code", "name", "altersgruppe"],
                 value_vars=["erwerbstaetig_m",
                             "erwerbstaetig_w",
                             "erwerbslos_m",
                             "erwerbslos_w",
                             "nichtErwerbsP_m",
                             "nichtErwerbsP_w"],
                 var_name="employment",
                 value_name="anzahl")

# Infos aus employment Spalte (z.B.: erwerbstaetig_m) in employment und sex aufspalten
melted[["employment", "sex"]] = melted["employment"].str.split(
    "_", expand=True)
melted["region_type"] = ""

melted = melted[["code", "name", "altersgruppe",
                 "employment", "sex", "region_type", "anzahl"]]

# region_type für Staedte = 1, für Landkreise = 3
melted.loc[melted["name"].str.contains(r"Berlin") == True, "region_type"] = 1
melted.loc[melted["name"].str.contains(r"Stadt") == True, "region_type"] = 3
#melted.loc[melted["name"].str.contains(r"Cottbus") == True, "region_type"] = 3
melted.loc[melted["name"].str.contains(r"Landkreis") == True, "region_type"] = 3

# Pro Anzahl eine Person erstellen mit jeweiligen Auspraegungen
region_data = melted.loc[melted.index.repeat(melted.anzahl)].reset_index(drop=True)

# Anzahl Spalte loeschen, Spaltennamen korrigieren
region_data = region_data.drop(["anzahl", "name"], axis=1)
region_data = region_data.rename(columns={"altersgruppe": "age"})

region_data["sex"] = region_data["sex"].replace("w", "f")
region_data.code = region_data.code.astype(str)

# Jahresgruppen angleichen zum mergen
zuordnung1 = {"65 bis unter 70 Jahre": "65 bis unter 75 Jahre",
              "70 bis unter 75 Jahre": "65 bis unter 75 Jahre",
              "75 bis unter 80 Jahre": "75 Jahre und mehr",
              "80 bis unter 85 Jahre": "75 Jahre und mehr",
              "85 bis unter 90 Jahre": "75 Jahre und mehr",
              "90 Jahre und mehr": "75 Jahre und mehr"}

region_data["mergeAge"] = region_data.age.replace(zuordnung1)

# Berechnung der employments je Altersgruppe, Landkreis, Geschlecht
emplDist = pd.DataFrame(region_data.groupby(["code", "sex", "mergeAge"]).employment.value_counts()/region_data.groupby(["code", "sex", "mergeAge"]).employment.count(), columns=["dist"])
emplDist = emplDist.reset_index()


# %%% DATA 2022 PREPARATION
data2022 = data2022.replace(to_replace=["x", ".", "-", "/", "", np.NaN], value=0)
data2022 = data2022[data2022.columns.drop(list(data2022.filter(regex="insgesamt")))]
data2022.name = data2022.name.str.replace("  ", "")
data2022 = data2022[data2022.columns.drop("stichtag")]

melted = pd.melt(data2022,
                 id_vars=["code", "name"],
                 value_vars=data2022.columns[2:],
                 var_name="age",
                 value_name="anzahl")

melted[["age", "sex"]] = melted["age"].str.split("_", expand=True)

melted.anzahl = melted.anzahl.astype(int)
melted.code = melted.code.astype(str)

# Berlin den Regiontype 1 geben, allen anderen 3
melted["region_type"] = 3
melted.loc[(melted["name"].str.contains(r"Berlin") == True) & (melted["name"].str.contains(r"bei") == False), "region_type"] = 1

# Berliner Bezirke und, Brandenburg und Landkreise raus
data2022 = melted[(melted["name"].str.contains("Berlin-|Landkreis") == False) & (melted.code != "12")]

# Landkreis Code
data2022["LK-code"] = data2022.code.str[0:5]

# Altersgruppen angleichen zum mergen
zuordnung2 = {"unter 3 Jahre": "unter 5 Jahre",
              "3 bis unter 6 Jahre": "unter 5 Jahre",
              "6 bis unter 10 Jahre": "5 bis unter 10 Jahre",
              "15 bis unter 18 Jahre": "15 bis unter 20 Jahre",
              "18 bis unter 20 Jahre": "15 bis unter 20 Jahre"}
data2022["mergeAge"] = data2022.age.replace(zuordnung2)

# jeder Ausprägung die entsprechende Employment Verteilung zumergen
data2022 = pd.merge(data2022, emplDist, how="left", left_on=["LK-code", "sex", "mergeAge"], right_on=["code", "sex", "mergeAge"], suffixes=("", "_y"))

data2022.anzahl = round(data2022.anzahl * data2022.dist)
data2022.age = data2022.age.astype(str)

data2022 = data2022[data2022["LK-code"] != "11"]

# Pro Person eine Zeile
data2022 = data2022.loc[data2022.index.repeat(data2022.anzahl)].reset_index(drop=True)
data2022 = data2022.drop(["dist", "code_y", "mergeAge", "anzahl"], axis=1)

data2022 = data2022[["LK-code", "code", "name", "region_type", "age", "sex", "employment"]]
region_data2022 = data2022

# combine ages and save
region_data_int = m.convertToInteger(m.replaceOver65(data2022), swissCategories=False)
region_data_int.to_csv("./output/basecase/region_data_int_2022.csv")


# %% SCENARIO 2030 DATASETS
# %%% ALTERSGRUPPEN VERTEILUNG VON 2022
data2022_scenario2030 = data2022

# Berlin rausfiltern, nicht relevant
data2022_scenario2030 = data2022_scenario2030[data2022_scenario2030["LK-code"] != 11]

# get distribution of Gemeindeeinwohner per Landkreis
codeDist = (data2022.groupby("LK-code").code.value_counts() / data2022.groupby("LK-code").code.count()).reset_index().rename(columns = {0:"dist"})

# Landkreis Code als Strings
data2022_scenario2030["LK-code"] = data2022_scenario2030["LK-code"].astype(str)

zuordnung = {"unter 3 Jahre":"0-18", 
             "3 bis unter 6 Jahre":"0-18", 
            "6 bis unter 10 Jahre":"0-18", 
            "10 bis unter 15 Jahre":"0-18",
            "15 bis unter 18 Jahre":"0-18",
            "18 bis unter 20 Jahre":"18-25",
            "20 bis unter 25 Jahre":"18-25",
            "25 bis unter 30 Jahre":"25-45",
            "30 bis unter 35 Jahre":"25-45",
            "35 bis unter 40 Jahre":"25-45",
            "40 bis unter 45 Jahre":"25-45",
            "45 bis unter 50 Jahre":"45-65",
            "50 bis unter 55 Jahre":"45-65",
            "55 bis unter 60 Jahre":"45-65",
            "60 bis unter 65 Jahre":"45-65",
            "65 bis unter 75 Jahre":"65-80",
            "75 Jahre und mehr":"80 und älter"}

data2022_scenario2030["mergeAge"] = data2022_scenario2030["age"].apply(lambda x: zuordnung[x])

# Verteilung der Altersgruppen innerhalb der neuen, größeren Altersgruppen je Landkreis und Geschlecht
ageDist = pd.DataFrame(data2022_scenario2030.groupby(["code","mergeAge", "sex"]).age.value_counts()/data2022_scenario2030.groupby(["code", "mergeAge", "sex"]).age.count(), columns=["dist"])
ageDist = ageDist.reset_index()

# Verteilung der employments je Geschlecht und Altersgruppe und Landkreis
emplDist = pd.DataFrame(data2022_scenario2030.groupby(["code", "age", "sex"]).employment.value_counts()/data2022_scenario2030.groupby(["code", "age", "sex"]).employment.count(), columns=["dist"])
emplDist = emplDist.reset_index()

# %%% PROGNOSE DATEN DER LANDKREISE MIT ALTERSGRUPPENVERTEILUNG UND EMPLOYMENT MERGEN
varianten = [mittlereVariante2030, obereVariante2030, untereVariante2030]

for i in varianten:
    rawData = i
    varianteName = i.name
    rawData.code = rawData.code.astype(str)
    
    # Spalten mit Einwohnerzahlen * 1000 für absolute Werte
    rawData.iloc[:,2:] = (rawData.iloc[:,2:]*1000).astype(int)
    
    # Erstellung der Spalten für männlich und löschen der insgesamt Spalten
    for col in rawData.columns[2:8]:
        rawData[col+"_m"] = rawData[col]-rawData[col+"_f"]
        rawData = rawData.drop([col], axis = 1)
    
    # Format von viele Spalten in viele Zeilen ändern
    melted = pd.melt(rawData, 
                     id_vars = ["landkreis", "code"],
                     value_vars=rawData.columns,
                     var_name = "age",
                     value_name = "anzahl")
    
    # Info Alter_Geschlecht aufsplitten in zwei Spalten
    melted[["age","sex"]]= melted["age"].str.split("_",expand=True)
    
    melted = melted.rename(columns={"age":"mergeAge", "code":"LK-code"})
    
    # Anzahl aufsplitten nach Gemeinde innerhalb Landkreis
    region_data = pd.merge(melted[["LK-code", "mergeAge", "sex", "anzahl"]], codeDist, how = "left", on=["LK-code"])
    region_data.anzahl = round(region_data.anzahl * region_data.dist)
    region_data = region_data.drop("dist", axis = 1)

    # Anzahl auf die detaillierteren Altergruppen aufsplitten mit Verteilung aus Ursprungsdaten
    # je Landkreis, grober Altersgruppe und Geschlecht wird eine Verteilung in deteillierter Altersgruppen angenommen von 2022   
    region_data = pd.merge(region_data, ageDist, how = "left", left_on=["code", "mergeAge", "sex"], right_on=["code", "mergeAge", "sex"])   
    region_data.anzahl = region_data["dist"] * region_data["anzahl"]
    region_data = region_data.drop(["mergeAge", "dist"], axis = 1)

    # da die Altersgruppe 65-75 schwer zu matchen ist mit 65-80 wurden zunaechst alle 65-80 Jährigen eingestuft als 65 bis 75
    # dementsprechend sind in der Gruppe 65-75 ein fuenftel zu viele Menschen, die werden jetzt abgezogen und in 75 und mehr gesteckt
    abzug = region_data[region_data.age == "65 bis unter 75 Jahre"]
    abzug.anzahl = abzug.anzahl * 0.2
    abzug = abzug.rename(columns = {"anzahl" : "abzug"})
    abzug.age = abzug.age.str.replace("65 bis unter 75 Jahre", "75 Jahre und mehr")
    region_data.anzahl = np.where(region_data.age == "65 bis unter 75 Jahre", region_data.anzahl * 0.8, region_data.anzahl)
    
    region_data = pd.merge(region_data, abzug[["code", "age", "sex", "abzug"]], how = "left", on = ["code", "age", "sex"])
    region_data = region_data.fillna(0)
    region_data.anzahl = region_data.anzahl + region_data.abzug
    region_data = region_data.drop("abzug", axis = 1)
        
    # Employment Verteilung aus Ursprungsdaten
    # je Landkreis, Altersgruppe und Geschlecht anhand der 2022 (bzw. 2011) employment Verteilung die Anzahl aufsplitten
    region_data = pd.merge(region_data, emplDist,  how = "left", left_on=["code", "age", "sex"], right_on=["code", "age", "sex"])
    
    region_data.anzahl = round(region_data["dist"] * region_data["anzahl"])
    
    # je Person eine Spalte
    region_data = region_data.loc[region_data.index.repeat(region_data.anzahl)].reset_index(drop=True)
    region_data = region_data.drop(["dist", "anzahl"], axis = 1)
    
    region_data = region_data[["LK-code", "code", "age", "sex", "employment"]]


    region_data["region_type"] = 3
    
    region_data_int_2030 = m.convertToInteger(m.replaceOver65(region_data))     
    region_data_int_2030.to_csv("./output/scenario2030/region_data_int_2030_"+varianteName+".csv")

# %% SCENARIO 2050 DATASETS
# %%% DATA 2050 FIRST STEPS
for col in data2050.columns:
    try:
        data2050[col] = data2050[col].str.strip()
    except:
        pass
    
data2050 = data2050[(data2050.Jahr == 2050) & (data2050.Alter.str.contains("Insgesamt") == False) & (data2050.Region.str.contains("Brandenburg"))].reset_index(drop = True)
data2050 = data2050.drop(["Statistik", "Jahr", "Region"], axis = 1)
data2050 = data2050.rename(columns = {"Bevoelkerungsstand_in1000":"anzahl", "Alter":"age"})
data2050.anzahl = data2050.anzahl * 1000

data2050.age = data2050.age.replace({"unter 20":"unter 20 Jahre" ,
                                     "20 - 66":"20 bis unter 67 Jahre",
                                     "67 und älter":"67 Jahre und mehr"})

# %%% AGE, SEX, EMPLOYMENT DISTRIBUTIONS FROM 2022 DATA FOR 2050 SCENARIO

zuordnung2050 = {"unter 3 Jahre":"unter 20 Jahre","3 bis unter 6 Jahre":"unter 20 Jahre", 
                 "6 bis unter 10 Jahre":"unter 20 Jahre","10 bis unter 15 Jahre":"unter 20 Jahre", 
                 "15 bis unter 18 Jahre":"unter 20 Jahre","18 bis unter 20 Jahre":"unter 20 Jahre",
                 "20 bis unter 25 Jahre":"20 bis unter 67 Jahre","25 bis unter 30 Jahre":"20 bis unter 67 Jahre",
                 "30 bis unter 35 Jahre":"20 bis unter 67 Jahre","35 bis unter 40 Jahre":"20 bis unter 67 Jahre",
                 "40 bis unter 45 Jahre":"20 bis unter 67 Jahre","45 bis unter 50 Jahre":"20 bis unter 67 Jahre",
                 "50 bis unter 55 Jahre":"20 bis unter 67 Jahre","55 bis unter 60 Jahre":"20 bis unter 67 Jahre",
                 "60 bis unter 65 Jahre":"20 bis unter 67 Jahre","65 bis unter 75 Jahre":"67 Jahre und mehr",
                 "75 Jahre und mehr":"67 Jahre und mehr"}

# data2022 distribution of LK-code and code
codeDist = (region_data2022["code"].value_counts()/len(region_data2022)).reset_index().rename(columns = {"count":"code_dist"})
codeDist["id"]  = 0
data2050["id"]  = 0

# split dataset with code-distribution
data2050 = pd.merge(codeDist, data2050, how = "left", on = "id").drop("id", axis = 1)
data2050.anzahl = round(data2050.anzahl * data2050.code_dist)
data2050 = data2050.drop("code_dist", axis = 1)
data2050["LK-code"] = data2050.code.str[0:5]

region_data2022["mergeAge2050"] = region_data2022.age.replace(zuordnung2050)
region_data2022 = region_data2022[region_data2022["LK-code"] != 11]
ageDist2050 = pd.DataFrame(region_data2022.groupby(["code", "mergeAge2050"]).age.value_counts()/region_data2022.groupby(["code", "mergeAge2050"]).age.count(), columns=["dist"]).reset_index()
sexDist2050 = pd.DataFrame(region_data2022.groupby(["code", "age"]).sex.value_counts()/region_data2022.groupby(["code", "age"]).sex.count(), columns=["dist"]).reset_index()
emplDist2050 = pd.DataFrame(region_data2022.groupby(["code", "age", "sex"]).employment.value_counts()/region_data2022.groupby(["code", "age", "sex"]).employment.count(), columns=["dist"]).reset_index()

# split age groups in smaller groups with data2022 distribution
data2050 = pd.merge(data2050, ageDist2050, how = "left", left_on=["code", "age"], right_on=["code", "mergeAge2050"])
data2050.anzahl = round(data2050.anzahl * data2050.dist)
data2050 = data2050.drop(["dist", "age_x", "mergeAge2050"], axis = 1).rename(columns = {"age_y":"age"})

# split into female and male from 2022 distribution
data2050 = pd.merge(data2050, sexDist2050, how = "left", on=["code", "age"])
data2050.anzahl = round(data2050.anzahl * data2050.dist)
data2050 = data2050.drop("dist", axis = 1)

# split into employment types
data2050 = pd.merge(data2050, emplDist2050, how = "left", on=["code", "age", "sex"])
data2050.anzahl = round(data2050.anzahl * data2050.dist)
data2050 = data2050.drop("dist", axis = 1)
data2050["region_type"] = 3

# %%% VERSCHIEDENE VARIANTEN
data2050_mittlereVariante = data2050[data2050.Variante == 2].drop("Variante", axis = 1).reset_index(drop = True)
data2050_jungeVariante = data2050[data2050.Variante == 5].drop("Variante", axis = 1).reset_index(drop = True)
data2050_alteVariante = data2050[data2050.Variante == 4].drop("Variante", axis = 1).reset_index(drop = True)

data2050_mittlereVariante = data2050_mittlereVariante.loc[data2050_mittlereVariante.index.repeat(data2050_mittlereVariante.anzahl)].reset_index(drop=True)
data2050_mittlereVariante = data2050_mittlereVariante.drop(["anzahl"], axis = 1)
data2050_mittlereVariante = m.convertToInteger(m.replaceOver65(data2050_mittlereVariante))
data2050_mittlereVariante.to_csv("./output/scenario2050/region_data_int_2050_mittlereVariante.csv")

data2050_alteVariante = data2050_alteVariante.loc[data2050_alteVariante.index.repeat(data2050_alteVariante.anzahl)].reset_index(drop=True)
data2050_alteVariante = data2050_alteVariante.drop(["anzahl"], axis = 1)
data2050_alteVariante = m.convertToInteger(m.replaceOver65(data2050_alteVariante))
data2050_alteVariante.to_csv("./output/scenario2050/region_data_int_2050_alteVariante.csv")

data2050_jungeVariante = data2050_jungeVariante.loc[data2050_jungeVariante.index.repeat(data2050_jungeVariante.anzahl)].reset_index(drop=True)
data2050_jungeVariante = data2050_jungeVariante.drop(["anzahl"], axis = 1)
data2050_jungeVariante = m.convertToInteger(m.replaceOver65(data2050_jungeVariante))
data2050_jungeVariante.to_csv("./output/scenario2050/region_data_int_2050_jungeVariante.csv")


# %% LEARNING DATA SRV
# activities sortieren nach Person und chronologischer Reihenfolge
activities = activities.rename(columns={"type": "activity", "duration": "activityDuration"})
activities = activities.sort_values(["p_id", "n"], ascending=[True, True])

# Aktivities zusammenfassen
activity_replacement = {"edu_kiga":"education_school",
                        "edu_secondary" : "education_school",
                        "edu_primary" : "education_school",
                        "edu_higher" : "education_higher",
                        "edu_other" : "education_higher",
                        "work_business" : "work",
                        "shop_daily" : "shopping",
                        "shop_other" : "shopping",
                        "outside_recreation" : "leisure"}

activities.activity = activities.activity.replace(activity_replacement)

# Alter und Fahrentfernung in Gruppen einteilen (gleiche wie in Regionalstatistik Daten)
persons["age_values"] = persons.age
persons["age"] = pd.cut(persons["age"],
                        bins=[0, 3, 6, 10, 15, 18, 20, 25, 30, 35,
                              40, 45, 50, 55, 60, 65, 75, float("inf")],
                        labels=["unter 3 Jahre",
                                "3 bis unter 6 Jahre",
                                "6 bis unter 10 Jahre",
                                "10 bis unter 15 Jahre",
                                "15 bis unter 18 Jahre",
                                "18 bis unter 20 Jahre",
                                "20 bis unter 25 Jahre",
                                "25 bis unter 30 Jahre",
                                "30 bis unter 35 Jahre",
                                "35 bis unter 40 Jahre",
                                "40 bis unter 45 Jahre",
                                "45 bis unter 50 Jahre",
                                "50 bis unter 55 Jahre",
                                "55 bis unter 60 Jahre",
                                "60 bis unter 65 Jahre",
                                "65 bis unter 75 Jahre",
                                "75 Jahre und mehr"],
                        right=False)

# Umstrukturierung von vielen Zeilen einer Zeile pro Tag einer Person ============================================
# anstatt pro Activity eine Zeile: fuer die ersten x Activities eine Spalte
person_activities = activities.groupby("p_id")["activity"].apply(list).apply(pd.Series)
for col in person_activities.columns:
    person_activities = person_activities.rename(columns={col: "activity"+str(col)})
person_activities = person_activities.iloc[:, :9]


# gleiches fuer distance
person_distances = activities.groupby("p_id")["leg_dist"].apply(list).apply(pd.Series)
for col in person_distances.columns:
    person_distances = person_distances.rename(columns={col: "distance"+str(col)})
person_distances = person_distances.iloc[:, :8]

# gleiches fuer activityDuration
person_activityDurations = activities.groupby("p_id")["activityDuration"].apply(list).apply(pd.Series)
for col in person_activityDurations.columns:
    person_activityDurations = person_activityDurations.rename(columns={col: "activityDuration"+str(col)})
person_activityDurations = person_activityDurations.iloc[:, :8]

# gleiches fuer legDurations
person_legDurations = activities.groupby("p_id")["leg_duration"].apply(list).apply(pd.Series)
for col in person_legDurations.columns:
    person_legDurations = person_legDurations.rename(columns={col: "legDuration"+str(col)})
person_legDurations = person_legDurations.iloc[:, :8]


# zusammen joinen der activities, distances, activityDurations und legDurations
trips_person = pd.merge(person_activities, person_distances, on="p_id", how="left")
trips_person = pd.merge(trips_person, person_activityDurations, on="p_id", how="left")
trips_person = pd.merge(trips_person, person_legDurations, on="p_id", how="left")

# leavingHomeTime =================
trips_person["leavingHomeTime"] = trips_person.activityDuration0

# startTimeOfActivity =======================================================================
# erstellen der startTimeOfActivitys Spalten
trips_person["startTimeOfActivity1"] = trips_person.activityDuration0 +  trips_person.legDuration1

for i in range(2, 8):
    trips_person["startTimeOfActivity"+str(i)] = trips_person["startTimeOfActivity"+str(i-1)] + trips_person["activityDuration"+str(i-1)] + trips_person["legDuration"+str(i)]


# trips mit Personenattributen joinen
trips_person = pd.merge(trips_person, persons,
                        left_on="p_id", right_on="idx", how="left")
trips_person = trips_person.rename(columns={"gender": "sex"})

# max 7 Trips am Tag==================================================================================
# nur Personen mit maximal 8 Activities zulassen (inklusive "home")
trips_person = trips_person[trips_person.activity8.notnull() == False]

# nur Personen, deren Tag zu Hause startet
trips_person = trips_person[trips_person.activity0 == "home"]


# umbenennen der activityDuration
for i in range(1, 8):
    trips_person = trips_person.rename(columns={"activityDuration"+str(i): "durationOfActivity"+str(i)})

# startTimeOfActivity =========================================================================
avgActivityStartTimes = pd.DataFrame() # for calculation of avg start time per group
for i in range(1, 8):
    avgActivityStartTimes["activity" + str(i)] = trips_person["activity" + str(i)]
    avgActivityStartTimes["startTimeOfActivity" + str(i)] = trips_person["startTimeOfActivity" + str(i)]
    trips_person["startTimeOfActivity" + str(i)] = pd.cut(trips_person["startTimeOfActivity" + str(i)],
                                                          bins=[-1, 6*60, 10*60, 14*60, 18*60, 22*60, float("inf")],
                                                          labels=["0-6Uhr", "6-10Uhr", "10-14Uhr", "14-18Uhr", "18-22Uhr", ">22Uhr"],
                                                          right=False)  # von ... bis unter [)
    trips_person["startTimeOfActivity" + str(i)] = trips_person["startTimeOfActivity" + str(i)].astype(str)
    avgActivityStartTimes["groupedStartTimeOfActivity" + str(i)] = trips_person["startTimeOfActivity" + str(i)]

# leavingHomeTime =============================
avgLeavingHomeTime = pd.DataFrame()
avgLeavingHomeTime["leavingHomeTime"] = trips_person.leavingHomeTime

trips_person["leavingHomeTime"] = pd.cut(trips_person["leavingHomeTime"],
                                        bins=[-1, 6*60, 10*60, 14*60, 18*60, 22*60, float("inf")],
                                        labels=["0-6Uhr", "6-10Uhr", "10-14Uhr", "14-18Uhr", "18-22Uhr", ">22Uhr"],
                                        right=False)  # von ... bis unter [)
trips_person["leavingHomeTime"] = trips_person["leavingHomeTime"].astype(str)


avgLeavingHomeTime["groupedLeavingHomeTime"] = trips_person.leavingHomeTime

# durationOfActivities =======================================================
trips_person["durationOfActivitiesTotal"] = 0
dayEndsWithHome = trips_person.loc[:, "activity1":"activity7"].fillna(method="ffill", axis=1).activity7 == "home"
trips_person.durationOfActivitiesTotal[dayEndsWithHome] = trips_person.loc[:, "durationOfActivity1":"durationOfActivity7"].sum(axis=1, min_count=1)-1440  # last home activity is always 1440 minutes and substracted
trips_person.durationOfActivitiesTotal[-dayEndsWithHome] = trips_person.loc[:,"durationOfActivity1":"durationOfActivity7"].sum(axis=1, min_count=1)

for i in range(1, 8):
    trips_person["durationOfActivity" + str(i)] = pd.cut(trips_person["durationOfActivity" + str(i)],
                                                         bins=[-1, 5, 15, 30, 60, 2*60, 3*60, 5*60,
                                                               7*60, 8*60, 9*60, 12*60, float("inf")],
                                                         labels=["0-5min", "5-15min", "15-30min", "30-60min", "1-2h",
                                                                 "2-3h", "3-5h", "5-7h", "7-8h", "8-9h", "9-12h", ">12h"],
                                                         right=False)  # von ... bis unter [)
    trips_person["durationOfActivity" + str(i)] = trips_person["durationOfActivity" + str(i)].astype(str)

trips_person["durationOfActivitiesTotal"] = pd.cut(trips_person["durationOfActivitiesTotal"],
                                                   bins=[-1, 5, 15, 30, 60, 2*60, 3*60, 5*60,
                                                         7*60, 8*60, 9*60, 12*60, float("inf")],
                                                   labels=["0-5min", "5-15min", "15-30min", "30-60min", "1-2h",
                                                           "2-3h", "3-5h", "5-7h", "7-8h", "8-9h", "9-12h", ">12h"],
                                                   right=False).astype(str)  # von ... bis unter [)
# Vorbereitung distances csv =============================================

# nur mit regionalstädtischen Daten lernen lassen
trips_person = trips_person[trips_person.region_type == 3]

# Vorbereitung der Distances df
dist_cols = [x for x in trips_person.columns if x.startswith("distance") and x != "distance0"]
distances = pd.DataFrame(columns=["distance"])
for dist_col in dist_cols:
    distances["distance"] = pd.concat([distances["distance"], trips_person[dist_col]], axis = 0, ignore_index=True)

# totalDistances ============================================================
# die totalDistances nach jeder Distance einfügen

dist_cols = [x for x in trips_person.columns if x.startswith("distance") and x != "distance0"]
trips_person["totalDistance_value"] = trips_person[dist_cols].sum(axis = 1, skipna = True)

trips_person["totalDistance"] = pd.cut(trips_person["totalDistance_value"],
                                       bins=[-1, 1, 2, 5, 10,
                                             20, 50, float("inf")],
                                       labels=["0-1km", "1-2km", "2-5km", "5-10km", "10-20km", "20-50km", ">50km"],
                                       right=False)  # von ... bis unter [)
trips_person["totalDistance"] = trips_person["totalDistance"].astype(str)
trips_person.totalDistance[trips_person.totalDistance_value == 0] = np.NaN


# Distanzen gruppieren und Median der jeweiligen Gruppe speichern
for i in range(1, 8):
    trips_person["distance" + str(i) + "_value"] = trips_person["distance" + str(i)]
    trips_person["distance" + str(i)] = pd.cut(trips_person["distance" + str(i)],
                                               bins=[-1, 1, 2, 5, 10,
                                                     20, 50, float("inf")],
                                               labels=["0-1km", "1-2km", "2-5km", "5-10km", "10-20km", "20-50km", ">50km"],
                                               right=False)  # von ... bis unter [)
    trips_person["distance" + str(i)] = trips_person["distance" + str(i)].astype(str)
    
# legDuration ===================================================================
avgLegDuration = pd.DataFrame()
for i in range(1, 8):
    avgLegDuration["distance"+str(i)] = trips_person["distance"+str(i)]
    avgLegDuration["legDuration"+str(i)] = trips_person["legDuration"+str(i)]

    trips_person["legDuration" + str(i)] = pd.cut(trips_person["legDuration" + str(i)],
                                                          bins=[0, 5, 15, 30, 60, 90, 2*60, 3*60, 5*60, 7*60, float("inf")],
                                                          labels=["0-5min", "5-15min", "15-30min", "30-60min", "60-90min", "90-120min",
                                                                  "2-3h", "3-5h", "5-7h", ">7h"],
                                                          right=False)  # von ... bis unter [)
    trips_person["legDuration" + str(i)] = trips_person["legDuration" + str(i)].astype(str)

    avgLegDuration["groupedLegDuration" + str(i)] = trips_person["legDuration" + str(i)]

# employment Bezeichnungen ====================================================================


# employment anpassen:
# erwerbstaetig: jemand, der arbeitet
# erwerbslos: jemand, der arbeiten moechte, aber keinen Job hat
# nicht-Erwerbsperson: die, die weder erwerbstaetig nich erwerbslos sind

employment_dict = {"job_part_time": "erwerbstaetig",
                   "job_full_time": "erwerbstaetig",
                   "trainee": "erwerbstaetig",
                   "unemployed": "erwerbslos",
                   "school": "nichtErwerbsP",
                   "retiree": "nichtErwerbsP",
                   "student": "nichtErwerbsP",
                   "child": "nichtErwerbsP",
                   "homemaker": "nichtErwerbsP",
                   "other": "nichtErwerbsP"}
try:
    trips_person["employment"] = trips_person["employment"].apply(
        lambda x: employment_dict[x])
except:
    pass


layout = ["age", "sex", "region_type", "employment", "economic_status", "driving_license",
          "leavingHomeTime",
          "activity1", "activity2", "activity3", "activity4", "activity5", "activity6", "activity7",
          "distance1", "distance2", "distance3", "distance4", "distance5", "distance6", "distance7",
          "distance1_value", "distance2_value", "distance3_value", "distance4_value", "distance5_value", "distance6_value", "distance7_value",
          "legDuration1", "legDuration2", "legDuration3", "legDuration4", "legDuration5", "legDuration6", "legDuration7",
          "startTimeOfActivity1", "startTimeOfActivity2", "startTimeOfActivity3", "startTimeOfActivity4", "startTimeOfActivity5", "startTimeOfActivity6", "startTimeOfActivity7",
          "totalDistance", "totalDistance_value",
          "durationOfActivity1", "durationOfActivity2", "durationOfActivity3", "durationOfActivity4", "durationOfActivity5", "durationOfActivity6", "durationOfActivity7", "durationOfActivitiesTotal"]

trips_person = m.changeLayout(trips_person, layout)

# String Einträge der Dateien in Integer umwandeln
trips_int = m.convertToInteger(m.replaceOver65(trips_person), swissCategories=False)


trips_int["weight"] = 1

trips_int.to_csv("./output/learningData_SRV_int.csv")



# %% DISTANCES PER GROUP

distances["distance_group"] = pd.cut(distances.distance,
                                     bins=[-1, 1, 2, 5, 10, 20, 50, float("inf")],
                                     labels=["0-1km", "1-2km", "2-5km",
                                             "5-10km", "10-20km", "20-50km", ">50km"],
                                     precision=1,
                                     right=False)  # von ... bis unter [)

distances.to_csv("./output/distances_valuesAndGroups.csv")


