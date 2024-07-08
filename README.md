
# Generierung und Prognostizierung von Aktivitätenketten und Wegedistanzen unter Verwendung Bayes’scher Netze

This project uses Bayesian Networks to generate daily patterns of a synthetic population in Brandenburg, Germany.

The data is generated for various scenarios in the years of 2022, 2030 and 2050.

The Bayesian Network is learned with data from the SrV traffic survey, the traffic data is generated based on public zensus data.

To reproduce the results, the folling steps need to be executed:



## Authors

- [@MoritzOff](https://www.github.com/MoritzOff)


## 1. Data preparation
To prepare the training set and the regional population data for the different scenarios, the **preparation.py** file must be executed first. The resulting data is saved in the *./output* folder. 

run time of the file: approx 5 minutes

memory used: 3 GB RAM
## 2. Test and build a Bayesian Network
With the prepared training dataset, different model structures are tested in the **createModels.py** file. The documentation of the tests is saved in the *./output/model_tests_fScores* folder, the resulting model is saved as model_complete.bif in the *./output* folder. The resulting model structure is visualized in *./output/Graph-model-complete.html*

run time of the file: approx 4 minutes

memory used: 5 GB RAM
## 3. Validate model and run scenarioAnalysis
The following steps are independant from each other and can be executed on a hpc-cluster at the same time.
### 3.1 Validate model with cross validation
The **crossValidation.py** file is used to validate the trained model. Teh data is split in 80% train data and 20% test data. The model parameters are trained with the train data and traffic patterns are generated based on the test data. All resulting data sets, the originals and the generated as well are saved in the *./output/crossValidation* folder and in the *./output/scenarioAnalysis/outputData* folder.

run time of the file: approx 10 minutes

memory used: 3 GB RAM
### 3.2 Run Basecase-Scenario 2022
The **basecase.py** file is used to load the learned Bayesian Network and generate the data for the basecase population. The results are saved in *./output/basecase* folder and in the *./output/scenarioAnalysis/outputData* folder.

run time of the file: approx 15 minutes

memory used: 5 GB RAM
### 3.3 Run 2030 scenarios
The **scenario2030.py** file is used to load the learned Bayesian Network and generate the data for the three 2030 populations. The results are saved in *./output/scenario2030* folder and in the *./output/scenarioAnalysis/outputData* folder.

run time of the file: approx 30 minutes

memory used: 5 GB RAM
### 3.4 Run 2050 scenarios
The **scenario2050.py** file is used to load the learned Bayesian Network and generate the data for the basecase population. The results are saved in *./output/scenario2050* folder and in the *./output/scenarioAnalysis/outputData* folder.

run time of the file: approx 20 minutes

memory used: 6 GB RAM
## 4. Analyse the scenario results
The **scenarioAnalysis.py** file loads the output data from *./output/scenarioAnalysis/outputData* and generated various plots to analyse the data. The plots are saved in the different folders in *./output/scenarioAnalysis*

run time of the file: approx 45 minutes

memory used: 10 GB RAM

This the generation of various heatmaps takes a long time, they can be commented out if not needed
## Further information
The **methods.py** file contains various methods to manipulate, analyse or visualize the data. 

To keep further, unused data in the data set available, a forward sampling algorithm modified by Aurore Sallard and Milos Balac is used. This one can be found in the **FW-sampling.py** file.

The required python packages can be found in **requirements.txt**
