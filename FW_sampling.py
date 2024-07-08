import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pgmpy.estimators import HillClimbSearch, K2Score, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import itertools as it

import warnings
warnings.filterwarnings("ignore")

def _return_samples(samples, state_names_map=None):
    df = pd.DataFrame.from_records(samples)
    if state_names_map is not None:
        for var in df.columns:
            if var != "_weight":
                df[var] = df[var].map(state_names_map[var])
    return df
    
def _adjusted_weights(weights):
    error = 1 - np.sum(weights)
    if abs(error) > 1e-3:
        raise ValueError("The probability values do not sum to 1.")
    elif error != 0:
        warnings.warn(
            f"Probability values don't exactly sum to 1. Differ by: {error}. Adjusting values."
        )
        weights[-1] += error

    return weights
    
def sample_discrete(values, weights, size=1, seed=None):
    if seed is not None:
        np.random.seed(seed)

    weights = np.array(weights)
    if weights.ndim == 1:
        return np.random.choice(values, size=size, p=_adjusted_weights(weights))
    else:
        samples = np.zeros(size, dtype=int)
        unique_weights, counts = np.unique(weights, axis=0, return_counts=True)
        for index, size in enumerate(counts):
            samples[(weights == unique_weights[index]).all(axis=1)] = np.random.choice(
                values, size=size, p=_adjusted_weights(unique_weights[index])
            )
        return samples
        
def sample_discrete_maps(states, weight_indices, index_to_weight, size, seed = None):
    if seed is not None:
        np.random.seed(seed)
        
    samples = np.zeros(size, dtype = int)
    unique_weight_indices, counts = np.unique(weight_indices, return_counts = True)
    
    for weight_size, weight_index in zip(counts, unique_weight_indices):
        samples[weight_indices == weight_index] = np.random.choice(
            states, size=weight_size, p=index_to_weight[weight_index]
        )
    return samples

def simulate(bn, 
        n_samples = 1000,
        do = None,
        evidence = None,
        virtual_evidence = None,
        virtual_intervention = None,
        include_latents = False,
        partial_samples = None,
        seed = None,
        show_progress = False,):
    
    my_bn = bn.copy()
    state_names = bn.states
    
    if evidence is None:
        evidence = {}
    
    # Step 3: If no evidence do a forward sampling
    if len(evidence) == 0:
        samples = my_forward_sample(my_bn,
            size=n_samples,
            include_latents=include_latents,
            seed=seed,
            show_progress=show_progress,
            partial_samples=partial_samples,
        )

    # Step 5: Postprocess and return
    if include_latents:
        return samples
    else:
        return samples#.loc[:, list(set(bn.nodes()) - bn.latents)]



def pre_compute_reduce_maps(bn, variable):
    variable_cpd = bn.get_cpds(variable)
    variable_evid = variable_cpd.variables[:0:-1]
    cardinality = variable_cpd.variable_card

    state_combinations = [
        tuple(sc)
        for sc in it.product(
            *[range(bn.get_cpds(var).variable_card) for var in variable_evid]
        )
    ]
    
    weights_list = np.array(
        [
            variable_cpd.reduce(
                list(zip(variable_evid, sc)), inplace=False, show_warnings=False
            ).values
            for sc in state_combinations
        ]
    )

    unique_weights, weights_indices = np.unique(
        weights_list, axis=0, return_inverse=True
    )

    # convert weights to index; make mapping of state to index
    state_to_index = dict(zip(state_combinations, weights_indices))

    # make mapping of index to weights
    index_to_weight = dict(enumerate(unique_weights))

    # return mappings of state to index, and index to weight
    return state_to_index, index_to_weight
    
    

def my_forward_sample(bn, size = 1, include_latents = False, seed = None, show_progress = False, partial_samples = None):
    topological_order = list(nx.topological_sort(bn))
    sampled = pd.DataFrame(columns=list(bn.nodes()))
    
    if (partial_samples is not None):
        for column in partial_samples.columns:
            if not (column in topological_order):
                print("    " + column + " is in partial_samples but is not a node, copying the values")
                sampled[column] = partial_samples.loc[:, column].values


    for node in topological_order:
        print("  ", node)
        # If values specified in partial_samples, use them. Else generate the values.
        if (partial_samples is not None) and (node in partial_samples.columns):
            sampled[node] = partial_samples.loc[:, node].values
            size = len(partial_samples)
            print("    Not sampling, in the partial samples data set")
        else:
            print("    Sampling from BN")
            cpd = bn.get_cpds(node)
            states = range(cpd.variable_card) # variable_card = Anzahl der möglichen Ausprägungen des Knotens
            evidence = cpd.variables[:0:-1] # Auf den Knoten zeigende Knoten
            
            name_to_index = cpd.name_to_no[node] # dictionary, welches jeder Ausprägung einen Zahlwert zuordnet
            index_to_name = {v: k for k, v in name_to_index.items()} # dictionary, welches jede Ausprägung von Zahl zu String zurückwandelt
            
            if evidence: # wenn Knoten auf den Knoten zeigen/wenn es kein erster Knoten ist
                evidence_values = np.vstack([sampled[i] for i in evidence]) # transponiert alle auf den Knoten zeigende Spalten und schreibt sie in Zeilen untereinander               
                evidence_values = evidence_values.astype("str")
                
                unique, inverse = np.unique( # erstellt einen df mit allen existierenden Kombinationen der Ausprägungen der Spalten und ein Array, das für jeder Zeile den entsprechendenunique Wert zuweist
                    evidence_values.T, axis=0, return_inverse=True
                )
                #%
                dic_evid_name_to_index = {}
                dic_index_to_evid      = {}
                i = 0
                
                for var in evidence: # für jeden Knoten, der auf den Knoten zeigt:
                    dic_aux              = bn.get_cpds(var).name_to_no[var] # dictionary mit für jede Ausprägung des Knotens einen Zahlenwert
                    dic_index_to_evid[i] = var # dict, welches von 0 hochzählend die draufzeigenden Knotennamen enthält
                    i = i+1
                    dic_aux_2            = {}
                    for k,v in dic_aux.items(): # schreibt jeden key von dict_aux als string
                        if type(k)!= str:
                            dic_aux_2[str(k)] = v
                        else:
                            dic_aux_2[k] = v
                                                        
                    dic_evid_name_to_index[var] = dic_aux_2 # dict, in dem jeder draufzeigende Knoten ein weiteres dict mit Umwandlung der Werte in Zahl enthält
                                                            # müsste self.state_names_map ersetzen
                state_to_index, index_to_weight = pre_compute_reduce_maps( 
                    bn, variable=node
                )
                # state_to_index ist dict, welches jeder Kombination der evidence Ausprägungen einen index gibt
                # index_to_weight ist dict, welches für jede Kombination der draufzeigenden Knoten ein Array der Outputwahrscheinlichkeiten enthält
                
                dic1 = dic_evid_name_to_index
                dic2 = dic_index_to_evid
                
                unique = [[dic1[dic2[i]][s[i]]  for i in range(len(s))] for s in unique] # Liste mit Arrays der möglichen Kombinationen der draufzeigenden Knoten
                
                weight_index = np.array([state_to_index[tuple(u)] for u in unique])[
                    inverse
                ] # Array mit Länge von trips_int index-Einträgen, welche auf die entsprechende evidence Kombination in state_to_index zeigen
                
                aux_column = node + "_aux"
                
                sampled[aux_column] = sample_discrete_maps( # eigentliches sampling
                    states, weight_index, index_to_weight, size
                )
                
                sampled[node] = [index_to_name[no] for no in sampled[aux_column]] # Umwandlung von Zahlen in ursprüngliche Strings
                
                del sampled[aux_column]
                
            else:
                weights = cpd.values
                aux_column = node + "_aux"
                sampled[aux_column] = sample_discrete(states, weights, size)
                sampled[node] = [index_to_name[no] for no in sampled[aux_column]]
                del sampled[aux_column]

    samples_df = sampled
    if not include_latents:
        samples_df.drop(bn.latents, axis=1, inplace=True)
    return samples_df