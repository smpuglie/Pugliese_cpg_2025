# Copy and pasted from the hub_circuits branch which I'm sure is not good git protocol
import numpy as np
import yaml
import sys, os
import argparse
import matplotlib.pyplot as plt
#from src.sim_utils_Jax import *
from src.sim_utils_test import *
import pandas as pd


def removal_probability(maxFrs,neuronsToExclude=[]):
    """assign a probability of removal to each neuron based on their max FR"""
    tmp = 1/maxFrs
    tmp[~np.isfinite(tmp)] = 0
    tmp[neuronsToExclude] = 0
    p = tmp/np.sum(tmp)

    #Manually adjusting to ensure p sums to exactly 1
    #TODO: Figure out a better way to fix this issue
    """
    if np.sum(p) != 1:
        for i in range(len(p)):
            if p[i] > 0:
                print(f'adjust {i}')
                p[i] += 1-np.sum(p)
                break
                
    """

    return p

#Manually doing choice because the numpy version 
#Doesn't like that p[i] doesn't sum exactly to 1
def choice(a, p):
    p_sum = np.cumsum(p)
    event = np.random.uniform()
    event_num = 0
    while event > p_sum[event_num]: # Figure our which event happened
        event_num += 1
    return a[event_num]

def run_pruning_from_start(params,oscillationThreshold=0.1):
    #TODO there's no way of changing the oscillationThreshold right now

    wTable = pd.read_csv(params["dfPath"],index_col=0)
    allNeurons = wTable.index.to_numpy()
    mnIdxs = wTable.loc[wTable["class"]=="motor neuron"].index
    # inIdxs = np.setdiff1d(allNeurons,np.concatenate((np.array(params["stimNeurons"]),mnIdxs.to_numpy()))) # OK so this isn't *exactly* INs... but it's what we want
    inIdxs = np.setdiff1d(allNeurons,mnIdxs.to_numpy())
    removedStimNeurons = [[] for i in range(params["nSims"])]
    prevNeuronsPutBack = [[] for i in range(params["nSims"])]
    
    
    minRs, minimalCircuitParams = run_pruning_from_params(params,0,prevNeuronsPutBack,mnIdxs,inIdxs,oscillationThreshold,removedStimNeurons)

    # Save minimal circuit
    screenStr = minimalCircuitParams["screen"]
    simStr = screenStr+'_min'
    simDir = f"/data/users/jkl/vnc-closedloop/results/jax/{screenStr}/{simStr}"
    save_sim_data(minRs,minimalCircuitParams,saveDir=simDir)
    #wTable.loc[np.setdiff1d(wTable.index,minimalCircuitParams["removeNeurons"])].to_csv(f"../results/jax/{params["name"]}/minimalCircuit.csv")


def run_pruning_from_params(params,iter,prevNeuronsPutBack,mnIdxs,inIdxs,oscillationThreshold,removedStimNeurons):
    
    #nIters = len(params["allSeeds"])
    screenStr = params["name"]
    level = np.zeros(params["nSims"])
    lastRemoved = [None]*params["nSims"]
    start = 0
    end = int((params["T"])//params["dt"])
    neuronsPutBack = [[] for i in range(params["nSims"])]
    min_circuit = np.full(params["nSims"], False)

    #TODO: Figure this part out
    p = np.zeros([params["nSims"], len(inIdxs)])
    for i in range(params["nSims"]):
        p[i] = removal_probability(np.ones(len(inIdxs)),neuronsToExclude=np.in1d(inIdxs,np.union1d(params["removeNeurons"][i],removedStimNeurons[i])))

    prevParams = params.copy()

    while not np.all(min_circuit):
    #for iteration in range(nIters):

        # Run and save simulation
        simStr = screenStr+f'_iter{iter}'
        params["name"] = simStr
        params["screen"] = screenStr

        #TODO: Make sure params has "nSims"
        #Dont resample seeds when converged to minimum  circuit
        resample_idxs = np.where(min_circuit == False)[0]
        for sim in resample_idxs:
            params["seed"][sim] = int(np.random.default_rng().integers(100000))
            #params["seed"][sim] = params["allSeeds"][iter][sim]

        simDir = f"/data/users/jkl/vnc-closedloop/results/jax/{screenStr}/{simStr}"
        fullConfigPath = save_config(params,dir=simDir) # save config directly into the sim directory

        Rs, params_arr, params = run_sim_from_params(params)
        save_sim_data(Rs,params,saveDir=simDir)
        print(f"Completed run {iter}, results saved in {simDir}")
        # os.system(f"tsp python {os.path.abspath('../src/run_from_config.py')} '{fullConfigPath}'") # Run

        #params["stimI"] = prevParams["stimI"].copy()
        print(f'stimI after resetting: {params["stimI"]}')

        # Load results and evaluate score
        prevParams = params.copy()

        prevRemoveNeurons = prevParams["removeNeurons"]

        

        for i in range(params["nSims"]):

            # Do not update minimum circuits
            if min_circuit[i]:
                continue
            

            R = Rs[i]
            maxFrs = np.max(R,axis=1)
            activeMnIdxs = mnIdxs[np.where(maxFrs[mnIdxs]>0)]

            oscillationScore = sim_oscillation_score(R,activeMnIdxs,start,end)
            print(f"Oscillation score sim {i}: {oscillationScore}")
            if (oscillationScore < oscillationThreshold) or np.isnan(oscillationScore):
                # Want to RESET and try another

                
                params["removeNeurons"][i] = np.setdiff1d(np.array(prevRemoveNeurons[i]),lastRemoved[i]).astype(int).tolist()
                

                if lastRemoved[i] is not None:
                    neuronsPutBack[i] += [lastRemoved[i]] # keep track!
                    print(f"Neuron {lastRemoved[i]} put back")

                    # Set the probability of re-removing this neuron this round to 0
                    #TODO: Make sure inIdxs should be the same across all sims
                    p[i, np.where(inIdxs==lastRemoved[i])] = 0
                    p[i] = p[i]/np.sum(p[i])

                if ~np.isfinite(np.sum(p[i])):
                    # NO NEURONS LEFT TO REMOVE
                    if set(neuronsPutBack[i]) == set(prevNeuronsPutBack[i]):
                        print(f"sim {i} converged to minimal circuit")
                        min_circuit[i] = True
                        continue
                    else:
                        #TODO: see what else I need to add here
                        level[i] += 1
                        print(f"running level {level[i]}")
                        prevNeuronsPutBack[i] = neuronsPutBack[i].copy()
                        lastRemoved[i] = None
                        neuronsPutBack[i] = []
                        p[i] = removal_probability(np.ones(len(inIdxs)),neuronsToExclude=np.in1d(inIdxs,np.union1d(params["removeNeurons"][i],removedStimNeurons[i])))
                        continue

                        #TODO: No recursion!!!
                        #return run_pruning_from_params(params,iter,neuronsPutBack,mnIdxs,inIdxs,oscillationThreshold,screenStr,removedStimNeurons,level+1)

                neuronToRemove = choice(inIdxs,p[i])
                #neuronToRemove = inIdxs[np.argmax(p[i])]

            else:
                # Remove silent interneurons. (TODO Should I try a version where I leave them in?)
                silentINs = np.intersect1d(np.where(maxFrs==0)[0],inIdxs)
                params["removeNeurons"][i] = np.union1d(np.array(prevRemoveNeurons[i]),silentINs).astype(int).tolist()
                print(f"Neuron {lastRemoved[i]} removed")
                #TODO: Does stimNeurons change across sims???
                if lastRemoved[i] in params["stimNeurons"]:
                    removedStimNeurons[i] += [lastRemoved[i]]
                
                # Choose an interneuron to remove
                p[i] = removal_probability(maxFrs[inIdxs],neuronsToExclude=np.in1d(inIdxs,np.union1d(neuronsPutBack[i],removedStimNeurons[i])))
                print(np.sum(p[i]))
                if ~np.isfinite(np.sum(p[i])): # not sure I need this in the else block
                    # NO NEURONS LEFT TO REMOVE
                    if set(neuronsPutBack[i]) == set(prevNeuronsPutBack[i]):
                        print(f"sim {i} converged to minimal circuit")
                        min_circuit[i] = True
                        continue
                    else:

                        #TODO: make this the same as the other block like it
                        level[i] += 1
                        print(f"running level {level[i]}")
                        prevNeuronsPutBack[i] = neuronsPutBack[i].copy()
                        lastRemoved[i] = None
                        neuronsPutBack[i] = []
                        p[i] = removal_probability(np.ones(len(inIdxs)),neuronsToExclude=np.in1d(inIdxs,np.union1d(params["removeNeurons"][i],removedStimNeurons[i])))
                        continue
                        #return run_pruning_from_params(params,iter,neuronsPutBack,mnIdxs,inIdxs,oscillationThreshold,screenStr,removedStimNeurons,level+1)
                neuronToRemove = choice(inIdxs,p[i])
                #neuronToRemove = inIdxs[np.argmax(p[i])]

            lastRemoved[i] = neuronToRemove
            params["removeNeurons"][i] = params["removeNeurons"][i] + [int(neuronToRemove)]

        iter += 1
    
    return Rs, params

def get_args():
    parser = argparse.ArgumentParser("Run multiple simulations in task spooler")
    parser.add_argument("paramFile")

    return parser.parse_args()

def get_params(args):
    paramFile = args.paramFile
    with open(paramFile, 'r') as file:
        runParams = yaml.safe_load(file)
    
    paramsToIterate = runParams["paramsToIterate"]

    simParams = runParams.copy()
    for paramName in simParams.keys():
        if paramName in paramsToIterate:
            simParams[paramName] = runParams[paramName][args.idx]

    return simParams

def main():
    args = get_args()
    params = get_params(args)
    run_pruning_from_start(params)

if __name__ == "__main__":
    main()