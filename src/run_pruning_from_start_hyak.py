# Copy and pasted from the hub_circuits branch which I'm sure is not good git protocol
import numpy as np
import yaml
import sys, os
import argparse
import matplotlib.pyplot as plt
from sim_utils import *
import pandas as pd


def removal_probability(maxFrs,neuronsToExclude=[]):
    """assign a probability of removal to each neuron based on their max FR"""
    tmp = 1/maxFrs
    tmp[~np.isfinite(tmp)] = 0
    tmp[neuronsToExclude] = 0
    p = tmp/sum(tmp)

    return p

def run_pruning_from_start(params,oscillationThreshold=0.2):
    #TODO there's no way of changing the oscillationThreshold right now

    wTable = pd.read_csv(params["dfPath"],index_col=0)
    allNeurons = wTable.index.to_numpy()
    try:
        mnIdxs = wTable.loc[wTable["class"]=="motor neuron"].index
    except:
        mnIdxs = wTable.loc[wTable["motor module"].notna()].index
    # inIdxs = np.setdiff1d(allNeurons,np.concatenate((np.array(params["stimNeurons"]),mnIdxs.to_numpy()))) # OK so this isn't *exactly* INs... but it's what we want
    inIdxs = np.setdiff1d(allNeurons,mnIdxs.to_numpy())
    screenStr = params["name"]

    print("Running "+screenStr)
    
    minimalCircuitParams = run_pruning_from_params(params,0,[],mnIdxs,inIdxs,oscillationThreshold,screenStr,[],0)

    # Save minimal circuit
    wTable.loc[np.setdiff1d(wTable.index,minimalCircuitParams["removeNeurons"])].to_csv(f"../results/hyak/{screenStr}/minimalCircuit.csv")

def run_pruning_from_ongoing(params,oscillationThreshold=0.2):
    #TODO there's no way of changing the oscillationThreshold right now

    wTable = pd.read_csv(params["dfPath"],index_col=0)
    allNeurons = wTable.index.to_numpy()
    try:
        mnIdxs = wTable.loc[wTable["class"]=="motor neuron"].index
    except:
        mnIdxs = wTable.loc[wTable["motor module"].notna()].index
    # inIdxs = np.setdiff1d(allNeurons,np.concatenate((np.array(params["stimNeurons"]),mnIdxs.to_numpy()))) # OK so this isn't *exactly* INs... but it's what we want
    inIdxs = np.setdiff1d(allNeurons,mnIdxs.to_numpy())
    
    paramName = params["name"]
    iter = int(params["name"].split("iter")[-1])
    print("Running "+paramName)
    screenStr = paramName.split("_iter")[0]
    
    minimalCircuitParams = run_pruning_from_params(params,iter,[],mnIdxs,inIdxs,oscillationThreshold,screenStr,[],0)

    # Save minimal circuit
    wTable.loc[np.setdiff1d(wTable.index,minimalCircuitParams["removeNeurons"])].to_csv(f"../results/hyak/{screenStr}/minimalCircuit.csv")


def run_pruning_from_params(params,iter,prevNeuronsPutBack,mnIdxs,inIdxs,oscillationThreshold,screenStr,removedStimNeurons,level):
    lastRemoved = None
    start = 250
    end = -1
    neuronsPutBack = []

    p = removal_probability(np.ones(len(inIdxs)),neuronsToExclude=np.in1d(inIdxs,np.union1d(params["removeNeurons"],removedStimNeurons)))

    prevParams = params.copy()

    while True:

        # Run and save simulation
        simStr = screenStr+f'_iter{iter:>03}'
        params["name"] = simStr
        params["screen"] = screenStr
        # params["seed"] = int(np.random.default_rng().integers(100000)) 2025-03-06 we don't want to update the seed!!

        simDir = f"../results/hyak/{screenStr}/"
        fullConfigPath = save_config(params,dir=simDir) # save config directly into the sim directory

        sim, params = run_sim_from_params(params)
        # save_sim_data(sim,params,saveDir=simDir)
        # save_config(params,dir="../../configs")
        print(f"Completed run {iter}, results saved in {simDir}")
        # os.system(f"hyak python {os.path.abspath('../src/run_from_config.py')} '{fullConfigPath}'") # Run

        # Load results and evaluate score
        prevParams = params.copy()
        R = sim.R

        maxFrs = np.max(R,axis=1)
        activeMnIdxs = mnIdxs[np.where(maxFrs[mnIdxs]>0.001)]

        prevRemoveNeurons = prevParams["removeNeurons"]

        oscillationScore = sim_oscillation_score(R,activeMnIdxs,start,end)
        print(f"Oscillation score: {oscillationScore}")
        if (oscillationScore < oscillationThreshold) or np.isnan(oscillationScore):
            if iter == 0:
                raise Exception("BAD INITIALIZATION, QUITTING!")
            
            # Want to RESET and try another
            params["removeNeurons"] = np.setdiff1d(np.array(prevRemoveNeurons),lastRemoved).astype(int).tolist()

            # Reset the stim strength
            # tmp = load_from_yaml(fullConfigPath)

            # Keep track of removed neurons
            if lastRemoved is not None:
                neuronsPutBack += [lastRemoved] 
                print(f"Neuron {lastRemoved} put back")

                # Set the probability of re-removing this neuron this round to 0
                p[np.where(inIdxs==lastRemoved)] = 0
                p = p/np.sum(p)

            if ~np.isfinite(np.sum(p)):
                # NO NEURONS LEFT TO REMOVE
                if set(neuronsPutBack) == set(prevNeuronsPutBack):
                    print("converged to minimal circuit")
                    return params
                else:
                    print(f"running level {level+1}")
                    return run_pruning_from_params(params,iter,neuronsPutBack,mnIdxs,inIdxs,oscillationThreshold,screenStr,removedStimNeurons,level+1)
            neuronToRemove = np.random.choice(inIdxs,p=p)

        else:
            # Remove silent interneurons. (TODO Should I try a version where I leave them in?)
            silentINs = np.intersect1d(np.where(maxFrs==0)[0],inIdxs)
            params["removeNeurons"] = np.union1d(np.array(prevRemoveNeurons),silentINs).astype(int).tolist()
            print(f"Neuron {lastRemoved} removed")
            if lastRemoved in params["stimNeurons"]:
                removedStimNeurons += [lastRemoved]
            
            # Choose an interneuron to remove
            p = removal_probability(maxFrs[inIdxs],neuronsToExclude=np.in1d(inIdxs,np.union1d(neuronsPutBack,removedStimNeurons)))
            if ~np.isfinite(np.sum(p)): # not sure I need this in the else block
                # NO NEURONS LEFT TO REMOVE
                if set(neuronsPutBack) == set(prevNeuronsPutBack):
                    print("converged to minimal circuit")
                    return params
                else:
                    print(f"running level {level+1}")
                    return run_pruning_from_params(params,iter,neuronsPutBack,mnIdxs,inIdxs,oscillationThreshold,screenStr,removedStimNeurons,level+1)
            neuronToRemove = np.random.choice(inIdxs,p=p)

        lastRemoved = neuronToRemove
        params["removeNeurons"] = params["removeNeurons"] + [int(neuronToRemove)]

        iter += 1

def get_args():
    parser = argparse.ArgumentParser("Run multiple simulations on hyak")
    parser.add_argument("--batchFolder",type=str)
    parser.add_argument("--idx",type=int)

    return parser.parse_args()

def get_params(args):
    batchFolder = args.batchFolder
    paramFiles = [batchFolder+"/"+s for s in sorted(os.listdir(batchFolder))]

    paramFile = paramFiles[args.idx]
    simParams = load_from_yaml(paramFile)
    
    # paramsToIterate = runParams["paramsToIterate"]
    # simParams = runParams.copy()
    # for paramName in simParams.keys():
    #     if paramName in paramsToIterate:
    #         simParams[paramName] = runParams[paramName][args.idx]

    return simParams

def main():
    args = get_args()
    params = get_params(args)
    # ON HYAK: CHECK IF YOU ALREADY RAN THIS JOB AND GOT BOOTED
    resultsDir = f"../results/hyak/{params['name']}"
    if os.path.exists(resultsDir):
        if os.path.exists(resultsDir+"/minimalCircuit.csv"):
            print("You already found the minimal circuit!")
        else:
            #TODO I should have these have leading zeros. Will want to add :>03 to the fstring
            latestFile = sorted(os.listdir(resultsDir))[-1]
            if os.path.splitext(latestFile)[-1]!=".yaml":
                FileExistsError("Could not find param file to run :(")
            # latestFile = f"{params['name']}_iter{len(os.listdir(resultsDir))-1}.yaml"
            # if not os.path.exists(f"{resultsDir}/{latestFile}"):
            #     print("boo")
            else:
                params = load_from_yaml(f"{resultsDir}/{latestFile}")
                run_pruning_from_ongoing(params)
                # Right now I am not keeping track of things like level, prevRemoved, etc.
                # So this will slightly throw off the screen compared to what it would have been.
                # I will try to fix this in the future.
    else:
        run_pruning_from_start(params)

if __name__ == "__main__":
    main()