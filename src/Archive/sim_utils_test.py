import yaml
import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
#import src.vncNetParallel as vncNetParallel
import src.vncNetShuffle as vncNetTest
import pandas as pd
import sparse
from datetime import date
import jax.numpy as jnp

def load_from_yaml(yamlPath):
    """load data from .yml file path"""
    with open(yamlPath,'r') as file:
        data = yaml.safe_load(file)
    return data

def save_str(simStr):
    dateStr = date.today().strftime(format="%Y%m%d")
    return dateStr+"_"+simStr

def save_params(params, configPath='/data/users/jkl/vnc-closedloop/configs/'):
    #saveDir = f"/data/users/jkl/vnc-closedloop/configs/{params["screen"]}"
    saveDir = configPath+params["screen"]

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    fullConfigPath = saveDir + f"/{params["name"]}.yaml"
    with open(fullConfigPath,"w") as outfile:
        yaml.dump(params,outfile)

    print(fullConfigPath)

def getParams(simStr, batchNum, nSims, paramPath = "/data/users/jkl/vnc-closedloop/configs/20250122_mancBDN2activation.yaml"):
    params = load_from_yaml(paramPath)
    saveStr = save_str(simStr)
    params["screen"] = saveStr
    if batchNum < 10:
        params["name"] = saveStr + f'_0{batchNum}'
    else:
        params["name"] = saveStr + f'_{batchNum}'

    params["seed"] = np.random.randint(100000,size=nSims).tolist()

    return params


def get_sampled_param(params, paramToIter = ''):
    wTable = pd.read_csv(params["dfPath"],index_col=0)
    nNeurons = len(wTable)  
    nSims = len(params["seed"])
    stimNeurons = np.array(params["stimNeurons"])

    params_arr = vncNetTest.generate_params_from_config(wTable, params, nSims, nNeurons, stimNeurons)
    tAxis, T, dt, input, pulseStart, pulseEnd, tau, threshold, a, frCap, excSynapseMultiplier, inhSynapseMultiplier = params_arr
    if paramToIter == 'a':
        return a
    elif paramToIter == 'threshold':
        return threshold
    elif paramToIter == 'frcap':
        return frCap
    elif paramToIter == 'tau':
        return tau
    else:
        return a


def save_config(params,dir="../../configs"):
    """save parameters to a .yaml file"""

    if not os.path.exists(dir):
        os.makedirs(dir)

    simName = params["name"]
    fullConfigPath = dir + f"/{simName}.yaml"
    
    with open(fullConfigPath,"w") as outfile:
        yaml.dump(params,outfile,sort_keys=False)

    return fullConfigPath

def neuron_oscillation_score_old(activity,calculateFrequency=False):
    """calculate oscillation score for a neuron"""
    
    # Normalize
    activity = activity-np.min(activity)
    activity = activity/np.max(activity)
    
    # Get autocorrelation
    autocorr = signal.correlate(activity,activity)
    lags = signal.correlation_lags(len(activity),len(activity))
    autocorr = autocorr[lags>0]
    lags = lags[lags>0]
    
    peaks, peakProperties = signal.find_peaks(autocorr,height=(None,None),prominence=(None,None))

    if len(peaks) > 1:
        oscillationScore = np.max(peakProperties["prominences"])
    else:
        oscillationScore = 0

    if calculateFrequency:
        peaks = signal.find_peaks(activity,prominence=0.2)[0]
        try:
            frequency = np.mean(1/np.diff(peaks))
        except:
            frequency = np.nan
        
        return oscillationScore, frequency

    return oscillationScore

def neuron_oscillation_score_helper(activity,prominence):
    activity = activity-np.min(activity)
    activity = 2 * activity/np.max(activity) - 1

    autocorr = np.correlate(activity,activity,mode="full") / np.inner(activity,activity)
    lags = signal.correlation_lags(len(activity),len(activity))
    autocorr = autocorr[lags>0]
    lags = lags[lags>0]

    peaks, peakProperties = signal.find_peaks(autocorr,height=(None,None),prominence=prominence)
    if len(peaks) > 0:
        score = np.min([np.max(peakProperties["peak_heights"]),np.max(peakProperties["prominences"])])
        frequency = 1 / peaks[np.argmax(peakProperties["prominences"])]
    else:
        score = 0
        frequency = 0

    return score, frequency

def neuron_oscillation_score(activity,returnFrequency=False,prominence=0.05):
    rawScore, frequency = neuron_oscillation_score_helper(activity,prominence)
    # normalize to sine wave of the same frequency and duration
    if rawScore == 0:
        score = 0
    else:
        refSinScore, _ = neuron_oscillation_score_helper(np.sin(2*np.pi*frequency*np.arange(len(activity))),prominence)
        refCosScore, _ = neuron_oscillation_score_helper(np.cos(2*np.pi*frequency*np.arange(len(activity))),prominence)
        refScore = np.max((refSinScore,refCosScore))
        score = rawScore / refScore

    if returnFrequency:
        return score, frequency
    else:
        return score

def sim_oscillation_score(R,activeMnIdxs,start=None,end=None,returnFrequency=False):
    """calculate oscillation score for a simulation"""
    if start is None:
        start = 0
    if end is None:
        end = -1

    if returnFrequency:
        neuronOscillationScores = []
        frequencies = []

        for j in activeMnIdxs:
            score, freq = neuron_oscillation_score(R[j][start:end],returnFrequency=True)
            neuronOscillationScores.append(score)
            frequencies.append(freq)

        return np.mean(neuronOscillationScores), np.nanmean(frequencies)
        
    else:
        neuronOscillationScores = [neuron_oscillation_score(R[j][start:end]) for j in activeMnIdxs] # scores for each neuron
        return np.mean(neuronOscillationScores) # average for the simulation
    
def sim_oscillation_score_old(R,activeMnIdxs,start=None,end=None,calculateFrequency=False):
    """calculate oscillation score for a simulation"""
    if start is None:
        start = 0
    if end is None:
        end = -1

    if calculateFrequency:
        neuronOscillationScores = []
        frequencies = []

        for j in activeMnIdxs:
            score, freq = neuron_oscillation_score_old(R[j][start:end],calculateFrequency=True)
            neuronOscillationScores.append(score)
            frequencies.append(freq)

        return np.mean(neuronOscillationScores), np.nanmean(frequencies)
        
    else:
        neuronOscillationScores = [neuron_oscillation_score_old(R[j][start:end]) for j in activeMnIdxs] # scores for each neuron
        return np.mean(neuronOscillationScores) # average for the simulation

def save_sim_data(Rs,params,saveDir=None,idx=None):

    # argDict = args.__dict__
    # paramFile = argDict.pop("paramFile")
    # name = os.path.splitext(os.path.split(paramFile)[1])[0]
    # name = os.path.splitext(paramFile.split("configs/")[1])[0]

    # saveStr = params["name"]
    # for arg in argDict:
    #     saveStr += f"_{arg}{argDict[arg]}"
    if saveDir is None:
        try:
            saveDir = f'/data/users/jkl/vnc-closedloop/results/jax/{params["screen"]}/{params["name"]}'
        except:
            saveDir = f'/data/users/jkl/vnc-closedloop/results/jax/{params["name"]}/{params["name"]}'
    
    
    
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    with open(f"{saveDir}/params.yaml","w") as outfile:
        yaml.dump(params,outfile)

    sparse.save_npz(f'{saveDir}/Rs.npz', sparse.COO.from_numpy(Rs))

    """
    for i, R in enumerate(Rs):
        print(f'save sim {i}')
        idxSaveDir = saveDir + 'R_idx' + str(i).zfill(4)

        if not os.path.exists(idxSaveDir):
            os.makedirs(idxSaveDir)
    
        print(f"{idxSaveDir}/R.npy")

        np.save(f"{idxSaveDir}/R.npy",R)

        if params["saveFigs"]:
            plt.matshow(R,aspect=0.05)
            plt.colorbar()
            plt.gcf().set_figheight(3)
            plt.savefig(f"{idxSaveDir}/R.svg")
            plt.savefig(f"{idxSaveDir}/R.png")
    """

def load_W(wPath):
    wExt = os.path.splitext(wPath)[1]

    if wExt == ".npy":
        W = np.load(wPath)
    elif wExt == ".csv":
        W = pd.read_csv(wPath).drop(columns="bodyId_pre").to_numpy().astype(float)
    else:
        raise ValueError("Cannot read W file type.")
        # TODO add more file types that can be read

    return W

def load_wTable(dfPath):
    dfExt = os.path.splitext(dfPath)[1]
    if dfExt == ".pkl":
        wTable = pd.read_pickle(dfPath)
    elif dfExt == ".csv":
        wTable = pd.read_csv(dfPath,index_col=0)
    else:
        raise ValueError("Cannot read wTable file type.")
        # TODO add more file types that can be read

    return wTable



def setup_sim_from_params(params):
    """Sets up a sim from params typical of a config file and returns but does not run it. This gives you the opportunity to modify the params if you want."""
    # TODO I could consider parsing out different input formats based on file extension. For now I'll just assume all the same format.

    W = load_W(params["wPath"])
    wTable = load_wTable(params["dfPath"])
    

    
    #sim.set_time(params["T"],params["dt"])

    #TODO: Should be able to handle if we arent iterating over seeds
    nSims = len(params["seed"])
    print(nSims)

    stimNeurons = np.array(params["stimNeurons"])
    if stimNeurons.ndim == 0:
        stimNeurons = np.expand_dims(stimNeurons,0)

    #sim.set_input(sim.pulse_input(params["pulseStart"],params["pulseEnd"],params["stimI"],stimNeurons))

    
    """
    Wt = np.transpose(W) # correct orientation for matrix multiplication
    
    # Reweight W
    Wt_exc = np.maximum(Wt,0)
    Wt_inh = np.minimum(Wt,0)
    W = params["excitatoryMultiplier"]*Wt_exc + params["inhibitoryMultiplier"]*Wt_inh
    """
        

    nNeurons = len(W[0])

    params_arr = vncNetTest.generate_params_from_config(wTable, params, nSims, nNeurons, stimNeurons)
    return W, params_arr, nSims, stimNeurons


def run_sim_from_params(params):

    Ws, params_arr, nSims, stimNeurons = setup_sim_from_params(params)

    print('start run')

    if "adjustStimI" in params: # in here for now for older configs without "adjustStimI"
        if params["adjustStimI"]:

            print('stim adjust')

            Rs, stimIs = vncNetTest.run_with_stim_adjustment(Ws, params_arr, nSims, stimNeurons,
                                                                 maxIters=params["maxIters"],nActiveUpper=params["nActiveUpper"],
                                                                 nActiveLower=params["nActiveLower"],nHighFrUpper=params["nHighFrUpper"])
            # this will adjust stimulus strength. since we know it's a pulse input (and right now I just have a single stimI) can just look for the max
            params["stimI"] = stimIs.tolist()

        else:
            Rs = vncNetTest.simulate(Ws, params_arr)
    else:
        Rs = vncNetTest.simulate(Ws, params_arr)

    return Rs, params_arr, params