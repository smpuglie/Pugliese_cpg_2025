import yaml
import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import src.vncNet as vncNet
# import src.vncNetJax as vncNetJax # TODO I haven't set up jax in my env on mycroft workstation. I wonder if there's a better way to package this for either/or?
import pandas as pd

def fancTypeToIdx(wTable,typeStr):
    return wTable[wTable["w_type"] == typeStr]["w_idx"].to_numpy()

def load_from_yaml(yamlPath):
    """load data from .yml file path"""
    with open(yamlPath,'r') as file:
        data = yaml.safe_load(file)
    return data

def save_config(params,dir="../../configs"):
    """save parameters to a .yaml file"""

    if not os.path.exists(dir):
        os.makedirs(dir)

    simName = params["name"]
    fullConfigPath = dir + f"/{simName}.yaml"
    
    with open(fullConfigPath,"w") as outfile:
        yaml.dump(params,outfile,sort_keys=False)

    return fullConfigPath

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
            if score > 0.2:
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

def save_sim_data(sim,params,saveDir=None,idx=None):

    # argDict = args.__dict__
    # paramFile = argDict.pop("paramFile")
    # name = os.path.splitext(os.path.split(paramFile)[1])[0]
    # name = os.path.splitext(paramFile.split("configs/")[1])[0]

    # saveStr = params["name"]
    # for arg in argDict:
    #     saveStr += f"_{arg}{argDict[arg]}"
    if saveDir is None:
        if params["useTsp"]:
            try:
                saveDir = f'../results/tsp/{params["screen"]}/{params["name"]}'
            except:
                saveDir = f'../results/tsp/{params["name"]}/{params["name"]}'
        else:
            try:
                saveDir = f'../results/{params["screen"]}/{params["name"]}'
            except:
                saveDir = f'../results/{params["name"]}/{params["name"]}'
    
    if idx is not None:
        saveDir = saveDir + '_idx' + str(idx).zfill(4)
    
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    
    print(f"{saveDir}/R.npy")

    R = sim.R
    np.save(f"{saveDir}/R.npy",R)
    with open(f"{saveDir}/params.yaml","w") as outfile:
        yaml.dump(params,outfile)

    if params["saveFigs"]:
        plt.matshow(R,aspect=prominence)
        plt.colorbar()
        plt.gcf().set_figheight(3)
        plt.savefig(f"{saveDir}/R.svg")
        plt.savefig(f"{saveDir}/R.png")


def setup_sim_from_params(params,useJax=False):
    """Sets up a sim from params typical of a config file and returns but does not run it. This gives you the opportunity to modify the params if you want."""
    # TODO I could consider parsing out different input formats based on file extension. For now I'll just assume all the same format.

    wPath = params["wPath"]
    wExt = os.path.splitext(wPath)[1]

    if wExt == ".npy":
        W = np.load(wPath)
    elif wExt == ".csv":
        W = pd.read_csv(wPath).drop(columns="bodyId_pre").to_numpy().astype(float)
    else:
        raise ValueError("Cannot read W file type.")
        # TODO add more file types that can be read

    dfPath = params["dfPath"]
    dfExt = os.path.splitext(dfPath)[1]
    if dfExt == ".pkl":
        wTable = pd.read_pickle(dfPath)
    elif dfExt == ".csv":
        wTable = pd.read_csv(dfPath,index_col=0)
    else:
        raise ValueError("Cannot read wTable file type.")
        # TODO add more file types that can be read

    if useJax:
        sim = vncNetJax.Simulation(W)
    else:
        sim = vncNet.Simulation(W)
    sim.set_time(params["T"],params["dt"])

    stimNeurons = np.array(params["stimNeurons"])
    if stimNeurons.ndim == 0:
        stimNeurons = np.expand_dims(stimNeurons,0)

    sim.set_input(sim.pulse_input(params["pulseStart"],params["pulseEnd"],params["stimI"],stimNeurons))
    
    removeNeurons = np.array(params["removeNeurons"])
    if removeNeurons.ndim == 0:
        removeNeurons = np.expand_dims(removeNeurons,0)
    if len(removeNeurons)>0:
        sim.silence_neurons(removeNeurons)

    if params["seed"] is not None:
        seeds = np.random.default_rng(params["seed"]).integers(10000,size=4)
        sim.set_tau_distribution(params["tauMean"],params["tauStdv"],seed=seeds[0])
        sim.set_a_distribution(params["aMean"],params["aStdv"],seed=seeds[1])
        sim.set_threshold_distribution(params["thresholdMean"],params["thresholdStdv"],seed=seeds[2])
        sim.set_fr_distribution(params["frcapMean"],params["frcapStdv"],seed=seeds[3])
    else:
        sim.set_tau_distribution(params["tauMean"],params["tauStdv"])
        sim.set_a_distribution(params["aMean"],params["aStdv"])
        sim.set_threshold_distribution(params["thresholdMean"],params["thresholdStdv"])
        sim.set_fr_distribution(params["frcapMean"],params["frcapStdv"])

    sim.set_synapse_multipliers(params["excitatoryMultiplier"],params["inhibitoryMultiplier"])

    try:
        sim.set_sizes(wTable["size"])
    except:
        sim.set_sizes(wTable["surf_area_um2"])

    return sim


def run_sim_from_params(params,useJax=False):

    sim = setup_sim_from_params(params,useJax)

    if "adjustStimI" in params: # in here for now for older configs without "adjustStimI"
        if params["adjustStimI"]:
            sim.run_with_stim_adjustment(maxIters=params["maxIters"],nActiveUpper=params["nActiveUpper"],nActiveLower=params["nActiveLower"],
                                         nHighFrUpper=params["nHighFrUpper"])
            
            # this will adjust stimulus strength. since we know it's a pulse input (and right now I just have a single stimI) can just look for the max
            params["stimI"] = float(np.max(sim.inputs))

        else:
            sim.run()
    else:
        sim.run()

    return sim, params