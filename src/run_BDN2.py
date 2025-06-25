import pandas as pd
import yaml
import numpy as np
from datetime import date
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # Use GPU 0
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import matplotlib.pyplot as plt
from scipy import signal
#from src.sim_utils_Jax import *
from src.sim_utils_test import *
from src.plot_utils import *
import src.sim_utils as sim_utils

def load_params(paramPath):
    with open(paramPath, 'r') as file:
        defaultParams = yaml.safe_load(file)
    return defaultParams

def save_str(fanc = False):
    if fanc:
        simStr = 'fancBDN2activation'
    else:
        simStr = 'mancBDN2activation'
    dateStr = date.today().strftime(format="%Y%m%d")
    return dateStr+"_"+simStr

def getParams(batchNum, nSims, paramPath = "/data/users/jkl/vnc-closedloop/configs/20250122_mancBDN2activation.yaml", fanc=False):
    params = load_params(paramPath)
    saveStr = save_str(fanc=fanc)
    params["screen"] = saveStr
    if batchNum < 10:
        params["name"] = saveStr + f'_0{batchNum}'
    else:
        params["name"] = saveStr + f'_{batchNum}'

    
    params["seed"] = np.random.randint(100000,size=nSims).tolist()
    #params["aMean"] = 0.8
    #params["thresholdMean"] = 7.5

    return params

def save_params(params):
    saveDir = f"/data/users/jkl/vnc-closedloop/configs/{params["screen"]}"

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    fullConfigPath = saveDir + f"/{params["name"]}.yaml"
    with open(fullConfigPath,"w") as outfile:
        yaml.dump(params,outfile)

def runBDN2():
    nSims = 100

    fanc = True

    if fanc:
        paramPath = '/data/users/jkl/vnc-closedloop/configs/20250227_fancBDN2activation.yaml'
    else:
        paramPath = '/data/users/jkl/vnc-closedloop/configs/20250122_mancBDN2activation.yaml'
    for i in range(10):
        print(f'running batch {i}')
        params = getParams(i, nSims, paramPath=paramPath, fanc=fanc)
        save_params(params)
        Rs, params_arr, params = run_sim_from_params(params)
        save_sim_data(Rs,params)

def gain_sweep():
    nSims = 100
    for i in range(15):
        a = (i+1)/100
        params = getParams(a, nSims)
        params["a132"] = a
        save_params(params)

        #Run Sim
        Rs, params_arr, params = run_sim_from_params(params)
        save_sim_data(Rs,params)

def o_score(R):
    start = 0
    end = int(0.5//0.005)
    print(end)
    wTable = pd.read_csv("../data/MANC weights/20231020_DNtoMN/wTable_20231020_DNtoMN_unsorted_withModules.csv",index_col=0)
    mnIdxs = wTable.loc[wTable["class"]=="motor neuron"].index
    maxFrs = np.max(R,axis=1)
    activeMnIdxs = mnIdxs[np.where(maxFrs[mnIdxs]>0)]
    return sim_oscillation_score(R,activeMnIdxs,start,end)

def rerun_pruning_iter():
    j=0
    i=121
    resultsDir = '/data/users/jkl/vnc-closedloop/results'
    Rjax = sparse.load_npz(f'{resultsDir}/jax/20250218_mancBDN2pruning/20250218_mancBDN2pruning_iter{i}/Rs.npz').todense()[j]
    Rtsp = np.load(f'{resultsDir}/tsp/20250218_mancBDN2pruning_{j}/20250218_mancBDN2pruning_{j}_iter{i}/R.npy')

    with open(f'{resultsDir}/tsp/20250218_mancBDN2pruning_{j}/20250218_mancBDN2pruning_{j}_iter{i}/params.yaml', 'r') as file:
        tspParams = yaml.safe_load(file)
    with open(f'{resultsDir}/jax/20250218_mancBDN2pruning/20250218_mancBDN2pruning_iter{i}/params.yaml', 'r') as file:
        jaxParams = yaml.safe_load(file)

    jaxParams["stimI"] = [200 for i in range(jaxParams["nSims"])]
    tspParams["stimI"] = 400

    sim, tspParams = sim_utils.run_sim_from_params(tspParams)
    print(tspParams["stimI"])
    RsJax, params_arr, jaxParams = run_sim_from_params(jaxParams)
    print(jaxParams["stimI"])
    

    Rjax = RsJax[0]
    Rtsp = sim.R

    oScoreJax = o_score(Rjax)
    oScoreTsp = o_score(Rtsp)
    print(oScoreJax, oScoreTsp)

    print(np.sum((Rjax-Rtsp)**2))

def run_screen():

    loadDir = '/data/users/jkl/vnc-closedloop/configs/'
    screenStr = f'20250624_mancBDN2activation_W_shuffle'


    nBatches = 11
    for i in range(nBatches):
        print(f'running batch {i}')
        if i < 10:
            loadFile = f'{screenStr}_0{i}.yaml'
        else:
            loadFile = f'{screenStr}_{i}.yaml'
        paramPath = loadDir + screenStr + '/' + loadFile
        params = load_params(paramPath)

        #Run Sim
        Rs, params_arr, params = run_sim_from_params(params)
        save_sim_data(Rs,params)




def main():
    start = time.time()
    #gain_sweep()
    #rerun_pruning_iter()
    run_screen()
    #runBDN2()

    runtime = time.time() - start
    print(f'Total runtime: {runtime}')
    

if __name__ == "__main__":
    main()