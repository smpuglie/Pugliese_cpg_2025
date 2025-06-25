#from plot_utils import *
#from sim_utils import *
import sys, os

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # Use GPU 0
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"


import pandas as pd
from scipy import io
from scipy.integrate import solve_ivp # ODE solver
import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import time
from diffrax import diffeqsolve, ODETerm, SaveAt, Dopri5, PIDController
import jax.numpy as jnp
from jax import vmap
from jax.experimental import sparse
from jax.lax import cond
from jax.lax import div
import jax.random as random

import gc

from src.plot_utils import *

def load_minicircuit():
    W = np.load("./data/MANC weights/20241118_T1Lminicircuit/W_20241118_T1Lminicircuit.npy")
    wTable = pd.read_csv("./data/MANC weights/20241118_T1Lminicircuit/wTable_20241118_T1Lminicircuit.csv",index_col=0)
    return W, wTable

def load_fullMANC():
    Wpd = pd.read_csv("data/MANC weights/20231020_DNtoMN/W_20231020_DNtoMN_unsorted.csv")
    W = Wpd.drop(columns="bodyId_pre").to_numpy().astype(float) # changing the type to float is what allows it to work, TODO catch this and fix
    wTable = pd.read_csv("data/MANC weights/20231020_DNtoMN/wTable_20231020_DNtoMN_unsorted_withModules.csv",index_col=0)
    return W, wTable

def display_W(W,clim=20):
    """ plot adjacency matrix """
    fig = plt.figure(figsize=(3,3))
    im = plt.imshow(W,cmap="seismic",vmin=-clim,vmax=clim,interpolation="none")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("Postsynaptic neuron")
    plt.ylabel("Presynaptic neuron")
    plt.colorbar(label = "# synapses")

    return fig, im

def sample_positive_truncnorm(mean,stdev,nSamples,nSims,seeds=None):
    """Sample a truncated normal distribution that is constrained to be positive"""
    a = -mean/stdev # number of standard deviations to truncate on the left side -- needs to truncate at 0, so that is mean/stdev standard deviations away
    b = 100 # arbitrarily high number of standard deviations, basically we want infinity but can't pass that in I don't think
    samples = np.zeros([nSims, nSamples])
    for sim in range(nSims):

        # vector of randomly sampled values
        if seeds is not None:
            samples[sim] = truncnorm.rvs(a,b,loc=mean,scale=stdev,size=nSamples,random_state=np.random.RandomState(seed=seeds[sim]))
        else:
            samples[sim] = truncnorm.rvs(a,b,loc=mean,scale=stdev,size=nSamples,random_state=np.random.RandomState(seed=seeds))
             
    return samples

def set_sizes(sizes, a, threshold):
        """Should correspond to surface area. This method is going to need a lot of improvement, very much testing out"""

        normSize = np.nanmedian(sizes) #np.nanmean(sizes)
        sizes[sizes.isna()] = normSize
        sizes[sizes == 0] = normSize #TODO this isn't great
        sizes = sizes/normSize
        sizes = np.asarray(sizes)

        a = a / sizes
        threshold = threshold * sizes
        return a, threshold

def pulse_input(start,stop,amp,stimNeurons, nNeurons, nSims, tAxis, dt, iterStimNeurons = False):
        """Create a simple pulse/block input.
        - start and stop: pulse start and stop time (s)
        - amp: pulse amplitude
        - stimNeurons: neurons to receive the pulse

        Returns a nNeurons x time ndarray with [nNeurons,start:stop] set equal to amp, and 0 everywhere else
        """
        startF = round(start/dt)   # start frame
        stopF = round(stop/dt)     # stop frame
        input = np.zeros([nSims, nNeurons, len(tAxis)])

        print(f'stim neurons: {stimNeurons}')

        #TODO This can probably be vectorized
        for i in np.arange(nSims):


            #TODO: Can probably handle this better
            if type(amp) == int:
                if iterStimNeurons:
                    input[i][np.ix_([stimNeurons[i]],np.arange(startF,stopF))] = amp
                else:
                    input[i][np.ix_(stimNeurons,np.arange(startF,stopF))] = amp
            elif iterStimNeurons:
                input[i][np.ix_([stimNeurons[i]],np.arange(startF,stopF))] = amp[i]
            else:
                input[i][np.ix_(stimNeurons,np.arange(startF,stopF))] = amp[i]
                
                
        return input

def make_input(nNeurons, stimNeurons, stimI):
    input = np.zeros(nNeurons)
    input[stimNeurons] = stimI
    return jnp.array(input)

def arrayInterpJax(x, xp, fp):
    """Interpolate an array over one dimension, using jax.numpy, used in diffrax solver"""
    #TODO: All this is right now is a for loop of np.interp(). By putting it here I'm holding out for myselfto create a better version.
    return jnp.array([jnp.interp(x, xp, fp[i]) for i in range(len(fp))])



def rate_equation_half_tanh(t, R, args):
    #rate_sp = sparse.sparsify(rate_half_tanh_helper)
    #return sparse.BCOO.todense(rate_sp(t, R, args))

    inputs, pulseStart, pulseEnd, tau, weightedW, threshold, a, frCap = args

    #Pulse Input
    pred = jnp.logical_and(t >= pulseStart, t <= pulseEnd)
    true_fun = lambda: inputs
    false_fun = lambda: inputs*0
    I = cond(pred, true_fun, false_fun)

    totalInput = I + jnp.dot(weightedW, R)
    activation = jnp.maximum(frCap*jnp.tanh((a/frCap)*(totalInput-threshold)),0)

    return (activation - R) /tau

def rate_equation_relu(t, R, args):
    
    timePoints, inputs, tau, weightedW, threshold, a, frCap = args

    I = arrayInterpJax(t, timePoints, inputs) # interpolate to get input values
    totalInput = I + jnp.dot(weightedW,R)
    activation = jnp.minimum(jnp.maximum(a*(totalInput-threshold),0), frCap) #Testing capping relu output at frCap

    return (activation - R) /tau

def run(W, tAxis, T, dt, inputs, pulseStart, pulseEnd, tau, threshold, a, frCap, seed, stdvProp, r_tol=1e-7, a_tol=1e-9):
    """run the simulation"""
    

    """
    if len(clampedNeurons) == 0:
        R0 = jnp.zeros([len(W[0]),]) # initialize firing rates
    else:
        raise NameError("clamping rates is no longer implemented")
    """

   

    #W_rw = silence_neurons_jax(W, removeNeurons)
    #pred = stdvProp <= 0.01
    #true_fun = lambda: W
    #false_fun = resample_W(W, seed, stdvProp)
    #W_rw = cond(pred, true_fun, resample_W(W, seed, stdvProp))

    W_rw = resample_W(W, seed, stdvProp)

    del W
    gc.collect()

    R0 = jnp.zeros([len(W_rw[0]),]) # initialize firing rates
        
    #Solve using Diffrax, call activation function
    term = ODETerm(rate_equation_half_tanh)
    solver = Dopri5()
    saveat = SaveAt(ts=jnp.array(tAxis))
    odeSolution = diffeqsolve(term, solver, 0, T, dt, R0, 
                              args=(inputs,pulseStart,pulseEnd,tau,W_rw,threshold,a,frCap), saveat=saveat,
                              stepsize_controller=PIDController(rtol=r_tol, atol=a_tol), max_steps = 5000000)
        
    return jnp.transpose(odeSolution.ys)

def simulate(W, args):
    tAxis, T, dt, input, pulseStart, pulseEnd, tau, threshold, a, frCap, excSynapseMultiplier, inhSynapseMultiplier, seed, stdvProp = args
    print(f'stdvProp: {stdvProp}')
    print(f'seed: {seed}')
    seed = jnp.array(seed)
    
    W = jnp.array(W)
    tAxis = jnp.array(tAxis)
    #input = sparse.BCOO.fromdense(input, n_batch=input.ndim)
    input - jnp.array(input)
    tau = jnp.array(tau)
    threshold = jnp.array(threshold)
    a = jnp.array(a)
    frCap = jnp.array(frCap)
    
    # Reweight W
    Wt_exc = jnp.maximum(W,0)
    Wt_inh = jnp.minimum(W,0)
    W_rw = jnp.transpose(excSynapseMultiplier*Wt_exc + inhSynapseMultiplier*Wt_inh)

    del W, Wt_exc, Wt_inh
    gc.collect()
    
    vmap_axes = (None, None, None, None, None, None, None, 0, 0, 0, 0, 0, None)
    start_time = time.time()
    print('run')
    Rs = vmap(run, in_axes=vmap_axes)(W_rw, tAxis, T, dt, input, pulseStart, pulseEnd, tau, threshold, a, frCap, seed, stdvProp)
    end_time = time.time()
    print('Time to solve system vmap: ' + str(end_time-start_time))

    del W_rw, tAxis, T, dt, input, tau, threshold, a, frCap, excSynapseMultiplier, inhSynapseMultiplier, seed, stdvProp, args
    gc.collect()

    return np.array(Rs)


def silence_neurons(W, idxsToRemove):
    W[idxsToRemove,:] = 0
    W[:,idxsToRemove] = 0
    return W

def silence_neurons_jax(W, idxsToRemove):
    #return jnp.multiply(jnp.multiply(W, idxsToRemove), jnp.transpose(idxsToRemove))
    return jnp.multiply(jnp.transpose(jnp.multiply(W, idxsToRemove)), idxsToRemove)

def generate_params(wTable, nSims, nNeurons, stimNeurons, seeds):
    
    #Input
    stimI = 200 

    #Time axis
    T = 1
    dt = 0.001
    tAxis = np.arange(0,T,dt)

    #Generate nSims sets of parameters
    input = make_input(nNeurons, stimNeurons, stimI)
    pulseStart = 0.02
    pulseEnd = 0.9-dt #To keep consistency with old input method
    if seeds is not None:
        tau = sample_positive_truncnorm(0.02, 0.002, nNeurons, nSims, seeds=seeds[0])
        a = sample_positive_truncnorm(1, 0.1, nNeurons, nSims, seeds=seeds[1])
        threshold = sample_positive_truncnorm(6, 0.6, nNeurons, nSims, seeds=seeds[2])
        frCap = sample_positive_truncnorm(200, 10, nNeurons, nSims, seeds=seeds[3])
    else:
        tau = sample_positive_truncnorm(0.02, 0.002, nNeurons, nSims)
        a = sample_positive_truncnorm(1, 0.1, nNeurons, nSims)
        threshold = sample_positive_truncnorm(6, 0.6, nNeurons, nSims)
        frCap = sample_positive_truncnorm(200, 10, nNeurons, nSims)
    excSynapseMultiplier = 0.03
    inhSynapseMultiplier = 0.03
    

    a, threshold = set_sizes(wTable["size"], a, threshold)

    removeNeurons = jnp.full(nSims, 1)

    return (tAxis, T, dt, input, pulseStart, pulseEnd, tau, threshold, a, frCap, excSynapseMultiplier, inhSynapseMultiplier, removeNeurons)

def make_remove_arr(removeNeurons, nSims, nNeurons):
    rnSims = []
    rnIdxs = []

    for sim, neurons in enumerate(removeNeurons):
        rnSims = rnSims + [sim] * len(neurons)
        rnIdxs = rnIdxs + neurons

    return jnp.full([nSims, nNeurons], 1).at[(rnSims, rnIdxs)].set(0)

#Add random noise to W
def resample_W(W, seed, stdvProp):
    stdvs = W*stdvProp
    key = random.key(seed)
    samples = random.truncated_normal(key, lower = div(-1.0, stdvProp), upper = float('inf'), shape=W.shape)
    return W + stdvs*samples


def generate_params_from_config(wTable, params, nSims, nNeurons, stimNeurons):
    
    #Set time
    T = params["T"]
    dt = params["dt"]
    tAxis = np.arange(0,T,dt)

    #TODO: Shape handling could be better
    iterStimNeurons = False
    if "stimNeurons" in params["paramsToIterate"]:
        iterStimNeurons = True
    
    #TODO: handle iterStimNeurons
    input = make_input(nNeurons, stimNeurons, params["stimI"])
    pulseStart = params["pulseStart"]
    pulseEnd = params["pulseEnd"]-params["dt"] #To keep consistency with old input method

    """
    removeNeurons = jnp.full(nSims, 1)
    if "removeNeurons" in params["paramsToIterate"]:
        removeNeurons = make_remove_arr(params["removeNeurons"], nSims, nNeurons)
    elif len(params["removeNeurons"]) == nSims:
        print('hi')
    elif len(params["removeNeurons"]) > 0:
        print('hi2')
    """
    
    if params["seed"] is not None:

        #Generate seeds for each parameter
        #TODO: Maybe this can be vectorized?
        seeds = np.zeros([4, nSims], dtype=int)
        for sim in range(nSims):
            seeds[:,sim] = np.random.default_rng(params["seed"][sim]).integers(10000,size=4)

        #Sample parameters
        tau = sample_positive_truncnorm(params["tauMean"], params["tauStdv"], nNeurons, nSims, seeds=seeds[0])
        a = sample_positive_truncnorm(params["aMean"], params["aStdv"], nNeurons, nSims, seeds=seeds[1])
        threshold = sample_positive_truncnorm(params["thresholdMean"], params["thresholdStdv"], nNeurons, nSims, seeds=seeds[2])
        frCap = sample_positive_truncnorm(params["frcapMean"], params["frcapStdv"], nNeurons, nSims, seeds=seeds[3])
    else:

        #No seed given
        tau = sample_positive_truncnorm(params["tauMean"], params["tauStdv"], nNeurons, nSims)
        a = sample_positive_truncnorm(params["aMean"], params["aStdv"], nNeurons, nSims)
        threshold = sample_positive_truncnorm(params["thresholdMean"], params["thresholdStdv"], nNeurons, nSims)
        frCap = sample_positive_truncnorm(params["frcapMean"], params["frcapStdv"], nNeurons, nSims)

    excSynapseMultiplier = params["excitatoryMultiplier"]
    inhSynapseMultiplier = params["inhibitoryMultiplier"]

    #FANC has different size variable name
    if "size" in wTable:
        a, threshold = set_sizes(wTable["size"], a, threshold)
    else:
        a, threshold = set_sizes(wTable["surf_area_um2"], a, threshold)


    return (tAxis, T, dt, input, pulseStart, pulseEnd, tau, threshold, a, frCap, excSynapseMultiplier, inhSynapseMultiplier, params["seed"], params["stdvProp"])


def run_with_stim_adjustment(Ws, args, nSims, stimNeurons,
                             maxIters=10,clampedNeurons=[],clampedRates=None,nActiveUpper=500,nActiveLower=5,nHighFrUpper=100):
    
    #TODO: I just need to pull inputs
    tAxis, T, dt, input, tau, threshold, a, frCap, excSynapseMultiplier, inhSynapseMultiplier = args

    nextHighest = np.full(input.shape, None)
    nextLowest = np.full(input.shape, None)

    stimIs = np.zeros(nSims)

    goodStims = np.full(nSims, False)

    for i in range(maxIters):

        #TODO: Handle clamped neurons

        print(f'adjust iter {i}')

        args = (tAxis, T, dt, input, tau, threshold, a, frCap, excSynapseMultiplier, inhSynapseMultiplier)
        Rs = simulate(Ws, args)

        #Adjust stims for all simulations
        for sim in range(nSims):
            R = Rs[sim]

            nActive = sum(np.sum(R,1)>0)
            nHighFr = sum(np.max(R,1)>100)

            currInputs = input[sim].copy()
            print(f"sim {sim} max stimI = {np.max(currInputs)}")
            print(f'sim {sim} nActive: {nActive}')
            print(f'sim {sim} nHighFr: {nHighFr}')

            if (nActive > nActiveUpper) or (nHighFr > nHighFrUpper): # too strong
                if np.any(nextLowest[sim]) is None:
                    newInputs = currInputs/2
                else:
                    newInputs = (currInputs+nextLowest[sim])/2
                nextHighest[sim] = currInputs
                input[sim] = newInputs
            elif (nActive < nActiveLower): # too weak
                if np.any(nextHighest[sim]) is None:
                    newInputs = currInputs*2
                else:
                    newInputs = (currInputs+nextHighest[sim])/2
                nextLowest[sim] = currInputs
                input[sim] = newInputs
            elif not goodStims[sim]:
                goodStims[sim] = True

            stimIs[sim] = float(np.max(input[sim]))
        
        print()

        #Break loop if all the inputs are good
        if np.all(goodStims):
            break

                    
            
             
            
    return Rs, stimIs





def main():

    #Run 2 simulations on minicircuit
    W, wTable = load_minicircuit()

    nNeurons = len(W[0])
    nSims = 2

    seedOfSeeds = np.random.randint(100000)
    seeds = np.random.default_rng(seedOfSeeds).integers(10000,size=nSims)

    bdn2_idx = wTable.loc[wTable["bodyId"]==10093].index[0]
    stimNeurons = [bdn2_idx]

    args = generate_params(wTable, nSims, nNeurons, stimNeurons, seeds)
    
    R = simulate(W, args)

    

    R = np.array(R)
    print(R.shape)

    nonMns = wTable.loc[wTable["class"]!="motor neuron"]
    for i  in range(1):
        plot_R_traces(R[i, nonMns.index],nonMns,activeOnly=True)
        #plt.savefig(f'vmaptest{i}.png')
    


    #x1 = np.arange(9.0).reshape((3, 3))
    #x2 = np.arange(3.0)
    #print(x1, x2)
    #print(np.divide(x1, x2))
    
    #sim.set_a_distribution(1,0.1,seed=seeds[1])
    #sim.set_threshold_distribution(6,0.6,seed=seeds[2])
    #sim.set_synapse_multipliers(0.03,0.03)
    #sim.set_fr_distribution(200,10,seed=seeds[3])
    #sim.set_sizes(wTable["size"])
    






if __name__ == "__main__":
    main()