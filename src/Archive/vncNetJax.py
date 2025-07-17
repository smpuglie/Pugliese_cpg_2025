""" Functionality for running ventral nerve cord (VNC) network """

# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf

import sys, os


os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Use GPU 0
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import pandas as pd
from scipy import io
from scipy.integrate import solve_ivp # ODE solver
# import src.vncNet as vncNet
#import tensorflow as tfv
import numpy as np
# from numpy import random
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
#from numba import jit
import time
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, Dopri5, PIDController
import jax.numpy as jnp

def arrayInterpJax(x, xp, fp):
    """Interpolate an array over one dimension, using jax.numpy, used in diffrax solver"""
    #TODO: All this is right now is a for loop of np.interp(). By putting it here I'm holding out for myselfto create a better version.
    return jnp.array([jnp.interp(x, xp, fp[i]) for i in range(len(fp))])

def arrayInterp(x, xp, fp):
    """Interpolate an array over one dimension"""
    #TODO: All this is right now is a for loop of np.interp(). By putting it here I'm holding out for myselfto create a better version.
    return np.array([np.interp(x, xp, fp[i]) for i in range(len(fp))])

def create_time_axis(T,dt):
    """ create axis of time values given total time and time step"""
    tAxis = np.arange(0,T,dt)
    return tAxis

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

def sample_positive_truncnorm(mean,stdev,nSamples,seed=None):
    """Sample a truncated normal distribution that is constrained to be positive"""
    a = -mean/stdev # number of standard deviations to truncate on the left side -- needs to truncate at 0, so that is mean/stdev standard deviations away
    b = 100 # arbitrarily high number of standard deviations, basically we want infinity but can't pass that in I don't think
    return truncnorm.rvs(a,b,loc=mean,scale=stdev,size=nSamples,random_state=np.random.RandomState(seed=seed)) # vector of randomly sampled values

def activation_function(x, theta, a, rmax, form):
    """nonlinear activation function called by rate equation"""
    # as of Oct 2024 I'm adding a few options to play with. The default will be the rectified tanh

    if form=="half-tanh":
        return jnp.maximum(rmax*jnp.tanh((a/rmax)*(x-theta)),0)
    
    elif form=="relu":
        return jnp.maximum(a*(x-theta),0)
    
    elif form=="capped-relu":
        return jnp.minimum(jnp.maximum(a*(x-theta),0),rmax)
    
    elif form=="elu":
        term1 = jnp.maximum(a*(x-theta-1),0)
        term2 = jnp.minimum(a*(jnp.exp(x-theta-1)-1),0)
        return term1+term2+a
    
    elif form=="softplus":
        return a*jnp.log(1+jnp.exp(x-theta))

    elif form=="sigmoid":
        # I DOUBT THIS WILL WORK EVEN AT ALL
        # for now using 6a/rmax rather than 4a/rmax
        expTerm = (-6*a/rmax) * (x - (rmax/(2*a)) - theta)
        return rmax/(1+jnp.exp(expTerm))
    
    elif form=="linear":
        # just clowning around for now
        return a*(x-theta)

    else:
        raise ValueError("Invalid value for form")
        


# @staticmethod
# @jit(nopython=True)
def rate_equation(t, R, timePoints, inputs, tau, weightedW, threshold, a, frCap, form):

    # m = a/frCap
    I = arrayInterp(t, timePoints, inputs) # interpolate to get input values
    if form=="linear":
        totalInput = I+np.dot(weightedW,np.maximum(R,0))
    else:
        totalInput = I + np.dot(weightedW,R)

    # totalInput = I + np.dot(weightedW,R)

    R = (activation_function(totalInput, threshold, a, frCap, form) - R) /tau
    # R = (np.maximum(frCap*np.tanh(m*totalInput),0) - R) / tau
    return R

    


def rate_equation_diffrax(t, R, args):
    
    timePoints, inputs, tau, weightedW, threshold, a, frCap, form = args

    I = arrayInterpJax(t, timePoints, inputs) # interpolate to get input values
    totalInput = I + jnp.dot(weightedW,R)
    activation = activation_function(totalInput, threshold, a, frCap, form)

    return (activation - R) /tau

def rate_equation_diffrax_tanh(t, R, args):
    
    timePoints, inputs, tau, weightedW, threshold, a, frCap = args

    totalInput = arrayInterpJax(t, timePoints, inputs) + jnp.dot(weightedW,R)# interpolate to get input values
    

    #half-tanh activation function
    activation = jnp.maximum(frCap*jnp.tanh((a/frCap)*(totalInput-threshold)),0)
    
    return (activation - R) /tau

def rate_equation_diffrax_relu(t, R, args):
    
    timePoints, inputs, tau, weightedW, threshold, a, frCap = args

    totalInput = arrayInterpJax(t, timePoints, inputs) + jnp.dot(weightedW,R)# interpolate to get input values

    #relu activation function
    activation = jnp.maximum(inputs*(totalInput-timePoints),0)
    
    return (activation - R) /tau

# TODO should I split into Network and NetworkSimulation with one inheriting from the other?
def load_W_from_matlab(filename,varname):
    """Import connectivity weight matrix from a .mat file as a Tensor"""
    W = io.loadmat(filename)[varname]
    W[np.where(np.isnan(W))] = 0 # NOTE: comment out for reproducing old results
    # return tf.constant(W)
    return W


class Simulation():
    def __init__(self, W):
        """ initialize simulation from connectivity matrix W """
        self.W = W # orientation = [presynaptic, postsynaptic]
        self.nNeurons = len(W[0]) # number of neurons
        self.R = None
        self.inputs = None
        self.create_default_biophys_params()
        
    def create_default_biophys_params(self):
        """ called during __init__ so they don't have to be set manually, although they can if desired """
        self.tau = 5            # time constant
        self.threshold = 4      # threshold of weighted synaptic inputs that produces an increase in firing rate
        self.frCap = 100        # maximum firing rate
        
    def set_tau_distribution(self, mean, stdev, seed=None):
        self.tau = sample_positive_truncnorm(mean, stdev, self.nNeurons, seed=seed)
    
    def set_threshold_distribution(self, mean, stdev, seed=None):
        self.threshold = sample_positive_truncnorm(mean, stdev, self.nNeurons, seed=seed)
    
    def set_a_distribution(self, mean, stdev, seed=None):
        self.a = sample_positive_truncnorm(mean, stdev, self.nNeurons, seed=seed)

    def set_synapse_multipliers(self, excMultiplier, inhMultiplier):
        """scalar value by which to scale synaptic weights"""
        self.excSynapseMultiplier = excMultiplier
        self.inhSynapseMultiplier = inhMultiplier

    def set_sizes(self,sizes):
        """Should correspond to surface area. This method is going to need a lot of improvement, very much testing out"""

        normSize = np.nanmedian(sizes) #np.nanmean(sizes)
        sizes[sizes.isna()] = normSize
        sizes[sizes == 0] = normSize #TODO this isn't great
        sizes = sizes/normSize
        sizes = np.asarray(sizes)

        self.a = self.a / sizes
        self.threshold = self.threshold * sizes
        # self.threshold = self.threshold * sizes**2
    
    def set_fr_distribution(self, mean, stdev, seed=None):
        self.frCap = sample_positive_truncnorm(mean, stdev, self.nNeurons, seed=seed)

    def set_time(self,T,dt):
        """ set the time for your simulation by passing in the total time and time step """
        self.T = T
        self.dt = dt
        self.tAxis = create_time_axis(T,dt)
        if self.inputs is None:
            self.set_input(np.zeros((self.nNeurons,len(self.tAxis)))) # default = no input

    def pulse_input(self,start,stop,amp,stimNeurons):
        """Create a simple pulse/block input.
        - start and stop: pulse start and stop time (s)
        - amp: pulse amplitude
        - stimNeurons: neurons to receive the pulse

        Returns a nNeurons x time ndarray with [nNeurons,start:stop] set equal to amp, and 0 everywhere else
        """
        startF = round(start/self.dt)   # start frame
        stopF = round(stop/self.dt)     # stop frame
        input = np.zeros([self.nNeurons, len(self.tAxis)])
        input[np.ix_(stimNeurons,np.arange(startF,stopF))] = amp
        return input
    

    def ramp_input(self,start,stop,amp,stimNeurons):
        """Create a ramp input input.
        - start and stop: ramp start and stop time (s)
        - amp: ramp maximum value (assumed to start at 0)
        - stimNeurons: neurons to receive the ramp

        Returns a nNeurons x time ndarray with [nNeurons,start:stop] set equal to the ramp stimulus, and 0 everywhere else
        """
        startF = round(start/self.dt)   # start frame
        stopF = round(stop/self.dt)     # stop frame
        input = np.zeros([self.nNeurons, len(self.tAxis)])
        input[np.ix_(stimNeurons,np.arange(startF,stopF))] = np.linspace(0,amp,num=len(np.arange(startF,stopF)))
        return input

    def set_input(self,inputs):
        """ Input should be a numpy array with dimensions nNeurons x len(tAxis) """
        self.inputs = inputs

    def silence_neurons(self,idxsToRemove):
        """Set all pre- and post-synaptic weights to 0 for a group of neurons to remove them from the simulation"""
        # TODO this seemed like the most straightforward way to me because if you start np.delete()ing rows and columns then you
        # have to deal with reindexing and matching up to the wTable, which this code doesn't consider. However, I imagine it would
        # be nice to have a version that does this in order to be less data-intensive and potentially faster
        W = self.W.copy()
        W[idxsToRemove,:] = 0
        W[:,idxsToRemove] = 0

        self.W = W



    def run(self, r_tol=1e-7, a_tol=1e-9, clampedNeurons=[],clampedRates=None,form="half-tanh"):
        """run the simulation"""
        Wt = np.transpose(self.W) # correct orientation for matrix multiplication

        # I think this is faster than accessing self.var in the loop?
        # dt = self.dt
        tAxis = self.tAxis
        T = self.T
        inputs = self.inputs
        tau = self.tau
        threshold = self.threshold
        a = self.a
        frCap = self.frCap

        # Reweight W
        Wt_exc = np.maximum(Wt,0)
        Wt_inh = np.minimum(Wt,0)
        Wt_reweighted = self.excSynapseMultiplier*Wt_exc + self.inhSynapseMultiplier*Wt_inh

        if len(clampedNeurons) == 0:
            R0 = np.zeros([self.nNeurons,]) # initialize firing rates
        else:
            raise NameError("clamping rates is no longer implemented")
            # try:
            #     R = clampedRates
            # except:
            #     NameError("if you are clamping neurons you must provide clamped R")
            
        # R0 = np.zeros([self.nNeurons,])
        # Simulate using forward Euler
        # R = self.run_helper(R,nT,dt,tau,inputs,Wt_reweighted,threshold,a,frCap,clampedNeurons=clampedNeurons)

        # self.R = R

        #Solve using Diffrax, call activation function
        start_time = time.time()
        term = ODETerm(rate_equation_diffrax)
        solver = Dopri5()
        saveat = SaveAt(ts=jnp.array(tAxis))
        odeSolution = diffeqsolve(term, solver, 0, T, self.dt, R0, 
                                  args=(tAxis,inputs,tau,Wt_reweighted,threshold,a,frCap, form), saveat=saveat,
                                  stepsize_controller=PIDController(rtol=r_tol, atol=a_tol), max_steps=5000000)
        end_time = time.time()
        print('Time to solve ODEs Diffrax: ' + str(end_time-start_time))
        
        self.R = np.array(odeSolution.ys).T

    # # this doesn't get used anymore
    # @staticmethod
    # @jit(nopython=True)
    # def run_helper(R,nT,dt,tau,inputs,W_weighted,threshold,a,frCap,clampedNeurons):
    #     # synaptic weighting of W is done before passing into this function

    #     m = a/frCap # transform desired gain into the actual coefficient needed

    #     # TODO it may be cleaner to separate the activation into another function entirely. 
    #     # But I will need to redo the code for ODE solver anyway so I am holding off for now.
    #     for t in range(1,nT):
    #         prevR = R[:,t-1]
    #         prevR[clampedNeurons] = R[clampedNeurons,t] # TODO THIS IS BAD CODING, BAD MATH, ETC!
    #         activationValue = np.maximum(frCap*np.tanh(m*(inputs[:,t] + np.dot(W_weighted,prevR) - threshold)),0)

    #         R[:,t] = prevR + (dt/tau) * (activationValue - prevR)
    #     return R
    
    def run_with_stim_adjustment(self,maxIters=10,clampedNeurons=[],clampedRates=None,nActiveUpper=500,nActiveLower=5,nHighFrUpper=100):
        nextHighest = None
        nextLowest = None

        for i in range(maxIters):
            self.run(clampedNeurons=clampedNeurons,clampedRates=clampedRates)
            R = self.R

            nActive = sum(np.sum(R,1)>0)
            nHighFr = sum(np.max(R,1)>100)

            currInputs = self.inputs.copy()

            print(f"Run {i}")
            print(f"max stimI = {np.max(currInputs)}")
            print(f"nActive: {nActive}")
            print(f"nHighFr: {nHighFr}")

            if (nActive > nActiveUpper) or (nHighFr > nHighFrUpper): # too strong
                if nextLowest is None:
                    newInputs = currInputs/2
                else:
                    newInputs = (currInputs+nextLowest)/2
                nextHighest = currInputs
            elif (nActive < nActiveLower): # too weak
                if nextHighest is None:
                    newInputs = currInputs*2
                else:
                    newInputs = (currInputs+nextHighest)/2
                nextLowest = currInputs
            else:
                break

            self.set_input(newInputs)