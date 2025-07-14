import time

import jax.numpy as jnp
import numpy as np
from jax import vmap, jit
from scipy.stats import truncnorm

import os
import pandas as pd
from jax.lax import cond


def sample_positive_truncnorm(mean, stdev, nSamples, nSims, seeds=None):
    """For sampling the neuron parameters from distribution."""

    a = (
        -mean / stdev
    )  # number of standard deviations to truncate on the left side -- needs to truncate at 0, so that is mean/stdev standard deviations away
    b = 100  # arbitrarily high number of standard deviations, basically we want infinity but can't pass that in I don't think
    samples = np.zeros([nSims, nSamples])
    for sim in range(nSims):

        current_seed = seeds[sim] if seeds is not None else None

        samples[sim] = truncnorm.rvs(
            a,
            b,
            loc=mean,
            scale=stdev,
            size=nSamples,
            random_state=np.random.RandomState(seed=current_seed),
        )

    return samples


def arrayInterpJax(x, xp, fp):
    return vmap(lambda f: jnp.interp(x, xp, f))(fp)


def rate_equation_half_tanh(t, R, args):
    inputs, pulseStart, pulseEnd, tau, weightedW, threshold, a, frCap = args

    pulse_active = (t >= pulseStart) & (t <= pulseEnd)
    I = inputs * pulse_active

    totalInput = I + jnp.dot(weightedW, R)
    activation = jnp.maximum(
        frCap * jnp.tanh((a / frCap) * (totalInput - threshold)), 0
    )

    return (activation - R) / tau


def set_sizes(sizes, a, threshold):
    """Should correspond to surface area. This method is going to need a lot of improvement, very much testing out"""

    normSize = np.nanmedian(sizes)  # np.nanmean(sizes)
    sizes[sizes.isna()] = normSize
    sizes[sizes == 0] = normSize  # TODO this isn't great
    sizes = sizes / normSize
    sizes = np.asarray(sizes)

    a = a / sizes
    threshold = threshold * sizes
    return a, threshold


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
        wTable = pd.read_csv(dfPath, index_col=0)
    else:
        raise ValueError("Cannot read wTable file type.")
        # TODO add more file types that can be read

    return wTable


def make_input(nNeurons, stimNeurons, stimI):
    input = np.zeros(nNeurons)
    input[stimNeurons] = stimI
    return jnp.array(input)
