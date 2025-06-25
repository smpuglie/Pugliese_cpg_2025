# Copy and pasted from the hub_circuits branch which I'm sure is not good git protocol

import numpy as np
import yaml
import sys, os
import argparse
import matplotlib.pyplot as plt
from sim_utils import *
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser("Run multiple simulations in task spooler")
    parser.add_argument("paramFile")
    parser.add_argument("--idx",type=int)
    parser.add_argument("--jax",action=argparse.BooleanOptionalAction)
    # parser.add_argument("--seedIdx",type=int)
    return parser.parse_args()

def get_params(args):
    paramFile = args.paramFile
    runParams = load_from_yaml(paramFile)
    
    paramsToIterate = runParams["paramsToIterate"]

    simParams = runParams.copy()
    for paramName in simParams.keys():
        if paramName in paramsToIterate:
            simParams[paramName] = runParams[paramName][args.idx]

    return simParams

def main():
    args = get_args()
    params = get_params(args)

    if args.jax:
        params["jax"] = True
        sim, params = run_sim_from_params(params,useJax=True)
    else:
        sim, params = run_sim_from_params(params)

    save_sim_data(sim,params,idx=args.idx)

if __name__ == "__main__":
    main()