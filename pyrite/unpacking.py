#!/usr/bin/env python3
import numpy as np
from constants import LAYERS
from tools import slice_by_species
"""
to do:
- get rid of letter labels for param estimates
"""

def unpack_state_estimates(tracers, params, state_elements, xhat, Ckp1):

    xhat_e = np.sqrt(np.diag(Ckp1))
    
    tracer_estimates = unpack_by_slicing(tracers, state_elements, xhat, xhat_e)
    residual_estimates = unpack_by_slicing(
        tracers, state_elements, xhat, xhat_e, prefix='R')
    param_estimates = unpack_param_estimates(
        params, state_elements, xhat, xhat_e)
    
    return tracer_estimates, residual_estimates, param_estimates

def unpack_by_slicing(dictionary, state_elements, xhat, xhat_e, prefix=''):
    
    sliced = {k: {} for k in dictionary}
    
    for k in sliced:
        kp = f'{prefix}{k}'
        sliced[k]['posterior'] = slice_by_species(xhat, kp, state_elements)
        sliced[k]['posterior_e'] = slice_by_species(xhat_e, kp, state_elements)
        
    return sliced

def unpack_param_estimates(params, state_elements, xhat, xhat_e):
    
    param_estimates = {p: {} for p in params}
    
    for p in params:
        if params[p]['dv']:
            param_estimates[p]['posterior'] = {}
            param_estimates[p]['posterior_e'] = {}
            for l in LAYERS:
                layer_param = '_'.join([p, l])
                i = state_elements.index(layer_param)
                param_estimates[p]['posterior'][l] = xhat[i]
                param_estimates[p]['posterior_e'][l] = xhat_e[i]
        else:
            i = state_elements.index(p)
            param_estimates[p]['posterior'] = xhat[i]
            param_estimates[p]['posterior_e'] = xhat_e[i]
    
    return param_estimates