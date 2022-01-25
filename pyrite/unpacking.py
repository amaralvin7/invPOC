#!/usr/bin/env python3
import numpy as np
from constants import LAYERS

def unpack_state_estimates(
    tracers, residuals, params, state_elements, xhat, Ckp1):

    xhat_e = np.sqrt(np.diag(Ckp1))
    
    unpack_by_slicing(tracers, state_elements, xhat, xhat_e)
    unpack_by_slicing(residuals, state_elements, xhat, xhat_e, prefix='R')
    unpack_param_estimates(params, state_elements, xhat, xhat_e)

def unpack_by_slicing(dict, state_elements, xhat, xhat_e, prefix=''):
    
    for k in dict:
        kp = f'{prefix}{k}'
        dict[k]['posterior'] = slice_by_species(xhat, kp, state_elements)
        dict[k]['posterior_e'] = slice_by_species(xhat_e, kp, state_elements)

def slice_by_species(to_slice, species, state_elements):

    sliced = [to_slice[i] for i, e in enumerate(
        state_elements) if e.split('_')[0] == species]

    return sliced

def unpack_param_estimates(params, state_elements, xhat, xhat_e):
    
    for p in params:
        if params[p]['dv']:
            for l in LAYERS:
                layer_param = '_'.join([p, l])
                i = state_elements.index(layer_param)
                params[p]['posterior'][l] = xhat[i]
                params[p]['posterior_e'][l] = xhat_e[i]
        else:
            i = state_elements.index(p)
            params[p]['posterior'] = xhat[i]
            params[p]['posterior_e'] = xhat_e[i]