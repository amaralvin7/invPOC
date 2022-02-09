#!/usr/bin/env python3
import numpy as np
from constants import LAYERS

def unpack_state_estimates(tracers, params, state_elements, xhat, Ckp1):

    xhat_e = np.sqrt(np.diag(Ckp1))
    
    tracer_estimates = unpack_tracers(tracers, state_elements, xhat, xhat_e)
    residual_estimates = unpack_resids(tracers, state_elements, xhat, xhat_e)
    param_estimates = unpack_params(params, state_elements, xhat, xhat_e)
    
    return tracer_estimates, residual_estimates, param_estimates

def slice_by_species(to_slice, species, state_elements):

    sliced = [to_slice[i] for i, e in enumerate(
        state_elements) if e.split('_')[0] == species]
    
    return sliced

def unpack_tracers(tracers, state_elements, xhat, xhat_e, prefix=''):
    
    tracer_estimates = {t: {} for t in tracers}
    
    for t in tracer_estimates:
        tracer_estimates[t]['posterior'] = slice_by_species(
            xhat, t, state_elements)
        tracer_estimates[t]['posterior_e'] = slice_by_species(
            xhat_e, t, state_elements)
        
    return tracer_estimates

def unpack_resids(tracers, state_elements, xhat, xhat_e, prefix=''):
    
    resid_estimates = {t: {} for t in tracers}
    
    for t in resid_estimates:
        posterior = slice_by_species(xhat, f'R{t}', state_elements)
        posterior_e = slice_by_species(xhat_e, f'R{t}', state_elements)
        for l in LAYERS:
            resid_estimates[t][l] = (posterior[l], posterior_e[l])
        
    return resid_estimates

def unpack_params(params, state_elements, xhat, xhat_e):
    
    param_estimates = {p: {} for p in params}
    
    for p in params:
        if params[p]['dv']:
            param_estimates[p]['posterior'] = slice_by_species(
                xhat, p, state_elements)
            param_estimates[p]['posterior_e'] = slice_by_species(
                xhat_e, p, state_elements)
        else:
            i = state_elements.index(p)
            param_estimates[p]['posterior'] = xhat[i]
            param_estimates[p]['posterior_e'] = xhat_e[i]
    
    return param_estimates