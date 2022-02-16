#!/usr/bin/env python3
import numpy as np
from itertools import product

def define_state_elements(tracers, params, layers):

    state_elements = []

    for t, l in product(tracers, layers):
        state_elements.append(f'{t}_{l}')  # POC concentrations
        state_elements.append(f'R{t}_{l}')  # residuals for model equations
    
    # put all residuals after all concentrations
    state_elements = state_elements[::2] + state_elements[1::2]

    for p in params:
        if params[p]['dv']:
            state_elements.extend([f'{p}_{l}' for l in layers])
        else:
            state_elements.append(f'{p}')
    
    return state_elements

def define_equation_elements(tracers, layers):

    equation_elements = [f'{t}_{l}' for t, l in product(tracers, layers)]
    
    return equation_elements

def define_prior_vector(tracers, residuals, params, layers):

    xo = []

    for t in tracers:
        xo.extend(tracers[t]['prior'])

    for r in residuals:
        xo.extend(np.ones(len(layers)) * residuals[r]['prior'])

    for p in params:
        if params[p]['dv']:
            xo.extend(np.ones(len(layers)) * params[p]['prior'])
        else:
            xo.append(params[p]['prior'])

    return np.array(xo)
        
def define_cov_matrix(tracers, residuals, params, layers):

    Co = []

    for t in tracers:
        Co.extend(tracers[t]['prior_e']**2)
    
    for r in residuals:
        Co.extend(np.ones(len(layers)) * residuals[r]['prior_e']**2)

    for p in params:
        if params[p]['dv']:
            Co.extend(np.ones(len(layers)) * params[p]['prior_e']**2)
        else:
            Co.append(params[p]['prior_e']**2)

    return np.diag(Co)
