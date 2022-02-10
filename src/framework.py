#!/usr/bin/env python3
import numpy as np
from itertools import product

from src.constants import GRID, LAYERS

def define_state_elements(tracers, params):

    state_elements = []

    for t, l in product(tracers, LAYERS):
        state_elements.append(f'{t}_{l}')  # POC concentrations
        state_elements.append(f'R{t}_{l}')  # residuals for model equations
    
    # put all residuals after all concentrations
    state_elements = state_elements[::2] + state_elements[1::2]

    for p in params:
        if params[p]['dv']:
            state_elements.extend([f'{p}_{l}' for l in LAYERS])
        else:
            state_elements.append(f'{p}')
    
    return state_elements

def define_equation_elements(tracers):

    equation_elements = [f'{t}_{l}' for t, l in product(tracers, LAYERS)]
    
    return equation_elements

def define_prior_vector(tracers, residuals, params):

    xo = []

    for t in tracers:
        xo.extend(tracers[t]['prior'])

    for r in residuals:
        xo.extend(np.ones(len(GRID)) * residuals[r]['prior'])

    for p in params:
        if params[p]['dv']:
            xo.extend(np.ones(len(GRID)) * params[p]['prior'])
        else:
            xo.append(params[p]['prior'])

    return np.array(xo)
        
def define_cov_matrix(tracers, residuals, params):

    Co = []

    for t in tracers:
        Co.extend(tracers[t]['prior_e']**2)
    
    for r in residuals:
        Co.extend(np.ones(len(GRID)) * residuals[r]['prior_e']**2)

    for p in params:
        if params[p]['dv']:
            Co.extend(np.ones(len(GRID)) * params[p]['prior_e']**2)
        else:
            Co.append(params[p]['prior_e']**2)

    return np.diag(Co)
