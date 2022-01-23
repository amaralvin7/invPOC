#!/usr/bin/env python3
from constants import MLD, GAMMA, GRID, LAYERS
from itertools import product
import numpy as np

def define_state_elements(tracers, params):

    state_elements = []

    for t, l in product(tracers, LAYERS):
        state_elements.append(f'{t}_{l}')  # POC concentrations
        state_elements.append(f'R{t}_{l}')  # residuals for model equations
    
    # put all residuals after all concentrations
    state_elements = state_elements[::2] + state_elements[1::2]

    for p in params:
        if params[p]['dv']:
            for l in LAYERS:
                state_elements.append(f'{p}_{l}')
        else:
            state_elements.append(f'{p}')
    
    return state_elements

def define_equation_elements(tracers):

    equation_elements = [f'{t}_{l}' for t, l in product(tracers, LAYERS)]
    
    return equation_elements

def define_prior_vector(tracers, params):

    xo = []

    for t in tracers:  # tracer concentrations
        xo.extend(tracers[t]['prior'])

    xo.extend(np.zeros(len(GRID) * len(tracers)))  # residuals

    for p in params:
        if params[p]['dv']:
            xo.extend(np.ones(len(GRID)) * params[p]['prior'])
        else:
            xo.append(params[p]['prior'])

    return np.array(xo)
        
def define_cov_matrix(tracers, params):

    Co = []

    for t in tracers:  # tracer concentrations
        Co.extend(tracers[t]['prior_e']**2)
    
    Co.extend(np.ones(len(GRID) * len(tracers))  # residuals
              * (GAMMA * params['P30']['prior'] * MLD)**2)

    for p in params:
        if params[p]['dv']:
            Co.extend(np.ones(len(GRID)) * params[p]['prior_e']**2)
        else:
            Co.append(params[p]['prior_e']**2)

    return np.diag(Co)
