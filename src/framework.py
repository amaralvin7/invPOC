"""Building blocks of model machinery."""
from itertools import product

import numpy as np


def define_state_elements(
    tracers, params, layers, soft_constraint=False, Th_fluxes=None):
    """Build a list of string identifiers for all state elements."""
    state_elements = []

    for t, l in product(tracers, layers):
        state_elements.append(f'{t}_{l}')  # POC concentrations

    if soft_constraint:
        for t, l in product(tracers, layers):
            state_elements.append(f'R{t}_{l}')  # residuals in model equations

    for p in params:
        if params[p]['dv']:
            state_elements.extend([f'{p}_{l}' for l in layers])
        else:
            state_elements.append(f'{p}')

    if Th_fluxes is not None:
        add_Th_elements(state_elements, Th_fluxes)

    return state_elements


def define_equation_elements(tracers, layers, Th_fluxes=None):
    """Define which state elements have associated equations (tracers only)."""
    equation_elements = [f'{t}_{l}' for t, l in product(tracers, layers)]
    
    if Th_fluxes is not None:
        add_Th_elements(equation_elements, Th_fluxes)

    return equation_elements


def add_Th_elements(element_list, Th_fluxes):

    for _, r in Th_fluxes.iterrows(): 
        element_list.append(f'Y_{int(r["layer"])}')
        
            
def define_prior_vector(tracers, params, layers, residuals=None, Th_fluxes=None):
    """Build the vector of prior state estimates."""
    xo = []

    for t in tracers:
        xo.extend(tracers[t]['prior'])

    if residuals is not None:
        for r in residuals:
            xo.extend(np.ones(len(layers)) * residuals[r]['prior'])

    for p in params:
        if params[p]['dv']:
            xo.extend(np.ones(len(layers)) * params[p]['prior'])
        else:
            xo.append(params[p]['prior'])

    if Th_fluxes is not None:
        xo.extend(Th_fluxes['flux'].values)

    return np.array(xo)


def define_cov_matrix(tracers, params, layers, residuals=None, Th_fluxes=None):
    """Build the error covariance matrix of prior estimates."""
    Co = []

    for t in tracers:
        Co.extend(tracers[t]['prior_e']**2)

    if residuals is not None:
        for r in residuals:
            Co.extend(np.ones(len(layers)) * residuals[r]['prior_e']**2)

    for p in params:
        if params[p]['dv']:
            Co.extend(np.ones(len(layers)) * params[p]['prior_e']**2)
        else:
            Co.append(params[p]['prior_e']**2)

    if Th_fluxes is not None:
        Co.extend([f**2 for f in Th_fluxes['flux'].values])

    return np.diag(Co)
