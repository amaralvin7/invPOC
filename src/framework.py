"""Building blocks of model machinery."""
from itertools import product

import numpy as np


def define_state_elements(
    tracers, params, layers, soft_constraint=False, flux_constraint=False):
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

    if flux_constraint:
        state_elements.append('ppzf')

    return state_elements


def define_equation_elements(tracers, layers, flux_constraint_layer=None):
    """Define which state elements have associated equations (tracers only)."""
    equation_elements = [f'{t}_{l}' for t, l in product(tracers, layers)]
    
    if flux_constraint_layer is not None:
        equation_elements.append(f'ppzf_{flux_constraint_layer}')

    return equation_elements


def define_prior_vector(tracers, params, layers, residuals=None, ppz_flux=None):
    """Build the vector of prior state estimates."""
    xo = []

    for t in tracers:
        xo.extend(tracers[t]['prior'])

    if residuals:
        for r in residuals:
            xo.extend(np.ones(len(layers)) * residuals[r]['prior'])

    for p in params:
        if params[p]['dv']:
            xo.extend(np.ones(len(layers)) * params[p]['prior'])
        else:
            xo.append(params[p]['prior'])

    if ppz_flux:
        xo.append(ppz_flux)

    return np.array(xo)


def define_cov_matrix(tracers, params, layers, residuals=None, ppz_flux=None):
    """Build the error covariance matrix of prior estimates."""
    Co = []

    for t in tracers:
        Co.extend(tracers[t]['prior_e']**2)

    if residuals:
        for r in residuals:
            Co.extend(np.ones(len(layers)) * residuals[r]['prior_e']**2)

    for p in params:
        if params[p]['dv']:
            Co.extend(np.ones(len(layers)) * params[p]['prior_e']**2)
        else:
            Co.append(params[p]['prior_e']**2)

    if ppz_flux:
        flux_error = 0.5
        Co.append((ppz_flux * flux_error)**2)

    return np.diag(Co)
