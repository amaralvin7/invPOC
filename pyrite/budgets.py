#!/usr/bin/env python3
"""
to do:
- turn common parts of integrate_* into functions?
"""

from constants import LAYERS, GRID
from itertools import product
import sympy as sym
import numpy as np

def eval_sym_expression(
    y, state_elements, Ckp1, tracers=[], residuals=[], params=[]):

    x_symbolic = list(y.free_symbols)
    x_numerical = []
    x_indices = []

    for x in x_symbolic:
        x_indices.append(state_elements.index(x.name))
        if '_' in x.name:  # if it varies with depth
            element, layer = x.name.split('_')
            li = LAYERS.index(layer)
            if element in tracers:
                x_numerical.append(tracers[element]['posterior'][li])
            elif element[1:] in residuals:
                x_numerical.append(residuals[element[1:]]['posterior'][li])
            else:  # if it's a depth-varying parameter
                x_numerical.append(params[element][layer]['posterior'])
        else:  # if it's a depth-independent parameter
            x_numerical.append(params[x.name]['posterior'])

    variance_sym = 0  # symbolic expression for variance of y
    derivs = [y.diff(x) for x in x_symbolic]
    
    # sub-CVM corresponding to state elements in y
    cvm = Ckp1[np.ix_(x_indices, x_indices)]
    
    for i, row in enumerate(cvm):
        for j, _ in enumerate(row):
            if i > j:
                continue
            if i == j:
                variance_sym += (derivs[i]**2)*cvm[i, j]
            else:
                variance_sym += 2*derivs[i]*derivs[j]*cvm[i, j]

    result = sym.lambdify(x_symbolic, y)(*x_numerical)
    variance = sym.lambdify(x_symbolic, variance_sym)(*x_numerical)
    error = np.sqrt(variance)

    return result, error

def get_symbolic_residuals(residuals):
    
    residuals_sym = {r: {} for r in residuals}
    
    for r in residuals_sym:
        # resids_sym[r]['profile'] = [sym.symbols(f'R{r}_{l}') for l in LAYERS]
        profile = [sym.symbols(f'R{r}_{l}') for l in LAYERS]
        residuals_sym[r]['EZ'] = np.sum(profile[:3])
        residuals_sym[r]['UMZ'] = np.sum(profile[3:])
    
    return residuals_sym

def integrate_resids_by_zone(residuals, residuals_sym, state_elements, Ckp1):
    
    integrated = {r: {} for r in residuals}
    
    for (r, z) in product(residuals, ('EZ', 'UMZ')):
        y = residuals_sym[r][z]
        integral, error = eval_sym_expression(
            y, state_elements, Ckp1, residuals=residuals)
        integrated[r][z] = integral
        integrated[r][f'{z}_e'] = error
    
    return integrated
        
def get_symbolic_inventories(tracers):
    
    inventories_sym = {t: {} for t in tracers}
    grid_with_surface = (0,) + GRID
    thickness = np.diff(grid_with_surface)
    
    for t in tracers:  
        concentrations = [sym.symbols(f'{t}_{l}') for l in LAYERS]
        profile = [concentrations[0] * thickness[0]]  # mixed layer 
        for i, h in enumerate(thickness[1:], 1):  # all other layers
            avg_conc = np.mean([concentrations[i], concentrations[i-1]])
            profile.append(avg_conc * h)
        inventories_sym[t]['profile'] = profile
        inventories_sym[t]['EZ'] = np.sum(profile[:3])
        inventories_sym[t]['UMZ'] = np.sum(profile[3:])
        
    return inventories_sym

def integrate_inventories(inventories_sym, state_elements, Ckp1, tracers):
    
    inventories = {tracer: {} for tracer in inventories_sym}
    
    for (tracer, z) in product(inventories, ('EZ', 'UMZ')):
        y = inventories_sym[tracer][z]
        integral, error = eval_sym_expression(
            y, state_elements, Ckp1, tracers=tracers)
        inventories[tracer][z] = integral
        inventories[tracer][f'{z}_e'] = error

    for tracer in inventories:
        inventories[tracer]['posterior'] = []
        inventories[tracer]['posterior_e'] = []
        for y in inventories_sym[tracer]['profile']:
            integral, error = eval_sym_expression(
                y, state_elements, Ckp1, tracers)
            inventories[tracer]['posterior'].append(integral)
            inventories[tracer]['posterior_e'].append(error)
    
    return inventories