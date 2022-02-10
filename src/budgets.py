#!/usr/bin/env python3
import sympy as sym
import numpy as np
from itertools import product

from src.constants import THICK, LAYERS

def eval_sym_expression(
    y, state_elements, Ckp1, tracers=[], residuals=[], params=[]):

    x_symbolic = list(y.free_symbols)
    x_numerical = []
    x_indices = []

    for x in x_symbolic:
        x_indices.append(state_elements.index(x.name))
        if '_' in x.name:  # if it varies with depth
            element, layer = x.name.split('_')
            layer = int(layer)
            if element in tracers:
                x_numerical.append(tracers[element]['posterior'][layer])
            elif element[1:] in residuals:
                x_numerical.append(residuals[element[1:]][layer][0])
            else:  # if it's a depth-varying parameter
                x_numerical.append(params[element]['posterior'][layer])
        else:  # if it's a depth-independent parameter
            x_numerical.append(params[x.name]['posterior'])

    variance_sym = 0  # symbolic expression for variance of y
    derivs = [y.diff(x) for x in x_symbolic]
    
    # sub-CVM corresponding to state elements in y
    cvm = Ckp1[np.ix_(x_indices, x_indices)]
    nrows, ncols = cvm.shape
       
    for (i, j) in product(range(nrows), range(ncols)):
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
    
    resids_sym = {r: {} for r in residuals}
    
    for r in resids_sym:
        profile = [sym.symbols(f'R{r}_{l}') for l in LAYERS]
        resids_sym[r]['EZ'] = np.sum(profile[:3])
        resids_sym[r]['UMZ'] = np.sum(profile[3:])
        for l in LAYERS:
            resids_sym[r][l] = profile[l]
    
    return resids_sym
       
def get_symbolic_inventories(tracers):
    
    inventories_sym = {t: {} for t in tracers}
    
    for t in tracers:  
        concentrations = [sym.symbols(f'{t}_{l}') for l in LAYERS]
        profile = [concentrations[0] * THICK[0]]  # mixed layer 
        for i, h in enumerate(THICK[1:], 1):  # all other layers
            avg_conc = np.mean([concentrations[i], concentrations[i-1]])
            profile.append(avg_conc * h)
        inventories_sym[t]['EZ'] = np.sum(profile[:3])
        inventories_sym[t]['UMZ'] = np.sum(profile[3:])
        for l in LAYERS:
            inventories_sym[t][l] = profile[l]
        
    return inventories_sym

def integrate_by_zone(symbolic, state_elements, Ckp1, **state_element_types):
    
    integrated = {k: {} for k in symbolic}

    for (k, z) in product(integrated, ('EZ', 'UMZ')):
        y = symbolic[k][z]
        integrated[k][z] = eval_sym_expression(
            y, state_elements, Ckp1, **state_element_types)
    
    return integrated

def integrate_by_zone_and_layer(
    symbolic, state_elements, Ckp1, **state_element_types):
    
    integrated = integrate_by_zone(
        symbolic, state_elements, Ckp1, **state_element_types)

    for (k, l) in product(integrated, LAYERS):
        y = symbolic[k][l]
        integrated[k][l] = eval_sym_expression(
            y, state_elements, Ckp1, **state_element_types)
    
    return integrated