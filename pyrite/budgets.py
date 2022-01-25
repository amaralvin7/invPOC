#!/usr/bin/env python3
from constants import LAYERS
from itertools import product
import sympy as sym
import numpy as np

# def integrate_residuals(residuals):

#     zone_dict = {'EZ': self.zone_names[:3], 'UMZ': self.zone_names[3:]}

#     int_resids_sym = {}

#     for t in self.tracer_names:
#         int_resids_sym[t] = {}
#         for sz, zones in zone_dict.items():
#             to_integrate = 0
#             for z in zones:
#                 int_resids_sym[t][z] = sym.symbols(f'R{t}_{z}')
#                 to_integrate += sym.symbols(f'R{t}_{z}')
#             int_resids_sym[t][sz] = to_integrate
#             run.integrated_resids[t][sz] = self.eval_symbolic_func(
#                 run, to_integrate)

#     return integrated_residuals_sym

# def create_layer_zone_dict():
    
#     dictionary = dict.fromkeys(LAYERS)
#     dictionary['EZ'] = None
#     dictionary['UMZ'] = None
    
#     return dictionary

def eval_sym_expression(y, state_elements, tracers, residuals, params, Ckp1):

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

def integrate_residuals_by_zone(
    residuals, residuals_sym, state_elements, tracers, params, Ckp1):
    
    for (r, z) in product(residuals, ('EZ', 'UMZ')):
        y = residuals_sym[r][z]
        integral, error = eval_sym_expression(
            y, state_elements, tracers, residuals, params, Ckp1)
        residuals[r][z] = integral
        residuals[r][f'{z}_e'] = error
        