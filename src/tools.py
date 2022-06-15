import numpy as np
from sympy import symbols

from src.budgets import eval_sym_expression

def merge_by_keys(merge_this, into_this):
    
    for i in into_this:
        for j in merge_this[i]:
            into_this[i][j] = merge_this[i][j]

def calculate_B2(grid, state_elements, Ckp1, tracers, params):

    params['B2'] = {'dv': True, 'posterior': [], 'posterior_e': []}

    for i in range(len(grid)):
            
        B2p, Psi = symbols(f'B2p_{i} POCS_{i}')
        if i == 0:
            Psa = Psi
        else:
            Psim1 = symbols(f'POCS_{i-1}')
            Psa = (Psi + Psim1)/2
        y = B2p*Psa
        est, err = eval_sym_expression(
            y, state_elements, Ckp1, tracers=tracers, params=params)
        
        params['B2']['posterior'].append(est)
        params['B2']['posterior_e'].append(err)

def normalized_state_residuals(xhat, xo, Co):

        x_resids = list((xhat - xo)/np.sqrt(np.diag(Co)))

        return x_resids

def get_layer_bounds(layer, grid):
    
    zi = grid[layer]
    zim1 = grid[grid.index(zi) - 1] if layer > 0 else 0
    
    return zi, zim1


def nonnegative_check(state_elements, xhat):
    
    indexes = [i for i, s in enumerate(state_elements) if 'R' not in s]
    nonresidual_estimates = [xhat[i] for i in indexes]
    nonnegative = all(i >= 0 for i in nonresidual_estimates)
    
    return nonnegative
            