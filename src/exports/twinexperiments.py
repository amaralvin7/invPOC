import numpy as np
import pickle

from src.constants import DPY
from src.unpacking import slice_by_species
from src.modelequations import evaluate_model_equations, get_layer_bounds
from src.exports.constants import *


def load_targets(priors_from):
    """Load state element values with which to generate pseudodata.

    Args:
        priors_from (str): Location from which to pick B2p and Bm2 priors. Can
        be NA (North Atlantic) or SP (Station P).

    Returns:
        targets (dict): Results from the inversion defined by priors_from, and
        by gamma and relative error values of 0.5.
    """
    results_path = '../../results/exports'
    with open(f'{results_path}/{priors_from}_0.5_0.5.pkl', 'rb') as pickled:
        targets = pickle.load(pickled)

    return targets


def generate_linear_solution(targets, n_tracer_elements):
    """Obtain estimates of the tracers with a least-squares approach.

    Uses linear formulations of the model equations, which require a
    first-order aggregation term and the assumption of perfectly-known particle
    production. DVM and residuals are ignored.

    Args:
        n_tracer_elements (int): Number of state elements that are tracers
        (i.e., POCS and POCL at every grid depth).
        targets (dict): Results from the inversion defined by priors_from, and
        by gamma and relative error values of 0.5.

    Returns:
        x (numpy.ndarray): State vector with generated state estiamtes.
    """

    A = np.zeros((n_tracer_elements, n_tracer_elements))
    b = np.zeros(n_tracer_elements)
    element_index = targets['state_elements']

    for i, element in enumerate(element_index[:n_tracer_elements]):

        tracer, layer = element.split('_')
        l = int(layer)
        zi, zim1 = get_layer_bounds(l, GRID)
        h = zi - zim1

        iPsi = element_index.index(f'POCS_{l}')
        iPli = element_index.index(f'POCL_{l}')

        B2 = 0.8 / DPY  # from Murnane 1994, JGR
        Bm2 = targets['params']['Bm2']['posterior'][l]
        Bm1s = targets['params']['Bm1s']['posterior'][l]
        Bm1l = targets['params']['Bm1l']['posterior'][l]
        Po = targets['params']['Po']['posterior']
        Lp = targets['params']['Lp']['posterior']
        ws = targets['params']['ws']['posterior'][l]
        wl = targets['params']['wl']['posterior'][l]

        if l != 0:
            iPsim1 = element_index.index(f'POCS_{l - 1}')
            iPlim1 = element_index.index(f'POCL_{l - 1}')
            wsm1 = targets['params']['ws']['posterior'][l - 1]
            wlm1 = targets['params']['wl']['posterior'][l - 1]

        if tracer == 'POCS':
            if l == 0:
                A[i, iPsi] = ws + (Bm1s + B2) * h
                A[i, iPli] = -Bm2 * h
                b[i] = Po * MLD
            else:
                A[i, iPsi] = ws + 0.5 * (Bm1s + B2) * h
                A[i, iPsim1] = -wsm1 + 0.5 * (Bm1s + B2) * h
                A[i, iPli] = -0.5 * Bm2 * h
                A[i, iPlim1] = -0.5 * Bm2 * h
                b[i] = Lp * Po * (np.exp(-(zim1 - MLD) / Lp)
                                  - np.exp(-(zi - MLD) / Lp))
        else:
            if l == 0:
                A[i, iPli] = wl + (Bm1l + Bm2) * h
                A[i, iPsi] = -B2 * h
            else:
                A[i, iPli] = wl + 0.5 * (Bm1l + Bm2) * h
                A[i, iPlim1] = -wlm1 + 0.5 * (Bm1l + Bm2) * h
                A[i, iPsi] = -0.5 * B2 * h
                A[i, iPsim1] = -0.5 * B2 * h

    x = np.linalg.solve(A, b)
    x = np.clip(x, 10**-10, None)

    return x


def generate_nonlinear_solution(targets):
    """Obtain estimates of the tracers with an iterative approach.

    Takes the previously generated solution to the linear model equations and
    uses it as a prior estimate in an iterative approach to obtain estimates of
    the model tracers from the nonlinear model equations that are considered in
    the real data inversions.

    Args:
        targets (dict): Results from the inversion defined by priors_from, and
        by gamma and relative error values of 0.5.

    Returns:
        x (numpy.ndarray): State vector with generated state estiamtes.
    """
    max_iterations = 20
    max_change_limit = 10**-6
    n_tracer_elements = len(targets['tracers']) * len(GRID)
    xk = generate_linear_solution(targets, n_tracer_elements)

    Po = targets['params']['Po']['posterior']
    Lp = targets['params']['Lp']['posterior']
    b = np.zeros(n_tracer_elements)

    for layer in LAYERS:
        l = int(layer)
        zi, zim1 = get_layer_bounds(l, GRID)
        if l == 0:
            b[l] = -Po * MLD
        else:
            b[l] = -Lp * Po * (np.exp(-(zim1 - MLD) / Lp) -
                               np.exp(-(zi - MLD) / Lp))

    tracers = targets['tracers']
    state_elements = targets['state_elements']
    equation_elements = targets['equation_elements']

    for _ in range(max_iterations):
        f, F = evaluate_model_equations(
            tracers, state_elements, equation_elements, xk, GRID, ZG,
            UMZ_START, MLD, targets=targets)
        xkp1 = np.linalg.solve(F, (F @ xk - f + b))
        change = np.abs((xkp1 - xk) / xk)
        if np.max(change) < max_change_limit:
            break
        xk = xkp1

    return xkp1


def generate_pseudodata(priors_from, targets):
    """Generate pseudodata to be used for the twin experiment.

    Args:
        priors_from (str): Location from which to pick B2p and Bm2 priors. Can
        be NA (North Atlantic) or SP (Station P).
        targets (dict): Results from the inversion defined by priors_from, and
        by gamma and relative error values of 0.5.

    Returns:
        pseudodata (dict): Data to be used for twin experiment.
    """
    targets = load_targets(priors_from)
    x = generate_nonlinear_solution(targets)

    pseudodata = targets['tracers'].copy()
    equation_elements = targets['equation_elements']

    for tracer, subkeys in pseudodata.items():
        subkeys.pop('posterior', None)
        subkeys.pop('posterior_e', None)
        re = subkeys['prior_e'] / subkeys['prior']
        subkeys['prior'] = slice_by_species(x, tracer, equation_elements)
        subkeys['prior_e'] = subkeys['prior'] * re

    return pseudodata
