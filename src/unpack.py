"""Unpack the ATI solution into data containers."""
import numpy as np


def unpack_state_estimates(tracers, params, state_elements, x, C, layers,
                           soft_constraint=False):
    """Unpack solution estimates for all state elements.

    Args:
        tracers (dict): Data container for tracers.
        params (dict): Data container for model parameters.
        state_elements (list[str]): Names of state elements.
        x (np.ndarray): State vector.
        C (np.ndarray): Error covariance matrix.
        layers (tuple[int]): Integer indexes of model layers.

    Returns:
        tracer_estimates (dict): The unpacked tracer estimates.
        residual_estimates (dict): The unpacked residual estimates.
        parameter_estimates (dict): The unpacked model parameter estimates.
    """
    x_e = np.sqrt(np.diag(C))

    tracer_estimates = unpack_tracers(tracers, state_elements, x, x_e)
    param_estimates = unpack_params(params, state_elements, x, x_e)
    
    unpacked = [tracer_estimates, param_estimates]
    
    if soft_constraint:
        residual_estimates = unpack_resids(tracers, state_elements, x, x_e, layers)
        unpacked.append(residual_estimates)
    
    if 'ppzf' in state_elements:
        i = state_elements.index('ppzf')
        ppzf_estimate = (x[i], x_e[i])      
        unpacked.append(ppzf_estimate)

    return unpacked


def slice_by_species(to_slice, element, state_elements):
    """Get all values from a list that correspond to a given state element."""
    sliced = [to_slice[i] for i, e in enumerate(
        state_elements) if e.split('_')[0] == element]

    return sliced


def unpack_tracers(tracers, state_elements, x, x_e):
    """Unpack tracer estimates into a dictionary."""
    tracer_estimates = {t: {} for t in tracers}

    for t in tracer_estimates:
        tracer_estimates[t]['posterior'] = slice_by_species(
            x, t, state_elements)
        tracer_estimates[t]['posterior_e'] = slice_by_species(
            x_e, t, state_elements)

    return tracer_estimates


def unpack_resids(tracers, state_elements, x, x_e, layers):
    """Unpack residual estimates into a dictionary."""
    resid_estimates = {t: {} for t in tracers}

    for t in resid_estimates:
        posterior = slice_by_species(x, f'R{t}', state_elements)
        posterior_e = slice_by_species(x_e, f'R{t}', state_elements)
        for l in layers:
            resid_estimates[t][l] = (posterior[l], posterior_e[l])

    return resid_estimates


def unpack_params(params, state_elements, x, x_e):
    """Unpack model paramter estimates into a dictionary."""
    param_estimates = {p: {} for p in params}

    for p in params:
        if params[p]['dv']:  # if param varies with depth
            param_estimates[p]['posterior'] = slice_by_species(
                x, p, state_elements)
            param_estimates[p]['posterior_e'] = slice_by_species(
                x_e, p, state_elements)
        else:
            i = state_elements.index(p)
            param_estimates[p]['posterior'] = x[i]
            param_estimates[p]['posterior_e'] = x_e[i]

    return param_estimates


def merge_by_keys(merge_this, into_this):
    """Merge one dict into another. Useful for merging prior/posterior data."""
    for i in into_this:
        for j in merge_this[i]:
            into_this[i][j] = merge_this[i][j]
